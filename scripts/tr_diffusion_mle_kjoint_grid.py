#!/usr/bin/env python3
"""MLE (Poisson-NLL) reconstruction from k-rotation-joint low-dose projections.

Instead of averaging repeated same-angle measurements (SW-FBP,
`temporal_average_sequence`/`tr_diffusion_recon_swfbp.py`), every projection
from `k` consecutive rotations is kept at its own exact angle (wrapped mod
360deg, static-volume assumption) and fed directly into a Poisson-likelihood
reconstruction -- k times more distinct photon measurements than a single
rotation, exploited through the physics rather than through projection-domain
averaging. Compares, for k in {5,10,20,40}: plain Poisson-MLE gradient
descent, GD+TV, and ADMM+TV (grid search over TV weight), all warm-started
from the existing baseline UNet-with-context denoiser's own reconstruction,
against a native (full-dose) FBP ground truth.

  python scripts/tr_diffusion_mle_kjoint_grid.py --profile wunderkerze2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sdate.tr_diffusion import reconstruct as R
from sdate.tr_diffusion import mle_reconstruct as M
from sdate.tr_diffusion.profiles import DatasetProfile
from sdate.tr_diffusion.frames import MemmapFrameSource
from sdate.tr_naf.metrics import make_circular_mask, masked_psnr, masked_ssim

DEFAULT_CKPT = "/mydata/sdate/shared/checkpoints/tr_denoise_baseline_k1_dose005_poissonhead.pt"
# Shifted sharply downward from an earlier [0.01 .. 100] sweep: even the SMALLEST
# multiplier tested there already visibly over-smoothed the reconstruction (plain
# GD with NO TV at all showed genuinely sharp structure, just noisy -- the useful
# regularisation regime is well BELOW where that sweep started).
TV_GRID_MULT = [1e-3, 3e-3, 5e-3, 7e-3, 1e-2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="wunderkerze2")
    p.add_argument("--ckpt", default=DEFAULT_CKPT, help="baseline UNet-with-context denoiser checkpoint")
    p.add_argument("--frame_start", type=int, default=410_000,
                  help="first frame of the joint k-window -- ~1/10 into the profile's usable "
                       "[400000,500000) range, matching the portion used for prior reconstruction "
                       "movies (parameterised on purpose, explore freely)")
    p.add_argument("--k_list", type=int, nargs="+", default=[5, 10, 20, 40])
    p.add_argument("--det_bin", type=int, default=2)
    p.add_argument("--dose", type=float, default=0.05)
    p.add_argument("--noise_seed", type=int, default=12345)
    p.add_argument("--gt_window_deg", type=float, default=180.0)
    p.add_argument("--gd_iters", type=int, default=400)
    p.add_argument("--gd_lr", type=float, default=1e-3)
    p.add_argument("--admm_outer", type=int, default=25)
    p.add_argument("--admm_inner", type=int, default=8)
    p.add_argument("--admm_lr", type=float, default=1e-3)
    p.add_argument("--rho_mult", type=float, default=1.0,
                  help="multiplier on the auto-scaled ADMM rho (see mle_reconstruct.auto_admm_rho)")
    p.add_argument("--gd_tv_grid", type=float, nargs="+", default=None,
                  help="override TV_GRID_MULT for the GD+TV sweep only (e.g. a finer local search)")
    p.add_argument("--admm_tv_grid", type=float, nargs="+", default=None,
                  help="override TV_GRID_MULT for the ADMM+TV sweep only")
    p.add_argument("--skip_admm", action="store_true", help="skip the ADMM+TV sweep entirely")
    p.add_argument("--denoise_batch", type=int, default=32)
    p.add_argument("--denoise_workers", type=int, default=4)
    p.add_argument("--out_dir", default="/myhome/data/sdate/shared/time_resolved/mle_kjoint")
    p.add_argument("--tag", default=None)
    p.add_argument("--force", action="store_true", help="recompute even if a cached .npz exists")
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


def cache_path(out_dir: Path, tag: str, k, name: str) -> Path:
    kstr = f"k{k}" if k is not None else "shared"
    return out_dir / f"{tag}_{kstr}_{name}.npz"


def save_vol(path: Path, mu: torch.Tensor, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mu=mu.detach().cpu().numpy().astype(np.float32), **extra)


def load_vol(path: Path, device):
    d = np.load(path)
    return torch.from_numpy(d["mu"]).to(device=device, dtype=torch.float32)


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prof = DatasetProfile.load(a.profile)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"{prof.name}_mlekjoint_f{a.frame_start}"
    t0 = time.time()

    # --- real per-pixel flat/dark calibration (native crop resolution) ---
    dark_mov = Path(prof.mov_path).with_name(f"{prof.name}_darks.mov")
    flat_mov = Path(prof.mov_path).with_name(f"{prof.name}_flats.mov")
    dark_native = torch.from_numpy(R.load_calibration_average(
        str(dark_mov), prof.crop, prof.rot_axis_col, height=prof.height, width=prof.width,
    )).to(device=device, dtype=torch.float32)
    flat_native = torch.from_numpy(R.load_calibration_average(
        str(flat_mov), prof.crop, prof.rot_axis_col, height=prof.height, width=prof.width,
    )).to(device=device, dtype=torch.float32)
    print(f"flat/dark loaded: dark mean={dark_native.mean():.4g} flat mean={flat_native.mean():.4g}",
          flush=True)

    src = MemmapFrameSource(prof.memmap_path, prof.mov_path)
    win_length = lambda k: int(round(k * prof.period_360))
    win_max = win_length(max(a.k_list))
    vol_shape = (prof.crop[0] // a.det_bin, prof.crop[1] // a.det_bin, prof.crop[1] // a.det_bin)
    nslices, hplane = vol_shape[0], vol_shape[1]
    mask = make_circular_mask(hplane, hplane, device=device)
    mid_slice = nslices // 2
    gt_win = R.window_length_frames(a.gt_window_deg, prof.deg_per_frame)

    # --- baseline-denoiser reconstruction, computed once over the union window,
    #     then sliced per-k (window_k is a prefix of window_{k_max} for fixed frame_start) ---
    den_mm_path = out_dir / f"{tag}_denoised.f16"
    margin = 300
    if not Path(str(den_mm_path) + ".meta.npz").exists() or a.force:
        print("=== baseline denoiser (union window) ===", flush=True)
        first, n, meta = R.denoise_sequence(
            a.ckpt, prof.mov_path, prof.memmap_path, str(den_mm_path),
            frame_start=a.frame_start - margin, frame_end=a.frame_start + win_max + margin,
            dose=a.dose, noise_seed=a.noise_seed, batch=a.denoise_batch,
            num_workers=a.denoise_workers, deg_per_frame=prof.deg_per_frame,
            axis_col=prof.rot_axis_col, device=device,
        )
    else:
        m = np.load(str(den_mm_path) + ".meta.npz")
        first, n = int(m["first_index"]), int(m["num_frames"])
    assert first <= a.frame_start and first + n >= a.frame_start + win_max, (
        f"denoised memmap [{first}, {first + n}) doesn't cover the requested union window "
        f"[{a.frame_start}, {a.frame_start + win_max}) -- increase `margin`"
    )
    den_mm = np.memmap(den_mm_path, dtype=np.float16, mode="r", shape=(n, prof.crop[0], prof.crop[1]))

    results = {}  # (k, variant) -> dict(psnr, ssim, tv_weight, ...)
    for k in a.k_list:
        win_k = win_length(k)
        idx = np.arange(a.frame_start, a.frame_start + win_k)
        angles = R.projection_angles(idx, deg_per_frame=prof.deg_per_frame)

        # --- GT: native (dose=1) FBP, ~180deg window centred in THIS k-window (e.g. for
        # k=20, that's around revolution 10 of 20) -- not anchored at the window's start,
        # so it reflects the object's state at the joint reconstruction's effective "middle
        # of time" rather than possibly-mismatched dynamics right at the window edge.
        gt_center = a.frame_start + win_k / 2.0
        gt_lo = int(round(gt_center - gt_win / 2.0))
        gt_idx = np.arange(gt_lo, gt_lo + gt_win)
        gt_angles = R.projection_angles(gt_idx, deg_per_frame=prof.deg_per_frame)
        gt_path = cache_path(out_dir, tag, k, "gt")
        if gt_path.exists() and not a.force:
            gt_vol = load_vol(gt_path, device)
        else:
            gt_native = R.native_window_gpu(src, gt_idx, prof.crop, prof.rot_axis_col, device)
            gt_atten = R.counts_to_attenuation_flatdark(gt_native, dark_native, flat_native)
            gt_vol = R.reconstruct(gt_atten, gt_angles, det_bin=a.det_bin, method="fbp", device=device)
            save_vol(gt_path, gt_vol)
        print(f"k={k}: GT window [{gt_idx[0]},{gt_idx[-1]}] (centred at revolution {k / 2:g} of {k})",
              flush=True)

        native = R.native_window_gpu(src, idx, prof.crop, prof.rot_axis_col, device)
        gen = torch.Generator(device=device).manual_seed(a.noise_seed + a.frame_start)
        noisy_counts = R.noisy_window_gpu(native, a.dose, generator=gen)
        y_native = R.bin_detector(noisy_counts, a.det_bin)
        I0_native = R.bin_detector(flat_native.unsqueeze(0), a.det_bin)
        dark_b = R.bin_detector(dark_native.unsqueeze(0), a.det_bin)

        # plain-FBP-on-the-joint-sinogram baseline (no Poisson optimisation at all)
        joint_fbp_path = cache_path(out_dir, tag, k, "joint_fbp")
        if joint_fbp_path.exists() and not a.force:
            joint_fbp_vol = load_vol(joint_fbp_path, device)
        else:
            noisy_atten = R.counts_to_attenuation_flatdark(noisy_counts, dark_native, flat_native)
            joint_fbp_vol = R.reconstruct(noisy_atten, angles, det_bin=a.det_bin, method="fbp", device=device)
            save_vol(joint_fbp_path, joint_fbp_vol)

        # baseline-denoiser warm start (this k's prefix of the union denoised memmap)
        warm_path = cache_path(out_dir, tag, k, "warm_start")
        if warm_path.exists() and not a.force:
            mu0 = load_vol(warm_path, device)
        else:
            den_counts = R.denoised_window_gpu(den_mm, first, idx, device)
            den_atten = R.counts_to_attenuation_flatdark(den_counts, dark_native, flat_native)
            mu0 = R.reconstruct(den_atten, angles, det_bin=a.det_bin, method="fbp", device=device)
            save_vol(warm_path, mu0)
        print(f"k={k}: win={win_k} views, warm-start mu0 [{mu0.min():.4g},{mu0.max():.4g}]", flush=True)

        # baseline denoiser, but reconstructed from ONLY the GT's own (single-revolution)
        # window -- isolates "how much sharpness is lost going from full-dose GT to the
        # baseline denoiser at dose=0.05" on its own, decoupled from the k-window MLE question.
        baseline_gt_path = cache_path(out_dir, tag, k, "baseline_gt_window")
        if baseline_gt_path.exists() and not a.force:
            baseline_gt_vol = load_vol(baseline_gt_path, device)
        else:
            den_gt_counts = R.denoised_window_gpu(den_mm, first, gt_idx, device)
            den_gt_atten = R.counts_to_attenuation_flatdark(den_gt_counts, dark_native, flat_native)
            baseline_gt_vol = R.reconstruct(den_gt_atten, gt_angles, det_bin=a.det_bin,
                                            method="fbp", device=device)
            save_vol(baseline_gt_path, baseline_gt_vol)

        # auto-scaled TV weight reference (data-loss vs TV-loss gradient balance at mu0)
        from astra_torch.lamino import build_lamino_projector
        probe_layer = build_lamino_projector(
            vol_shape=vol_shape, det_shape=(y_native.shape[1], y_native.shape[2]),
            angles_deg=angles, lamino_angle_deg=0.0, device=device,
        )
        tv0 = M.auto_tv_weight(mu0, probe_layer, y_native, I0_native, dark_b, a.dose)
        rho = M.auto_admm_rho(mu0, tv0) * a.rho_mult
        print(f"k={k}: auto tv0={tv0:.4g}  auto rho={rho:.4g}", flush=True)

        def score(vol):
            dr = float(gt_vol[..., mask].max() - gt_vol[..., mask].min())
            return masked_psnr(gt_vol, vol, mask, dr), masked_ssim(gt_vol, vol, mask, dr)

        psnr, ssim = score(joint_fbp_vol)
        results[(k, "joint_fbp", None)] = dict(psnr=psnr, ssim=ssim)
        psnr, ssim = score(mu0)
        results[(k, "warm_start", None)] = dict(psnr=psnr, ssim=ssim)
        psnr, ssim = score(baseline_gt_vol)
        results[(k, "baseline_gt_window", None)] = dict(psnr=psnr, ssim=ssim)
        print(f"k={k} baseline_gt_window: PSNR {psnr:.2f} SSIM {ssim:.3f}", flush=True)

        # --- pure GD (no TV) ---
        p = cache_path(out_dir, tag, k, "gd")
        gd_vol = load_vol(p, device) if p.exists() and not a.force else None
        if gd_vol is None:
            gd_vol = M.gd_mle_reconstruct(
                y_native, angles, I0_native, dark_b, a.dose, vol_shape=vol_shape,
                tv_weight=0.0, mu_init=mu0, n_iters=a.gd_iters, lr=a.gd_lr,
                device=device, log_every=a.log_every,
            )
            save_vol(p, gd_vol)
        psnr, ssim = score(gd_vol)
        results[(k, "gd", None)] = dict(psnr=psnr, ssim=ssim)
        print(f"k={k} gd: PSNR {psnr:.2f} SSIM {ssim:.3f}", flush=True)

        # --- GD+TV grid (every point kept, not just the PSNR-argmax -- PSNR and visual
        # sharpness disagree badly at the high-TV end, see module docstring / smoke tests) ---
        gd_tv_vols = []
        gd_tv_grid = a.gd_tv_grid if a.gd_tv_grid is not None else TV_GRID_MULT
        for mult in gd_tv_grid:
            tvw = tv0 * mult
            p = cache_path(out_dir, tag, k, f"gdtv_{mult:g}")
            vol = load_vol(p, device) if p.exists() and not a.force else None
            if vol is None:
                vol = M.gd_mle_reconstruct(
                    y_native, angles, I0_native, dark_b, a.dose, vol_shape=vol_shape,
                    tv_weight=tvw, mu_init=mu0, n_iters=a.gd_iters, lr=a.gd_lr,
                    device=device, log_every=a.log_every,
                )
                save_vol(p, vol)
            psnr, ssim = score(vol)
            results[(k, "gd_tv", mult)] = dict(psnr=psnr, ssim=ssim, tv_weight=tvw)
            print(f"k={k} gd_tv mult={mult:g} (tv={tvw:.4g}): PSNR {psnr:.2f} SSIM {ssim:.3f}", flush=True)
            gd_tv_vols.append((mult, tvw, vol))

        # --- ADMM+TV grid (same treatment) ---
        admm_tv_vols = []
        admm_tv_grid = [] if a.skip_admm else (a.admm_tv_grid if a.admm_tv_grid is not None else TV_GRID_MULT)
        for mult in admm_tv_grid:
            tvw = tv0 * mult
            p = cache_path(out_dir, tag, k, f"admmtv_{mult:g}")
            vol = load_vol(p, device) if p.exists() and not a.force else None
            if vol is None:
                vol = M.admm_tv_mle_reconstruct(
                    y_native, angles, I0_native, dark_b, a.dose, vol_shape=vol_shape,
                    tv_weight=tvw, rho=rho, mu_init=mu0, n_outer=a.admm_outer,
                    inner_iters=a.admm_inner, inner_lr=a.admm_lr,
                    device=device, log_every=a.log_every,
                )
                save_vol(p, vol)
            psnr, ssim = score(vol)
            results[(k, "admm_tv", mult)] = dict(psnr=psnr, ssim=ssim, tv_weight=tvw, rho=rho)
            print(f"k={k} admm_tv mult={mult:g} (tv={tvw:.4g}): PSNR {psnr:.2f} SSIM {ssim:.3f}", flush=True)
            admm_tv_vols.append((mult, tvw, vol))

        def save_figure(rows, out_path, title):
            fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)), squeeze=False)
            mask_np = mask.cpu().numpy()
            gts = gt_vol[mid_slice].cpu().numpy()
            # outside the inscribed reconstruction circle, rays never fully determine the
            # volume (interior-tomography / truncation artifacts) -- blank it out for display
            # so it doesn't visually dominate or skew the diff-map colour scale.
            gts_disp = np.where(mask_np, gts, np.nan)
            vmin, vmax = np.nanpercentile(gts_disp, [1, 99])
            for i, (name, vol) in enumerate(rows):
                rec = vol[mid_slice].cpu().numpy()
                diff = rec - gts
                dmax = np.nanpercentile(np.where(mask_np, np.abs(diff), np.nan), 99) + 1e-6
                rec_disp = np.where(mask_np, rec, np.nan)
                diff_disp = np.where(mask_np, diff, np.nan)
                axes[i, 0].imshow(gts_disp, cmap="gray", vmin=vmin, vmax=vmax)
                axes[i, 1].imshow(rec_disp, cmap="gray", vmin=vmin, vmax=vmax)
                axes[i, 2].imshow(diff_disp, cmap="coolwarm", vmin=-dmax, vmax=dmax)
                # axis('off') would also hide the ylabel, so strip ticks/spines by hand instead
                for j, t in enumerate(["GT", name, "diff"]):
                    ax = axes[i, j]
                    ax.set_title(t, fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
                axes[i, 0].set_ylabel(name, fontsize=9)
            fig.suptitle(title)
            fig.tight_layout()
            fig.savefig(out_path, dpi=130)
            plt.close(fig)
            print(f"k={k}: wrote {out_path}", flush=True)

        title = f"k={k} ({win_k} views)  frame_start={a.frame_start}"
        save_figure(
            [("joint FBP", joint_fbp_vol), ("baseline (full k-window)", mu0),
             ("baseline (GT window only)", baseline_gt_vol), ("GD (no TV)", gd_vol)],
            out_dir / f"{tag}_k{k}_arms.png", title + " -- reference arms",
        )
        if gd_tv_vols:
            save_figure(
                [(f"GD+TV mult={mult:g}", vol) for mult, _, vol in gd_tv_vols],
                out_dir / f"{tag}_k{k}_tv_sweep_gd.png", title + " -- GD+TV sweep",
            )
        if admm_tv_vols:
            save_figure(
                [(f"ADMM+TV mult={mult:g}", vol) for mult, _, vol in admm_tv_vols],
                out_dir / f"{tag}_k{k}_tv_sweep_admm.png", title + " -- ADMM+TV sweep",
            )

    # --- summary ---
    summary = {"tag": tag, "profile": prof.name, "frame_start": a.frame_start,
              "det_bin": a.det_bin, "dose": a.dose, "minutes": round((time.time() - t0) / 60, 1),
              "results": [
                  {"k": k, "variant": v, "tv_mult": m, **res}
                  for (k, v, m), res in results.items()
              ]}
    summary_path = out_dir / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
