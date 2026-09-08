#!/usr/bin/env python3
"""Infer + reconstruct + score the T5 native-noise Gaussian-floor-loss
denoiser (checkpoint: tr_denoise_baseline_jointfbpctx_T5_native_poissonhead_
gaussianfloor.pt -- see project memory project-tr-diffusion-jointfbpctx and
nb_head.nb_nll_gaussian's docstring for why this checkpoint exists).

Unlike the dose=0.05 evaluations elsewhere in this project, native (dose=1,
unthinned) data has no independently-cleaner "ground truth" to score
against -- a single-180deg-window native FBP already uses the best available
photon statistics for one revolution; there is no synthetic thinning to
"undo". So this mirrors the k=21 joint-FBP baseline
(tr_diffusion_jointfbp_k_sweep.py) but joins NATIVE (unthinned) revolutions
instead of dose-thinned ones, giving a genuinely lower-noise structural
reference (temporal blur traded for photon-count redundancy) to score both
the single-window native floor and the T5-denoised reconstruction against.
Given that, validation here leans qualitative (movies + PNG stills), same as
every quantitative number this project produces gets backed by a visual --
just more so here since the "ground truth" itself is a soft reference.

Arms per window:
  * floor      -- single ~180deg-window native FBP, no denoising (what a
                  bare full-dose acquisition gives you today).
  * t5native   -- same window, but the CENTRAL frame stream comes from the
                  T5-context-conditioned Gaussian-floor-loss denoiser.
  * pseudo_gt  -- k=21 joined NATIVE revolutions (all kept at their own exact
                  wrapped angle, no averaging -- static-volume assumption),
                  single joint FBP. Low noise, blurred wherever the sample
                  moved across the window -- a reference, not a perfect truth.

Outputs (matching this project's established comparison format):
  * recon movie       -- pseudo_gt | floor | denoised, reconstructed slice over time
  * projection movie  -- GT(raw native) | denoised | noisy(extra native draw), raw frames over time
  * recon stills PNG  -- rows = a few spread-out windows, cols = pseudo_gt | floor | denoised | diff
  * projection stills PNG -- rows = a few spread-out frames, cols = raw | denoised | diff

Safe to re-run: every stage is skipped if its output already exists.

    python scripts/tr_diffusion_jointfbpctx_t5native_pipeline.py
"""
from __future__ import annotations

import os

os.environ["PATH"] = "/myhome/bin:" + os.environ.get("PATH", "")

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")

from sdate.tr_diffusion import reconstruct as R  # noqa: E402
from sdate.tr_diffusion.frames import MemmapFrameSource  # noqa: E402
from sdate.tr_diffusion.profiles import DatasetProfile  # noqa: E402
from sdate.tr_naf.metrics import make_circular_mask, masked_psnr, masked_ssim  # noqa: E402

DATA = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2"
CACHE = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache"
CKDIR = "/mydata/sdate/shared/checkpoints"
TAG = "jointfbpctx_T5_native_gaussianfloor"

MOV = f"{DATA}/212_Wunderkerze2.mov"
MEMMAP = f"{DATA}/frames_400k_500k.u16"

CKPT = f"{CKDIR}/tr_denoise_baseline_jointfbpctx_T5_native_poissonhead_gaussianfloor_full_snapshot.pt"
DENOISED_MM = f"{DATA}/denoised_212_Wunderkerze2_{TAG}.f16"

# bounded by the T5 native context-tap cache's coverage -- widened 2026-08-31 to give
# ~250 eval windows (was ~51) starting from the same place (centre 414899)
INFER_FRAME_START, INFER_FRAME_END = 412_000, 468_000
DOSE, NOISE_SEED = 1.0, 12345  # native: poisson_dose=1.0 for the posterior combination; no synthetic thinning
DET_BIN = 2
K_PSEUDO_GT = 21  # joined NATIVE revolutions for the low-noise reference (same k as the project's dose=0.05 floor)
GT_WINDOW_DEG = 180.0
N_STILLS = 5  # rows in each PNG, evenly spread across the eval range

RESULTS = Path(CACHE) / f"recon_results_{TAG}.npz"
SUMMARY = Path(CACHE) / f"recon_summary_{TAG}.json"
STILLS_NPZ = Path(CACHE) / f"recon_stills_{TAG}.npz"
RECON_MOVIE = Path(CACHE) / f"recon_{TAG}_vs_floor_vs_pseudogt.mov"
PROJ_MOVIE = Path(CACHE) / f"proj_{TAG}_vs_raw.mov"
RECON_PNG = Path(CACHE) / f"{TAG}_recon_stills.png"
PROJ_PNG = Path(CACHE) / f"{TAG}_proj_stills.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def stage_infer() -> None:
    if Path(str(DENOISED_MM) + ".meta.npz").exists():
        log(f"[skip] infer: denoised memmap already exists at {DENOISED_MM}")
        return
    assert Path(CKPT).exists(), f"checkpoint not found: {CKPT}"
    log(f"=== stage 1/5: denoise eval range with {TAG} checkpoint (native_noise=True) ===")
    t0 = time.time()
    R.denoise_sequence(
        CKPT, MOV, MEMMAP, DENOISED_MM,
        frame_start=INFER_FRAME_START, frame_end=INFER_FRAME_END,
        dose=DOSE, native_noise=True, noise_seed=NOISE_SEED,
        batch=96, num_workers=8, device=device, log_every=100,
    )
    log(f"[infer] done in {(time.time() - t0) / 60:.1f} min -> {DENOISED_MM}")


def stage_reconstruct() -> dict:
    if SUMMARY.exists() and RESULTS.exists() and STILLS_NPZ.exists():
        log(f"[skip] reconstruct: summary already exists at {SUMMARY}")
        return json.loads(SUMMARY.read_text())
    log(f"=== stage 2/5: sliding-window FBP -- floor vs {TAG} vs k={K_PSEUDO_GT} joint-native pseudo-GT ===")
    t0 = time.time()
    prof = DatasetProfile.load("wunderkerze2")
    dark = torch.from_numpy(np.load(f"{CACHE}/dark_map.npy")).to(device=device, dtype=torch.float32)
    flat = torch.from_numpy(np.load(f"{CACHE}/flat_map.npy")).to(device=device, dtype=torch.float32)
    src = MemmapFrameSource(prof.memmap_path, prof.mov_path)

    m = np.load(str(DENOISED_MM) + ".meta.npz")
    d_first, d_n = int(m["first_index"]), int(m["num_frames"])
    crop = (int(m["crop"][0]), int(m["crop"][1]))
    d_mm = np.memmap(DENOISED_MM, dtype=np.float16, mode="r", shape=(d_n, *crop))

    gt_win = R.window_length_frames(GT_WINDOW_DEG, prof.deg_per_frame)
    win_k = int(round(K_PSEUDO_GT * prof.period_360))
    half = win_k // 2
    stride = int(round(prof.period_360))

    lo_center = max(d_first + gt_win // 2 + 1, d_first + half + 1)
    hi_center = min(d_first + d_n - (gt_win - gt_win // 2) - 1, d_first + d_n - (win_k - half) - 1)
    centers = np.arange(lo_center, hi_center, stride)
    assert len(centers) > 0, f"empty centre range [{lo_center},{hi_center})"
    log(f"gt_win={gt_win}f  pseudo_gt_win={win_k}f (k={K_PSEUDO_GT})  centres={len(centers)} "
        f"[{centers[0]},{centers[-1]}]  stride={stride}f")

    vol_shape = (crop[0] // DET_BIN, crop[1] // DET_BIN, crop[1] // DET_BIN)
    nslices, hplane = vol_shape[0], vol_shape[1]
    mask = make_circular_mask(hplane, hplane, device=device)
    mid_slice = nslices // 2

    still_idx = set(int(i) for i in np.linspace(0, len(centers) - 1, N_STILLS).round())

    arms = ("floor", TAG)
    metrics = {a: {"psnr": [], "ssim": []} for a in arms}
    movie_rows = []
    stills = {"pseudo_gt": [], "floor": [], TAG: [], "centre": []}
    starts = []
    for wi, c in enumerate(centers):
        c = int(c)
        gt_idx = np.arange(c - gt_win // 2, c - gt_win // 2 + gt_win)
        angles = R.projection_angles(gt_idx, deg_per_frame=prof.deg_per_frame)

        floor_native = R.native_window_gpu(src, gt_idx, prof.crop, prof.rot_axis_col, device)
        floor_atten = R.counts_to_attenuation_flatdark(floor_native, dark, flat)
        floor_vol = R.reconstruct(floor_atten, angles, det_bin=DET_BIN, method="fbp", device=device)

        den_counts = torch.from_numpy(np.asarray(d_mm[gt_idx - d_first]).astype(np.float32)).to(device)
        den_atten = R.counts_to_attenuation_flatdark(den_counts, dark, flat)
        den_vol = R.reconstruct(den_atten, angles, det_bin=DET_BIN, method="fbp", device=device)

        pg_idx = np.arange(c - half, c - half + win_k)
        pg_angles = R.projection_angles(pg_idx, deg_per_frame=prof.deg_per_frame)
        pg_native = R.native_window_gpu(src, pg_idx, prof.crop, prof.rot_axis_col, device)
        pg_atten = R.counts_to_attenuation_flatdark(pg_native, dark, flat)
        pg_vol = R.reconstruct(pg_atten, pg_angles, det_bin=DET_BIN, method="fbp", device=device)

        dr = float(pg_vol[..., mask].max() - pg_vol[..., mask].min())
        metrics["floor"]["psnr"].append(masked_psnr(pg_vol, floor_vol, mask, dr))
        metrics["floor"]["ssim"].append(masked_ssim(pg_vol, floor_vol, mask, dr))
        metrics[TAG]["psnr"].append(masked_psnr(pg_vol, den_vol, mask, dr))
        metrics[TAG]["ssim"].append(masked_ssim(pg_vol, den_vol, mask, dr))

        movie_rows.append({"pseudo_gt": pg_vol[mid_slice].cpu(), "floor": floor_vol[mid_slice].cpu(),
                           TAG: den_vol[mid_slice].cpu()})
        if wi in still_idx:
            stills["pseudo_gt"].append(pg_vol[mid_slice].cpu().numpy())
            stills["floor"].append(floor_vol[mid_slice].cpu().numpy())
            stills[TAG].append(den_vol[mid_slice].cpu().numpy())
            stills["centre"].append(c)
        starts.append(c)
        if wi % 10 == 0:
            log(f"  window {wi + 1}/{len(centers)} centre {c}  floor {metrics['floor']['psnr'][-1]:.2f}dB  "
                f"{TAG} {metrics[TAG]['psnr'][-1]:.2f}dB")

    metrics = {a: {k_: np.array(v) for k_, v in d.items()} for a, d in metrics.items()}
    for a in arms:
        log(f"  {a:24s} PSNR {metrics[a]['psnr'].mean():6.2f}  SSIM {metrics[a]['ssim'].mean():.3f}")

    np.savez(RESULTS, window_starts=np.array(starts),
             **{f"{a}_psnr": metrics[a]["psnr"] for a in arms},
             **{f"{a}_ssim": metrics[a]["ssim"] for a in arms})
    np.savez(STILLS_NPZ, centre=np.array(stills["centre"]),
             pseudo_gt=np.stack(stills["pseudo_gt"]), floor=np.stack(stills["floor"]),
             **{TAG: np.stack(stills[TAG])})

    summary = {
        "tag": TAG, "n_windows": len(centers), "det_bin": DET_BIN,
        "k_pseudo_gt": K_PSEUDO_GT, "gt_window_deg": GT_WINDOW_DEG,
        "frame_range": [int(centers[0]), int(centers[-1])],
        "minutes": round((time.time() - t0) / 60, 1),
        "correction": "flat_dark (no destripe)",
        "caveat": "pseudo_gt is a k=21 joint-native FBP (low noise, some temporal blur), "
                  "not an independent ground truth -- both floor and denoised are scored against it. "
                  "Given that, validation here leans qualitative -- see the recon/projection movies and PNGs.",
    }
    for a in arms:
        summary[f"{a}_psnr"] = float(metrics[a]["psnr"].mean())
        summary[f"{a}_ssim"] = float(metrics[a]["ssim"].mean())
    SUMMARY.write_text(json.dumps(summary, indent=2))
    log(f"[reconstruct] done in {summary['minutes']} min -> {SUMMARY}")
    log("SUMMARY " + json.dumps(summary, indent=2))

    vmin, vmax = np.percentile(np.stack([r["pseudo_gt"].numpy() for r in movie_rows]), [1, 99])
    combined = [torch.cat([r["pseudo_gt"], r["floor"], r[TAG]], dim=1) for r in movie_rows]
    R.write_slice_movie(combined, RECON_MOVIE, float(vmin), float(vmax))
    log(f"[reconstruct] movie written -> {RECON_MOVIE}  panels: pseudo_gt | floor | {TAG}")
    return summary


def stage_projection_movie() -> None:
    if PROJ_MOVIE.exists():
        log(f"[skip] projection movie already exists at {PROJ_MOVIE}")
        return
    log(f"=== stage 3/5: projection-domain movie (GT raw native | {TAG} | noisy) ===")
    t0 = time.time()
    m = np.load(str(DENOISED_MM) + ".meta.npz")
    d_first, d_n = int(m["first_index"]), int(m["num_frames"])
    R.write_projection_movie(
        MOV, MEMMAP, {TAG: DENOISED_MM}, PROJ_MOVIE,
        frame_start=d_first, frame_end=d_first + d_n,
        dose=DOSE, crop=(128, 512), noise_seed=NOISE_SEED, device=device,
    )
    log(f"[projection movie] done in {(time.time() - t0) / 60:.1f} min -> {PROJ_MOVIE}")
    log("  NOTE: the 'noisy' panel here is GT + one extra independent native-scale Poisson "
      "draw (dose=1.0 is not a true passthrough -- see project memory feedback on "
      "add_poisson_noise), included only as a 'what does another native measurement's own "
      "randomness look like' visual reference, not a synthetically-thinned low-dose panel.")


def stage_pngs() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not RECON_PNG.exists():
        log("=== stage 4/5: reconstruction stills PNG ===")
        d = np.load(STILLS_NPZ)
        centres = d["centre"]
        n = len(centres)
        cols = ["pseudo_gt", "floor", TAG, f"{TAG} - floor"]
        vmin, vmax = np.percentile(d["pseudo_gt"], [1, 99])
        diff = d[TAG] - d["floor"]
        dmax = np.percentile(np.abs(diff), 99)
        fig, axes = plt.subplots(n, len(cols), figsize=(3 * len(cols), 3 * n))
        axes = np.atleast_2d(axes)
        for r in range(n):
            for c, name in enumerate(cols):
                ax = axes[r, c]
                if name.endswith("- floor"):
                    ax.imshow(diff[r], cmap="RdBu_r", vmin=-dmax, vmax=dmax)
                else:
                    ax.imshow(d[name][r], cmap="gray", vmin=vmin, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(name, fontsize=10)
                if c == 0:
                    ax.set_ylabel(f"centre {int(centres[r])}", fontsize=8)
        fig.suptitle(f"{TAG}: reconstruction slice, k={K_PSEUDO_GT} joint-native pseudo-GT reference "
                    "(not independent ground truth)", fontsize=10)
        fig.tight_layout()
        fig.savefig(RECON_PNG, dpi=110)
        plt.close(fig)
        log(f"[pngs] wrote {RECON_PNG}")
    else:
        log(f"[skip] recon stills PNG already exists at {RECON_PNG}")

    if not PROJ_PNG.exists():
        log("=== stage 4/5: projection stills PNG ===")
        prof = DatasetProfile.load("wunderkerze2")
        src = MemmapFrameSource(prof.memmap_path, prof.mov_path)
        m = np.load(str(DENOISED_MM) + ".meta.npz")
        d_first, d_n = int(m["first_index"]), int(m["num_frames"])
        crop = (int(m["crop"][0]), int(m["crop"][1]))
        d_mm = np.memmap(DENOISED_MM, dtype=np.float16, mode="r", shape=(d_n, *crop))
        frame_idx = np.linspace(d_first, d_first + d_n - 1, N_STILLS).round().astype(int)

        # native_window_gpu treats idx as a CONTIGUOUS [idx[0], idx[-1]] range (reads
        # idx[0]/idx[-1] only -- see its docstring/feedback_native_window_gpu_contiguous_only
        # memory note), so a sparse frame_idx must be looped one frame at a time, NOT
        # passed in directly (that silently returns the whole intervening block instead).
        raw = np.stack([
            R.native_window_gpu(src, np.array([fi]), prof.crop, prof.rot_axis_col, device)[0].cpu().numpy()
            for fi in frame_idx
        ])
        den = np.asarray(d_mm[frame_idx - d_first]).astype(np.float32)
        diff = den - raw
        vmin, vmax = np.percentile(raw, [1, 99])
        dmax = np.percentile(np.abs(diff), 99)

        cols = ["raw (native)", TAG, f"{TAG} - raw"]
        aspect = crop[1] / crop[0]  # crop=(128,512) -> wide panels, short figure
        panel_w = 4.5
        fig, axes = plt.subplots(N_STILLS, len(cols), figsize=(panel_w * len(cols), panel_w / aspect * N_STILLS))
        axes = np.atleast_2d(axes)
        for r in range(N_STILLS):
            for c, name in enumerate(cols):
                ax = axes[r, c]
                if name.endswith("- raw"):
                    ax.imshow(diff[r], cmap="RdBu_r", vmin=-dmax, vmax=dmax)
                else:
                    src_arr = raw[r] if name.startswith("raw") else den[r]
                    ax.imshow(src_arr, cmap="gray", vmin=vmin, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(name, fontsize=10)
                if c == 0:
                    ax.set_ylabel(f"frame {int(frame_idx[r])}", fontsize=8)
        fig.suptitle(f"{TAG}: raw projection frame vs denoised", fontsize=10)
        fig.tight_layout()
        fig.savefig(PROJ_PNG, dpi=110)
        plt.close(fig)
        log(f"[pngs] wrote {PROJ_PNG}")
    else:
        log(f"[skip] projection stills PNG already exists at {PROJ_PNG}")


def main() -> None:
    t_all = time.time()
    log(f"device={device}  tag={TAG}  ckpt={CKPT}")
    stage_infer()
    summary = stage_reconstruct()
    stage_projection_movie()
    stage_pngs()
    log("=== stage 5/5: done ===")
    log(f"TOTAL runtime: {(time.time() - t_all) / 60:.1f} min")
    log("FINAL SUMMARY " + json.dumps(summary, indent=2))
    log(f"denoised memmap:  {DENOISED_MM}")
    log(f"recon summary:    {SUMMARY}")
    log(f"recon movie:      {RECON_MOVIE}")
    log(f"projection movie: {PROJ_MOVIE}")
    log(f"recon PNG:        {RECON_PNG}")
    log(f"projection PNG:   {PROJ_PNG}")
    log("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
