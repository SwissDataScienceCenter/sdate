#!/usr/bin/env python3
"""Joint (non-averaged) k-revolution FBP baseline, swept across the whole
wunderkerze2 dynamical scene.

Unlike SW-FBP (`temporal_average_sequence`/`tr_diffusion_recon_swfbp.py`),
which AVERAGES repeated same-angle measurements into fewer, cleaner views,
this keeps every projection from `k` consecutive revolutions at its own
exact angle (wrapped mod 360deg, static-volume assumption -- same convention
as `tr_diffusion_mle_kjoint_grid.py`) and feeds all of them into a single
FBP reconstruction -- k times the distinct photon measurements of one
rotation, no learning, no iterative optimisation. Expected to be very sharp
wherever the sample didn't move across the k-revolution window, and blurred/
ghosted wherever it did -- this sweep slides the window's centre across the
ENTIRE usable frame range to show exactly where/how that tradeoff plays out
over the real dynamical scene, and renders a GT | joint_fbp movie the same
way prior baselines were rendered. Only 2 FBP calls per window (GT, joint_fbp)
-- the single-noisy-turn floor and the learned-denoiser baseline are already
characterised from earlier runs this project, no need to recompute them here.

k=21 full-scene result (mean over 479 windows): PSNR 27.10dB / SSIM 0.715 --
see README "Key experimental findings" -> "Joint (non-averaged) k-revolution
FBP baseline" for the full writeup and per-window breakdown; this is meant to
be the primary baseline future reconstruction methods on this dataset are
compared against, not just SW-FBP or a single noisy rotation.

  python scripts/tr_diffusion_jointfbp_k_sweep.py --k 21
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")

# write_slice_movie's HevcGray10Streamer invokes the bare "ffmpeg" command (PATH
# lookup), but the RunAI launcher runs via `bash --noprofile --norc`, so PATH never
# picks up /myhome/bin (where this project's ffmpeg actually lives) -- confirmed by
# the ffmpeg-based flat/dark calibration step working fine here since it calls an
# explicit absolute path instead. Prepend it so PATH-based lookups succeed too.
os.environ["PATH"] = f"/myhome/bin:{os.environ.get('PATH', '')}"

from sdate.tr_diffusion import reconstruct as R
from sdate.tr_diffusion.profiles import DatasetProfile
from sdate.tr_diffusion.frames import MemmapFrameSource
from sdate.tr_naf.metrics import make_circular_mask, masked_psnr, masked_ssim

VARIANTS = ("joint_fbp",)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="wunderkerze2")
    p.add_argument("--k", type=int, default=21, help="number of joint revolutions")
    p.add_argument("--det_bin", type=int, default=2)
    p.add_argument("--dose", type=float, default=0.05)
    p.add_argument("--noise_seed", type=int, default=12345)
    p.add_argument("--gt_window_deg", type=float, default=180.0)
    p.add_argument("--frame_start", type=int, default=None, help="default: profile's usable range start")
    p.add_argument("--frame_end", type=int, default=None, help="default: profile's usable range end")
    p.add_argument("--stride", type=int, default=None,
                   help="movie-frame spacing in source frames; default = 1 revolution (period_360)")
    p.add_argument("--out_dir", default="/myhome/data/sdate/shared/time_resolved/tr_recon_cache")
    p.add_argument("--tag", default=None)
    p.add_argument("--vmax_pctile", type=float, default=99.0)
    p.add_argument("--log_every", type=int, default=20)
    return p.parse_args()


def main():
    a = parse_args()
    prof = DatasetProfile.load(a.profile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"{prof.name}_jointfbp_k{a.k}"
    t0 = time.time()

    # --- real per-pixel flat/dark calibration (same convention as tr_diffusion_mle_kjoint_grid.py) ---
    dark_mov = Path(prof.mov_path).with_name(f"{prof.name}_darks.mov")
    flat_mov = Path(prof.mov_path).with_name(f"{prof.name}_flats.mov")
    dark_native = torch.from_numpy(R.load_calibration_average(
        str(dark_mov), prof.crop, prof.rot_axis_col, height=prof.height, width=prof.width,
    )).to(device=device, dtype=torch.float32)
    flat_native = torch.from_numpy(R.load_calibration_average(
        str(flat_mov), prof.crop, prof.rot_axis_col, height=prof.height, width=prof.width,
    )).to(device=device, dtype=torch.float32)
    print(f"flat/dark loaded: dark mean={dark_native.mean():.4g} flat mean={flat_native.mean():.4g}", flush=True)

    src = MemmapFrameSource(prof.memmap_path, prof.mov_path)
    win_k = int(round(a.k * prof.period_360))
    half = win_k // 2
    gt_win = R.window_length_frames(a.gt_window_deg, prof.deg_per_frame)

    fs = a.frame_start if a.frame_start is not None else prof.frame_start
    fe = a.frame_end if a.frame_end is not None else prof.frame_end
    lo_center = fs + half + (gt_win // 2) + 1
    hi_center = fe - (win_k - half) - (gt_win - gt_win // 2) - 1
    stride = a.stride if a.stride is not None else int(round(prof.period_360))
    centers = np.arange(lo_center, hi_center, stride)
    assert len(centers) > 0, f"empty centre range [{lo_center},{hi_center}) for k={a.k}"
    print(f"k={a.k}  win={win_k}f ({win_k / prof.period_360:.2f} turns)  "
          f"centres={len(centers)} [{centers[0]},{centers[-1]}]  stride={stride}f", flush=True)

    vol_shape = (prof.crop[0] // a.det_bin, prof.crop[1] // a.det_bin, prof.crop[1] // a.det_bin)
    nslices, hplane = vol_shape[0], vol_shape[1]
    mask = make_circular_mask(hplane, hplane, device=device)
    mid_slice = nslices // 2

    metrics = {name: {"psnr": [], "ssim": []} for name in VARIANTS}
    movie_rows = []
    for wi, c in enumerate(centers):
        c = int(c)
        gt_idx = np.arange(c - gt_win // 2, c - gt_win // 2 + gt_win)
        gt_angles = R.projection_angles(gt_idx, deg_per_frame=prof.deg_per_frame)
        gt_native = R.native_window_gpu(src, gt_idx, prof.crop, prof.rot_axis_col, device)
        gt_atten = R.counts_to_attenuation_flatdark(gt_native, dark_native, flat_native)
        gt_vol = R.reconstruct(gt_atten, gt_angles, det_bin=a.det_bin, method="fbp", device=device)

        # joint FBP: ALL k revolutions' projections at their own exact wrapped angle
        idx = np.arange(c - half, c - half + win_k)
        angles = R.projection_angles(idx, deg_per_frame=prof.deg_per_frame)
        native = R.native_window_gpu(src, idx, prof.crop, prof.rot_axis_col, device)
        gen = torch.Generator(device=device).manual_seed(a.noise_seed + c)
        noisy_counts = R.noisy_window_gpu(native, a.dose, generator=gen)
        noisy_atten = R.counts_to_attenuation_flatdark(noisy_counts, dark_native, flat_native)
        joint_vol = R.reconstruct(noisy_atten, angles, det_bin=a.det_bin, method="fbp", device=device)

        dr = float(gt_vol[..., mask].max() - gt_vol[..., mask].min())
        metrics["joint_fbp"]["psnr"].append(masked_psnr(gt_vol, joint_vol, mask, dr))
        metrics["joint_fbp"]["ssim"].append(masked_ssim(gt_vol, joint_vol, mask, dr))

        movie_rows.append({"GT": gt_vol[mid_slice].cpu(), "joint_fbp": joint_vol[mid_slice].cpu()})
        if a.log_every and wi % a.log_every == 0:
            print(f"  window {wi + 1}/{len(centers)} centre {c}  "
                  f"joint_fbp {metrics['joint_fbp']['psnr'][-1]:.2f}dB", flush=True)

    metrics = {name: {k_: np.array(v) for k_, v in d.items()} for name, d in metrics.items()}

    np.savez(out_dir / f"recon_results_{tag}.npz", window_starts=centers,
             **{f"{name}_psnr": metrics[name]["psnr"] for name in VARIANTS},
             **{f"{name}_ssim": metrics[name]["ssim"] for name in VARIANTS})
    summary = {
        "tag": tag, "profile": prof.name, "k": a.k, "win_frames": win_k,
        "det_bin": a.det_bin, "dose": a.dose, "n_windows": len(centers), "stride": stride,
        "frame_range": [int(centers[0]), int(centers[-1])],
        "minutes": round((time.time() - t0) / 60, 1),
    }
    for name in VARIANTS:
        summary[f"{name}_psnr"] = float(metrics[name]["psnr"].mean())
        summary[f"{name}_ssim"] = float(metrics[name]["ssim"].mean())
    (out_dir / f"recon_summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)

    gts = np.stack([r["GT"].numpy() for r in movie_rows])
    vmin, vmax = np.percentile(gts, [100 - a.vmax_pctile, a.vmax_pctile])
    combined = [torch.cat([r["GT"], r["joint_fbp"]], dim=1) for r in movie_rows]
    movie_path = out_dir / f"recon_{tag}.mov"
    R.write_slice_movie(combined, movie_path, float(vmin), float(vmax))
    print(f"panels: GT | joint_fbp_k{a.k}", flush=True)
    print(f"movie written: {movie_path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
