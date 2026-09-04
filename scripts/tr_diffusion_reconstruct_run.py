#!/usr/bin/env python3
"""Full reconstruction test of the tr_diffusion baseline denoiser.

Denoise the sequence once (cached), FBP-reconstruct sliding 180deg windows
(GT / denoised / noisy arms), optionally run iterative GD on the most dynamic
window, and render slice movies. All outputs go to OUT_DIR (kept OFF the NFS
home /myhome/sdate). Every knob is a CLI arg; ``--tag`` suffixes the outputs so
different configs (e.g. 256^2-all vs full-res-firsthalf) don't clobber.

Examples
--------
  # default: det_bin=2 (256^2), all disjoint windows, GD on the dynamic window
  python scripts/tr_diffusion_reconstruct_run.py --tag bin2_all

  # full-res, first half only, skip one window between recons, FBP only
  python scripts/tr_diffusion_reconstruct_run.py --det_bin 1 --frame_end 450000 \
      --window_skip 2 --gd_num 0 --tag fullres_firsthalf
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate"); sys.path.insert(0, "/myhome/astra-torch")
from sdate.tr_diffusion import reconstruct as R

DATA = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mov", default=f"{DATA}/212_Wunderkerze2.mov")
    p.add_argument("--memmap", default=f"{DATA}/frames_400k_500k.u16")
    p.add_argument("--baseline", default="/myhome/sdate/checkpoints/tr_denoise_baseline_k1_dose005.pt")
    p.add_argument("--denoised", default=f"{DATA}/denoised_baseline_dose005.f16")
    p.add_argument("--out_dir", default="/myhome/data/sdate/shared/time_resolved/tr_recon_cache")
    p.add_argument("--tag", default="bin2_all", help="suffix for output files")
    p.add_argument("--frame_start", type=int, default=400_000)
    p.add_argument("--frame_end", type=int, default=500_000)
    p.add_argument("--det_bin", type=int, default=2, help="1 -> full 512^2 volume; 2 -> 256^2")
    p.add_argument("--window_deg", type=float, default=180.0)
    p.add_argument("--window_skip", type=int, default=1,
                   help="stride = window_skip * window_length; 1=disjoint, 2=skip one between")
    p.add_argument("--stride", type=int, default=None, help="explicit stride in frames (overrides window_skip)")
    p.add_argument("--gd_num", type=int, default=5, help="windows around the most-dynamic point for GD; 0 disables GD")
    p.add_argument("--gd_epochs", type=int, default=200)
    p.add_argument("--n_movie_rows", type=int, default=3)
    return p.parse_args()


def main():
    a = parse_args()
    dev = torch.device("cuda")
    OUT = Path(a.out_dir); OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # stage 1: denoise full sequence (cached; independent of det_bin/stride)
    if not Path(a.denoised + ".meta.npz").exists():
        print("=== denoising full sequence ===", flush=True)
        R.denoise_sequence(a.baseline, a.mov, a.memmap, a.denoised,
                           frame_start=400_000, frame_end=500_000, batch=96, num_workers=8, device=dev)
    else:
        print("=== denoised memmap already cached ===", flush=True)

    win = R.window_length_frames(a.window_deg)
    stride = a.stride if a.stride is not None else a.window_skip * win
    print(f"=== FBP sweep  det_bin={a.det_bin} ({512 // a.det_bin}^2)  window={win}f  "
          f"stride={stride}f (skip x{a.window_skip})  range [{a.frame_start},{a.frame_end}) ===", flush=True)
    n_slices = 128 // a.det_bin
    movie_rows = None if a.n_movie_rows == 3 else list(
        np.linspace(0, n_slices - 1, a.n_movie_rows).round().astype(int))
    fbp = R.run_windows(a.mov, a.memmap, a.denoised, stride=stride, window_deg=a.window_deg,
                        det_bin=a.det_bin, method="fbp", frame_start=a.frame_start, frame_end=a.frame_end,
                        movie_rows=movie_rows, device=dev, log_every=25)
    nW = len(fbp["window_starts"])
    print(f"FBP done: {nW} windows | denoised {fbp['metrics']['denoised']['psnr'].mean():.2f}dB/"
          f"{fbp['metrics']['denoised']['ssim'].mean():.3f} | noisy "
          f"{fbp['metrics']['noisy']['psnr'].mean():.2f}dB/{fbp['metrics']['noisy']['ssim'].mean():.3f}", flush=True)

    dyn_i, change = R.most_dynamic_window(fbp["movie"]["GT"])
    dyn_start = fbp["window_starts"][dyn_i]
    print(f"most-dynamic window: index {dyn_i} (start {dyn_start})", flush=True)

    gd = None
    gd_starts = []
    if a.gd_num > 0:
        half = a.gd_num // 2
        gd_starts = [fbp["window_starts"][i] for i in range(max(0, dyn_i - half), min(nW, dyn_i + half + 1))]
        print(f"=== GD on {len(gd_starts)} windows {gd_starts} ===", flush=True)
        gd = R.run_windows(a.mov, a.memmap, a.denoised, window_starts=gd_starts, window_deg=a.window_deg,
                           det_bin=a.det_bin, method="gd", gd_kwargs=dict(max_epochs=a.gd_epochs, lr=1e-1, clamp_min=0.0),
                           device=dev, log_every=1)
        print(f"GD done: denoised {gd['metrics']['denoised']['psnr'].mean():.2f}dB | "
              f"noisy {gd['metrics']['noisy']['psnr'].mean():.2f}dB", flush=True)

    # save metrics first (persist before movies)
    np.savez(OUT / f"recon_results_{a.tag}.npz",
             window_starts=np.array(fbp["window_starts"]),
             den_psnr=fbp["metrics"]["denoised"]["psnr"], den_ssim=fbp["metrics"]["denoised"]["ssim"],
             noisy_psnr=fbp["metrics"]["noisy"]["psnr"], noisy_ssim=fbp["metrics"]["noisy"]["ssim"],
             change=change, dyn_index=dyn_i, dyn_start=dyn_start, gd_starts=np.array(gd_starts),
             I0=fbp["I0"], det_bin=a.det_bin, window_deg=a.window_deg, stride=stride)
    summary = dict(tag=a.tag, n_windows=nW, det_bin=a.det_bin, resolution=512 // a.det_bin,
                   window_frames=win, stride_frames=stride, window_skip=a.window_skip,
                   frame_range=[a.frame_start, a.frame_end],
                   fbp_denoised_psnr=float(fbp["metrics"]["denoised"]["psnr"].mean()),
                   fbp_denoised_ssim=float(fbp["metrics"]["denoised"]["ssim"].mean()),
                   fbp_noisy_psnr=float(fbp["metrics"]["noisy"]["psnr"].mean()),
                   fbp_noisy_ssim=float(fbp["metrics"]["noisy"]["ssim"].mean()),
                   dyn_start=int(dyn_start), minutes=round((time.time() - t0) / 60, 1))
    if gd is not None:
        summary["gd_denoised_psnr"] = float(gd["metrics"]["denoised"]["psnr"].mean())
        summary["gd_noisy_psnr"] = float(gd["metrics"]["noisy"]["psnr"].mean())
    (OUT / f"recon_summary_{a.tag}.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)

    # movies (arms side-by-side, middle slice)
    mid = a.n_movie_rows // 2
    gt_slices = np.stack([f[mid].numpy() for f in fbp["movie"]["GT"]])
    vmin, vmax = np.percentile(gt_slices, [1, 99])
    print(f"movie window [{vmin:.4f}, {vmax:.4f}]", flush=True)
    combined = [torch.cat([fbp["movie"][arm][i][mid] for arm in R.ARMS], dim=1) for i in range(nW)]
    R.write_slice_movie(combined, OUT / f"recon_sweep_{a.tag}.mov", float(vmin), float(vmax))
    if gd is not None:
        gd_comb = [torch.cat([gd["movie"][arm][i][mid] for arm in R.ARMS], dim=1) for i in range(len(gd_starts))]
        R.write_slice_movie(gd_comb, OUT / f"recon_gd_{a.tag}.mov", float(vmin), float(vmax))
    print("movies written", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
