#!/usr/bin/env python3
"""Coarse full-range FBP sweep to locate the most dynamic window for a dataset.

Generates a baseline-denoised sequence over the full profile frame range (fast,
single pass), then reconstructs every disjoint 180deg window at reduced
resolution (det_bin=2) purely to score inter-window change and PSNR/SSIM trends.
Reports the most dynamic window start and a change-vs-time curve; does not
render movies (see tr_diffusion_recon_pm.py for the full-res follow-up).

  python scripts/tr_diffusion_find_dynamic.py --profile asc_thixo
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate"); sys.path.insert(0, "/myhome/astra-torch")
from sdate.tr_diffusion import reconstruct as R
from sdate.tr_diffusion.profiles import DatasetProfile

CK = "/myhome/sdate/checkpoints"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", required=True)
    p.add_argument("--exp", default="asc_dose005")
    p.add_argument("--baseline_ckpt", default=None)
    p.add_argument("--dose", type=float, default=0.05)
    p.add_argument("--det_bin", type=int, default=2)
    p.add_argument("--out_dir", default="/myhome/data/sdate/shared/time_resolved/tr_recon_cache")
    p.add_argument("--batch", type=int, default=96, help="lower if the GPU is shared with training")
    p.add_argument("--num_workers", type=int, default=8)
    return p.parse_args()


def main():
    a = parse_args()
    prof = DatasetProfile.load(a.profile)
    dev = torch.device("cuda")
    OUT = Path(a.out_dir); OUT.mkdir(parents=True, exist_ok=True)
    dcap = prof.mov_path.rsplit("/", 1)[0]
    base_ck = a.baseline_ckpt or f"{CK}/tr_denoise_baseline_{a.exp}.pt"
    base_mm = f"{dcap}/denoised_{prof.name}_baseline_dose{int(a.dose*100):02d}_full.f16"
    t0 = time.time()

    if not Path(base_mm + ".meta.npz").exists():
        print("=== baseline denoise, full range ===", flush=True)
        R.denoise_sequence(base_ck, prof.mov_path, prof.memmap_path, base_mm,
                           frame_start=prof.frame_start, frame_end=prof.frame_end, dose=a.dose,
                           batch=a.batch, num_workers=a.num_workers, device=dev,
                           deg_per_frame=prof.deg_per_frame, axis_col=prof.rot_axis_col)

    print(f"=== coarse FBP sweep, det_bin={a.det_bin}, full range ===", flush=True)
    res = R.run_windows(prof.mov_path, prof.memmap_path, {"baseline": base_mm}, det_bin=a.det_bin,
                        method="fbp", axis_col=prof.rot_axis_col, deg_per_frame=prof.deg_per_frame,
                        device=dev, log_every=25)
    nW = len(res["window_starts"])
    dyn_i, change = R.most_dynamic_window(res["movie"]["GT"])
    dyn_start = res["window_starts"][dyn_i]
    print(f"windows {nW} | baseline {res['metrics']['baseline']['psnr'].mean():.2f}dB | "
          f"noisy {res['metrics']['noisy']['psnr'].mean():.2f}dB", flush=True)
    print(f"most-dynamic window: index {dyn_i} start {dyn_start} "
          f"(of range [{prof.frame_start},{prof.frame_end}))", flush=True)

    np.savez(OUT / f"dynamic_scan_{prof.name}.npz", window_starts=np.array(res["window_starts"]),
             change=change, baseline_psnr=res["metrics"]["baseline"]["psnr"],
             baseline_ssim=res["metrics"]["baseline"]["ssim"],
             noisy_psnr=res["metrics"]["noisy"]["psnr"], noisy_ssim=res["metrics"]["noisy"]["ssim"])
    summary = dict(profile=prof.name, n_windows=nW, dyn_index=int(dyn_i), dyn_start=int(dyn_start),
                   baseline_psnr=float(res["metrics"]["baseline"]["psnr"].mean()),
                   noisy_psnr=float(res["metrics"]["noisy"]["psnr"].mean()),
                   minutes=round((time.time() - t0) / 60, 1))
    (OUT / f"dynamic_scan_{prof.name}.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
