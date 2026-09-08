#!/usr/bin/env python3
"""Recompute reconstruction-domain PSNR/SSIM/sharpness for an already-finished
Annealed-N2N run, from its cached mmse_anchor.f16/final_estimate.f16 memmaps --
no retraining/re-inference needed. Written because the original run's summary
(recon_summary.json) omitted the sharpness metric run_windows already computes
(fixed in tr_diffusion_anneal_n2n.py going forward; this is the one-off backfill
for runs that finished before that fix).

  python scripts/tr_diffusion_anneal_n2n_sharpness_recheck.py --run_dir .../run1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")

from sdate.tr_diffusion import reconstruct as R
from sdate.tr_diffusion.profiles import DatasetProfile


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--profile", default="wunderkerze2")
    p.add_argument("--eval_frame_start", type=int, default=402_149)
    p.add_argument("--eval_frame_end", type=int, default=497_749)
    p.add_argument("--det_bin", type=int, default=2)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prof = DatasetProfile.load(args.profile)

    variants = {"annealed_n2n_final": str(run_dir / "final_estimate.f16"),
                "mmse_anchor": str(run_dir / "mmse_anchor.f16")}
    res = R.run_windows(prof.mov_path, prof.memmap_path, variants, det_bin=args.det_bin, method="fbp",
                        frame_start=args.eval_frame_start, frame_end=args.eval_frame_end,
                        axis_col=prof.rot_axis_col, deg_per_frame=prof.deg_per_frame,
                        device=device, log_every=100)

    summary = {"n_windows": len(res["window_starts"]), "eval_frame_range": [args.eval_frame_start, args.eval_frame_end]}
    for arm in list(variants) + ["noisy"]:
        m = res["metrics"][arm]
        summary[f"{arm}_psnr"] = float(m["psnr"].mean())
        summary[f"{arm}_ssim"] = float(m["ssim"].mean())
        summary[f"{arm}_sharpness"] = float(m["sharpness"].mean())
        print(f"  {arm:20s} PSNR {m['psnr'].mean():6.2f}  SSIM {m['ssim'].mean():.3f}  "
              f"sharpness {m['sharpness'].mean():.3f}", flush=True)
    (run_dir / "recon_summary_with_sharpness.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
