#!/usr/bin/env python3
"""Full tomographic-reconstruction comparison for the motion-compensated
("warped") temporal context experiment (angular-resolution-gap fix).

Denoises the full sequence with three checkpoints -- the existing (unretrained)
production control, Leg 2 (baseline recipe retrained with warped context), and
Leg 1 (refine: gamma-sample central + warped context) -- over the frame range
common to all three (gated by the warped-context caches' own valid range,
[400402, 499598)), then reconstructs the SAME sliding FBP windows used by every
prior ablation in this project (``tr_diffusion_recon_ablation.py``'s
window_skip=2/start_window_offset=1/det_bin=1 convention) and scores PSNR/SSIM
per variant vs the GT reconstruction, plus the noisy floor.

    python scripts/tr_diffusion_warped_context_recon.py
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
from sdate.tr_diffusion.geometry import usable_frame_range  # noqa: E402
from sdate.tr_diffusion.profiles import REGISTRY  # noqa: E402

CKDIR = "/myhome/data/sdate/shared/checkpoints"
CONTROL_CKPT = f"{CKDIR}/tr_denoise_baseline_k1_dose005_poissonhead.pt"
LEG2_CKPT = f"{CKDIR}/tr_denoise_baseline_k1_dose005_poissonhead_warpedctx.pt"
LEG1_CKPT = f"{CKDIR}/tr_denoise_refine_warpedctx.pt"

TR_DIR = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2"
OUT_DIR = Path("/myhome/data/sdate/shared/time_resolved/tr_recon_cache")

# The warped-context caches only cover [400402, 499598) (99196 frames) -- the
# intersection of the phase-1 registration cache's own usable range with the
# 4 temporal taps' bracket-frame requirements (see
# scripts/tr_diffusion_warped_context_precompute.py). All three variants must
# be compared over this SAME range for a fair, apples-to-apples comparison.
FRAME_START, FRAME_END = 400402, 499598
DOSE = 0.05
DET_BIN = 1
WINDOW_SKIP = 2
START_WINDOW_OFFSET = 1
DESTRIPE_K = 31  # matches tr_diffusion_noise2clean_pipeline.py / tr_diffusion_poissonhead_pipeline.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main():
    REAL = REGISTRY["wunderkerze2"]
    kw = dict(deg_per_frame=REAL.deg_per_frame, axis_col=REAL.rot_axis_col)
    t0 = time.time()

    variants = {}
    for name, ckpt in [("control", CONTROL_CKPT), ("leg2_warpedctx", LEG2_CKPT),
                       ("leg1_refine_warpedctx", LEG1_CKPT)]:
        mm_path = f"{TR_DIR}/denoised_212_Wunderkerze2_warpedctx_recon_{name}.f16"
        if not Path(mm_path + ".meta.npz").exists():
            log(f"=== denoising full sequence: {name} ===")
            R.denoise_sequence(ckpt, REAL.mov_path, REAL.memmap_path, mm_path,
                               frame_start=FRAME_START, frame_end=FRAME_END, dose=DOSE,
                               noise_seed=12345, batch=64, num_workers=8, device=device,
                               log_every=200, **kw)
        else:
            log(f"[skip] {name} sequence already cached at {mm_path}")
        variants[name] = mm_path

    win = R.window_length_frames(180.0, REAL.deg_per_frame)
    stride = WINDOW_SKIP * win
    ulo, _ = usable_frame_range(FRAME_START, FRAME_END, 1, period_360=REAL.period_360)
    recon_start = ulo + START_WINDOW_OFFSET * win
    dark = torch.from_numpy(np.load(f"{OUT_DIR}/dark_map.npy")).float()
    flat = torch.from_numpy(np.load(f"{OUT_DIR}/flat_map.npy")).float()
    log(f"=== reconstructing windows: det_bin={DET_BIN} stride={stride}f start={recon_start} "
        f"(flat/dark + destripe_k={DESTRIPE_K}) ===")
    res = R.run_windows(REAL.mov_path, REAL.memmap_path, variants, stride=stride, det_bin=DET_BIN,
                        method="fbp", frame_start=recon_start, frame_end=FRAME_END,
                        axis_col=REAL.rot_axis_col, deg_per_frame=REAL.deg_per_frame,
                        dark_map=dark, flat_map=flat, destripe_k=DESTRIPE_K,
                        device=device, log_every=25)
    nW = len(res["window_starts"])
    log(f"windows: {nW}")

    print()
    print("=== warped-context experiment: 247-window FBP reconstruction comparison ===")
    for arm in list(variants) + ["noisy"]:
        m = res["metrics"][arm]
        print(f"  {arm:24s} PSNR {m['psnr'].mean():6.2f}  SSIM {m['ssim'].mean():.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "wunderkerze2_warpedctx"
    np.savez(OUT_DIR / f"recon_results_{tag}.npz", window_starts=np.array(res["window_starts"]),
             **{f"{arm}_psnr": res["metrics"][arm]["psnr"] for arm in res["metrics"]},
             **{f"{arm}_ssim": res["metrics"][arm]["ssim"] for arm in res["metrics"]})
    summary = {"tag": tag, "n_windows": nW, "det_bin": DET_BIN,
               "frame_range": [FRAME_START, FRAME_END], "minutes": round((time.time() - t0) / 60, 1)}
    for arm in res["metrics"]:
        summary[f"{arm}_psnr"] = float(res["metrics"][arm]["psnr"].mean())
        summary[f"{arm}_ssim"] = float(res["metrics"][arm]["ssim"].mean())
    (OUT_DIR / f"recon_summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote recon_summary_{tag}.json")
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)

    mid = len(res["movie_rows"]) // 2
    gt = np.stack([f[mid].numpy() for f in res["movie"]["GT"]])
    vmin, vmax = np.percentile(gt, [1, 99])
    combined = [torch.cat([res["movie"][arm][i][mid] for arm in res["arms"]], dim=1) for i in range(nW)]
    R.write_slice_movie(combined, OUT_DIR / f"recon_{tag}.mov", float(vmin), float(vmax))
    log(f"panels: {' | '.join(res['arms'])}")
    log("PIPELINE COMPLETE (warped context reconstruction eval)")


if __name__ == "__main__":
    main()
