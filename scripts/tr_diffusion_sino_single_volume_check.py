#!/usr/bin/env python3
"""Quick ballpark reconstruction check for the --mode sinogram checkpoint:
ONE 200-projection (360-degree) volume window, PSNR/SSIM vs GT, det_bin=1,
real flat/dark + destripe correction (same convention as every other number
in this project) -- not the full 247-window sweep, just a fast sanity read.

200 projections ~= a full turn at wunderkerze2's 1.8014 deg/frame (~199.84
frames/360deg) -- deliberately NOT this project's usual 180deg/~100-frame
window (which minimises motion blur for the real sliding-window evaluations);
this is a quick, coarser ballpark only.

    python scripts/tr_diffusion_sino_single_volume_check.py
"""
import os
import time

os.environ["PATH"] = "/myhome/bin:" + os.environ.get("PATH", "")

import sys

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")

import numpy as np
import torch

from sdate.tr_diffusion import reconstruct as R

DATA = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2"
CACHE = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache"
CKPT = "/myhome/data/sdate/shared/checkpoints/tr_denoise_sinogram_k1dose005_v1.pt"
MOV = f"{DATA}/212_Wunderkerze2.mov"
MEMMAP = f"{DATA}/frames_400k_500k.u16"
SINO_MM = f"{DATA}/denoised_212_Wunderkerze2_sinogram_ballpark.f16"
BASE_MM = f"{DATA}/denoised_212_Wunderkerze2_poissonhead_k1_full_dose05.f16"  # already cached

INFER_FRAME_START, INFER_FRAME_END = 420_000, 420_900
WINDOW_START = 420_300
DOSE, NOISE_SEED = 0.05, 12345
DET_BIN = 1
WINDOW_DEG = 360.0  # ~200 frames at this dataset's deg/frame -- "200 projections"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main():
    log(f"device={device}")
    win = R.window_length_frames(WINDOW_DEG)
    log(f"window={win} frames at {WINDOW_DEG} deg (target ~200)")

    from pathlib import Path
    if Path(SINO_MM + ".meta.npz").exists():
        log(f"[skip] denoise: already exists at {SINO_MM}")
    else:
        t0 = time.time()
        first, n, meta = R.denoise_sequence(
            CKPT, MOV, MEMMAP, SINO_MM,
            frame_start=INFER_FRAME_START, frame_end=INFER_FRAME_END,
            dose=DOSE, noise_seed=NOISE_SEED,
            batch=8, num_workers=4, device=device, log_every=5,
        )
        log(f"denoised {n} frames in {(time.time() - t0) / 60:.1f} min")

    dark = torch.from_numpy(np.load(f"{CACHE}/dark_map.npy")).float()
    flat = torch.from_numpy(np.load(f"{CACHE}/flat_map.npy")).float()
    variants = {"sinogram_denoised": SINO_MM, "poissonhead": BASE_MM}

    # window_starts (if used) must exactly match an auto-generated candidate
    # (stride-aligned from the usable range's own start) -- simpler to just
    # constrain frame_start/frame_end so exactly ONE window falls out naturally.
    t1 = time.time()
    res = R.run_windows(
        MOV, MEMMAP, variants, window_deg=WINDOW_DEG,
        frame_start=WINDOW_START, frame_end=WINDOW_START + win + 5,
        det_bin=DET_BIN, method="fbp",
        dark_map=dark, flat_map=flat, destripe_k=31,
        device=device, log_every=50,
    )
    log(f"reconstruction done in {(time.time() - t1) / 60:.1f} min")

    for arm in list(variants) + ["noisy"]:
        m = res["metrics"][arm]
        log(f"  {arm:20s} PSNR {m['psnr'].mean():6.2f}  SSIM {m['ssim'].mean():.3f}  "
            f"sharpness {m['sharpness'].mean():.3f}")
    log("DONE")


if __name__ == "__main__":
    main()
