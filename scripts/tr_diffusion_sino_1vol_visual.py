#!/usr/bin/env python3
"""Quick visual for the single-volume (200-proj/360deg) sinogram-domain
ballpark check: GT / sinogram_denoised / poissonhead / noisy, 3 slice rows,
plus per-arm abs-error maps vs GT (same window as
tr_diffusion_sino_single_volume_check.py, whose denoised caches are reused).

    python scripts/tr_diffusion_sino_1vol_visual.py
"""
import os
import time

os.environ["PATH"] = "/myhome/bin:" + os.environ.get("PATH", "")

import sys

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sdate.tr_diffusion import reconstruct as R

DATA = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2"
CACHE = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache"
MOV = f"{DATA}/212_Wunderkerze2.mov"
MEMMAP = f"{DATA}/frames_400k_500k.u16"
SINO_MM = f"{DATA}/denoised_212_Wunderkerze2_sinogram_ballpark.f16"
BASE_MM = f"{DATA}/denoised_212_Wunderkerze2_poissonhead_k1_full_dose05.f16"

WINDOW_START = 420_300
DET_BIN = 1
WINDOW_DEG = 360.0
OUT_PNG = "/myhome/sdate/scripts/sino_1vol_visual.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main():
    log(f"device={device}")
    win = R.window_length_frames(WINDOW_DEG)
    dark = torch.from_numpy(np.load(f"{CACHE}/dark_map.npy")).float()
    flat = torch.from_numpy(np.load(f"{CACHE}/flat_map.npy")).float()
    variants = {"sinogram_denoised": SINO_MM, "poissonhead": BASE_MM}

    res = R.run_windows(
        MOV, MEMMAP, variants, window_deg=WINDOW_DEG,
        frame_start=WINDOW_START, frame_end=WINDOW_START + win + 5,
        det_bin=DET_BIN, method="fbp",
        dark_map=dark, flat_map=flat, destripe_k=31,
        device=device, log_every=50,
    )
    for arm in list(variants) + ["noisy"]:
        m = res["metrics"][arm]
        log(f"  {arm:20s} PSNR {m['psnr'].mean():6.2f}  SSIM {m['ssim'].mean():.3f}  "
            f"sharpness {m['sharpness'].mean():.3f}")

    arms = ["GT", "sinogram_denoised", "poissonhead", "noisy"]
    movie = res["movie"]
    row_idx = 1  # middle of the 3 cached movie_rows -> central slice
    imgs = {a: movie[a][0][row_idx].numpy() for a in arms}  # single window -> index 0
    gt = imgs["GT"]
    vmin, vmax = np.percentile(gt, [1, 99])

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for col, a in enumerate(arms):
        axes[0, col].imshow(imgs[a], cmap="gray", vmin=vmin, vmax=vmax)
        title = a
        if a != "GT":
            m = res["metrics"][a]
            title += f"\nPSNR {m['psnr'][0]:.2f}dB  SSIM {m['ssim'][0]:.3f}"
        axes[0, col].set_title(title, fontsize=10)
        axes[0, col].axis("off")

        if a == "GT":
            axes[1, col].axis("off")
        else:
            err = np.abs(imgs[a] - gt)
            im = axes[1, col].imshow(err, cmap="inferno", vmin=0, vmax=np.percentile(err, 99))
            axes[1, col].set_title(f"|{a} - GT|", fontsize=9)
            axes[1, col].axis("off")
            plt.colorbar(im, ax=axes[1, col], fraction=0.046)

    plt.suptitle(f"212_Wunderkerze2  window start {WINDOW_START}  200 proj / 360deg  det_bin={DET_BIN}", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120)
    log(f"saved {OUT_PNG}")
    log("DONE")


if __name__ == "__main__":
    main()
