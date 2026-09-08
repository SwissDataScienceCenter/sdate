#!/usr/bin/env python3
"""Side-by-side noisy (original) vs denoised (full-dataset, present=True)
projection visuals, for a handful of frames spread across the whole dataset.

    python scripts/sewellia_v2_noisy_vs_denoised_visuals.py
"""
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORIG_H5 = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01.h5"
DEN_H5 = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/denoised_v2_fullres_present_true/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01.h5"
OUT_DIR = Path("/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/test_images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAMES = [0, 2000, 4000, 8000, 10000, 12000, 16000, 19999]

with h5py.File(ORIG_H5, "r") as fo, h5py.File(DEN_H5, "r") as fd:
    theta = fo["exchange/theta"][:]
    orig = {i: fo["exchange/data"][i].astype(np.float32) for i in FRAMES}
    den = {i: fd["exchange/data"][i].astype(np.float32) for i in FRAMES}

# -- 1. one overview grid: all frames x {noisy, denoised, diff} --
fig, axes = plt.subplots(len(FRAMES), 3, figsize=(10.5, 3.4 * len(FRAMES)))
for r, i in enumerate(FRAMES):
    raw = orig[i]
    dn = den[i]
    diff = dn - raw
    vmin, vmax = np.percentile(raw, 1), np.percentile(raw, 99)
    dmax = np.percentile(np.abs(diff), 99)
    for c, (name, im, cmap, vlo, vhi) in enumerate([
        ("noisy (raw)", raw, "gray_r", vmin, vmax),
        ("denoised", dn, "gray_r", vmin, vmax),
        ("denoised - raw", diff, "RdBu_r", -dmax, dmax),
    ]):
        ax = axes[r, c]
        ax.imshow(im, cmap=cmap, vmin=vlo, vmax=vhi)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(name, fontsize=11)
        if c == 0:
            ax.set_ylabel(f"i={i}\ntheta={theta[i]:.1f}", fontsize=9)
fig.suptitle("Noisy vs denoised projections (full-dataset, present=True), gray_r inverted cmap")
fig.tight_layout()
out_overview = OUT_DIR / "noisy_vs_denoised_overview.png"
fig.savefig(out_overview, dpi=140)
plt.close(fig)
print("saved ->", out_overview)

# -- 2. one bigger side-by-side image per frame (noisy | denoised | diff), for closer inspection --
for i in FRAMES:
    raw = orig[i]
    dn = den[i]
    diff = dn - raw
    vmin, vmax = np.percentile(raw, 1), np.percentile(raw, 99)
    dmax = np.percentile(np.abs(diff), 99)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5))
    axes[0].imshow(raw, cmap="gray_r", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"noisy (raw) i={i}")
    axes[1].imshow(dn, cmap="gray_r", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"denoised i={i}")
    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
    axes[2].set_title("denoised - raw")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out_path = OUT_DIR / f"noisy_vs_denoised_i{i}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("saved ->", out_path)

    # standalone diff-only image too
    fig2, ax2 = plt.subplots(figsize=(5.2, 5))
    im3 = ax2.imshow(diff, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
    ax2.set_title(f"denoised - raw, i={i}")
    ax2.set_xticks([]); ax2.set_yticks([])
    fig2.colorbar(im3, ax=ax2, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    diff_path = OUT_DIR / f"diff_i{i}.png"
    fig2.savefig(diff_path, dpi=150)
    plt.close(fig2)
    print("saved ->", diff_path)

print("SUCCESS")
