#!/usr/bin/env python3
"""Render the angular-densification comparison: narrow-raw vs wide-raw vs
narrow-denoised vs densified (denoised-real + denoised-synthetic combined).

    python scripts/sewellia_v2_densify_visuals.py --npz densify_v2_i10000.npz
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata"

p = argparse.ArgumentParser()
p.add_argument("--npz", required=True)
a = p.parse_args()

d = np.load(a.npz)
target = int(d["target"])
n_real = int(d["n_real"])
n_synthetic = int(d["n_synthetic"])

panels = [
    ("narrow-raw (n~%d, real only)" % n_real, d["slice_narrow_raw"]),
    ("wide-raw (n~900, real only)", d["slice_wide_raw"]),
    ("narrow-denoised (present=True)", d["slice_narrow_denoised"]),
    ("densified (real-den + %d synthetic)" % n_synthetic, d["slice_densified"]),
]

all_vals = np.concatenate([im.ravel() for _, im in panels])
vmin = np.percentile(all_vals, 1)
vmax = np.percentile(all_vals, 99)

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for ax, (name, im) in zip(axes, panels):
    ax.imshow(im, cmap="gray_r", vmin=vmin, vmax=vmax)
    ax.set_title(name, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"Angular densification prototype, target i={target}")
fig.tight_layout()
out_path = f"{OUT_DIR}/{Path(a.npz).stem}_visuals.png"
fig.savefig(out_path, dpi=150)
print("saved ->", out_path)
