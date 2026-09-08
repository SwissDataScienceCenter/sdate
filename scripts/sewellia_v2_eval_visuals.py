#!/usr/bin/env python3
"""Render raw vs present=True-denoised vs present=False(context-only)-denoised
reconstruction comparisons from eval_v2_*.npz (see scripts/sewellia_real_eval_v2.py).

    python scripts/sewellia_v2_eval_visuals.py --npz eval_v2_sewellia_n2v_v2_fullres_step75500.npz
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
targets = d["targets"]

fig, axes = plt.subplots(len(targets), 3, figsize=(11, 3.6 * len(targets)))
for r, ti in enumerate(targets):
    raw = d[f"t{ti}_recon_raw"]
    den_p = d[f"t{ti}_recon_den_present"]
    den_a = d[f"t{ti}_recon_den_absent"]
    vmin = np.percentile(np.concatenate([raw, den_p, den_a]), 1)
    vmax = np.percentile(np.concatenate([raw, den_p, den_a]), 99)
    for c, (name, im) in enumerate([("raw (narrow, n~200)", raw),
                                    ("denoised present=True", den_p),
                                    ("denoised present=False (context-only)", den_a)]):
        ax = axes[r, c]
        ax.imshow(im, cmap="gray_r", vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(name, fontsize=10)
        if c == 0:
            ax.set_ylabel(f"i={ti}", fontsize=10)
fig.suptitle("Sewellia v2 N2V+context eval (step75500): narrow-window (n~200) FBP, raw vs denoised")
fig.tight_layout()
out_path = f"{OUT_DIR}/{Path(a.npz).stem}_visuals.png"
fig.savefig(out_path, dpi=150)
print("saved ->", out_path)
