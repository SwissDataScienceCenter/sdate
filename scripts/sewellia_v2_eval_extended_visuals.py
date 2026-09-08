#!/usr/bin/env python3
"""Render (1) the raw-vs-denoised PROJECTION-domain comparison (denoising
effect before any reconstruction) and (2) multi-z-slice reconstruction grids,
from eval_v2_extended_*.npz (see scripts/sewellia_v2_eval_extended.py).

    python scripts/sewellia_v2_eval_extended_visuals.py --npz eval_v2_extended_sewellia_n2v_v2_fullres_step75500.npz
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
slice_rows = d["slice_rows"]
stem = Path(a.npz).stem

# -- 1. projection-domain comparison: raw / den_present / den_absent, + signed diffs --
fig, axes = plt.subplots(len(targets), 5, figsize=(19, 3.6 * len(targets)))
for r, ti in enumerate(targets):
    raw = d[f"t{ti}_proj_raw"]
    dp = d[f"t{ti}_proj_den_present"]
    da = d[f"t{ti}_proj_den_absent"]
    vmin = np.percentile(np.concatenate([raw.ravel(), dp.ravel(), da.ravel()]), 1)
    vmax = np.percentile(np.concatenate([raw.ravel(), dp.ravel(), da.ravel()]), 99)
    diff_p = dp - raw
    diff_a = da - raw
    dmax = np.percentile(np.abs(np.concatenate([diff_p.ravel(), diff_a.ravel()])), 99)
    panels = [("raw", raw, "gray_r", vmin, vmax), ("den present=True", dp, "gray_r", vmin, vmax),
              ("den present=False", da, "gray_r", vmin, vmax),
              ("den(T) - raw", diff_p, "RdBu_r", -dmax, dmax), ("den(F) - raw", diff_a, "RdBu_r", -dmax, dmax)]
    for c, (name, im, cmap, vlo, vhi) in enumerate(panels):
        ax = axes[r, c]
        ax.imshow(im, cmap=cmap, vmin=vlo, vmax=vhi)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(name, fontsize=10)
        if c == 0:
            ax.set_ylabel(f"i={ti}", fontsize=10)
fig.suptitle("Projection-domain denoising effect (before any reconstruction)")
fig.tight_layout()
out1 = f"{OUT_DIR}/{stem}_projections.png"
fig.savefig(out1, dpi=150)
print("saved ->", out1)

# -- 2. multi-slice reconstruction grid, one figure per target --
for ti in targets:
    fig, axes = plt.subplots(3, len(slice_rows), figsize=(3.0 * len(slice_rows), 9.5))
    all_vals = np.concatenate([d[f"t{ti}_recon_{name}_row{row}"].ravel()
                               for name in ["raw", "den_present", "den_absent"] for row in slice_rows])
    vmin, vmax = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
    for rr, name in enumerate(["raw", "den_present", "den_absent"]):
        for c, row in enumerate(slice_rows):
            ax = axes[rr, c]
            ax.imshow(d[f"t{ti}_recon_{name}_row{row}"], cmap="gray_r", vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if rr == 0:
                ax.set_title(f"row={row}", fontsize=10)
            if c == 0:
                ax.set_ylabel(name, fontsize=10)
    fig.suptitle(f"Multi-slice reconstruction comparison, i={ti} (det_bin=2)")
    fig.tight_layout()
    out2 = f"{OUT_DIR}/{stem}_slices_i{ti}.png"
    fig.savefig(out2, dpi=150)
    print("saved ->", out2)
