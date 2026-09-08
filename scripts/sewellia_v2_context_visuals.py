#!/usr/bin/env python3
"""Render the confirmed-matching-z slice across the T=5 phi-context levels,
from the diagnostics the v2 build already saved (row=290, the native-full-res
equivalent of the earlier preview dataset's "slice 29" -- and the same row
the zscan investigation confirmed matches the paper's anatomy).

Colormap: gray_r (inverted), the confirmed display convention for this
dataset (see project memory project-sewellia-phi-context).

    python scripts/sewellia_v2_context_visuals.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SLICES_PATH = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/phi_context/sewellia_v2_phictx_slices.npz"
OUT_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/phi_context"

d = np.load(SLICES_PATH)
target_views = d["target_views"]
row = int(d["row"])
target_idx = d["target_idx"]
target_theta = d["target_theta"]
target_phase = d["target_phase"]

targets = [4000, 10000, 16000]
T = len(target_views)

fig, axes = plt.subplots(len(targets), T, figsize=(3.2 * T, 3.2 * len(targets)))
for r, ti in enumerate(targets):
    vols = d[f"target_{ti}"]  # (T, 576, 576)
    vmin = np.percentile(vols, 1)
    vmax = np.percentile(vols, 99)
    j = list(target_idx).index(ti)
    for c in range(T):
        ax = axes[r, c]
        ax.imshow(vols[c], cmap="gray_r", vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(f"n_views={target_views[c]}")
        if c == 0:
            ax.set_ylabel(f"i={ti}\ntheta={target_theta[j]:.1f} phase={target_phase[j]:.2f}", fontsize=9)
fig.suptitle(f"Sewellia v2 phi-context, T=5 levels, row={row} (native full-res, gray_r=inverted cmap)")
fig.tight_layout()
out_path = f"{OUT_DIR}/sewellia_v2_context_visuals_row{row}.png"
fig.savefig(out_path, dpi=140)
print("saved ->", out_path)

# also the raw projections at each target (transmission-domain), for reference
fig2, axes2 = plt.subplots(1, len(targets), figsize=(4.5 * len(targets), 4.5))
for c, ti in enumerate(targets):
    ax = axes2[c]
    im = d[f"proj_{ti}_transmission"]
    ax.imshow(im, cmap="gray_r")
    ax.set_title(f"i={ti} (transmission)")
    ax.set_xticks([]); ax.set_yticks([])
fig2.tight_layout()
out_path2 = f"{OUT_DIR}/sewellia_v2_target_projections.png"
fig2.savefig(out_path2, dpi=140)
print("saved ->", out_path2)
