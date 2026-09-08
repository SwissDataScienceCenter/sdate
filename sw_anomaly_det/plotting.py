"""Anomaly-score-vs-time plot, mirroring the project's PSNR-vs-time diagnostic
convention (see ``project-tr-diffusion-t5native-gaussianfloor`` memory).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_rows(log_path):
    with open(log_path, newline="") as fh:
        return list(csv.DictReader(fh))


def plot_anomaly_vs_time(log_path, out_path, T_list: Sequence[float]) -> Path:
    rows = _load_rows(log_path)
    out_path = Path(out_path)
    if not rows:
        return out_path
    t = [int(r["t"]) for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for metric, ax, label in [("mse", axes[0], "MSE"), ("mae", axes[1], "MAE")]:
        for T in T_list:
            key = f"{metric}_T{T:g}"
            ax.plot(t, [float(r[key]) for r in rows], label=f"T={T:g}")
        ax.plot(t, [float(r[f"{metric}_baseline"]) for r in rows], "--", color="gray",
               label="baseline (fixed t0)")
        ax.set_ylabel(f"{label} vs 1-rev FBP")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("frame (t)")
    fig.suptitle("sw_anomaly_det: 1-rev FBP vs T-joint FBP (higher = more scene motion)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
