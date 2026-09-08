#!/usr/bin/env python
"""Time-resolved NAF demo: fit a 4-D field to rotating limited-angle frames.

Builds a synthetic multi-sweep acquisition from a time-resolved tif volume,
reconstructs with the temporal-basis NAF field, and compares per-frame accuracy
against the sliding-window FBP baseline (SW-FBP).

Usage
-----
    python scripts/tr_naf_demo.py                     # defaults below
    python scripts/tr_naf_demo.py --frames 25 --K 6   # 5 sweeps, K=6

The model outputs, per spatial voxel (x,y,z), K cubic-B-spline coefficients that
describe that voxel's smooth attenuation curve over the scan.  See
``sdate/tr_naf/`` for the implementation.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# astra_torch / chip are external checkouts (mirrors the notebook path setup).
sys.path.insert(0, "/myhome/astra-torch")
sys.path.insert(0, "/myhome/chip-project")

from sdate.tr_naf import (
    build_acquisition,
    sliding_window_fbp,
    tr_naf_reconstruction,
    reconstruct_volume_at,
    generate_slice_movies,
)

DEFAULT_TIF = (
    "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2/timesteps/"
    "212_Wunderkerze2_rotate_04001.tif"
)


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tif", default=DEFAULT_TIF)
    ap.add_argument("--frames", type=int, default=15, help="number of time frames")
    ap.add_argument("--angle-range", type=float, default=36.0, help="wedge width per frame (deg)")
    ap.add_argument("--num-full-projs", type=int, default=720, help="proj density over 180 deg")
    ap.add_argument("--timestep-skip", type=int, default=4, help="tif files skipped between frames")
    ap.add_argument("--cube", type=int, default=128, help="reconstruct cube side")
    ap.add_argument("--K", type=int, default=6, help="temporal B-spline coefficients per voxel")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--reg-tv", type=float, default=1e-7)
    ap.add_argument("--reg-temporal-max", type=float, default=1e-2)
    ap.add_argument("--temporal-anneal-frac", type=float, default=0.6)
    ap.add_argument("--norm-max", type=float, default=233.0)
    ap.add_argument("--out", default="tr_naf_demo.png")
    ap.add_argument("--movie-dir", default=None,
                    help="If set, write HEVC movies of a few slices over time here.")
    ap.add_argument("--num-movie-slices", type=int, default=5)
    ap.add_argument("--movie-fps", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    frames, meta = build_acquisition(
        data_path=args.tif,
        num_frames=args.frames,
        angle_range_deg=args.angle_range,
        num_full_projs=args.num_full_projs,
        timestep_skip=args.timestep_skip,
        cube_size=args.cube,
        normalize_range=(0.0, args.norm_max),
        device=device,
    )
    print(f"Built {len(frames)} frames | {meta['num_sweeps']:.1f} sweeps | "
          f"{meta['num_la_projs']} projs/frame | cube={args.cube}")
    gt0, gt1 = frames[0].true_volume, frames[-1].true_volume
    print(f"GT motion first->last (rel L2): {rel_err(gt0, gt1):.4f}")

    result = tr_naf_reconstruction(
        frames, meta,
        K=args.K, n_iterations=args.iters, lr=args.lr,
        reg_tv=args.reg_tv, reg_temporal_max=args.reg_temporal_max,
        temporal_anneal_frac=args.temporal_anneal_frac,
        device=device, seed=0, verbose=True,
    )
    print(f"Trained in {result['time']:.1f}s | final loss {result['losses'][-1]:.3e}")

    sw = sliding_window_fbp(frames, meta, device=device)
    sw_errs, tr_errs = [], []
    tr_vols = []
    for f in frames:
        tr = reconstruct_volume_at(result, f.t_norm)
        tr_vols.append(tr)
        sw_errs.append(rel_err(sw, f.true_volume))
        tr_errs.append(rel_err(tr, f.true_volume))
    print(f"\nMean per-frame rel-error   SW-FBP: {np.mean(sw_errs):.4f}   "
          f"TR-NAF: {np.mean(tr_errs):.4f}")

    # ---- figure: mid-slice of first / mid / last frame: GT vs SW-FBP vs TR-NAF ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = args.cube // 2
    picks = [0, len(frames) // 2, len(frames) - 1]
    fig, axes = plt.subplots(3, len(picks), figsize=(4 * len(picks), 11))
    for col, fi in enumerate(picks):
        gt = frames[fi].true_volume[z].cpu()
        axes[0, col].imshow(gt, cmap="gray"); axes[0, col].set_title(f"GT  frame {fi}")
        axes[1, col].imshow(sw[z].cpu(), cmap="gray")
        axes[1, col].set_title(f"SW-FBP (static)  err={sw_errs[fi]:.3f}")
        axes[2, col].imshow(tr_vols[fi][z].cpu(), cmap="gray")
        axes[2, col].set_title(f"TR-NAF  err={tr_errs[fi]:.3f}")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"Saved comparison figure to {args.out}")

    # ---- optional HEVC movies of a few slices over time ----
    if args.movie_dir:
        print(f"\nWriting movies to {args.movie_dir} ...")
        try:
            generate_slice_movies(
                args.movie_dir, times,
                recon_volumes=tr_vols, gt_volumes=[f.true_volume for f in frames],
                sw_fbp=sw, num_movie_slices=args.num_movie_slices, fps=args.movie_fps,
            )
        except Exception as e:  # e.g. ffmpeg not installed
            print(f"[movies skipped] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
