#!/usr/bin/env python3
"""Quick visual survey: real single-revolution FBP central slices at N
uniformly-spaced sample frames, for picking WHERE to run the super-time-
resolution sweep (tr_diffusion_super_tr_movie.py). No model, no synthesis --
just plain real-data FBP so this is fast and cheap.

  python scripts/tr_diffusion_frame_survey.py --lo 400200 --hi 442000 --n 20
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")
os.environ["PATH"] = f"/myhome/bin:{os.environ.get('PATH', '')}"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sdate.tr_diffusion import reconstruct as R  # noqa: E402
from sdate.tr_diffusion.frames import MemmapFrameSource  # noqa: E402
from sdate.tr_diffusion.profiles import DatasetProfile  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="wunderkerze2")
    p.add_argument("--lo", type=int, default=400200)
    p.add_argument("--hi", type=int, default=442000)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--win_T", type=int, default=200, help="frames per single-revolution FBP window")
    p.add_argument("--det_bin", type=int, default=1)
    p.add_argument("--out", default="/myhome/data/sdate/shared/time_resolved/tr_recon_cache/frame_survey.png")
    return p.parse_args()


def main():
    a = parse_args()
    prof = DatasetProfile.load(a.profile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    half = a.win_T // 2

    dark_mov = Path(prof.mov_path).with_name(f"{prof.name}_darks.mov")
    flat_mov = Path(prof.mov_path).with_name(f"{prof.name}_flats.mov")
    dark_native = torch.from_numpy(R.load_calibration_average(
        str(dark_mov), prof.crop, prof.rot_axis_col, height=prof.height, width=prof.width,
    )).to(device=device, dtype=torch.float32)
    flat_native = torch.from_numpy(R.load_calibration_average(
        str(flat_mov), prof.crop, prof.rot_axis_col, height=prof.height, width=prof.width,
    )).to(device=device, dtype=torch.float32)
    print(f"flat/dark loaded: dark mean={dark_native.mean():.4g} flat mean={flat_native.mean():.4g}", flush=True)

    src = MemmapFrameSource(prof.memmap_path, prof.mov_path)

    samples = np.linspace(a.lo, a.hi, a.n, dtype=int)
    lo_needed, hi_needed = int(samples.min()) - half, int(samples.max()) + half
    assert lo_needed >= prof.frame_start and hi_needed <= prof.frame_end, (
        f"survey range needs real data [{lo_needed},{hi_needed}) outside usable "
        f"profile range [{prof.frame_start},{prof.frame_end})")
    print(f"survey: {a.n} samples in [{a.lo},{a.hi}]  win_T={a.win_T}  "
          f"needs real data [{lo_needed},{hi_needed})", flush=True)

    slices = []
    for i, f in enumerate(samples):
        f = int(f)
        idx = np.arange(f - half, f - half + a.win_T)
        angles = R.projection_angles(idx, deg_per_frame=prof.deg_per_frame)
        native = R.native_window_gpu(src, idx, prof.crop, prof.rot_axis_col, device)
        atten = R.counts_to_attenuation_flatdark(native, dark_native, flat_native)
        vol = R.reconstruct(atten, angles, det_bin=a.det_bin, method="fbp", device=device)
        mid = vol.shape[0] // 2
        slices.append(vol[mid].cpu())
        print(f"  {i + 1}/{a.n}  frame={f}", flush=True)

    stacked = torch.stack(slices).numpy()
    vmin, vmax = np.percentile(stacked, [1, 99])

    ncols = 5
    nrows = int(np.ceil(a.n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)
    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        if i < a.n:
            ax.imshow(stacked[i], cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(f"frame {int(samples[i])}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    print(f"saved survey grid -> {out_path}", flush=True)
    print("PIPELINE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
