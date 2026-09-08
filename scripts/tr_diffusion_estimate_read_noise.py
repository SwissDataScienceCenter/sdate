#!/usr/bin/env python3
"""Estimate the detector's additive Gaussian read-noise floor (per-pixel
variance) from the dark-frame calibration stream already used for flat/dark
correction elsewhere in this project.

The detector's actual noise model is Poisson (photon/dark-current shot noise)
convolved with Gaussian (electronic read noise) -- the existing poisson_head
NB-NLL loss (sdate/tr_diffusion/nb_head.py) only models the Poisson part. With
no incident x-ray flux, dark frames isolate the Gaussian floor: their per-pixel
VARIANCE across many dark exposures IS (an estimate of) sigma_read^2 (plus a
typically-negligible dark-current shot-noise contribution, since dark current
itself is usually a small, roughly Poisson-distributed rate).

Saves a per-pixel (H, W) float32 .npy map, cropped/axis-aligned exactly like
the flat/dark maps used everywhere else in this project (reconstruct.py's
load_calibration_average / _center_crop), so it lines up pixel-for-pixel with
the training data.

    python scripts/tr_diffusion_estimate_read_noise.py --profile wunderkerze2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ["PATH"] = f"/myhome/bin:{os.environ.get('PATH', '')}"
sys.path.insert(0, "/myhome/sdate")

import numpy as np

from sdate.tr_diffusion.frames import denormalize, load_norm_sidecar
from sdate.tr_diffusion.geometry import FRAME_H, FRAME_W
from sdate.tr_diffusion.profiles import DatasetProfile
from sdate.tr_diffusion.reconstruct import _center_crop


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="wunderkerze2")
    p.add_argument("--drop_first", type=int, default=1,
                   help="drop the first N dark frames -- frame 0 is a known decoder/settling "
                        "transient outlier (see load_calibration_average).")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    a = parse_args()
    prof = DatasetProfile.load(a.profile)
    dark_mov = Path(prof.mov_path).with_name(f"{prof.name}_darks.mov")
    out = Path(a.out) if a.out else dark_mov.with_name(f"{prof.name}_dark_var.npy")

    side = load_norm_sidecar(str(dark_mov))
    n = side["per_frame_min"].shape[0]
    height = prof.height or FRAME_H
    width = prof.width or FRAME_W
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(dark_mov), "-pix_fmt", "gray16le", "-f", "rawvideo", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    arr = np.frombuffer(proc.stdout, np.uint16).reshape(-1, height, width)
    assert arr.shape[0] == n, f"{dark_mov}: decoded {arr.shape[0]} frames, sidecar has {n}"
    counts = np.stack([denormalize(arr[k], side["per_frame_min"][k], side["per_frame_max"][k])
                       for k in range(n)])
    counts = counts[a.drop_first:]
    cropped = np.stack([_center_crop(c, prof.crop[0], prof.crop[1], prof.rot_axis_col) for c in counts])

    mean_map = cropped.mean(axis=0).astype(np.float32)
    var_map = cropped.var(axis=0).astype(np.float32)
    print(f"{n - a.drop_first} dark frames, cropped {cropped.shape[1:]}")
    print(f"dark mean: overall={cropped.mean():.3f}  per-pixel range=[{mean_map.min():.2f},{mean_map.max():.2f}]")
    print(f"dark var:  overall={var_map.mean():.3f}  median={np.median(var_map):.3f}  "
          f"range=[{var_map.min():.2f},{var_map.max():.2f}]  (std~{np.sqrt(var_map.mean()):.3f} counts)")

    np.save(out, var_map)
    print(f"wrote {out}  shape={var_map.shape}  dtype={var_map.dtype}")


if __name__ == "__main__":
    main()
