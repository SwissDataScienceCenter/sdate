#!/usr/bin/env python3
"""Movie of the reconstruction as phi sweeps through the full periodic cycle,
at a FIXED gating radius (one of the T=11 radii already used to build the
joint-FBP context taps) -- the classic gated periodic-tomography movie.

For each of the n_bins phi-bin centers, reconstructs a volume from all
projections within the fixed radius of that phi (raw AND denoised sinogram),
keeps the middle detector-row slice, and renders a raw|denoised side-by-side
HEVC movie sweeping phi from 0 to 2*pi.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")
os.environ["PATH"] = f"/myhome/bin:{os.environ.get('PATH', '')}"

from sdate.tr_diffusion import phi_context as PC
from sdate.tr_diffusion.reconstruct import write_slice_movie

DATA_PATH = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01.h5"
DENOISED_PATH = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/denoised_sewellia_n2v_phictx.npy"
OUT_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata"
PCT = 0.05  # one of the T=11 radii already used for the context taps (radius = 5% of 2*pi)
N_BINS = 180  # phi-sweep resolution (half of the 360-bin context grid -- smooth enough, half the compute)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    with h5py.File(DATA_PATH, "r") as f:
        sino = f["sinogram"][:]
        theta = f["theta"][:].astype(np.float64)
        phase = f["phase"][:].astype(np.float64)
    denoised = np.load(DENOISED_PATH)

    dark = torch.zeros((), device=device, dtype=torch.float32)
    flat = torch.ones((), device=device, dtype=torch.float32)
    n_rows, n_pix = sino.shape[1], sino.shape[2]
    vol_shape = (n_rows, n_pix, n_pix)
    slice_row = n_rows // 2
    max_views = 12000
    radius = PCT * PC.TWO_PI

    bin_centers = PC.phi_bin_centers(N_BINS)
    log(f"sweeping {N_BINS} phi values, fixed radius pct={PCT:.0%} ({radius:.4f} rad)")

    raw_frames, den_frames = [], []
    t0 = time.time()
    for bi, phi_c in enumerate(bin_centers):
        vol_raw, n_raw = PC.reconstruct_phi_gated(sino, theta, phase, float(phi_c), radius, dark, flat,
                                                  device, vol_shape=vol_shape, max_views=max_views)
        vol_den, n_den = PC.reconstruct_phi_gated(denoised, theta, phase, float(phi_c), radius, dark, flat,
                                                  device, vol_shape=vol_shape, max_views=max_views)
        raw_frames.append(vol_raw[slice_row].cpu())
        den_frames.append(vol_den[slice_row].cpu())
        if bi % 20 == 0:
            elapsed = time.time() - t0
            log(f"  bin {bi+1}/{N_BINS} phi={phi_c:.3f} n_sel={n_raw} "
                f"elapsed={elapsed/60:.1f}min est_total={elapsed/(bi+1)*N_BINS/60:.1f}min")

    log("caching reconstructed slices before attempting movie write (expensive to recompute)...")
    stacked_raw = torch.stack(raw_frames)
    stacked_den = torch.stack(den_frames)
    np.savez_compressed(Path(OUT_DIR) / f"sewellia_phi_sweep_pct{int(PCT*100)}_slices.npz",
                        bin_centers=bin_centers, pct=PCT, radius=radius,
                        raw=stacked_raw.numpy().astype(np.float16),
                        den=stacked_den.numpy().astype(np.float16))
    log("cached slices -> sewellia_phi_sweep_pct{}_slices.npz".format(int(PCT * 100)))

    log("computing display range and assembling side-by-side frames...")
    vmin, vmax = float(np.percentile(stacked_raw.numpy(), 1)), float(np.percentile(stacked_raw.numpy(), 99.5))
    combo_frames = [torch.cat([r, d], dim=-1) for r, d in zip(raw_frames, den_frames)]

    out_path = Path(OUT_DIR) / f"sewellia_phi_sweep_pct{int(PCT*100)}_raw_vs_denoised.mov"
    write_slice_movie(combo_frames, out_path, vmin=vmin, vmax=vmax)
    log(f"wrote movie -> {out_path}")
    log("SUCCESS")


if __name__ == "__main__":
    main()
