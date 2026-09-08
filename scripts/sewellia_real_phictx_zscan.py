#!/usr/bin/env python3
"""Diagnostic: reconstruct the FULL z-stack (all detector rows, not just the middle
one) for one representative frame, at two resolutions, to check (a) whether a
DIFFERENT z-height matches the PeriodRecon paper's figure better than the arbitrary
middle row we've been showing, and (b) whether det_bin=4 (our disk/memory-budget
choice for the full 360-bin cache) is itself suppressing contrast/dynamic range
relative to a higher-resolution one-off reconstruction.

Only ONE frame is reconstructed here (this is a diagnostic, not part of the cache
build) so a much higher resolution (det_bin=2, 290x288) is affordable -- the
memory ceiling that forced det_bin=4 for the full 360-bin job was about the
WIDEST gate's view count (up to 12000), not about resolution per se; at a more
modest max_views=2000 (already confirmed safe empirically at det_bin=2), det_bin=2
easily fits in a single one-off call.

    python scripts/sewellia_real_phictx_zscan.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")

from sdate.tr_diffusion import phi_context as PC

FULL_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01"
DATA_PATH = f"{FULL_DIR}/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01.h5"
PHASE_TXT = f"{FULL_DIR}/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01_sound_phase.txt"
CALIB_PATH = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/sewellia_real_calibration.npz"
OUT_DIR = Path("/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/phi_context")

N_BINS = 360
TARGET = 10000
PCT_WIDE = 0.5  # the cleanest/widest gate we build taps for


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    with h5py.File(DATA_PATH, "r") as f:
        sinogram = f["exchange/data"][:]
        theta = f["exchange/theta"][:].astype(np.float64)
    phase = np.loadtxt(PHASE_TXT).astype(np.float64)
    log(f"loaded sinogram {sinogram.shape}")

    z = np.load(CALIB_PATH)
    dark_full = torch.from_numpy(z["dark_mean"]).to(device=device, dtype=torch.float32)
    flat_full = torch.from_numpy(z["white_mean"]).to(device=device, dtype=torch.float32)
    n_rows, n_pix = sinogram.shape[1], sinogram.shape[2]

    bin_centers = PC.phi_bin_centers(N_BINS)
    bin_idx = PC.snap_to_bins(phase, N_BINS)
    b = int(bin_idx[TARGET])
    phi_c = bin_centers[b]
    radius = PCT_WIDE * PC.TWO_PI
    log(f"target i={TARGET}: bin={b} theta={theta[TARGET]:.2f} phase={phase[TARGET]:.3f}")

    configs = [
        ("detbin4_mv6000", 4, 6000),
        ("detbin2_mv2000", 2, 2000),
    ]
    for name, det_bin, max_views in configs:
        rows_bin, pix_bin = n_rows // det_bin, n_pix // det_bin
        vol_shape = (rows_bin, pix_bin, pix_bin)
        t0 = time.time()
        vol, n_sel = PC.reconstruct_phi_gated(sinogram, theta, phase, phi_c, radius, dark_full, flat_full,
                                              device, det_bin=det_bin, vol_shape=vol_shape, max_views=max_views)
        log(f"{name}: n_sel={n_sel} vol_shape={tuple(vol.shape)} recon_s={time.time()-t0:.2f} "
            f"vol[min={vol.min():.4g} max={vol.max():.4g} mean={vol.mean():.4g} std={vol.std():.4g}]")
        np.save(OUT_DIR / f"zscan_{TARGET}_{name}.npy", vol.cpu().numpy().astype(np.float32))
        del vol
        torch.cuda.empty_cache()

    log("SUCCESS")


if __name__ == "__main__":
    main()
