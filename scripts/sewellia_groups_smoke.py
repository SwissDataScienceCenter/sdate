#!/usr/bin/env python3
"""Validate reconstruct_phi_gated_groups: for one wide gate (h > 3600, so
m=4), run 4 sequential disjoint full-resolution (det_bin=1) reconstructions
in the SAME process and confirm none of them OOM -- the demanding case for
the new multi-group context design (up to 20 channels total: 5 T-levels x
up to 4 disjoint groups each).

    python scripts/sewellia_groups_smoke.py
"""
from __future__ import annotations

import sys
import time

import cupy as cp
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

TARGET = 10000
N_BINS = 360
# pick a radius wide enough that h clears 3600 (m=4) -- ~4500 projections at pct~11%
PCT_WIDE = 0.11


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
    vol_shape = (n_rows, n_pix, n_pix)

    bin_centers = PC.phi_bin_centers(N_BINS)
    bin_idx = PC.snap_to_bins(phase, N_BINS)
    b = int(bin_idx[TARGET])
    phi_c = bin_centers[b]
    radius = PCT_WIDE * PC.TWO_PI
    log(f"target i={TARGET}: bin={b}, radius pct={PCT_WIDE:.0%}")

    t0 = time.time()
    volumes, n_sel = PC.reconstruct_phi_gated_groups(sinogram, theta, phase, phi_c, radius, dark_full, flat_full,
                                                      device, det_bin=1, vol_shape=vol_shape,
                                                      group_size=900, max_groups=4)
    log(f"n_sel={n_sel} (total available in gate) -> got {len(volumes)} disjoint groups, "
        f"total_time={time.time()-t0:.1f}s")
    for i, vol in enumerate(volumes):
        log(f"  group {i}: shape={tuple(vol.shape)} min={vol.min():.4g} max={vol.max():.4g} mean={vol.mean():.4g}")
    np.savez_compressed("/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/phi_context/groups_smoke.npz",
                        **{f"vol{i}": v.cpu().numpy().astype(np.float32) for i, v in enumerate(volumes)})
    log("SUCCESS")


if __name__ == "__main__":
    main()
