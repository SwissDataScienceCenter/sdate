#!/usr/bin/env python3
"""Test the hypothesis that exchange/data is ALREADY dark/flat-corrected and
log-taken (so our counts_to_attenuation_flatdark step is a spurious SECOND
transform): feed the raw stored uint16 values straight into FBP, no calibration,
no log at all, and save the resulting volume for visual comparison.

    python scripts/sewellia_rawnolog_test.py
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
from sdate.tr_diffusion import reconstruct as R

FULL_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01"
DATA_PATH = f"{FULL_DIR}/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01.h5"
PHASE_TXT = f"{FULL_DIR}/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01_sound_phase.txt"
OUT_DIR = Path("/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/phi_context")

TARGET = 10000
N_BINS = 360
N_VIEWS = 2000
DET_BIN = 2


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

    n_rows, n_pix = sinogram.shape[1], sinogram.shape[2]
    rows_bin, pix_bin = n_rows // DET_BIN, n_pix // DET_BIN

    bin_centers = PC.phi_bin_centers(N_BINS)
    bin_idx = PC.snap_to_bins(phase, N_BINS)
    b = int(bin_idx[TARGET])
    phi_c = bin_centers[b]
    log(f"target i={TARGET}: bin={b}")

    mask = PC.phi_gate_mask(phase, phi_c, N_VIEWS * np.pi / phase.shape[0])
    idx = np.flatnonzero(mask)
    if idx.size > N_VIEWS:
        stride = idx.size / N_VIEWS
        idx = idx[np.floor(np.arange(N_VIEWS) * stride).astype(np.int64)]
    log(f"n_sel={idx.size}")

    p_raw = torch.from_numpy(np.ascontiguousarray(sinogram[idx])).to(device=device, dtype=torch.float32)
    angles = theta[idx]
    vol_raw = R.reconstruct(p_raw, angles, det_bin=DET_BIN, method="fbp",
                            vol_shape=(rows_bin, pix_bin, pix_bin), device=device, clamp=False)
    log(f"raw-no-log vol: shape={tuple(vol_raw.shape)} "
        f"min={vol_raw.min():.4g} max={vol_raw.max():.4g} mean={vol_raw.mean():.4g} std={vol_raw.std():.4g}")
    np.save(OUT_DIR / f"rawnolog_{TARGET}_detbin{DET_BIN}.npy", vol_raw.cpu().numpy().astype(np.float32))
    log(f"saved -> rawnolog_{TARGET}_detbin{DET_BIN}.npy")
    log("SUCCESS")


if __name__ == "__main__":
    main()
