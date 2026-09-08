#!/usr/bin/env python3
"""Recover the T-level context visualization data (reconstructed slices per tap +
raw/transmission projection samples) for a few target frames, WITHOUT rebuilding
the main tap cache -- that cache (all 360 bins x 11 taps) already completed
successfully; only the optional in-run visualization snapshot was lost to a bug
in save_partial_slices (np.savez's silent ".npz" suffix-append broke the atomic
rename, so it never actually landed on disk -- see sewellia_real_phi_context.py).

This reruns just the handful of (bin, tap) reconstructions needed for the
requested target frames -- cheap (a few seconds each) once the raw data is
loaded -- and writes {tag}_slices.npz directly, matching what the main script
would have produced had the bug not existed.

    python scripts/sewellia_real_phictx_recover_slices.py
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
TAG = "sewellia_real_phictx"

N_BINS = 360
PCTS = list(PC.DEFAULT_PCTS)
DET_BIN = 4
MAX_VIEWS = 6000
TARGETS = [4000, 10000, 16000]


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
    rows_bin, pix_bin = n_rows // DET_BIN, n_pix // DET_BIN
    vol_shape = (rows_bin, pix_bin, pix_bin)
    det_shape_bin = (rows_bin, pix_bin)
    slice_row = rows_bin // 2
    radii = PC.radii_from_pcts(PCTS)
    T = len(PCTS)

    bin_centers = PC.phi_bin_centers(N_BINS)
    bin_idx = PC.snap_to_bins(phase, N_BINS)

    slices, proj_samples = {}, {}
    for ti in TARGETS:
        b = int(bin_idx[ti])
        phi_c = bin_centers[b]
        log(f"target i={ti}: bin={b} theta={theta[ti]:.2f} phase={phase[ti]:.3f}")

        raw_frame = sinogram[ti].astype(np.float32)
        span = np.clip((flat_full - dark_full).cpu().numpy(), 1e-3, None)
        proj_samples[ti] = dict(raw=raw_frame, transmission=(raw_frame - dark_full.cpu().numpy()) / span,
                                theta=float(theta[ti]), phase=float(phase[ti]))

        slices[ti] = np.zeros((T, pix_bin, pix_bin), dtype=np.float32)
        for c, (pct, r) in enumerate(zip(PCTS, radii)):
            vol, n_sel = PC.reconstruct_phi_gated(sinogram, theta, phase, phi_c, r, dark_full, flat_full,
                                                  device, det_bin=DET_BIN, vol_shape=vol_shape,
                                                  max_views=MAX_VIEWS)
            slices[ti][c] = vol[slice_row].cpu().numpy()
            log(f"  tap{c} (pct={pct:.0%}, n_sel={n_sel}): vol[min={vol.min():.4g} max={vol.max():.4g}]")
            del vol
            torch.cuda.empty_cache()

    save_kwargs = {f"target_{ti}": arr for ti, arr in slices.items()}
    for ti, d in proj_samples.items():
        save_kwargs[f"proj_{ti}_raw"] = d["raw"]
        save_kwargs[f"proj_{ti}_transmission"] = d["transmission"]
        save_kwargs[f"proj_{ti}_theta"] = d["theta"]
        save_kwargs[f"proj_{ti}_phase"] = d["phase"]
    out_path = OUT_DIR / f"{TAG}_slices.npz"
    np.savez_compressed(out_path, pcts=np.array(PCTS), row=slice_row,
                        target_idx=np.array(TARGETS),
                        target_theta=theta[np.array(TARGETS)],
                        target_phase=phase[np.array(TARGETS)],
                        **save_kwargs)
    log(f"wrote -> {out_path}")
    log("SUCCESS")


if __name__ == "__main__":
    main()
