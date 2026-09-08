#!/usr/bin/env python3
"""Empirically probe GPU memory usage of a full-resolution (det_bin=1) phi-gated
FBP reconstruction at increasing view counts, to find the real safe ceiling for
"100% = N projections" in the new T=5 full-resolution context design -- rather
than extrapolating from the det_bin=2/4 crash data, which showed the dominant
memory cost is NOT the ramp-filter FFT buffer (that part scales predictably) but
something else internal to ASTRA's BP3D_CUDA call that doesn't extrapolate
cleanly across resolutions.

    python scripts/sewellia_probe_fullres_memory.py
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
PROBE_VIEWS = [300, 500, 800, 900, 1000, 1100, 1200, 1500, 2000]

from pathlib import Path as _Path
OUT_DIR_GLOBAL = _Path("/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/phi_context")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")
    props = torch.cuda.get_device_properties(0)
    log(f"GPU: {props.name}, total_memory={props.total_memory/1e9:.1f}GB")

    with h5py.File(DATA_PATH, "r") as f:
        sinogram = f["exchange/data"][:]
        theta = f["exchange/theta"][:].astype(np.float64)
    phase = np.loadtxt(PHASE_TXT).astype(np.float64)
    log(f"loaded sinogram {sinogram.shape}")

    z = np.load(CALIB_PATH)
    dark_full = torch.from_numpy(z["dark_mean"]).to(device=device, dtype=torch.float32)
    flat_full = torch.from_numpy(z["white_mean"]).to(device=device, dtype=torch.float32)
    n_rows, n_pix = sinogram.shape[1], sinogram.shape[2]
    vol_shape = (n_rows, n_pix, n_pix)  # det_bin=1 -- native full resolution

    bin_centers = PC.phi_bin_centers(N_BINS)
    bin_idx = PC.snap_to_bins(phase, N_BINS)
    b = int(bin_idx[TARGET])
    phi_c = bin_centers[b]
    log(f"target i={TARGET}: bin={b}")

    cp_pool = cp.get_default_memory_pool()
    for n_target in PROBE_VIEWS:
        radius = n_target * np.pi / phase.shape[0]  # rough radius guess (uniform-density formula), just to select ~n_target views
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        cp_pool.free_all_blocks()
        t0 = time.time()
        try:
            vol, n_sel = PC.reconstruct_phi_gated(sinogram, theta, phase, phi_c, radius, dark_full, flat_full,
                                                  device, det_bin=1, vol_shape=vol_shape, max_views=n_target)
            torch_peak = torch.cuda.max_memory_allocated() / 1e9
            cp_used = cp_pool.used_bytes() / 1e9
            cp_total = cp_pool.total_bytes() / 1e9
            log(f"n_target={n_target}: n_sel={n_sel} recon_s={time.time()-t0:.2f} "
                f"torch_peak={torch_peak:.2f}GB cupy_used={cp_used:.2f}GB cupy_pool_total={cp_total:.2f}GB -- OK")
            del vol
        except (torch.cuda.OutOfMemoryError, cp.cuda.memory.OutOfMemoryError) as e:
            log(f"n_target={n_target}: OOM -- {e!r}")
            log("stopping probe (higher targets will only be worse)")
            break
        torch.cuda.empty_cache()
        cp_pool.free_all_blocks()

    # ------------------------------------------------------------------- #
    # User hypothesis: the color-scheme inversion vs. the paper's figures might
    # mean exchange/data is ALREADY dark/flat-corrected and log-taken (like the
    # small precorrected preview file was), and our counts_to_attenuation_flatdark
    # step is a SECOND, spurious transform on top of that. Test directly: skip
    # calibration and the log entirely, feed the raw stored values straight into
    # FBP, at det_bin=2 (matches the earlier z-scan's slice indexing) so slice 29
    # is directly comparable to the det_bin=2 z-montage already shown.
    # ------------------------------------------------------------------- #
    log("raw-no-log test: feeding exchange/data straight into FBP, no calibration/log ...")
    from sdate.tr_diffusion import reconstruct as R
    det_bin_raw = 2
    rows_bin, pix_bin = n_rows // det_bin_raw, n_pix // det_bin_raw
    mask = PC.phi_gate_mask(phase, phi_c, 2000 * np.pi / phase.shape[0])
    idx = np.flatnonzero(mask)
    if idx.size > 2000:
        stride = idx.size / 2000
        idx = idx[np.floor(np.arange(2000) * stride).astype(np.int64)]
    p_raw = torch.from_numpy(np.ascontiguousarray(sinogram[idx])).to(device=device, dtype=torch.float32)
    angles = theta[idx]
    vol_raw = R.reconstruct(p_raw, angles, det_bin=det_bin_raw, method="fbp",
                            vol_shape=(rows_bin, pix_bin, pix_bin), device=device, clamp=False)
    log(f"raw-no-log vol: shape={tuple(vol_raw.shape)} n_sel={idx.size} "
        f"min={vol_raw.min():.4g} max={vol_raw.max():.4g} mean={vol_raw.mean():.4g}")
    np.save(OUT_DIR_GLOBAL / f"rawnolog_{TARGET}_detbin{det_bin_raw}.npy", vol_raw.cpu().numpy().astype(np.float32))
    log(f"saved -> rawnolog_{TARGET}_detbin{det_bin_raw}.npy")

    log("SUCCESS")


if __name__ == "__main__":
    main()
