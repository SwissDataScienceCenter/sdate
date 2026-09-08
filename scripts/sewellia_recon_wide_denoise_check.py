#!/usr/bin/env python3
"""Follow-up check: does N2V+context denoising show up at the WIDE (50%, all
20000 views) phi-gate, where angular sampling isn't the bottleneck? The
narrow-gate (1%, ~400 views) comparison showed almost no visible difference --
hypothesis: sparse-angle streak artifacts dominate there (a geometric
under-sampling problem N2V can't fix), masking any real denoising effect.
Reuses the already-saved denoised sinogram, no retraining/inference needed.
"""
from __future__ import annotations

import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")

from sdate.tr_diffusion import phi_context as PC

DATA_PATH = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01.h5"
DENOISED_PATH = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/denoised_sewellia_n2v_phictx.npy"
OUT_PATH = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/recon_compare_wide_denoise_check.npz"


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
    log(f"loaded raw {sino.shape} and denoised {denoised.shape}")

    dark = torch.zeros((), device=device, dtype=torch.float32)
    flat = torch.ones((), device=device, dtype=torch.float32)
    n_rows, n_pix = sino.shape[1], sino.shape[2]
    vol_shape = (n_rows, n_pix, n_pix)
    slice_row = n_rows // 2
    max_views = 12000

    targets = [4000, 10000, 16000]
    results = {}
    for ti in targets:
        phi_c = float(phase[ti])
        r_wide = 0.5 * PC.TWO_PI
        r_narrow = 0.01 * PC.TWO_PI
        log(f"target i={ti} theta={theta[ti]:.1f} phase={phase[ti]:.2f}")

        vol_raw_wide, n_w = PC.reconstruct_phi_gated(sino, theta, phase, phi_c, r_wide, dark, flat,
                                                      device, vol_shape=vol_shape, max_views=max_views)
        vol_den_wide, _ = PC.reconstruct_phi_gated(denoised, theta, phase, phi_c, r_wide, dark, flat,
                                                    device, vol_shape=vol_shape, max_views=max_views)
        vol_raw_narrow, n_n = PC.reconstruct_phi_gated(sino, theta, phase, phi_c, r_narrow, dark, flat,
                                                        device, vol_shape=vol_shape, max_views=max_views)
        vol_den_narrow, _ = PC.reconstruct_phi_gated(denoised, theta, phase, phi_c, r_narrow, dark, flat,
                                                      device, vol_shape=vol_shape, max_views=max_views)
        log(f"  n_wide={n_w} n_narrow={n_n}")
        results[ti] = dict(
            raw_wide=vol_raw_wide[slice_row].cpu().numpy(),
            den_wide=vol_den_wide[slice_row].cpu().numpy(),
            raw_narrow=vol_raw_narrow[slice_row].cpu().numpy(),
            den_narrow=vol_den_narrow[slice_row].cpu().numpy(),
            theta=float(theta[ti]), phase=float(phase[ti]), n_wide=n_w, n_narrow=n_n,
        )

    save_kwargs = {}
    for ti, r in results.items():
        for k, v in r.items():
            save_kwargs[f"t{ti}_{k}"] = v
    np.savez_compressed(OUT_PATH, targets=np.array(targets), **save_kwargs)
    log(f"saved -> {OUT_PATH}")
    log("SUCCESS")


if __name__ == "__main__":
    main()
