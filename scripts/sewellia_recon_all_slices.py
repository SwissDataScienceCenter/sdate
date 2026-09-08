#!/usr/bin/env python3
"""Reconstruct ALL 6 detector-row slices (not just the middle one) for one
representative target projection, at both the narrow and wide phi-gates,
raw vs. denoised input -- a full look at the little reconstructed volume
rather than a single cross-section.
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
OUT_PATH = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/recon_all_slices.npz"
TARGET = 10000


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
    max_views = 12000

    phi_c = float(phase[TARGET])
    gates = {"narrow": 0.01 * PC.TWO_PI, "wide": 0.5 * PC.TWO_PI}
    out = {}
    for gate_name, r in gates.items():
        for src_name, src in [("raw", sino), ("den", denoised)]:
            t0 = time.time()
            vol, n_sel = PC.reconstruct_phi_gated(src, theta, phase, phi_c, r, dark, flat, device,
                                                  vol_shape=vol_shape, max_views=max_views)
            out[f"{gate_name}_{src_name}"] = vol.cpu().numpy()  # (6, 556, 556) -- ALL rows
            log(f"{gate_name}/{src_name}: n_sel={n_sel} recon_s={time.time()-t0:.2f}")

    np.savez_compressed(OUT_PATH, target=TARGET, theta=theta[TARGET], phase=phase[TARGET], **out)
    log(f"saved -> {OUT_PATH}")
    log("SUCCESS")


if __name__ == "__main__":
    main()
