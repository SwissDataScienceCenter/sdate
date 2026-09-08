#!/usr/bin/env python3
"""Capture raw slice arrays (GT, warm start, and a few map_reconstruct attempts
at different lr/iteration counts) for one real window, saved as .npy to
persistent storage, for visual/local inspection of the divergence found in
tr_diffusion_map_reconstruct_debug.py.
"""
from __future__ import annotations

import os

os.environ["PATH"] = "/myhome/bin:" + os.environ.get("PATH", "")

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")

from sdate.tr_diffusion import reconstruct as R  # noqa: E402
from sdate.tr_diffusion.map_reconstruct import map_reconstruct  # noqa: E402

sys.path.insert(0, "/myhome/sdate/scripts")
import tr_diffusion_map_reconstruct_compare as C  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache/map_debug_slices.npz"


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main():
    log(f"device={device}")
    src = R.MemmapFrameSource(C.MEMMAP, C.MOV)
    dark = torch.from_numpy(np.load(f"{C.CACHE}/dark_map.npy")).float().to(device)
    flat = torch.from_numpy(np.load(f"{C.CACHE}/flat_map.npy")).float().to(device)

    s = 410300
    win = R.window_length_frames(180.0)
    idx = np.arange(s, s + win)
    ang = R.projection_angles(idx, deg_per_frame=R.DEG_PER_FRAME)

    gt = R.native_window_gpu(src, idx, C.CROP, C.AXIS_COL, device)
    g = torch.Generator(device=device).manual_seed(C.NOISE_SEED + int(s))
    noisy = R.noisy_window_gpu(gt, C.DOSE, generator=g)

    log("fetching MAP prior")
    mu_net, var_net = C.get_map_prior(idx)

    dark_b = R.bin_detector(dark.unsqueeze(0), C.DET_BIN)[0]
    flat_b = R.bin_detector(flat.unsqueeze(0), C.DET_BIN)[0]
    noisy_b = R.bin_detector(noisy, C.DET_BIN)
    mu_net_b = R.bin_detector(mu_net, C.DET_BIN)
    var_net_b = R.bin_detector(var_net, C.DET_BIN) / float(C.DET_BIN * C.DET_BIN)

    gt_atten = R.destripe_sinogram(R.counts_to_attenuation_flatdark(gt, dark, flat), 31)
    gt_vol = R.reconstruct(gt_atten, ang, det_bin=C.DET_BIN, method="fbp", device=device)

    baseline_atten = R.destripe_sinogram(R.counts_to_attenuation_flatdark(noisy, dark, flat), 31)
    # also grab poissonhead's own FBP (mu_net converted via the SAME flat/dark
    # transform then FBP -- i.e. what the existing pipeline already does well)
    mu_net_full = mu_net  # (V,R,C) at full res, pre-bin
    poissonhead_atten = R.destripe_sinogram(R.counts_to_attenuation_flatdark(mu_net_full, dark, flat), 31)
    poissonhead_vol = R.reconstruct(poissonhead_atten, ang, det_bin=C.DET_BIN, method="fbp", device=device)
    noisy_fbp_vol = R.reconstruct(baseline_atten, ang, det_bin=C.DET_BIN, method="fbp", device=device)

    mid = gt_vol.shape[0] // 2
    slices = {
        "GT": gt_vol[mid].cpu().numpy(),
        "noisy_fbp": noisy_fbp_vol[mid].cpu().numpy(),
        "poissonhead_fbp": poissonhead_vol[mid].cpu().numpy(),
    }

    # warm start alone (FBP of noisy, same as map_reconstruct's own internal start)
    from astra_torch.lamino import fbp_reconstruction_masked
    atten0 = R.counts_to_attenuation_flatdark(noisy_b, dark_b, flat_b)
    vol_shape = (noisy_b.shape[1], noisy_b.shape[2], noisy_b.shape[2])
    mu0 = fbp_reconstruction_masked(atten0, ang, lamino_angle_deg=0.0, vol_shape=vol_shape,
                                    det_spacing_mm=1.0, filter_type="hann", device=device).clamp_min(0.0)
    slices["warm_start"] = mu0[mid].cpu().numpy()

    for lr, n_iters in [(0.01, 200), (0.001, 200), (0.01, 2000), (0.001, 2000)]:
        log(f"map_reconstruct lr={lr} n_iters={n_iters}")
        mu_rec = map_reconstruct(
            noisy_b, mu_net_b, var_net_b, ang, flat_b, dark_b,
            vol_shape=None, n_iters=n_iters, lr=lr, warm_start="fbp",
            dose=C.DOSE, log_every=0, device=device,
        )
        slices[f"map_lr{lr}_it{n_iters}"] = mu_rec[mid].cpu().numpy()

    np.savez(OUT, **slices)
    log(f"saved -> {OUT}")
    for k, v in slices.items():
        log(f"  {k:22s} min={v.min():.4g} max={v.max():.4g} mean={v.mean():.4g}")
    log("DONE")


if __name__ == "__main__":
    main()
