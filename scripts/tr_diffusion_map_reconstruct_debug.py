#!/usr/bin/env python3
"""Debug variant of tr_diffusion_map_reconstruct_compare.py: ONE window, verbose
map_reconstruct diagnostics (alpha/beta/mu0/loss trajectory), and a couple of
learning rates compared side by side, to find why the real-data MAP arm came
out catastrophically wrong (PSNR -22dB, sharpness 12.7, i.e. worse than the raw
noisy floor) while baseline/poissonhead FBP arms reconstruct fine (28dB) on the
SAME window -- confirming the bug is specific to map_reconstruct's real-data
behaviour, not the geometry/calibration/window plumbing.
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

    log("fetching MAP prior (mu_net, var_net)")
    mu_net, var_net = C.get_map_prior(idx)

    dark_b = R.bin_detector(dark.unsqueeze(0), C.DET_BIN)[0]
    flat_b = R.bin_detector(flat.unsqueeze(0), C.DET_BIN)[0]
    noisy_b = R.bin_detector(noisy, C.DET_BIN)
    mu_net_b = R.bin_detector(mu_net, C.DET_BIN)
    var_net_b = R.bin_detector(var_net, C.DET_BIN) / float(C.DET_BIN * C.DET_BIN)

    # GT volume + mask, for scoring each attempt
    gt_atten = R.destripe_sinogram(R.counts_to_attenuation_flatdark(gt, dark, flat), 31)
    gt_vol = R.reconstruct(gt_atten, ang, det_bin=C.DET_BIN, method="fbp", device=device)
    nslices, hplane = C.CROP[0] // C.DET_BIN, C.CROP[1] // C.DET_BIN
    mask = R.make_mask(hplane, hplane).to(device)
    dr = float(gt_vol[..., mask].max() - gt_vol[..., mask].min())

    log(f"noisy_b: min={noisy_b.min().item():.4g} max={noisy_b.max().item():.4g} mean={noisy_b.mean().item():.4g}")
    log(f"flat_b: min={flat_b.min().item():.4g} max={flat_b.max().item():.4g} mean={flat_b.mean().item():.4g}")
    log(f"dark_b: min={dark_b.min().item():.4g} max={dark_b.max().item():.4g} mean={dark_b.mean().item():.4g}")

    ps_fbp, ss_fbp = R.masked_scores(gt_vol, R.reconstruct(
        R.destripe_sinogram(R.counts_to_attenuation_flatdark(noisy, dark, flat), 31),
        ang, det_bin=C.DET_BIN, method="fbp", device=device), mask, dr)
    log(f"(reference) FBP-of-noisy (no denoiser, no prior)  PSNR={ps_fbp:.2f}  SSIM={ss_fbp:.3f}")

    for lr, n_iters in [(1e-3, 2000), (3e-3, 2000), (1e-2, 2000)]:
        log(f"\n=== map_reconstruct lr={lr} n_iters={n_iters} ===")
        t0 = time.time()
        mu_rec = map_reconstruct(
            noisy_b, mu_net_b, var_net_b, ang, flat_b, dark_b,
            vol_shape=None, n_iters=n_iters, lr=lr, warm_start="fbp",
            dose=C.DOSE, log_every=n_iters // 8, device=device,
        )
        ps, ss = R.masked_scores(gt_vol, mu_rec, mask, dr)
        sh = R.masked_sharpness(gt_vol, mu_rec, mask)
        log(f"lr={lr} n_iters={n_iters}  PSNR={ps:.2f}  SSIM={ss:.3f}  sharpness={sh:.3f}  "
            f"mu_rec[min={mu_rec.min().item():.4g},max={mu_rec.max().item():.4g}]  "
            f"({time.time()-t0:.1f}s)")

    log("DONE")


if __name__ == "__main__":
    main()
