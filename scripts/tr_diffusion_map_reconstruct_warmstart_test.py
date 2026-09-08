#!/usr/bin/env python3
"""Does GD divergence depend on the starting point? Warm-start map_reconstruct
from the closed-form per-ray target's FBP (PSNR 28.4, see
tr_diffusion_map_reconstruct_perray_target_test.py) instead of FBP-of-raw-noisy
(PSNR ~11.3), via the new mu_init override, and see whether GD holds near that
much better starting point or still drifts away regardless. NOTE: this warm
start leaks mu_net/var_net info into the initialization -- a deliberate,
diagnostic-only violation of the module's normal anti-leakage warm-start
policy, done here purely to isolate "does starting point matter" from "is the
objective/optimizer itself the problem".
"""
from __future__ import annotations

import os

os.environ["PATH"] = "/myhome/bin:" + os.environ.get("PATH", "")

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")

from sdate.tr_diffusion import reconstruct as R  # noqa: E402
from sdate.tr_diffusion.map_reconstruct import gamma_shape_rate, map_reconstruct  # noqa: E402

sys.path.insert(0, "/myhome/sdate/scripts")
import tr_diffusion_map_reconstruct_compare as C  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_NPZ = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache/map_debug_warmstart.npz"
OUT_JSON = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache/map_debug_warmstart_metrics.json"


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
    nslices, hplane = C.CROP[0] // C.DET_BIN, C.CROP[1] // C.DET_BIN
    mask = R.make_mask(hplane, hplane).to(device)
    dr = float(gt_vol[..., mask].max() - gt_vol[..., mask].min())
    vol_shape = (noisy_b.shape[1], noisy_b.shape[2], noisy_b.shape[2])

    # closed-form per-ray target -> FBP, our PSNR=28.4 starting point
    alpha, beta = gamma_shape_rate(mu_net_b, var_net_b)
    n = C.DOSE * noisy_b
    alpha_post = alpha + n
    beta_post = beta + C.DOSE
    lam_star = (alpha_post - 1.0) / beta_post
    atten_star = R.destripe_sinogram(R.counts_to_attenuation_flatdark(lam_star, dark_b, flat_b), 31)
    star_vol = R.reconstruct(atten_star, ang, det_bin=1, method="fbp", device=device)
    ps0, ss0 = R.masked_scores(gt_vol, star_vol, mask, dr)
    log(f"closed-form target FBP (mu_init candidate)  PSNR={ps0:.2f}  SSIM={ss0:.3f}")

    mid = gt_vol.shape[0] // 2
    slices = {"GT": gt_vol[mid].cpu().numpy(), "closed_form_target_fbp": star_vol[mid].cpu().numpy()}
    metrics = {"closed_form_target_fbp": {"psnr": float(ps0), "ssim": float(ss0)}}

    for lr, n_iters in [(0.01, 200), (0.01, 2000), (0.001, 2000)]:
        t0 = time.time()
        mu_rec = map_reconstruct(
            noisy_b, mu_net_b, var_net_b, ang, flat_b, dark_b,
            vol_shape=vol_shape, n_iters=n_iters, lr=lr, warm_start="fbp",
            dose=C.DOSE, tv_weight=0.0, log_every=0, device=device,
            mu_init=star_vol,
        )
        ps, ss = R.masked_scores(gt_vol, mu_rec, mask, dr)
        sh = R.masked_sharpness(gt_vol, mu_rec, mask)
        tag = f"warmstart_lr{lr}_it{n_iters}"
        metrics[tag] = {"psnr": float(ps), "ssim": float(ss), "sharpness": float(sh), "max": float(mu_rec.max())}
        log(f"{tag}  PSNR={ps:.2f} SSIM={ss:.3f} sharp={sh:.3f} max={mu_rec.max().item():.4g}  ({time.time()-t0:.1f}s)")
        slices[tag] = mu_rec[mid].cpu().numpy()

    np.savez(OUT_NPZ, **slices)
    Path(OUT_JSON).write_text(json.dumps(metrics, indent=2))
    log(f"saved -> {OUT_NPZ}, {OUT_JSON}")
    log("SUMMARY " + json.dumps(metrics, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
