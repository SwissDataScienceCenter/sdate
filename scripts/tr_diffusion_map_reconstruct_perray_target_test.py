#!/usr/bin/env python3
"""Decisive model-vs-optimizer test. Both the alpha_post<1 and outlier-precision
hypotheses were falsified (see alphamask_test.py, precision_test.py). This test
skips GD/ASTRA-backprojection entirely: compute the closed-form per-ray MAP
target lam* = (alpha_post-1)/beta_post independently for every ray (no
tomographic consistency enforced at all), convert straight to an attenuation
sinogram, and do ONE plain FBP.

  - If that FBP looks clean (close to poissonhead_fbp/GT) -> the per-ray
    statistical targets are fine; the streaks are a GD/optimizer convergence
    bug (wrong lr schedule, bad warm start sensitivity, Adam momentum after
    the l_clip boundary, etc.), not a modeling problem. Masking would be the
    wrong lever.
  - If it ALREADY shows streaks with zero optimization involved -> the targets
    themselves are tomographically inconsistent as a set of independent
    per-ray numbers; GD is only faithfully reproducing that inconsistency.
    Masking/regularization is the right lever.
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
from sdate.tr_diffusion.map_reconstruct import gamma_shape_rate  # noqa: E402

sys.path.insert(0, "/myhome/sdate/scripts")
import tr_diffusion_map_reconstruct_compare as C  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_NPZ = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache/map_debug_perray_target.npz"
OUT_JSON = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache/map_debug_perray_target_metrics.json"


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

    poissonhead_atten = R.destripe_sinogram(R.counts_to_attenuation_flatdark(mu_net, dark, flat), 31)
    poissonhead_vol = R.reconstruct(poissonhead_atten, ang, det_bin=C.DET_BIN, method="fbp", device=device)
    ps, ss = R.masked_scores(gt_vol, poissonhead_vol, mask, dr)
    log(f"poissonhead_fbp (reference)  PSNR={ps:.2f}  SSIM={ss:.3f}")

    alpha, beta = gamma_shape_rate(mu_net_b, var_net_b)
    n = C.DOSE * noisy_b
    alpha_post = alpha + n
    beta_post = beta + C.DOSE
    lam_star = (alpha_post - 1.0) / beta_post
    log(f"lam_star: min={float(lam_star.min()):.4g} max={float(lam_star.max()):.4g} mean={float(lam_star.mean()):.4g}")
    log(f"noisy_b (raw, same units): min={float(noisy_b.min()):.4g} max={float(noisy_b.max()):.4g} mean={float(noisy_b.mean()):.4g}")
    log(f"mu_net_b (prior mean, same units): min={float(mu_net_b.min()):.4g} max={float(mu_net_b.max()):.4g} mean={float(mu_net_b.mean()):.4g}")

    atten_star = R.destripe_sinogram(R.counts_to_attenuation_flatdark(lam_star, dark_b, flat_b), 31)
    star_vol = R.reconstruct(atten_star, ang, det_bin=1, method="fbp", device=device)
    ps_star, ss_star = R.masked_scores(gt_vol, star_vol, mask, dr)
    sh_star = R.masked_sharpness(gt_vol, star_vol, mask)
    log(f"lam_star FBP (no GD, no tomography enforced)  PSNR={ps_star:.2f}  SSIM={ss_star:.3f}  sharpness={sh_star:.3f}")

    # also without destriping, in case destripe itself is masking/creating structure
    atten_star_nods = R.counts_to_attenuation_flatdark(lam_star, dark_b, flat_b)
    star_vol_nods = R.reconstruct(atten_star_nods, ang, det_bin=1, method="fbp", device=device)
    ps_nods, ss_nods = R.masked_scores(gt_vol, star_vol_nods, mask, dr)
    log(f"lam_star FBP (no destripe)  PSNR={ps_nods:.2f}  SSIM={ss_nods:.3f}")

    mid = gt_vol.shape[0] // 2
    slices = {
        "GT": gt_vol[mid].cpu().numpy(),
        "poissonhead_fbp": poissonhead_vol[mid].cpu().numpy(),
        "lam_star_fbp": star_vol[mid].cpu().numpy(),
        "lam_star_fbp_nodestripe": star_vol_nods[mid].cpu().numpy(),
    }
    metrics = {
        "poissonhead_fbp": {"psnr": float(ps), "ssim": float(ss)},
        "lam_star_fbp": {"psnr": float(ps_star), "ssim": float(ss_star), "sharpness": float(sh_star)},
        "lam_star_fbp_nodestripe": {"psnr": float(ps_nods), "ssim": float(ss_nods)},
    }
    np.savez(OUT_NPZ, **slices)
    Path(OUT_JSON).write_text(json.dumps(metrics, indent=2))
    log(f"saved -> {OUT_NPZ}, {OUT_JSON}")
    log("SUMMARY " + json.dumps(metrics, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
