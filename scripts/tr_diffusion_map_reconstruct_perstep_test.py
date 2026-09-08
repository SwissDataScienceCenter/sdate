#!/usr/bin/env python3
"""Per-step trace: warm-start map_reconstruct's GD loop from the closed-form
target FBP (PSNR 28.4) at lr=0.01, and record PSNR/SSIM/mu.max()/loss at
EVERY step from 0 to 10, to see whether the divergence is a single violent
first step (points to a gradient-scale/bug issue) or a gradual ramp
(points to a slower semi-convergence-style drift).
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
OUT_NPZ = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache/map_debug_perstep.npz"
OUT_JSON = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache/map_debug_perstep_metrics.json"


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

    alpha, beta = gamma_shape_rate(mu_net_b, var_net_b)
    n = C.DOSE * noisy_b
    alpha_post = alpha + n
    beta_post = beta + C.DOSE
    lam_star = (alpha_post - 1.0) / beta_post
    atten_star = R.destripe_sinogram(R.counts_to_attenuation_flatdark(lam_star, dark_b, flat_b), 31)
    star_vol = R.reconstruct(atten_star, ang, det_bin=1, method="fbp", device=device)
    ps0, ss0 = R.masked_scores(gt_vol, star_vol, mask, dr)
    log(f"step -1 (closed-form target FBP, the mu_init)  PSNR={ps0:.2f}  SSIM={ss0:.3f}  max={float(star_vol.max()):.4g}")

    from astra_torch.lamino import build_lamino_projector
    mu = star_vol.detach().clone().requires_grad_(True)
    proj_layer = build_lamino_projector(vol_shape=vol_shape, det_shape=(noisy_b.shape[1], noisy_b.shape[2]),
                                        angles_deg=ang, lamino_angle_deg=0.0, tilt_angle_deg=0.0,
                                        voxel_size_mm=1.0, det_spacing_mm=1.0, device=device)
    optimizer = torch.optim.Adam([mu], lr=0.01)
    eps, l_clip = 1e-6, 20.0

    mid = gt_vol.shape[0] // 2
    slices = {"GT": gt_vol[mid].cpu().numpy(), "step_-1_init": star_vol[mid].cpu().numpy()}
    metrics = {"step_-1": {"psnr": float(ps0), "ssim": float(ss0), "max": float(star_vol.max())}}

    for step in range(10):
        optimizer.zero_grad(set_to_none=True)
        l = proj_layer(mu.unsqueeze(0).unsqueeze(0))[0]
        l_raw_min, l_raw_max = float(l.min()), float(l.max())
        l = l.clamp(-l_clip, l_clip)
        lam = flat_b * torch.exp(-l) + dark_b
        loss = (beta_post * lam - (alpha_post - 1.0) * torch.log(lam.clamp_min(eps))).sum()
        loss.backward()
        grad_norm = float(mu.grad.norm())
        grad_max = float(mu.grad.abs().max())
        optimizer.step()
        with torch.no_grad():
            mu.clamp_(min=0.0)
        with torch.no_grad():
            ps, ss = R.masked_scores(gt_vol, mu.detach(), mask, dr)
        log(f"step {step}  loss={loss.item():.6g}  grad_norm={grad_norm:.4g}  grad_max={grad_max:.4g}  "
           f"l_raw[{l_raw_min:.4g},{l_raw_max:.4g}]  mu[max={float(mu.max()):.4g}]  PSNR={ps:.2f}  SSIM={ss:.3f}")
        metrics[f"step_{step}"] = {"psnr": float(ps), "ssim": float(ss), "max": float(mu.max()),
                                   "loss": float(loss.item()), "grad_norm": grad_norm, "grad_max": grad_max,
                                   "l_raw_min": l_raw_min, "l_raw_max": l_raw_max}
        slices[f"step_{step}"] = mu.detach()[mid].cpu().numpy()

    np.savez(OUT_NPZ, **slices)
    Path(OUT_JSON).write_text(json.dumps(metrics, indent=2))
    log(f"saved -> {OUT_NPZ}, {OUT_JSON}")
    log("DONE")


if __name__ == "__main__":
    main()
