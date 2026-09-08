#!/usr/bin/env python3
"""alpha_post<=1 hypothesis was falsified (0% of rays, see
tr_diffusion_map_reconstruct_alphamask_test.py). New hypothesis: beta_post ~=
mu_net/var_net is a per-ray PRECISION weight in the summed loss; context-only
var_net occasionally underestimates true uncertainty (network overconfident),
giving that ray an outsized weight relative to its neighbors -- a single such
ray dominates a GD step and blows up along its exact line (streak geometry).

This script:
  1. Reports the beta_post distribution (percentiles) -- look for a heavy tail.
  2. Backprojects an "is this ray in the top X% by beta_post" indicator via FBP,
     to see if its spatial pattern lines up with the visible streaks.
  3. Backprojects the per-ray loss evaluated AT THE WARM START (before any GD
     step) -- same idea, a more direct proxy for "which rays dominate step 0".
  4. Tests a fix: floor var_net_b at its Pth percentile (Winsorize the precision
     weight) for P in {1, 5, 10}, rerun map_reconstruct at lr=0.01, n_iters in
     {200, 2000}, compare metrics against the unfloored baseline.
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
OUT_NPZ = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache/map_debug_precision.npz"
OUT_JSON = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache/map_debug_precision_metrics.json"


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def pctiles(x, name):
    x = x.flatten()
    qs = [0, 0.01, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 99.99, 100]
    vals = np.percentile(x.cpu().numpy(), qs)
    log(f"  {name} percentiles " + " ".join(f"{q}%={v:.4g}" for q, v in zip(qs, vals)))


def run_map(y, mu_net, var_net, angles_deg, I0, dark, vol_shape, n_iters, lr, dose, device):
    from astra_torch.lamino import build_lamino_projector, fbp_reconstruction_masked
    from sdate.tr_diffusion.reconstruct import counts_to_attenuation_flatdark
    alpha, beta = gamma_shape_rate(mu_net, var_net)
    n = float(dose) * y
    alpha_post = alpha + n
    beta_post = beta + float(dose)
    atten0 = counts_to_attenuation_flatdark(y, dark, I0)
    mu0 = fbp_reconstruction_masked(atten0, angles_deg, lamino_angle_deg=0.0, vol_shape=vol_shape,
                                    det_spacing_mm=1.0, filter_type="hann", device=device).clamp_min(0.0)
    mu = mu0.detach().clone().requires_grad_(True)
    proj_layer = build_lamino_projector(vol_shape=vol_shape, det_shape=(y.shape[1], y.shape[2]),
                                        angles_deg=angles_deg, lamino_angle_deg=0.0, tilt_angle_deg=0.0,
                                        voxel_size_mm=1.0, det_spacing_mm=1.0, device=device)
    optimizer = torch.optim.Adam([mu], lr=lr)
    eps, l_clip = 1e-6, 20.0
    for step in range(n_iters):
        optimizer.zero_grad(set_to_none=True)
        l = proj_layer(mu.unsqueeze(0).unsqueeze(0))[0]
        l = l.clamp(-l_clip, l_clip)
        lam = I0 * torch.exp(-l) + dark
        loss = (beta_post * lam - (alpha_post - 1.0) * torch.log(lam.clamp_min(eps))).sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            mu.clamp_(min=0.0)
    return mu.detach()


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

    log("--- distributions ---")
    pctiles(var_net_b, "var_net_b")
    pctiles(mu_net_b, "mu_net_b")
    pctiles(beta, "beta (prior precision)")
    pctiles(beta_post, "beta_post")
    pctiles(alpha_post, "alpha_post")

    # per-ray loss AT WARM START, to find step-0 dominant rays
    from astra_torch.lamino import build_lamino_projector, fbp_reconstruction_masked
    atten0 = R.counts_to_attenuation_flatdark(noisy_b, dark_b, flat_b)
    mu0 = fbp_reconstruction_masked(atten0, ang, lamino_angle_deg=0.0, vol_shape=vol_shape,
                                    det_spacing_mm=1.0, filter_type="hann", device=device).clamp_min(0.0)
    proj_layer = build_lamino_projector(vol_shape=vol_shape, det_shape=(noisy_b.shape[1], noisy_b.shape[2]),
                                        angles_deg=ang, lamino_angle_deg=0.0, tilt_angle_deg=0.0,
                                        voxel_size_mm=1.0, det_spacing_mm=1.0, device=device)
    with torch.no_grad():
        l0 = proj_layer(mu0.unsqueeze(0).unsqueeze(0))[0].clamp(-20.0, 20.0)
        lam0 = flat_b * torch.exp(-l0) + dark_b
        loss0 = beta_post * lam0 - (alpha_post - 1.0) * torch.log(lam0.clamp_min(1e-6))
    pctiles(loss0, "per-ray loss @ warm start")
    top_frac = 0.001
    thresh = torch.quantile(loss0.flatten().float(), 1.0 - top_frac)
    top_mask = (loss0 > thresh).float()
    log(f"  top {top_frac*100}% loss rays: thresh={float(thresh):.4g}, count={int(top_mask.sum())}")
    top_bp = fbp_reconstruction_masked(top_mask, ang, lamino_angle_deg=0.0, vol_shape=vol_shape,
                                       det_spacing_mm=1.0, filter_type="hann", device=device)

    mid = gt_vol.shape[0] // 2
    slices = {"GT": gt_vol[mid].cpu().numpy(), "top_loss_ray_backprojection": top_bp[mid].cpu().numpy(),
             "warm_start": mu0[mid].cpu().numpy()}
    metrics = {}

    for lr, n_iters in [(0.01, 200), (0.01, 2000)]:
        # unfloored baseline
        mu_rec = run_map(noisy_b, mu_net_b, var_net_b, ang, flat_b, dark_b, vol_shape,
                         n_iters, lr, C.DOSE, device)
        ps, ss = R.masked_scores(gt_vol, mu_rec, mask, dr)
        sh = R.masked_sharpness(gt_vol, mu_rec, mask)
        tag = f"lr{lr}_it{n_iters}_nofloor"
        metrics[tag] = {"psnr": float(ps), "ssim": float(ss), "sharpness": float(sh), "max": float(mu_rec.max())}
        log(f"{tag}  PSNR={ps:.2f} SSIM={ss:.3f} sharp={sh:.3f} max={mu_rec.max().item():.4g}")
        slices[tag] = mu_rec[mid].cpu().numpy()

        for pct in (1, 5, 10):
            floor = float(np.percentile(var_net_b.cpu().numpy(), pct))
            var_floored = var_net_b.clamp_min(floor)
            mu_rec = run_map(noisy_b, mu_net_b, var_floored, ang, flat_b, dark_b, vol_shape,
                             n_iters, lr, C.DOSE, device)
            ps, ss = R.masked_scores(gt_vol, mu_rec, mask, dr)
            sh = R.masked_sharpness(gt_vol, mu_rec, mask)
            tag = f"lr{lr}_it{n_iters}_floorp{pct}"
            metrics[tag] = {"psnr": float(ps), "ssim": float(ss), "sharpness": float(sh),
                           "max": float(mu_rec.max()), "floor_value": floor}
            log(f"{tag}  PSNR={ps:.2f} SSIM={ss:.3f} sharp={sh:.3f} max={mu_rec.max().item():.4g} floor={floor:.4g}")
            slices[tag] = mu_rec[mid].cpu().numpy()

    np.savez(OUT_NPZ, **slices)
    Path(OUT_JSON).write_text(json.dumps(metrics, indent=2))
    log(f"saved -> {OUT_NPZ}, {OUT_JSON}")
    log("SUMMARY " + json.dumps(metrics, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
