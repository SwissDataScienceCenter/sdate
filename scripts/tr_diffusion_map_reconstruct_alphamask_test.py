#!/usr/bin/env python3
"""Test hypothesis: MAP reconstruction streaks come from rays where the Gamma
posterior shape alpha_post = alpha + dose*y falls below 1. For alpha_post<1 the
loss beta_post*lam - (alpha_post-1)*log(lam) is monotonically decreasing in
attenuation (no interior minimum) -- GD runs those rays' mu to infinity, which
backprojects into streaks. Fix tested here: mask those rays out of the loss
entirely (their voxels are still constrained by OTHER un-masked rays/angles).

Reports:
  - fraction of rays with alpha_post<=1 for window 410300
  - FBP of the bad-ray mask itself (does its backprojection look like the streaks?)
  - masked vs unmasked map_reconstruct at lr=0.01, n_iters in {200, 2000}
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
OUT_NPZ = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache/map_debug_alphamask.npz"
OUT_JSON = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache/map_debug_alphamask_metrics.json"


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def masked_map_reconstruct(y, mu_net, var_net, angles_deg, I0, dark, vol_shape,
                           n_iters, lr, dose, alpha_post_floor, device):
    """Copy of map_reconstruct's core loop, with an added per-ray mask on the
    loss wherever alpha_post <= alpha_post_floor (the ill-posed rays)."""
    from astra_torch.lamino import build_lamino_projector, fbp_reconstruction_masked
    from sdate.tr_diffusion.reconstruct import counts_to_attenuation_flatdark

    alpha, beta = gamma_shape_rate(mu_net, var_net)
    n = float(dose) * y
    alpha_post = alpha + n
    beta_post = beta + float(dose)

    good = (alpha_post > alpha_post_floor).float()
    n_bad = float((1.0 - good).sum())
    n_tot = float(good.numel())
    log(f"    alpha_post<= {alpha_post_floor}: {n_bad:.0f}/{n_tot:.0f} rays ({100*n_bad/n_tot:.2f}%)")

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
        per_ray = beta_post * lam - (alpha_post - 1.0) * torch.log(lam.clamp_min(eps))
        loss = (good * per_ray).sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            mu.clamp_(min=0.0)
    return mu.detach(), good


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
    log(f"gt_vol masked range: dr={dr:.4g}")

    # -- diagnostic: where are the alpha_post<=1 rays, spatially? --
    alpha, beta = gamma_shape_rate(mu_net_b, var_net_b)
    n = C.DOSE * noisy_b
    alpha_post = alpha + n
    bad = (alpha_post <= 1.0).float()
    log(f"overall bad-ray fraction (alpha_post<=1): {float(bad.mean())*100:.2f}%")
    from astra_torch.lamino import fbp_reconstruction_masked
    vol_shape = (noisy_b.shape[1], noisy_b.shape[2], noisy_b.shape[2])
    bad_bp = fbp_reconstruction_masked(bad, ang, lamino_angle_deg=0.0, vol_shape=vol_shape,
                                       det_spacing_mm=1.0, filter_type="hann", device=device)
    mid = gt_vol.shape[0] // 2
    slices = {"GT": gt_vol[mid].cpu().numpy(), "bad_ray_backprojection": bad_bp[mid].cpu().numpy()}

    metrics = {"bad_ray_fraction_overall": float(bad.mean())}
    for lr, n_iters in [(0.01, 200), (0.01, 2000)]:
        for masked in (False, True):
            floor = 1.0 if masked else -1e30  # -inf floor => nothing masked
            tag = f"lr{lr}_it{n_iters}_{'masked' if masked else 'unmasked'}"
            log(f"=== {tag} ===")
            mu_rec, good = masked_map_reconstruct(
                noisy_b, mu_net_b, var_net_b, ang, flat_b, dark_b, vol_shape,
                n_iters=n_iters, lr=lr, dose=C.DOSE, alpha_post_floor=floor, device=device)
            ps, ss = R.masked_scores(gt_vol, mu_rec, mask, dr)
            sh = R.masked_sharpness(gt_vol, mu_rec, mask)
            metrics[tag] = {"psnr": float(ps), "ssim": float(ss), "sharpness": float(sh),
                            "max": float(mu_rec.max())}
            log(f"  {tag}  PSNR={ps:.2f}  SSIM={ss:.3f}  sharpness={sh:.3f}  max={mu_rec.max().item():.4g}")
            slices[tag] = mu_rec[mid].cpu().numpy()

    np.savez(OUT_NPZ, **slices)
    Path(OUT_JSON).write_text(json.dumps(metrics, indent=2))
    log(f"saved -> {OUT_NPZ}, {OUT_JSON}")
    log("SUMMARY " + json.dumps(metrics, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
