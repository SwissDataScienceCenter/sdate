#!/usr/bin/env python3
"""Sweep tv_weight (new spatial-regularization knob added to map_reconstruct,
see map_reconstruct.py) on real wunderkerze2 window 410300, to find a value
that arrests the GD semi-convergence divergence (mu growing without bound,
confirmed generic to GD/SIRT on this data -- see
tr_diffusion_map_reconstruct_plaingd_test.py) without over-smoothing away
real structure. First a coarse log sweep at a moderate n_iters, then check
the best candidate(s) also hold up at n_iters=2000 (the setting that made the
untv'd divergence WORST) -- that's the real test of whether TV is fixing
semi-convergence itself, not just "less bad because fewer steps were taken".
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
from sdate.tr_diffusion.map_reconstruct import map_reconstruct  # noqa: E402

sys.path.insert(0, "/myhome/sdate/scripts")
import tr_diffusion_map_reconstruct_compare as C  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_NPZ = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache/map_debug_tv.npz"
OUT_JSON = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache/map_debug_tv_metrics.json"


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

    mid = gt_vol.shape[0] // 2
    slices = {"GT": gt_vol[mid].cpu().numpy()}
    metrics = {}

    lr = 0.01
    log("--- coarse sweep, n_iters=1000 (rescaled: data loss is O(1e9), TV is O(1e4), "
       "so tv_weight must be O(1e3)-O(1e6) to matter -- the earlier 1e-6..0.1 sweep "
       "never had a chance, see per-step trace) ---")
    for tv_weight in (0.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7):
        t0 = time.time()
        mu_rec = map_reconstruct(
            noisy_b, mu_net_b, var_net_b, ang, flat_b, dark_b,
            vol_shape=None, n_iters=1000, lr=lr, warm_start="fbp",
            dose=C.DOSE, tv_weight=tv_weight, log_every=0, device=device,
        )
        ps, ss = R.masked_scores(gt_vol, mu_rec, mask, dr)
        sh = R.masked_sharpness(gt_vol, mu_rec, mask)
        tag = f"tv{tv_weight:g}_it1000"
        metrics[tag] = {"psnr": float(ps), "ssim": float(ss), "sharpness": float(sh), "max": float(mu_rec.max())}
        log(f"{tag}  PSNR={ps:.2f} SSIM={ss:.3f} sharp={sh:.3f} max={mu_rec.max().item():.4g}  ({time.time()-t0:.1f}s)")
        slices[tag] = mu_rec[mid].cpu().numpy()

    best_tag = max((k for k in metrics if k.endswith("_it1000")), key=lambda k: metrics[k]["psnr"])
    best_tv = float(best_tag.replace("tv", "").split("_it")[0])
    log(f"best from coarse sweep: {best_tag} (tv_weight={best_tv:g})")

    log("--- does it hold up at n_iters=2000 (where untv'd divergence was WORST)? ---")
    for tv_weight in sorted({0.0, best_tv, best_tv * 3, best_tv / 3}):
        t0 = time.time()
        mu_rec = map_reconstruct(
            noisy_b, mu_net_b, var_net_b, ang, flat_b, dark_b,
            vol_shape=None, n_iters=2000, lr=lr, warm_start="fbp",
            dose=C.DOSE, tv_weight=tv_weight, log_every=0, device=device,
        )
        ps, ss = R.masked_scores(gt_vol, mu_rec, mask, dr)
        sh = R.masked_sharpness(gt_vol, mu_rec, mask)
        tag = f"tv{tv_weight:g}_it2000"
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
