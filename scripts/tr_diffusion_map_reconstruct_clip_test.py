#!/usr/bin/env python3
"""Same window/attempts as tr_diffusion_map_reconstruct_capture.py, but this time
clip each map_reconstruct volume to GT's (masked) max value and recompute
PSNR/SSIM/sharpness before vs after clipping, to test whether the real-data
divergence is "morally correct but unbounded" (i.e. clipping recovers a good
reconstruction) or genuinely wrong in structure (clipping still looks bad).
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
OUT_NPZ = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache/map_debug_slices_clipped.npz"
OUT_JSON = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache/map_debug_clip_metrics.json"


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

    noisy_atten = R.destripe_sinogram(R.counts_to_attenuation_flatdark(noisy, dark, flat), 31)
    noisy_fbp_vol = R.reconstruct(noisy_atten, ang, det_bin=C.DET_BIN, method="fbp", device=device)

    poissonhead_atten = R.destripe_sinogram(R.counts_to_attenuation_flatdark(mu_net, dark, flat), 31)
    poissonhead_vol = R.reconstruct(poissonhead_atten, ang, det_bin=C.DET_BIN, method="fbp", device=device)

    nslices, hplane = C.CROP[0] // C.DET_BIN, C.CROP[1] // C.DET_BIN
    mask = R.make_mask(hplane, hplane).to(device)
    dr = float(gt_vol[..., mask].max() - gt_vol[..., mask].min())
    gt_max = float(gt_vol[..., mask].max())
    log(f"gt_vol masked range: min={float(gt_vol[..., mask].min()):.4g} max={gt_max:.4g} (dr={dr:.4g})")

    metrics = {}
    for name, vol in [("noisy_fbp", noisy_fbp_vol), ("poissonhead_fbp", poissonhead_vol)]:
        ps, ss = R.masked_scores(gt_vol, vol, mask, dr)
        sh = R.masked_sharpness(gt_vol, vol, mask)
        metrics[name] = {"psnr": float(ps), "ssim": float(ss), "sharpness": float(sh)}
        log(f"  {name:20s} PSNR={ps:6.2f} SSIM={ss:.3f} sharpness={sh:.3f}")

    mid = gt_vol.shape[0] // 2
    slices = {
        "GT": gt_vol[mid].cpu().numpy(),
        "noisy_fbp": noisy_fbp_vol[mid].cpu().numpy(),
        "poissonhead_fbp": poissonhead_vol[mid].cpu().numpy(),
    }

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
        key = f"map_lr{lr}_it{n_iters}"
        ps_raw, ss_raw = R.masked_scores(gt_vol, mu_rec, mask, dr)
        sh_raw = R.masked_sharpness(gt_vol, mu_rec, mask)

        mu_clipped = mu_rec.clamp(min=0.0, max=gt_max)
        ps_clip, ss_clip = R.masked_scores(gt_vol, mu_clipped, mask, dr)
        sh_clip = R.masked_sharpness(gt_vol, mu_clipped, mask)

        metrics[key] = {
            "raw": {"psnr": float(ps_raw), "ssim": float(ss_raw), "sharpness": float(sh_raw),
                    "max": float(mu_rec.max())},
            "clipped_to_gt_max": {"psnr": float(ps_clip), "ssim": float(ss_clip), "sharpness": float(sh_clip)},
        }
        log(f"  {key:20s} RAW     PSNR={ps_raw:6.2f} SSIM={ss_raw:.3f} sharpness={sh_raw:.3f} max={mu_rec.max().item():.4g}")
        log(f"  {key:20s} CLIPPED PSNR={ps_clip:6.2f} SSIM={ss_clip:.3f} sharpness={sh_clip:.3f}")

        slices[key] = mu_rec[mid].cpu().numpy()
        slices[f"{key}_clipped"] = mu_clipped[mid].cpu().numpy()

    np.savez(OUT_NPZ, **slices)
    Path(OUT_JSON).write_text(json.dumps(metrics, indent=2))
    log(f"saved slices -> {OUT_NPZ}")
    log(f"saved metrics -> {OUT_JSON}")
    log("SUMMARY " + json.dumps(metrics, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
