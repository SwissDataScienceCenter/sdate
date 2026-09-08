#!/usr/bin/env python3
"""Does TV rescue the plain, prior-free MSE-GD reconstruction the same way it
rescued the Gamma-Poisson MAP-GD one (see tr_diffusion_map_reconstruct_tv_sweep.py)?
This is a much better-conditioned, quadratic (not exp-linked, not
precision-weighted) objective, so the right tv_weight scale is expected to be
completely different -- gd_reconstruction_masked's default data loss is
torch.mean((pred-meas)**2) over attenuation values O(0.01-0.05), so a MEAN-scale
loss O(1e-4 to 1e-3), vs the Poisson-MAP loss's SUM-scale O(1e9). Sweep widely
rather than assume.

Run on the baseline (existing Huber-denoiser) arm's attenuation sinogram, same
window (410300) as every other test, same setup as
tr_diffusion_map_reconstruct_plaingd_test.py's baseline_gd (PSNR 28.09 FBP ->
18.03 unregularized GD).
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
from sdate.tr_diffusion.map_reconstruct import _tv_loss  # noqa: E402

sys.path.insert(0, "/myhome/sdate/scripts")
import tr_diffusion_map_reconstruct_compare as C  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_NPZ = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache/map_debug_msegd_tv.npz"
OUT_JSON = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache/map_debug_msegd_tv_metrics.json"


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main():
    from astra_torch.lamino import gd_reconstruction_masked

    log(f"device={device}")
    src = R.MemmapFrameSource(C.MEMMAP, C.MOV)
    dark = torch.from_numpy(np.load(f"{C.CACHE}/dark_map.npy")).float().to(device)
    flat = torch.from_numpy(np.load(f"{C.CACHE}/flat_map.npy")).float().to(device)

    s = 410300
    win = R.window_length_frames(180.0)
    idx = np.arange(s, s + win)
    ang = R.projection_angles(idx, deg_per_frame=R.DEG_PER_FRAME)

    gt = R.native_window_gpu(src, idx, C.CROP, C.AXIS_COL, device)
    bmm = np.load(C.BASELINE_MM + ".meta.npz")
    baseline_counts = R.denoised_window_gpu(
        np.memmap(C.BASELINE_MM, dtype=np.float16, mode="r", shape=(int(bmm["num_frames"]), *C.CROP)),
        int(bmm["first_index"]), idx, device)

    gt_atten = R.destripe_sinogram(R.counts_to_attenuation_flatdark(gt, dark, flat), 31)
    gt_vol = R.reconstruct(gt_atten, ang, det_bin=C.DET_BIN, method="fbp", device=device)
    nslices, hplane = C.CROP[0] // C.DET_BIN, C.CROP[1] // C.DET_BIN
    mask = R.make_mask(hplane, hplane).to(device)
    dr = float(gt_vol[..., mask].max() - gt_vol[..., mask].min())

    baseline_atten = R.destripe_sinogram(R.counts_to_attenuation_flatdark(baseline_counts, dark, flat), 31)
    p = R.bin_detector(baseline_atten.to(device), C.DET_BIN)
    v, r, c = p.shape
    vol_shape = (r, c, c)

    mid = gt_vol.shape[0] // 2
    slices = {"GT": gt_vol[mid].cpu().numpy()}
    metrics = {}

    baseline_fbp_vol = R.reconstruct(baseline_atten, ang, det_bin=C.DET_BIN, method="fbp", device=device)
    ps0, ss0 = R.masked_scores(gt_vol, baseline_fbp_vol, mask, dr)
    log(f"baseline_fbp (reference)  PSNR={ps0:.2f}  SSIM={ss0:.3f}")
    slices["baseline_fbp"] = baseline_fbp_vol[mid].cpu().numpy()
    metrics["baseline_fbp"] = {"psnr": float(ps0), "ssim": float(ss0)}

    for tv_weight in (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        reg_fn = (lambda vol, w=tv_weight: w * _tv_loss(vol)) if tv_weight > 0 else None
        t0 = time.time()
        vol = gd_reconstruction_masked(
            p, ang, 0.0, vol_shape=vol_shape, det_spacing_mm=1.0, device=device, verbose=False,
            max_epochs=300, batch_size=int(v), lr=1e-1, clamp_min=0.0, regularization_fn=reg_fn,
        )
        if vol.dim() == 4:
            vol = vol.squeeze(0)
        vol = vol.clamp_min(0.0)
        ps, ss = R.masked_scores(gt_vol, vol, mask, dr)
        sh = R.masked_sharpness(gt_vol, vol, mask)
        tag = f"tv{tv_weight:g}"
        metrics[tag] = {"psnr": float(ps), "ssim": float(ss), "sharpness": float(sh), "max": float(vol.max())}
        log(f"{tag}  PSNR={ps:.2f} SSIM={ss:.3f} sharp={sh:.3f} max={vol.max().item():.4g}  ({time.time()-t0:.1f}s)")
        slices[tag] = vol[mid].cpu().numpy()

    np.savez(OUT_NPZ, **slices)
    Path(OUT_JSON).write_text(json.dumps(metrics, indent=2))
    log(f"saved -> {OUT_NPZ}, {OUT_JSON}")
    log("SUMMARY " + json.dumps(metrics, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
