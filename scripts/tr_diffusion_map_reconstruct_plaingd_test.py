#!/usr/bin/env python3
"""Does the "sharp but streaked" look of the failed MAP-GD attempt come from
the Gamma-Poisson prior specifically, or is it just a generic property of
doing ITERATIVE (GD/SIRT-style) tomographic reconstruction instead of FBP,
regardless of what sinogram you feed it? Test: run the existing plain linear
least-squares GD reconstruction (R.reconstruct(..., method="gd"),
gd_reconstruction_masked -- MSE against the attenuation sinogram, no Poisson
link, no prior at all) on (a) the raw noisy attenuation sinogram and (b) the
already-cached baseline (Huber-denoiser) arm's attenuation sinogram, for the
SAME window (410300). Compare PSNR/SSIM/sharpness and visual streaking against
FBP of the same inputs and against the failed Gamma-Poisson MAP-GD attempt.
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

sys.path.insert(0, "/myhome/sdate/scripts")
import tr_diffusion_map_reconstruct_compare as C  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_NPZ = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache/map_debug_plaingd.npz"
OUT_JSON = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache/map_debug_plaingd_metrics.json"


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

    bmm = np.load(C.BASELINE_MM + ".meta.npz")
    baseline_counts = R.denoised_window_gpu(
        np.memmap(C.BASELINE_MM, dtype=np.float16, mode="r", shape=(int(bmm["num_frames"]), *C.CROP)),
        int(bmm["first_index"]), idx, device)

    gt_atten = R.destripe_sinogram(R.counts_to_attenuation_flatdark(gt, dark, flat), 31)
    gt_vol = R.reconstruct(gt_atten, ang, det_bin=C.DET_BIN, method="fbp", device=device)
    nslices, hplane = C.CROP[0] // C.DET_BIN, C.CROP[1] // C.DET_BIN
    mask = R.make_mask(hplane, hplane).to(device)
    dr = float(gt_vol[..., mask].max() - gt_vol[..., mask].min())

    noisy_atten = R.destripe_sinogram(R.counts_to_attenuation_flatdark(noisy, dark, flat), 31)
    baseline_atten = R.destripe_sinogram(R.counts_to_attenuation_flatdark(baseline_counts, dark, flat), 31)

    mid = gt_vol.shape[0] // 2
    slices = {"GT": gt_vol[mid].cpu().numpy()}
    metrics = {}

    for name, atten in [("noisy", noisy_atten), ("baseline", baseline_atten)]:
        for method in ("fbp", "gd"):
            log(f"reconstructing {name} via {method}")
            t0 = time.time()
            vol = R.reconstruct(atten, ang, det_bin=C.DET_BIN, method=method, device=device)
            ps, ss = R.masked_scores(gt_vol, vol, mask, dr)
            sh = R.masked_sharpness(gt_vol, vol, mask)
            tag = f"{name}_{method}"
            metrics[tag] = {"psnr": float(ps), "ssim": float(ss), "sharpness": float(sh)}
            log(f"  {tag}  PSNR={ps:.2f}  SSIM={ss:.3f}  sharpness={sh:.3f}  ({time.time()-t0:.1f}s)")
            slices[tag] = vol[mid].cpu().numpy()

    np.savez(OUT_NPZ, **slices)
    Path(OUT_JSON).write_text(json.dumps(metrics, indent=2))
    log(f"saved -> {OUT_NPZ}, {OUT_JSON}")
    log("SUMMARY " + json.dumps(metrics, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
