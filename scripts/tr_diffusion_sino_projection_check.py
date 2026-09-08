#!/usr/bin/env python3
"""Quick sanity check for the --mode sinogram checkpoint: denoise a handful of
individual projection frames and compare against the noisy measurement and the
pre-thinning `reference` (the project's usual pseudo-GT for a single frame).

Uses the CURRENT checkpoint on disk (save_always=True, so this reflects
whatever epoch has most recently completed even while training continues in
its own job). Small frame range -> fast (no full 247-window reconstruction).

    python scripts/tr_diffusion_sino_projection_check.py
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["PATH"] = "/myhome/bin:" + os.environ.get("PATH", "")

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")
sys.path.insert(0, "/myhome/BaseTraining")

import numpy as np
import torch

from sdate.tr_diffusion import reconstruct as R
from sdate.tr_diffusion.data import TimeResolvedFrameDataset

DATA = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2"
CACHE = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache"
CKPT = "/myhome/data/sdate/shared/checkpoints/tr_denoise_sinogram_k1dose005_v1.pt"
MOV = f"{DATA}/212_Wunderkerze2.mov"
MEMMAP = f"{DATA}/frames_400k_500k.u16"
OUT_MM = f"{DATA}/denoised_212_Wunderkerze2_sinogram_check.f16"

FRAME_START, FRAME_END = 420_000, 420_900
DOSE, NOISE_SEED = 0.05, 12345

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main():
    log(f"device={device}")
    cfg = json.load(open(CKPT.replace(".pt", "_config.json")))
    lo_n, hi_n = float(cfg["norm_min"]), float(cfg["norm_max"])
    crop = tuple(cfg["crop"])
    if Path(OUT_MM + ".meta.npz").exists():
        m = np.load(OUT_MM + ".meta.npz")
        first, n = int(m["first_index"]), int(m["num_frames"])
        log(f"[skip] denoise: already exists, {n} frames {first}..{first+n-1}")
    else:
        t0 = time.time()
        first, n, meta = R.denoise_sequence(
            CKPT, MOV, MEMMAP, OUT_MM,
            frame_start=FRAME_START, frame_end=FRAME_END,
            dose=DOSE, noise_seed=NOISE_SEED,
            batch=8, num_workers=4, device=device, log_every=5,
        )
        log(f"denoised {n} frames in {(time.time() - t0) / 60:.1f} min")
    den_mm = np.memmap(OUT_MM, dtype=np.float16, mode="r", shape=(n, crop[0], crop[1]))

    # Re-build the same dataset (deterministic noise_seed) to get the matching
    # noisy-measurement + pre-thinning reference frames for these same indices.
    ds = TimeResolvedFrameDataset(
        mov_path=MOV, memmap_path=MEMMAP, k=int(cfg["k"]),
        frame_start=FRAME_START, frame_end=FRAME_END, crop=crop,
        norm_range=(lo_n, hi_n), extra_noise_dose=DOSE, noise_seed=NOISE_SEED,
    )
    idx_by_frame = {int(fi): i for i, fi in enumerate(ds.indices)}

    maes_noisy, maes_denoised = [], []
    examples = []
    for f in range(first, first + n):
        if f not in idx_by_frame:
            continue
        item = ds[idx_by_frame[f]]
        noisy_counts = (item["central"][0].numpy() + 1.0) * 0.5 * (hi_n - lo_n) + lo_n
        ref_counts = (item["reference"][0].numpy() + 1.0) * 0.5 * (hi_n - lo_n) + lo_n
        den_counts = den_mm[f - first].astype(np.float32)
        mae_noisy = float(np.abs(noisy_counts - ref_counts).mean())
        mae_denoised = float(np.abs(den_counts - ref_counts).mean())
        maes_noisy.append(mae_noisy)
        maes_denoised.append(mae_denoised)
        if len(examples) < 5:
            examples.append((f, mae_noisy, mae_denoised, ref_counts.mean()))

    log(f"n_compared={len(maes_noisy)}")
    log(f"MAE vs reference: noisy mean={np.mean(maes_noisy):.2f}  denoised mean={np.mean(maes_denoised):.2f}  "
        f"(improvement factor {np.mean(maes_noisy) / max(np.mean(maes_denoised), 1e-6):.2f}x)")
    for f, mn, md, refmean in examples:
        log(f"  frame {f}: ref_mean={refmean:.1f}  MAE noisy={mn:.2f}  MAE denoised={md:.2f}")

    # small comparison movie: GT | denoised | noisy, raw projection frames
    movie_path = Path(CACHE) / "sino_projection_check.mov"
    R.write_projection_movie(
        MOV, MEMMAP, {"sinogram_denoised": OUT_MM}, movie_path,
        frame_start=first, frame_end=first + n, dose=DOSE, crop=crop,
        noise_seed=NOISE_SEED, device=device,
    )
    log(f"movie written -> {movie_path}")
    log("DONE")


if __name__ == "__main__":
    main()
