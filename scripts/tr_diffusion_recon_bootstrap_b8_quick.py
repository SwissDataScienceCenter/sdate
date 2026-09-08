#!/usr/bin/env python3
"""Fast, small-range sanity check: does averaging just 8 independent Gamma
draws for the sample-conditioned bootstrap model (v2) move the needle at all
vs the already-good single-draw result? Deliberately NOT the full movie --
only a handful of reconstruction windows, so this finishes in minutes instead
of the ~9h the full-range num_samples=64 sweep would have taken.

Reuses the already-cached FULL-RANGE poissonhead/bootstrap_mean/bootstrap_sample
memmaps (they already cover this small frame range) and only computes the two
NEW small-range arms:
  - bootstrap_sample_b8:            v2, num_samples=8, no posterior combine.
  - bootstrap_sample_posterior_b8:  v2, num_samples=8, posterior-combined with
                                     the real observed pixel.

Same flat/dark+destripe FBP convention as every other ablation, just a much
smaller window count. Safe to re-run: stages skipped if outputs already exist.

    python scripts/tr_diffusion_recon_bootstrap_b8_quick.py
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

DATA = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2"
CACHE = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache"
CKDIR = "/myhome/data/sdate/shared/checkpoints"
TAG = "bootstrap_b8_quick_vs_poissonhead"

MOV = f"{DATA}/212_Wunderkerze2.mov"
MEMMAP = f"{DATA}/frames_400k_500k.u16"

V2_CKPT = f"{CKDIR}/tr_denoise_bootstrap_k1dose005_v2_gammasample.pt"

BASE_MM = f"{DATA}/denoised_212_Wunderkerze2_poissonhead_dose05.f16"                 # already cached (full range)
V1_MM = f"{DATA}/denoised_212_Wunderkerze2_bootstrap_v1mean_dose05.f16"              # already cached (full range)
V1_POST_MM = f"{DATA}/denoised_212_Wunderkerze2_bootstrap_v1mean_posterior_dose05.f16"  # already cached (full range)
V2_MM = f"{DATA}/denoised_212_Wunderkerze2_bootstrap_v2sample_dose05.f16"            # already cached (full range, num_samples=1)
V2_B8_MM = f"{DATA}/denoised_212_Wunderkerze2_bootstrap_v2sample_b8_quick_dose05.f16"            # NEW, small range only
V2_B8_POST_MM = f"{DATA}/denoised_212_Wunderkerze2_bootstrap_v2sample_b8_posterior_quick_dose05.f16"  # NEW, small range only

# Small subset -- just enough for a few reconstruction windows, not the full movie.
INFER_FRAME_START, INFER_FRAME_END = 400_000, 400_900
RECON_FRAME_START, RECON_FRAME_END = 400_300, 400_900
DOSE, NOISE_SEED = 0.05, 12345
DET_BIN, DESTRIPE_K, WINDOW_SKIP = 1, 31, 2
NUM_SAMPLES_B8 = 8

RECON_MOVIE = Path(CACHE) / f"recon_{TAG}.mov"
RECON_RESULTS = Path(CACHE) / f"recon_results_{TAG}.npz"
RECON_SUMMARY = Path(CACHE) / f"recon_summary_{TAG}.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


NEW_JOBS = [
    ("bootstrap_sample_b8", V2_CKPT, V2_B8_MM, dict(num_samples=NUM_SAMPLES_B8, poisson_posterior=False)),
    ("bootstrap_sample_posterior_b8", V2_CKPT, V2_B8_POST_MM,
     dict(num_samples=NUM_SAMPLES_B8, poisson_posterior=True)),
]


def stage_infer() -> None:
    for name, ckpt, mm, kw in NEW_JOBS:
        if Path(mm + ".meta.npz").exists():
            log(f"[skip] infer {name}: denoised memmap already exists at {mm}")
            continue
        log(f"=== denoise small eval range with {name} ({ckpt}, {kw}) ===")
        t0 = time.time()
        R.denoise_sequence(
            ckpt, MOV, MEMMAP, mm,
            frame_start=INFER_FRAME_START, frame_end=INFER_FRAME_END,
            dose=DOSE, noise_seed=NOISE_SEED,
            batch=64, num_workers=8, device=device, log_every=50,
            **kw,
        )
        log(f"[infer] {name} done in {(time.time() - t0) / 60:.1f} min -> {mm}")
    for mm in (BASE_MM, V1_MM, V1_POST_MM, V2_MM):
        if not Path(mm + ".meta.npz").exists():
            raise SystemExit(f"expected an already-cached memmap at {mm} -- not found; run "
                             "scripts/tr_diffusion_recon_bootstrap.py first.")


def stage_reconstruct() -> dict:
    if RECON_SUMMARY.exists() and RECON_RESULTS.exists():
        log(f"[skip] reconstruct: summary already exists at {RECON_SUMMARY}")
        return json.loads(RECON_SUMMARY.read_text())
    log("=== sliding-window FBP reconstruction (flat/dark + destripe) vs GT, small range ===")
    t0 = time.time()
    dark = torch.from_numpy(np.load(f"{CACHE}/dark_map.npy")).float()
    flat = torch.from_numpy(np.load(f"{CACHE}/flat_map.npy")).float()
    variants = {
        "poissonhead": BASE_MM,
        "bootstrap_mean": V1_MM,
        "bootstrap_mean_posterior": V1_POST_MM,
        "bootstrap_sample": V2_MM,
        "bootstrap_sample_b8": V2_B8_MM,
        "bootstrap_sample_posterior_b8": V2_B8_POST_MM,
    }
    win = R.window_length_frames(180.0)
    stride = WINDOW_SKIP * win
    res = R.run_windows(
        MOV, MEMMAP, variants, stride=stride, det_bin=DET_BIN, method="fbp",
        frame_start=RECON_FRAME_START, frame_end=RECON_FRAME_END,
        dark_map=dark, flat_map=flat, destripe_k=DESTRIPE_K,
        device=device, log_every=50,
    )
    nW = len(res["window_starts"])
    for arm in list(variants) + ["noisy"]:
        m = res["metrics"][arm]
        log(f"  {arm:32s} PSNR {m['psnr'].mean():6.2f}  SSIM {m['ssim'].mean():.3f}  "
            f"sharpness {m['sharpness'].mean():.3f}")

    np.savez(RECON_RESULTS,
             window_starts=np.array(res["window_starts"]),
             **{f"{arm}_psnr": res["metrics"][arm]["psnr"] for arm in res["metrics"]},
             **{f"{arm}_ssim": res["metrics"][arm]["ssim"] for arm in res["metrics"]},
             **{f"{arm}_sharpness": res["metrics"][arm]["sharpness"] for arm in res["metrics"]})

    summary = {
        "tag": TAG, "n_windows": nW, "det_bin": DET_BIN,
        "frame_range": [RECON_FRAME_START, RECON_FRAME_END],
        "minutes": round((time.time() - t0) / 60, 1),
        "correction": "flat_dark+destripe",
        "num_samples_b8": NUM_SAMPLES_B8,
    }
    for arm in res["metrics"]:
        summary[f"{arm}_psnr"] = float(res["metrics"][arm]["psnr"].mean())
        summary[f"{arm}_ssim"] = float(res["metrics"][arm]["ssim"].mean())
        summary[f"{arm}_sharpness"] = float(res["metrics"][arm]["sharpness"].mean())
    RECON_SUMMARY.write_text(json.dumps(summary, indent=2))
    log(f"[reconstruct] done in {summary['minutes']} min -> {RECON_SUMMARY}")
    log("SUMMARY " + json.dumps(summary, indent=2))

    mid = len(res["movie_rows"]) // 2
    gt = np.stack([f[mid].numpy() for f in res["movie"]["GT"]])
    vmin, vmax = np.percentile(gt, [1, 99])
    combined = [torch.cat([res["movie"][arm][i][mid] for arm in res["arms"]], dim=1) for i in range(nW)]
    R.write_slice_movie(combined, RECON_MOVIE, float(vmin), float(vmax))
    log(f"[reconstruct] movie written -> {RECON_MOVIE}  panels: {' | '.join(res['arms'])}")
    return summary


def main() -> None:
    t_all = time.time()
    log(f"device={device}  tag={TAG}")
    stage_infer()
    summary = stage_reconstruct()
    log(f"TOTAL runtime: {(time.time() - t_all) / 60:.1f} min")
    log("FINAL SUMMARY " + json.dumps(summary, indent=2))
    log(f"recon summary: {RECON_SUMMARY}")
    log(f"recon movie:   {RECON_MOVIE}")
    log("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
