#!/usr/bin/env python3
"""Reconstruct + score the new joint-FBP-context-conditioned ("jointfbpctx_T11")
baseline denoiser against the standard k=1 poisson_head baseline and the noisy
floor, over the same frame range its T=11 same-angle reprojection context taps
were cached for.

Mirrors `tr_diffusion_poissonhead_pipeline.py`'s stage structure, but:
- training is already done (checkpoint at CKPT, converged ~epoch 38, see
  project memory) -- this script only infers + reconstructs + scores.
- the reference k=1 poisson_head arm's denoised memmap
  (`denoised_212_Wunderkerze2_poissonhead_dose05.f16`, range [400201,449799),
  same noise_seed=12345) already fully covers this eval range, so it is reused
  as-is -- only the new jointfbpctx_T11 arm needs a fresh inference pass.
- eval range is bounded by the T=11 context-tap cache
  ([412000,428000), see `tr_diffusion_jointfbp_context_cache.py`) -- the new
  checkpoint's aux_channel_memmap paths are baked into its own config.json and
  read automatically by `denoise_sequence`, no aux args needed here.

Safe to re-run: every stage is skipped if its output already exists.

    python scripts/tr_diffusion_jointfbpctx_pipeline.py
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
CKDIR = "/mydata/sdate/shared/checkpoints"
TAG = "jointfbpctx_T11"

MOV = f"{DATA}/212_Wunderkerze2.mov"
MEMMAP = f"{DATA}/frames_400k_500k.u16"

CKPT = f"{CKDIR}/tr_denoise_baseline_jointfbpctx_T11_dose005_poissonhead.pt"
REF_MM = f"{DATA}/denoised_212_Wunderkerze2_poissonhead_dose05.f16"  # existing k=1 poisson_head arm, reused as-is

DENOISED_MM = f"{DATA}/denoised_212_Wunderkerze2_{TAG}_dose05.f16"

# bounded by the T=11 context-tap cache's coverage [412000,428000)
INFER_FRAME_START, INFER_FRAME_END = 412_000, 428_000
RECON_FRAME_START, RECON_FRAME_END = 412_300, 427_700
DOSE, NOISE_SEED = 0.05, 12345
DET_BIN, DESTRIPE_K, WINDOW_SKIP = 1, 31, 2

RECON_MOVIE = Path(CACHE) / f"recon_{TAG}_vs_poissonhead_k1.mov"
PROJ_MOVIE = Path(CACHE) / f"proj_{TAG}_vs_poissonhead_k1.mov"
RECON_RESULTS = Path(CACHE) / f"recon_results_{TAG}_vs_poissonhead_k1.npz"
RECON_SUMMARY = Path(CACHE) / f"recon_summary_{TAG}_vs_poissonhead_k1.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def stage_infer() -> None:
    if Path(str(DENOISED_MM) + ".meta.npz").exists():
        log(f"[skip] infer: denoised memmap already exists at {DENOISED_MM}")
        return
    assert Path(CKPT).exists(), f"checkpoint not found: {CKPT}"
    log("=== stage 1/4: denoise eval range with the jointfbpctx_T11 checkpoint ===")
    t0 = time.time()
    R.denoise_sequence(
        CKPT, MOV, MEMMAP, DENOISED_MM,
        frame_start=INFER_FRAME_START, frame_end=INFER_FRAME_END,
        dose=DOSE, noise_seed=NOISE_SEED,
        batch=96, num_workers=8, device=device, log_every=100,
    )
    log(f"[infer] done in {(time.time() - t0) / 60:.1f} min -> {DENOISED_MM}")


def stage_reconstruct() -> dict:
    if RECON_SUMMARY.exists() and RECON_RESULTS.exists():
        log(f"[skip] reconstruct: summary already exists at {RECON_SUMMARY}")
        return json.loads(RECON_SUMMARY.read_text())
    log("=== stage 2/4: sliding-window FBP reconstruction (flat/dark + destripe) vs GT ===")
    t0 = time.time()
    dark = torch.from_numpy(np.load(f"{CACHE}/dark_map.npy")).float()
    flat = torch.from_numpy(np.load(f"{CACHE}/flat_map.npy")).float()
    variants = {"poissonhead_k1": REF_MM, TAG: DENOISED_MM}
    win = R.window_length_frames(180.0)
    stride = WINDOW_SKIP * win
    res = R.run_windows(
        MOV, MEMMAP, variants, stride=stride, det_bin=DET_BIN, method="fbp",
        frame_start=RECON_FRAME_START, frame_end=RECON_FRAME_END,
        dark_map=dark, flat_map=flat, destripe_k=DESTRIPE_K,
        device=device, log_every=50,
    )
    nW = len(res["window_starts"])
    for arm in ("poissonhead_k1", TAG, "noisy"):
        m = res["metrics"][arm]
        log(f"  {arm:16s} PSNR {m['psnr'].mean():6.2f}  SSIM {m['ssim'].mean():.3f}")

    np.savez(RECON_RESULTS,
             window_starts=np.array(res["window_starts"]),
             **{f"{arm}_psnr": res["metrics"][arm]["psnr"] for arm in res["metrics"]},
             **{f"{arm}_ssim": res["metrics"][arm]["ssim"] for arm in res["metrics"]})

    summary = {
        "tag": f"{TAG}_vs_poissonhead_k1", "n_windows": nW, "det_bin": DET_BIN,
        "frame_range": [RECON_FRAME_START, RECON_FRAME_END],
        "minutes": round((time.time() - t0) / 60, 1),
        "correction": "flat_dark+destripe",
    }
    for arm in res["metrics"]:
        summary[f"{arm}_psnr"] = float(res["metrics"][arm]["psnr"].mean())
        summary[f"{arm}_ssim"] = float(res["metrics"][arm]["ssim"].mean())
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


def stage_projection_movie() -> None:
    if PROJ_MOVIE.exists():
        log(f"[skip] projection movie already exists at {PROJ_MOVIE}")
        return
    log("=== stage 3/4: projection-domain movie (GT | poissonhead_k1 | jointfbpctx_T11 | noisy) ===")
    t0 = time.time()
    variants = {"poissonhead_k1": REF_MM, TAG: DENOISED_MM}
    R.write_projection_movie(
        MOV, MEMMAP, variants, PROJ_MOVIE,
        frame_start=RECON_FRAME_START, frame_end=RECON_FRAME_END,
        dose=DOSE, crop=(128, 512), noise_seed=NOISE_SEED, device=device,
    )
    log(f"[projection movie] done in {(time.time() - t0) / 60:.1f} min -> {PROJ_MOVIE}")


def main() -> None:
    t_all = time.time()
    log(f"device={device}  tag={TAG}")
    stage_infer()
    summary = stage_reconstruct()
    stage_projection_movie()
    log("=== stage 4/4: done ===")
    log(f"TOTAL runtime: {(time.time() - t_all) / 60:.1f} min")
    log("FINAL SUMMARY " + json.dumps(summary, indent=2))
    log(f"checkpoint:        {CKPT}")
    log(f"denoised memmap:   {DENOISED_MM}")
    log(f"recon summary:     {RECON_SUMMARY}")
    log(f"recon movie:       {RECON_MOVIE}")
    log(f"projection movie:  {PROJ_MOVIE}")
    log("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
