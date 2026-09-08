#!/usr/bin/env python3
"""End-to-end pipeline for the single-channel Poisson-mean-only baseline
("poissonmean", loss_type="poisson") ablation on wunderkerze2: train from
scratch, denoise, reconstruct, score vs the plain-Huber baseline, and render
projection + reconstruction comparison movies.

This isolates the likelihood-mismatch fix ALONE (ablation "(a)" from the
loss-correction spec): same single-channel point-estimate head as the plain
Huber baseline, but trained with the correct (count-dependent-noise-aware)
Poisson NLL instead of a homoscedastic loss. No variance head, no posterior-
mean combination at inference -- see sdate.tr_diffusion.losses.BaselineN2VLoss
poisson_mean_only / sdate.tr_diffusion.nb_head.poisson_nll. Companion to (and
copy of the harness used for) tr_diffusion_poissonhead_pipeline.py, which
isolates the OTHER fix (the two-head NB-NLL + posterior combination).

Same recipe as the other wunderkerze2 ablations: 8 epochs from scratch, k=1,
dose=0.05, frame_start=400000, frame_end=500000 for training; reconstruction
scored over frame_start=400300/frame_end=450000, det_bin=1, stride=200
(window_skip=2), real flat/dark + destripe(k=31) correction, baseline listed
FIRST in the variants dict (the "noisy" arm's dose is read from whichever
variant is encountered first -- see run_windows/reconstruct.py).

Safe to re-run: every stage is skipped if its output already exists.

    python scripts/tr_diffusion_poissonmean_pipeline.py
"""
from __future__ import annotations

import os

# ffmpeg is NOT on the system PATH in this container (only the static build
# under /myhome/bin) -- HevcGray10Streamer/concat_hevc_segments shell out to a
# bare "ffmpeg", so this MUST be set before any movie-writing stage runs.
os.environ["PATH"] = "/myhome/bin:" + os.environ.get("PATH", "")

import json
import subprocess
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
CK = "/myhome/sdate/checkpoints"
TAG = "poissonmean"

MOV = f"{DATA}/212_Wunderkerze2.mov"
MEMMAP = f"{DATA}/frames_400k_500k.u16"

CKPT = f"{CK}/tr_denoise_baseline_k1_dose005_{TAG}.pt"

DENOISED_MM = f"{DATA}/denoised_212_Wunderkerze2_{TAG}_dose05.f16"
BASELINE_MM = f"{DATA}/denoised_baseline_dose005.f16"

INFER_FRAME_START, INFER_FRAME_END = 400_000, 450_000
RECON_FRAME_START, RECON_FRAME_END = 400_300, 450_000
DOSE, NOISE_SEED = 0.05, 12345
DET_BIN, DESTRIPE_K, WINDOW_SKIP = 1, 31, 2

PROJ_MOVIE = Path(CACHE) / f"proj_{TAG}_vs_baseline.mov"
RECON_MOVIE = Path(CACHE) / f"recon_{TAG}_vs_baseline.mov"
RECON_RESULTS = Path(CACHE) / f"recon_results_{TAG}_vs_baseline.npz"
RECON_SUMMARY = Path(CACHE) / f"recon_summary_{TAG}_vs_baseline.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def stage_train() -> None:
    if Path(CKPT).exists():
        log(f"[skip] train: checkpoint already exists at {CKPT}")
        return
    log("=== stage 1/5: train poisson-mean-only baseline (8 epochs, from scratch) ===")
    t0 = time.time()
    cmd = [
        sys.executable, "-m", "sdate.tr_diffusion.train",
        "--mode", "baseline", "--denoise_mode", "n2v",
        "--mov", MOV, "--memmap", MEMMAP,
        "--k", "1", "--frame_start", str(INFER_FRAME_START), "--frame_end", "500000",
        "--extra_noise_dose", str(DOSE),
        "--loss_type", "poisson",
        "--batch_size", "16", "--epochs", "8",
        "--exp_name", f"tr_diff_baseline_k1_dose005_{TAG}",
        "--save_checkpoint", CKPT,
        "--num_workers", "8",
    ]
    env = {**os.environ, "PYTHONPATH": "/myhome/sdate:/myhome/astra-torch"}
    subprocess.run(cmd, check=True, cwd="/myhome/sdate", env=env)
    log(f"[train] done in {(time.time() - t0) / 60:.1f} min -> {CKPT}")


def stage_infer() -> None:
    if Path(str(DENOISED_MM) + ".meta.npz").exists():
        log(f"[skip] infer: denoised memmap already exists at {DENOISED_MM}")
        return
    log("=== stage 2/5: denoise full eval range with the poisson-mean-only checkpoint ===")
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
    log("=== stage 3/5: sliding-window FBP reconstruction (flat/dark + destripe) vs GT ===")
    t0 = time.time()
    dark = torch.from_numpy(np.load(f"{CACHE}/dark_map.npy")).float()
    flat = torch.from_numpy(np.load(f"{CACHE}/flat_map.npy")).float()
    # baseline FIRST: run_windows/reconstruct.py's "noisy" arm reads its dose from
    # whichever variant is encountered first in this dict -- baseline is the real
    # dose-0.05 arm, so it must lead (see project memory on this exact bug).
    variants = {"baseline": BASELINE_MM, TAG: DENOISED_MM}
    win = R.window_length_frames(180.0)
    stride = WINDOW_SKIP * win
    res = R.run_windows(
        MOV, MEMMAP, variants, stride=stride, det_bin=DET_BIN, method="fbp",
        frame_start=RECON_FRAME_START, frame_end=RECON_FRAME_END,
        dark_map=dark, flat_map=flat, destripe_k=DESTRIPE_K,
        device=device, log_every=50,
    )
    nW = len(res["window_starts"])
    for arm in ("baseline", TAG, "noisy"):
        m = res["metrics"][arm]
        log(f"  {arm:12s} PSNR {m['psnr'].mean():6.2f}  SSIM {m['ssim'].mean():.3f}  sharpness {m['sharpness'].mean():.3f}")

    np.savez(RECON_RESULTS,
             window_starts=np.array(res["window_starts"]),
             **{f"{arm}_psnr": res["metrics"][arm]["psnr"] for arm in res["metrics"]},
             **{f"{arm}_ssim": res["metrics"][arm]["ssim"] for arm in res["metrics"]},
             **{f"{arm}_sharpness": res["metrics"][arm]["sharpness"] for arm in res["metrics"]})

    summary = {
        "tag": f"{TAG}_vs_baseline", "n_windows": nW, "det_bin": DET_BIN,
        "frame_range": [RECON_FRAME_START, RECON_FRAME_END],
        "minutes": round((time.time() - t0) / 60, 1),
        "correction": "flat_dark+destripe",
        "noisy_dose_note": "noisy arm uses the FIRST variant's dose metadata; "
                           "baseline is listed first so noisy = actual dose-0.05 measured data",
    }
    for arm in res["metrics"]:
        summary[f"{arm}_psnr"] = float(res["metrics"][arm]["psnr"].mean())
        summary[f"{arm}_ssim"] = float(res["metrics"][arm]["ssim"].mean())
        summary[f"{arm}_sharpness"] = float(res["metrics"][arm]["sharpness"].mean())
    RECON_SUMMARY.write_text(json.dumps(summary, indent=2))
    log(f"[reconstruct] done in {summary['minutes']} min -> {RECON_SUMMARY}")
    log("SUMMARY " + json.dumps(summary, indent=2))

    # reconstruction movie: GT | baseline | poissonmean | noisy, middle slice row
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
    log("=== stage 4/5: projection-domain movie (GT | baseline | poissonmean | noisy) ===")
    t0 = time.time()
    variants = {"baseline": BASELINE_MM, TAG: DENOISED_MM}
    R.write_projection_movie(
        MOV, MEMMAP, variants, PROJ_MOVIE,
        frame_start=RECON_FRAME_START, frame_end=RECON_FRAME_END,
        dose=DOSE, crop=(128, 512), noise_seed=NOISE_SEED, device=device,
    )
    log(f"[projection movie] done in {(time.time() - t0) / 60:.1f} min -> {PROJ_MOVIE}")


def main() -> None:
    t_all = time.time()
    log(f"device={device}  tag={TAG}")
    stage_train()
    stage_infer()
    summary = stage_reconstruct()
    stage_projection_movie()
    log("=== stage 5/5: done ===")
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
