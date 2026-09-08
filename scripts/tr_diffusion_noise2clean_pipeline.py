#!/usr/bin/env python3
"""End-to-end noise2clean auxiliary-channel experiment.

Hypothesis: a denoiser trained with REAL ground truth on a synthetic phantom
(rendered at wunderkerze2's own, non-integer-period angular resolution --
``profiles.REGISTRY["synthetic_wk2geom"]``) might learn sharper edge priors
than any self-supervised (N2V) model can, since it never has to hide
information from itself. Running that frozen model on REAL wunderkerze2 data
will have a domain gap and produce artifacts, but feeding its prediction as
one EXTRA input channel to the normal baseline model tests whether that prior
still helps -- the real model is free to learn to trust or discount it
per-pixel. See ``sdate/tr_diffusion/README.md``'s "Noise2clean auxiliary
channel" section (once written) for the full writeup.

Kept at k=1 throughout (matching the ALREADY-CACHED poisson_head baseline,
``tr_denoise_baseline_k1_dose005_poissonhead.pt`` / its denoised memmap
``denoised_212_Wunderkerze2_poissonhead_dose05.f16``) so that checkpoint is
reused as the control with NO retraining -- the only new real-data training
run is the k=1+aux variant, isolating the aux channel as the single variable
that changed (see project plan discussion: avoid conflating a k bump with the
aux-channel effect).

Stages (each skipped if its output already exists):
  1. generate the synthetic_wk2geom phantom dataset (analytic, noiseless).
  2. train the noise2clean model on it (supervised MSE against the real GT).
  3. cross-domain inference: run the frozen noise2clean checkpoint on REAL
     wunderkerze2 data -> cached aux-channel memmap.
  4. train the real k=1 poisson_head baseline WITH that aux channel appended.
  5. denoise the full eval range with the new checkpoint.
  6. 247-window flat/dark+destripe FBP reconstruction: existing poissonhead
     control vs. this new k=1+aux variant vs. noisy floor -- PSNR/SSIM/sharpness.

    python scripts/tr_diffusion_noise2clean_pipeline.py
"""
from __future__ import annotations

import os

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
from sdate.tr_diffusion.profiles import REGISTRY  # noqa: E402

TAG = "noise2clean_aux_vs_poissonhead"
CKDIR = "/myhome/data/sdate/shared/checkpoints"
CACHE = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache"

SYNTH = REGISTRY["synthetic_wk2geom"]
REAL = REGISTRY["wunderkerze2"]

N2C_CKPT = f"{CKDIR}/tr_denoise_noise2clean_synthwk2geom_k1_dose005.pt"
AUX_MM = f"{Path(REAL.memmap_path).parent}/aux_noise2clean_on_wunderkerze2_dose05.f16"
FINAL_CKPT = f"{CKDIR}/tr_denoise_baseline_k1_dose005_poissonhead_auxn2c.pt"
FINAL_MM = f"{Path(REAL.memmap_path).parent}/denoised_212_Wunderkerze2_poissonhead_auxn2c_dose05.f16"
# Already cached (see scripts/tr_diffusion_poissonhead_pipeline.py) -- the control, NOT retrained.
BASE_MM = f"{Path(REAL.memmap_path).parent}/denoised_212_Wunderkerze2_poissonhead_dose05.f16"

INFER_FRAME_START, INFER_FRAME_END = 400_000, 450_000
RECON_FRAME_START, RECON_FRAME_END = 400_300, 450_000
DOSE, NOISE_SEED = 0.05, 12345
DET_BIN, DESTRIPE_K, WINDOW_SKIP = 1, 31, 2

RECON_MOVIE = Path(CACHE) / f"recon_{TAG}.mov"
RECON_RESULTS = Path(CACHE) / f"recon_results_{TAG}.npz"
RECON_SUMMARY = Path(CACHE) / f"recon_summary_{TAG}.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _run(cmd) -> None:
    env = {**os.environ, "PYTHONPATH": "/myhome/sdate:/myhome/astra-torch"}
    subprocess.run(cmd, check=True, cwd="/myhome/sdate", env=env)


def stage_generate_synthetic() -> None:
    if Path(SYNTH.memmap_path + ".meta.npz").exists():
        log(f"[skip] synthetic dataset already exists at {SYNTH.memmap_path}")
        return
    log("=== stage 1/6: render synthetic_wk2geom phantom (wunderkerze2 angular resolution) ===")
    t0 = time.time()
    _run([sys.executable, "scripts/generate_synthetic_phantom.py", "--profile", "synthetic_wk2geom"])
    log(f"[generate] done in {(time.time() - t0) / 60:.1f} min -> {SYNTH.memmap_path}")


def stage_train_noise2clean() -> None:
    if Path(N2C_CKPT).exists():
        log(f"[skip] noise2clean checkpoint already exists at {N2C_CKPT}")
        return
    log("=== stage 2/6: train noise2clean model on synthetic_wk2geom (k=1, MSE vs real GT) ===")
    t0 = time.time()
    cmd = [
        sys.executable, "-m", "sdate.tr_diffusion.train",
        "--mode", "noise2clean", "--denoise_mode", "n2v",
        "--profile", "synthetic_wk2geom",
        "--k", "1", "--temporal_raw_pairs",
        "--extra_noise_dose", str(DOSE),
        "--batch_size", "16", "--epochs", "8",
        "--exp_name", "tr_diff_noise2clean_synthwk2geom_k1_dose005",
        "--save_checkpoint", N2C_CKPT,
        "--num_workers", "8",
    ]
    _run(cmd)
    log(f"[train noise2clean] done in {(time.time() - t0) / 60:.1f} min -> {N2C_CKPT}")


def stage_aux_infer() -> None:
    if Path(AUX_MM + ".meta.npz").exists():
        log(f"[skip] aux-channel cache already exists at {AUX_MM}")
        return
    log("=== stage 3/6: cross-domain inference: frozen noise2clean model on REAL wunderkerze2 ===")
    t0 = time.time()
    R.denoise_sequence(
        N2C_CKPT, REAL.mov_path, REAL.memmap_path, AUX_MM,
        frame_start=REAL.frame_start, frame_end=REAL.frame_end,   # full cache range, covers eval range too
        # The native (un-thinned) real measurement is NEVER actually available in
        # deployment -- it only exists here as a pseudo-GT for offline metric
        # evaluation (see data.py's extra_noise_dose branch: `reference` = native
        # frame, `central`/`context` = thinned). The only real input this whole
        # pipeline may assume access to is the dose=0.05 thinned view, same as
        # every other real-data model here -- so cross-domain inference must run
        # on that thinned view too, matching what the model was trained on.
        dose=DOSE, noise_seed=NOISE_SEED,
        axis_col=REAL.rot_axis_col,           # the REAL axis (269.85), not the checkpoint's own synthetic one
        deg_per_frame=REAL.deg_per_frame,     # already matches by construction (1.801402)
        poisson_posterior=False, var_out_path=None,
        batch=64, num_workers=8, device=device, log_every=100,
    )
    log(f"[aux infer] done in {(time.time() - t0) / 60:.1f} min -> {AUX_MM}")


def stage_train_final() -> None:
    if Path(FINAL_CKPT).exists():
        log(f"[skip] final k=1+aux checkpoint already exists at {FINAL_CKPT}")
        return
    log("=== stage 4/6: train real k=1 poisson_head baseline WITH the aux channel ===")
    t0 = time.time()
    cmd = [
        sys.executable, "-m", "sdate.tr_diffusion.train",
        "--mode", "baseline", "--denoise_mode", "n2v",
        "--profile", "wunderkerze2",
        "--k", "1", "--temporal_raw_pairs",
        "--extra_noise_dose", str(DOSE),
        "--poisson_head",
        "--aux_channel_memmap", AUX_MM,
        "--batch_size", "16", "--epochs", "8",
        "--exp_name", "tr_diff_baseline_k1_dose005_poissonhead_auxn2c",
        "--save_checkpoint", FINAL_CKPT,
        "--num_workers", "8",
    ]
    _run(cmd)
    log(f"[train final] done in {(time.time() - t0) / 60:.1f} min -> {FINAL_CKPT}")


def stage_infer_final() -> None:
    if Path(FINAL_MM + ".meta.npz").exists():
        log(f"[skip] infer final: denoised memmap already exists at {FINAL_MM}")
        return
    log("=== stage 5/6: denoise full eval range with the k=1+aux checkpoint ===")
    t0 = time.time()
    R.denoise_sequence(
        FINAL_CKPT, REAL.mov_path, REAL.memmap_path, FINAL_MM,
        frame_start=INFER_FRAME_START, frame_end=INFER_FRAME_END,
        dose=DOSE, noise_seed=NOISE_SEED,
        batch=64, num_workers=8, device=device, log_every=100,
    )
    log(f"[infer final] done in {(time.time() - t0) / 60:.1f} min -> {FINAL_MM}")
    if not Path(BASE_MM + ".meta.npz").exists():
        raise SystemExit(f"expected the already-cached poissonhead control memmap at {BASE_MM} -- not "
                         "found; run scripts/tr_diffusion_poissonhead_pipeline.py first.")


def stage_reconstruct() -> dict:
    if RECON_SUMMARY.exists() and RECON_RESULTS.exists():
        log(f"[skip] reconstruct: summary already exists at {RECON_SUMMARY}")
        return json.loads(RECON_SUMMARY.read_text())
    log("=== stage 6/6: sliding-window FBP reconstruction (flat/dark + destripe) vs GT ===")
    t0 = time.time()
    dark = torch.from_numpy(np.load(f"{CACHE}/dark_map.npy")).float()
    flat = torch.from_numpy(np.load(f"{CACHE}/flat_map.npy")).float()
    # poissonhead FIRST: the real dose-0.05 control arm (matches every other ablation's
    # convention for which variant the "noisy" arm's dose metadata is read from).
    variants = {"poissonhead": BASE_MM, "poissonhead_aux_n2c": FINAL_MM}
    win = R.window_length_frames(180.0)
    stride = WINDOW_SKIP * win
    res = R.run_windows(
        REAL.mov_path, REAL.memmap_path, variants, stride=stride, det_bin=DET_BIN, method="fbp",
        frame_start=RECON_FRAME_START, frame_end=RECON_FRAME_END,
        dark_map=dark, flat_map=flat, destripe_k=DESTRIPE_K,
        device=device, log_every=50,
    )
    nW = len(res["window_starts"])
    for arm in list(variants) + ["noisy"]:
        m = res["metrics"][arm]
        log(f"  {arm:20s} PSNR {m['psnr'].mean():6.2f}  SSIM {m['ssim'].mean():.3f}  "
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
    stage_generate_synthetic()
    stage_train_noise2clean()
    stage_aux_infer()
    stage_train_final()
    stage_infer_final()
    summary = stage_reconstruct()
    log(f"TOTAL runtime: {(time.time() - t_all) / 60:.1f} min")
    log("FINAL SUMMARY " + json.dumps(summary, indent=2))
    log(f"noise2clean checkpoint: {N2C_CKPT}")
    log(f"aux-channel cache:      {AUX_MM}")
    log(f"final checkpoint:       {FINAL_CKPT}")
    log(f"recon summary:          {RECON_SUMMARY}")
    log(f"recon movie:            {RECON_MOVIE}")
    log("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
