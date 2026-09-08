#!/usr/bin/env python3
"""Isolate the effect of non-integer-period angular resolution on noise2clean
recovery quality.

Trains a noise2clean model on ``synthetic_v3`` (deg_per_frame=2.0, exact
180-frame period) with the SAME hyperparameters used for the
``synthetic_wk2geom`` noise2clean model (see
``scripts/tr_diffusion_noise2clean_pipeline.py`` stage 2: k=1,
temporal_raw_pairs, extra_noise_dose=0.05, batch_size=16, epochs=8) -- the
phantom scene, crop, and dose are otherwise identical between the two
profiles (see profiles.py), so any PSNR/SSIM gap between the two trained
models isolates the effect of landing back on the same angle every 180
frames vs. never landing on it (wunderkerze2's real, calibrated rate).

Waits for the synthetic_wk2geom checkpoint (trained by a concurrent job) to
appear, then evaluates BOTH checkpoints on their own held-out test split
(same seed/test_fraction as train.py, so this is genuinely unseen data),
in raw-count units against the known clean phantom reference.

    python scripts/tr_diffusion_noise2clean_geom_ablation.py
"""
from __future__ import annotations

import os

os.environ["PATH"] = "/myhome/bin:" + os.environ.get("PATH", "")

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")

from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim  # noqa: E402

from sdate.tr_diffusion.data import TimeResolvedFrameDataset  # noqa: E402
from sdate.tr_diffusion.load import load_denoiser  # noqa: E402
from sdate.tr_diffusion.pipeline import denoise_frames_noise2clean  # noqa: E402
from sdate.tr_diffusion.profiles import REGISTRY  # noqa: E402

CKDIR = "/myhome/data/sdate/shared/checkpoints"
DOSE, K, BATCH_SIZE, EPOCHS = 0.05, 1, 16, 8
TEST_FRACTION, SEED = 0.05, 0

V3_CKPT = f"{CKDIR}/tr_denoise_noise2clean_synthv3_k1_dose005.pt"
WK2GEOM_CKPT = f"{CKDIR}/tr_denoise_noise2clean_synthwk2geom_k1_dose005.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def stage_train_v3() -> None:
    if Path(V3_CKPT).exists():
        log(f"[skip] synthetic_v3 noise2clean checkpoint already exists at {V3_CKPT}")
        return
    log("=== train noise2clean on synthetic_v3 (deg_per_frame=2.0, exact 180-frame period) ===")
    t0 = time.time()
    env = {**os.environ, "PYTHONPATH": "/myhome/sdate:/myhome/astra-torch"}
    cmd = [
        sys.executable, "-m", "sdate.tr_diffusion.train",
        "--mode", "noise2clean",
        "--profile", "synthetic_v3",
        "--k", str(K), "--temporal_raw_pairs",
        "--extra_noise_dose", str(DOSE),
        "--batch_size", str(BATCH_SIZE), "--epochs", str(EPOCHS),
        "--test_fraction", str(TEST_FRACTION), "--seed", str(SEED),
        "--save_checkpoint", V3_CKPT,
    ]
    subprocess.run(cmd, check=True, cwd="/myhome/sdate", env=env)
    log(f"[train] done in {(time.time() - t0) / 60:.1f} min -> {V3_CKPT}")


def wait_for(path: str, poll_s: int = 60, timeout_s: int = 3 * 3600) -> bool:
    t0 = time.time()
    while not Path(path).exists():
        if time.time() - t0 > timeout_s:
            return False
        time.sleep(poll_s)
    return True


def held_out_scores(ckpt_path: str, profile_name: str):
    model, cfg = load_denoiser(ckpt_path, device=device)
    prof = REGISTRY[profile_name]
    lo, hi = cfg["norm_min"], cfg["norm_max"]
    DR = hi - lo
    ds = TimeResolvedFrameDataset(
        prof.mov_path, memmap_path=prof.memmap_path, k=cfg["k"],
        frame_start=prof.frame_start, frame_end=prof.frame_end, crop=tuple(cfg["crop"]),
        norm_range=(lo, hi), extra_noise_dose=DOSE, temporal_raw_pairs=cfg.get("temporal_raw_pairs", True),
        axis_col=prof.rot_axis_col, deg_per_frame=prof.deg_per_frame,
    )
    n = len(ds)
    test_size = max(1, int(TEST_FRACTION * n))
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED)).tolist()
    test_idx = idx[-test_size:]

    def counts(x):
        return ((x.detach().float().cpu().clamp(-1, 1) + 1) * 0.5 * DR + lo).numpy()[:, 0]

    pin_p, pin_s, pred_p, pred_s = [], [], [], []
    bs = 32
    with torch.no_grad():
        for i in range(0, len(test_idx), bs):
            batch = [ds[j] for j in test_idx[i:i + bs]]
            cen = torch.stack([b["central"] for b in batch]).to(device)
            ctx = torch.stack([b["context"] for b in batch]).to(device)
            ref = torch.stack([b["reference"] for b in batch]).to(device)
            pred = denoise_frames_noise2clean(model, cen, ctx)
            in_c, ref_c, pred_c = counts(cen), counts(ref), counts(pred)
            for r, q_in, q_pred in zip(ref_c, in_c, pred_c):
                pin_p.append(psnr(r, q_in, data_range=DR)); pin_s.append(ssim(r, q_in, data_range=DR))
                pred_p.append(psnr(r, q_pred, data_range=DR)); pred_s.append(ssim(r, q_pred, data_range=DR))
    return {
        "n_test": len(test_idx),
        "noisy_input": (float(np.mean(pin_p)), float(np.mean(pin_s))),
        "noise2clean": (float(np.mean(pred_p)), float(np.mean(pred_s))),
    }


def main():
    stage_train_v3()
    log(f"waiting for concurrent job's checkpoint at {WK2GEOM_CKPT} ...")
    if not wait_for(WK2GEOM_CKPT):
        log(f"[error] timed out waiting for {WK2GEOM_CKPT}")
        return
    log("both checkpoints present -- scoring held-out test splits")

    results = {
        "synthetic_v3 (deg=2.0, periodic)": held_out_scores(V3_CKPT, "synthetic_v3"),
        "synthetic_wk2geom (deg=1.801402, non-periodic)": held_out_scores(WK2GEOM_CKPT, "synthetic_wk2geom"),
    }
    print()
    print("=== noise2clean recovery quality: angular-resolution ablation ===")
    for name, r in results.items():
        nin, nis = r["noisy_input"]
        p, s = r["noise2clean"]
        print(f"{name}  (n_test={r['n_test']})")
        print(f"  noisy input  PSNR {nin:6.2f}  SSIM {nis:.3f}")
        print(f"  noise2clean  PSNR {p:6.2f}  SSIM {s:.3f}   (delta PSNR {p - nin:+.2f})")
    print("PIPELINE COMPLETE (geom ablation)")


if __name__ == "__main__":
    main()
