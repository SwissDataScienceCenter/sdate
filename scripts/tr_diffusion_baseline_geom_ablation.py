#!/usr/bin/env python3
"""Isolate the effect of non-integer-period angular resolution on the
STANDARD self-supervised (N2V) baseline denoiser -- the counterpart to
``tr_diffusion_noise2clean_geom_ablation.py``, which ran this same ablation
for the noise2clean (supervised MSE) model.

Reuses the existing ``tr_denoise_baseline_synthetic_v3_final.pt`` checkpoint
(deg_per_frame=2.0, exact 180-frame period; extra_noise_dose=0.025,
poisson_head=False -- confirmed via its conv_out shape, [1, 64, 3, 3]) as one
arm, and trains its ``synthetic_wk2geom`` (deg_per_frame=1.801402,
non-periodic) twin with EXACTLY matched hyperparameters (read from that
checkpoint's own config JSON) as the other arm. Same phantom scene, crop,
and dose between the two profiles -- only deg_per_frame differs -- so any
PSNR/SSIM gap isolates the geometry effect, same logic as the noise2clean
ablation, but now for the model actually used in production (self-supervised
N2V, not supervised regression).

Evaluates both on their own held-out test split (same seed/test_fraction
default as train.py, so genuinely unseen data), scoring the BLIND-SPOT
denoiser's single-shot output (``present=True``, matching how every other
baseline checkpoint in this project is scored) against the known clean
phantom reference, in raw-count units.

    python scripts/tr_diffusion_baseline_geom_ablation.py
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
from sdate.tr_diffusion.pipeline import denoise_frames_baseline  # noqa: E402
from sdate.tr_diffusion.profiles import REGISTRY  # noqa: E402

CKDIR = "/myhome/data/sdate/shared/checkpoints"
# Matched to tr_denoise_baseline_synthetic_v3_final_config.json exactly.
DOSE, K = 0.025, 1
BATCH_SIZE, EPOCHS = 16, 8
TEST_FRACTION, SEED = 0.05, 0

V3_CKPT = f"{CKDIR}/tr_denoise_baseline_synthetic_v3_final.pt"
WK2GEOM_CKPT = f"{CKDIR}/tr_denoise_baseline_synthwk2geom_dose0025.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def stage_train_wk2geom() -> None:
    if Path(WK2GEOM_CKPT).exists():
        log(f"[skip] synthetic_wk2geom baseline checkpoint already exists at {WK2GEOM_CKPT}")
        return
    log("=== train baseline (N2V) on synthetic_wk2geom (deg_per_frame=1.801402, non-periodic) ===")
    t0 = time.time()
    env = {**os.environ, "PYTHONPATH": "/myhome/sdate:/myhome/astra-torch"}
    cmd = [
        sys.executable, "-m", "sdate.tr_diffusion.train",
        "--mode", "baseline", "--denoise_mode", "n2v",
        "--profile", "synthetic_wk2geom",
        "--k", str(K), "--temporal_raw_pairs",
        "--extra_noise_dose", str(DOSE),
        "--no-poisson_head",  # match the v3 checkpoint's architecture (conv_out out_channels=1)
        "--batch_size", str(BATCH_SIZE), "--epochs", str(EPOCHS),
        "--test_fraction", str(TEST_FRACTION), "--seed", str(SEED),
        "--save_checkpoint", WK2GEOM_CKPT,
    ]
    subprocess.run(cmd, check=True, cwd="/myhome/sdate", env=env)
    log(f"[train] done in {(time.time() - t0) / 60:.1f} min -> {WK2GEOM_CKPT}")


def held_out_scores(ckpt_path: str, profile_name: str):
    model, cfg = load_denoiser(ckpt_path, device=device)
    prof = REGISTRY[profile_name]
    lo, hi = cfg["norm_min"], cfg["norm_max"]
    DR = hi - lo
    ds = TimeResolvedFrameDataset(
        prof.mov_path, memmap_path=prof.memmap_path, k=cfg["k"],
        frame_start=prof.frame_start, frame_end=prof.frame_end, crop=tuple(cfg["crop"]),
        norm_range=(lo, hi), extra_noise_dose=DOSE, noise_seed=999,
        temporal_raw_pairs=cfg.get("temporal_raw_pairs", True),
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
            pred = denoise_frames_baseline(
                model, cen, ctx, present=True,
                poisson_head=bool(cfg.get("poisson_head", False)),
                poisson_mean_only=(cfg.get("loss_type") == "poisson"),
                norm_min=lo, norm_max=hi,
            )
            in_c, ref_c, pred_c = counts(cen), counts(ref), counts(pred)
            for r, q_in, q_pred in zip(ref_c, in_c, pred_c):
                pin_p.append(psnr(r, q_in, data_range=DR)); pin_s.append(ssim(r, q_in, data_range=DR))
                pred_p.append(psnr(r, q_pred, data_range=DR)); pred_s.append(ssim(r, q_pred, data_range=DR))
    return {
        "n_test": len(test_idx),
        "noisy_input": (float(np.mean(pin_p)), float(np.mean(pin_s))),
        "baseline": (float(np.mean(pred_p)), float(np.mean(pred_s))),
    }


def main():
    stage_train_wk2geom()
    log("both checkpoints present -- scoring held-out test splits")

    results = {
        "synthetic_v3 (deg=2.0, periodic)": held_out_scores(V3_CKPT, "synthetic_v3"),
        "synthetic_wk2geom (deg=1.801402, non-periodic)": held_out_scores(WK2GEOM_CKPT, "synthetic_wk2geom"),
    }
    print()
    print("=== baseline (N2V) recovery quality: angular-resolution ablation ===")
    for name, r in results.items():
        nin, nis = r["noisy_input"]
        p, s = r["baseline"]
        print(f"{name}  (n_test={r['n_test']})")
        print(f"  noisy input  PSNR {nin:6.2f}  SSIM {nis:.3f}")
        print(f"  baseline     PSNR {p:6.2f}  SSIM {s:.3f}   (delta PSNR {p - nin:+.2f})")
    print("PIPELINE COMPLETE (baseline geom ablation)")


if __name__ == "__main__":
    main()
