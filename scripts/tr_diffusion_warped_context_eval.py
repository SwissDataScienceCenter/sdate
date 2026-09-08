#!/usr/bin/env python3
"""Projection-domain held-out PSNR/SSIM/sharpness comparison for the
motion-compensated ("warped") temporal context experiment (angular-resolution
gap fix -- see the plan / README section of the same name).

Compares four arms on the SAME held-out real wunderkerze2 frames:
  - noisy input (no denoising)
  - control: existing (unretrained) tr_denoise_baseline_k1_dose005_poissonhead.pt
  - leg2: the exact same baseline recipe, retrained from scratch with the 4
    "temporal" context channels replaced by their motion-compensated (warped)
    counterparts
  - leg1 (refine): central channel replaced by a fresh Gamma-posterior sample
    drawn LIVE from the frozen control checkpoint's context-only belief, same
    warped context as leg2, trained to predict the real noisy measurement

The held-out split is computed exactly as train.py does (test_fraction=0.05,
seed=0, ``torch.randperm(n, generator=seed).tolist()[-test_size:]``) over a
dataset built WITH the warped-context memmaps attached -- since leg1/leg2 were
trained on exactly this dataset (same geometry, same warped_temporal_memmaps),
this reproduces their genuine held-out test split. The control checkpoint was
trained on a larger (ungated) frame range with its own independent split, so
this slice is not guaranteed unseen-by-control -- acceptable here since control
is a fixed frozen reference, not something being re-tuned; what matters is
scoring all three models on the identical set of frames.

    python scripts/tr_diffusion_warped_context_eval.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)  # per-sample psnr/ssim/sharpness calls otherwise oversubscribe threads
                          # in a CPU-limited container -- see the precompute job's identical bug

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")

from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim  # noqa: E402

from sdate.tr_diffusion.data import TimeResolvedFrameDataset, resolve_warped_context_dir  # noqa: E402
from sdate.tr_diffusion.load import load_denoiser  # noqa: E402
from sdate.tr_diffusion.pipeline import denoise_frames_baseline, denoise_frames_refine  # noqa: E402
from sdate.tr_diffusion.profiles import REGISTRY  # noqa: E402
from sdate.tr_naf.metrics import masked_sharpness_ratio  # noqa: E402

CKDIR = "/myhome/data/sdate/shared/checkpoints"
CONTROL_CKPT = f"{CKDIR}/tr_denoise_baseline_k1_dose005_poissonhead.pt"
LEG2_CKPT = f"{CKDIR}/tr_denoise_baseline_k1_dose005_poissonhead_warpedctx.pt"
LEG1_CKPT = f"{CKDIR}/tr_denoise_refine_warpedctx.pt"
WARPED_DIR = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2/warped_context_dose05"
OUT_DIR = Path("/myhome/data/sdate/shared/time_resolved/tr_recon_cache")

DOSE = 0.05
EVAL_NOISE_SEED = 999  # independent of training's seed=0 draw -- matches prior ablation scripts
TEST_FRACTION, SEED = 0.05, 0
BATCH = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataset(cfg: dict) -> TimeResolvedFrameDataset:
    REAL = REGISTRY["wunderkerze2"]
    warped = resolve_warped_context_dir(
        WARPED_DIR, k=int(cfg["k"]), include_mirror=bool(cfg.get("include_mirror", False)),
        neighborhoods=cfg.get("neighborhoods", "both"),
        temporal_raw_pairs=bool(cfg.get("temporal_raw_pairs", True)),
        deg_per_frame=REAL.deg_per_frame,
    )
    return TimeResolvedFrameDataset(
        REAL.mov_path, memmap_path=REAL.memmap_path, k=int(cfg["k"]),
        include_mirror=bool(cfg.get("include_mirror", False)),
        frame_start=REAL.frame_start, frame_end=REAL.frame_end, crop=tuple(cfg["crop"]),
        neighborhoods=cfg.get("neighborhoods", "both"), norm_range=(cfg["norm_min"], cfg["norm_max"]),
        extra_noise_dose=DOSE, noise_seed=EVAL_NOISE_SEED,
        temporal_raw_pairs=bool(cfg.get("temporal_raw_pairs", True)),
        axis_col=REAL.rot_axis_col, deg_per_frame=REAL.deg_per_frame,
        warped_temporal_memmaps=warped,
    )


def held_out_indices(ds: TimeResolvedFrameDataset) -> list:
    n = len(ds)
    test_size = max(1, int(TEST_FRACTION * n))
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED)).tolist()
    return idx[-test_size:]


def main():
    control_model, control_cfg = load_denoiser(CONTROL_CKPT, device=device)
    leg2_model, leg2_cfg = load_denoiser(LEG2_CKPT, device=device)
    refine_model, refine_cfg = load_denoiser(LEG1_CKPT, device=device)

    lo, hi = control_cfg["norm_min"], control_cfg["norm_max"]
    assert (leg2_cfg["norm_min"], leg2_cfg["norm_max"]) == (lo, hi), "leg2 norm range must match control"
    assert (refine_cfg["norm_min"], refine_cfg["norm_max"]) == (lo, hi), "leg1 norm range must match control"
    DR = hi - lo
    crop = tuple(control_cfg["crop"])
    mask = torch.ones(crop, dtype=torch.bool)

    ds = build_dataset(control_cfg)
    test_idx = held_out_indices(ds)
    print(f"held-out test set: {len(test_idx)} / {len(ds)} frames "
          f"(range [{ds.indices.min()}, {ds.indices.max()}])")

    control_present = float(control_cfg.get("conditioning_probability", 1.0)) > 0.0
    leg2_present = float(leg2_cfg.get("conditioning_probability", 1.0)) > 0.0
    refine_present = float(refine_cfg.get("conditioning_probability", 1.0)) > 0.0

    def counts(x: torch.Tensor) -> torch.Tensor:
        return (x.detach().float().cpu().clamp(-1, 1) + 1) * 0.5 * DR + lo

    arms = ["noisy_input", "control", "leg2_warpedctx", "leg1_refine_warpedctx"]
    psnr_v = {a: [] for a in arms}
    ssim_v = {a: [] for a in arms}
    sharp_v = {a: [] for a in arms}

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(test_idx), BATCH):
            if i % (BATCH * 5) == 0:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {i}/{len(test_idx)} "
                      f"({(time.time() - t0):.0f}s elapsed)", flush=True)
            batch = [ds[j] for j in test_idx[i:i + BATCH]]
            cen = torch.stack([b["central"] for b in batch]).to(device)
            ctx = torch.stack([b["context"] for b in batch]).to(device)
            ctx_w = torch.stack([b["context_warped"] for b in batch]).to(device)
            ref = torch.stack([b["reference"] for b in batch]).to(device)

            pred_control = denoise_frames_baseline(
                control_model, cen, ctx, present=control_present,
                poisson_head=bool(control_cfg.get("poisson_head", False)),
                norm_min=lo, norm_max=hi, poisson_dose=DOSE,
            )
            pred_leg2 = denoise_frames_baseline(
                leg2_model, cen, ctx_w, present=leg2_present,
                poisson_head=bool(leg2_cfg.get("poisson_head", False)),
                norm_min=lo, norm_max=hi, poisson_dose=DOSE,
            )
            pred_leg1 = denoise_frames_refine(
                refine_model, control_model, cen, ctx, ctx_w,
                input_mode="sample", num_samples=16, present=refine_present,
                norm_min=lo, norm_max=hi, poisson_posterior=True, poisson_dose=DOSE,
            )

            ref_t = counts(ref)[:, 0]
            preds_t = {
                "noisy_input": counts(cen)[:, 0],
                "control": counts(pred_control)[:, 0],
                "leg2_warpedctx": counts(pred_leg2)[:, 0],
                "leg1_refine_warpedctx": counts(pred_leg1)[:, 0],
            }
            for arm, pred_t in preds_t.items():
                for r_t, q_t in zip(ref_t, pred_t):
                    r, q = r_t.numpy(), q_t.numpy()
                    psnr_v[arm].append(psnr(r, q, data_range=DR))
                    ssim_v[arm].append(ssim(r, q, data_range=DR))
                    sharp_v[arm].append(masked_sharpness_ratio(r_t, q_t, mask))

    print()
    print("=== warped-context experiment: projection-domain held-out scores ===")
    summary = {"n_test": len(test_idx)}
    for arm in arms:
        p, s, sh = float(np.mean(psnr_v[arm])), float(np.mean(ssim_v[arm])), float(np.mean(sharp_v[arm]))
        print(f"{arm:24s} PSNR {p:6.2f}  SSIM {s:.3f}  sharpness {sh:.3f}")
        summary[arm] = {"psnr": p, "ssim": s, "sharpness": sh}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "warped_context_eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT_DIR / 'warped_context_eval_summary.json'}")
    print("PIPELINE COMPLETE (warped context projection-domain eval)")


if __name__ == "__main__":
    main()
