#!/usr/bin/env python3
"""Add sharpness (gradient-energy ratio vs GT) to the angular-resolution
ablation, for both the noise2clean and baseline models -- the earlier
``tr_diffusion_noise2clean_geom_ablation.py`` /
``tr_diffusion_baseline_geom_ablation.py`` runs only scored PSNR/SSIM.

Pure eval over the four ALREADY-TRAINED checkpoints (no retraining): reuses
``sdate.tr_naf.metrics.masked_sharpness_ratio`` (the same gradient-energy-ratio
metric the main pipeline's tomographic reconstruction eval uses: 1.0 = matched
sharpness, <1 = blurred, >1 = over-sharp/noisy), applied here directly on
projection-domain frames with a full (unmasked) region.

    python scripts/tr_diffusion_geom_ablation_sharpness.py
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")

from sdate.tr_diffusion.data import TimeResolvedFrameDataset  # noqa: E402
from sdate.tr_diffusion.load import load_denoiser  # noqa: E402
from sdate.tr_diffusion.pipeline import denoise_frames_baseline, denoise_frames_noise2clean  # noqa: E402
from sdate.tr_diffusion.profiles import REGISTRY  # noqa: E402
from sdate.tr_naf.metrics import masked_sharpness_ratio  # noqa: E402

CKDIR = "/myhome/data/sdate/shared/checkpoints"
TEST_FRACTION, SEED = 0.05, 0

RUNS = [
    ("noise2clean", f"{CKDIR}/tr_denoise_noise2clean_synthv3_k1_dose005.pt",
     "synthetic_v3 (deg=2.0, periodic)", "synthetic_v3", 0.05),
    ("noise2clean", f"{CKDIR}/tr_denoise_noise2clean_synthwk2geom_k1_dose005.pt",
     "synthetic_wk2geom (deg=1.801402, non-periodic)", "synthetic_wk2geom", 0.05),
    ("baseline", f"{CKDIR}/tr_denoise_baseline_synthetic_v3_final.pt",
     "synthetic_v3 (deg=2.0, periodic)", "synthetic_v3", 0.025),
    ("baseline", f"{CKDIR}/tr_denoise_baseline_synthwk2geom_dose0025.pt",
     "synthetic_wk2geom (deg=1.801402, non-periodic)", "synthetic_wk2geom", 0.025),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score(kind: str, ckpt_path: str, profile_name: str, dose: float):
    model, cfg = load_denoiser(ckpt_path, device=device)
    prof = REGISTRY[profile_name]
    lo, hi = cfg["norm_min"], cfg["norm_max"]
    DR = hi - lo
    ds = TimeResolvedFrameDataset(
        prof.mov_path, memmap_path=prof.memmap_path, k=cfg["k"],
        frame_start=prof.frame_start, frame_end=prof.frame_end, crop=tuple(cfg["crop"]),
        norm_range=(lo, hi), extra_noise_dose=dose, noise_seed=999,
        temporal_raw_pairs=cfg.get("temporal_raw_pairs", True),
        axis_col=prof.rot_axis_col, deg_per_frame=prof.deg_per_frame,
    )
    n = len(ds)
    test_size = max(1, int(TEST_FRACTION * n))
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED)).tolist()
    test_idx = idx[-test_size:]
    mask = torch.ones(tuple(cfg["crop"]), dtype=torch.bool)

    def counts(x):
        return (x.detach().float().cpu().clamp(-1, 1) + 1) * 0.5 * DR + lo

    sh_in, sh_pred = [], []
    bs = 32
    with torch.no_grad():
        for i in range(0, len(test_idx), bs):
            batch = [ds[j] for j in test_idx[i:i + bs]]
            cen = torch.stack([b["central"] for b in batch]).to(device)
            ctx = torch.stack([b["context"] for b in batch]).to(device)
            ref = torch.stack([b["reference"] for b in batch]).to(device)
            if kind == "noise2clean":
                pred = denoise_frames_noise2clean(model, cen, ctx)
            else:
                pred = denoise_frames_baseline(
                    model, cen, ctx, present=True,
                    poisson_head=bool(cfg.get("poisson_head", False)),
                    poisson_mean_only=(cfg.get("loss_type") == "poisson"),
                    norm_min=lo, norm_max=hi,
                )
            ref_c, in_c, pred_c = counts(ref)[:, 0], counts(cen)[:, 0], counts(pred)[:, 0]
            for r, q_in, q_pred in zip(ref_c, in_c, pred_c):
                sh_in.append(masked_sharpness_ratio(r, q_in, mask))
                sh_pred.append(masked_sharpness_ratio(r, q_pred, mask))
    return float(np.mean(sh_in)), float(np.mean(sh_pred)), len(test_idx)


def main():
    print("=== sharpness (gradient-energy ratio vs GT; 1.0=matched, <1=blurred, >1=over-sharp/noisy) ===")
    for kind, ckpt, label, profile_name, dose in RUNS:
        sh_in, sh_pred, n_test = score(kind, ckpt, profile_name, dose)
        print(f"[{kind}] {label}  (n_test={n_test})")
        print(f"  noisy input  sharpness {sh_in:.3f}")
        print(f"  model output sharpness {sh_pred:.3f}")
    print("PIPELINE COMPLETE (geom ablation sharpness)")


if __name__ == "__main__":
    main()
