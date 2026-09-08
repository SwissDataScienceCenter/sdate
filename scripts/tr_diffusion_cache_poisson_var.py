#!/usr/bin/env python3
"""Cache the poisson_head posterior VARIANCE alongside the existing mean cache.

The mean cache (denoised_212_Wunderkerze2_poissonhead_dose05.f16) was already
computed without --return_uncertainty, so the posterior variance was never
saved. Re-runs the SAME checkpoint/dose/noise_seed (deterministic -> the mean
output should come out byte-identical, a free sanity check) and additionally
writes var_post to a companion memmap, needed as the per-pixel heteroscedastic
"sigma_tn^2" for the Ambient-Tweedie training (see plan
radiant-singing-scott.md).

  python scripts/tr_diffusion_cache_poisson_var.py
"""
from __future__ import annotations

import os
os.environ["PATH"] = "/myhome/bin:" + os.environ.get("PATH", "")

import sys
sys.path.insert(0, "/myhome/sdate")

import torch

from sdate.tr_diffusion.reconstruct import denoise_sequence

DATA = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2"
CK = "/myhome/data/sdate/shared/checkpoints"

CKPT = f"{CK}/tr_denoise_baseline_k1_dose005_poissonhead.pt"
MOV = f"{DATA}/212_Wunderkerze2.mov"
MEMMAP = f"{DATA}/frames_400k_500k.u16"
MEAN_OUT = f"{DATA}/denoised_212_Wunderkerze2_poissonhead_dose05.f16"
VAR_OUT = f"{DATA}/denoised_212_Wunderkerze2_poissonhead_dose05_var.f16"

INFER_FRAME_START, INFER_FRAME_END = 400_000, 450_000
DOSE, NOISE_SEED = 0.05, 12345


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first, n, meta = denoise_sequence(
        CKPT, MOV, MEMMAP, MEAN_OUT,
        frame_start=INFER_FRAME_START, frame_end=INFER_FRAME_END,
        dose=DOSE, noise_seed=NOISE_SEED, poisson_posterior=True,
        var_out_path=VAR_OUT, device=device,
    )
    print(f"done: first={first} n={n}")


if __name__ == "__main__":
    main()
