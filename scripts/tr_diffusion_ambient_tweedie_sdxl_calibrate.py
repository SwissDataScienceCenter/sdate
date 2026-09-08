#!/usr/bin/env python3
"""Calibrate `timestep_nature` for the SDXL-LoRA ambient-Tweedie port.

Their released pipeline (github.com/giannisdaras/ambient-tweedie) manufactures its
"ambient" training example by taking a CLEAN image, VAE-encoding it, and injecting a
SYNTHETIC Gaussian noise draw at a chosen `timestep_nature` -- `timestep_nature` is
just a hyperparameter in that setting, since they know the true clean image and choose
how corrupted to pretend their observation is.

We don't have that luxury: our "nature" image IS a real noisy measurement (`y` =
poisson_head's posterior mean, real residual uncertainty `sigma_tn_map`, already
calibrated in PIXEL space via the Anscombe transform -- see
`sdate.tr_diffusion.data.TimeResolvedFrameDataset(anscombe=True)`). We skip their
synthetic `add_noise` call entirely (our cached `model_input` will be
`vae.encode(y_pixel).sample()` directly, see `train_lora_sdxl.py`'s
`compute_vae_encodings` replacement) -- but we still need a single scalar
`timestep_nature` for their (otherwise UNCHANGED) ADSM/consistency formulas, which
assume `model_input` sits at a KNOWN noise level relative to the true clean latent.

This script measures that noise level EMPIRICALLY, the same spirit as the existing
Anscombe/`sigma_tn_eff` pixel-space calibration, just propagated through the frozen
SDXL VAE instead of assumed analytically:

  1. Take a handful of representative frames' `y` (already Anscombe-normalized to
     [-1,1], per-frame constant `sigma_tn_map`).
  2. For each frame, draw K independent noise realizations consistent with the REAL
     measurement uncertainty: `y_k = y + sigma_tn_map * eta_k`.
  3. Replicate to 3 channels (matching their `.convert("RGB")` requirement) and
     VAE-encode each realization using `.mode()` (the encoder's deterministic MEAN,
     NOT `.sample()`) -- deliberately excluding the VAE's own internal posterior-
     sampling noise from this measurement, since that's a separate, always-present
     artifact of the frozen encoder unrelated to our actual measurement uncertainty,
     and we want to isolate JUST the real noise's propagation through the encoder.
     (The ACTUAL cached training `model_input` still uses `.sample()`, matching
     their literal code -- this `.mode()` choice is specific to calibration.)
  4. Measure the empirical std across the K realizations, per frame, then average
     across frames -> a single global scalar `sigma_nature_latent`.
  5. Scan candidate timesteps 0..999, compute each one's scheduler-implied sigma
     `sqrt(1 - alphas_cumprod[t])` (matching `ambient_utils.diffusers_utils.
     timesteps_to_sigma` exactly), and report the argmin |sigma(t) - measured|.

Usage:
  python scripts/tr_diffusion_ambient_tweedie_sdxl_calibrate.py \
      --n_frames 8 --n_realizations 16 --frame_start 430000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/BaseTraining")

from sdate.tr_diffusion.data import TimeResolvedFrameDataset  # noqa: E402

DATA = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2"
OUT = Path("/myhome/data/sdate/shared/time_resolved/tr_recon_cache")

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frame_start", type=int, default=430_000)
    p.add_argument("--n_frames", type=int, default=8, help="number of distinct frames to average over")
    p.add_argument("--n_realizations", type=int, default=16, help="noise draws per frame")
    p.add_argument("--crop", type=int, nargs=2, default=[128, 512])
    p.add_argument("--k", type=int, default=0, help="context taps (0, matching the rest of this project's ambient-tweedie work)")
    p.add_argument("--noise_seed", type=int, default=999)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    a = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)

    from diffusers import AutoencoderKL, DDPMScheduler

    print(f"Loading VAE ({VAE_MODEL}) and scheduler ({BASE_MODEL})...")
    vae = AutoencoderKL.from_pretrained(VAE_MODEL, torch_dtype=torch.float32).to(dev)
    vae.eval()
    scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
    alphas_cumprod = scheduler.alphas_cumprod.to(dev)

    ds = TimeResolvedFrameDataset(
        f"{DATA}/212_Wunderkerze2.mov", memmap_path=f"{DATA}/frames_400k_500k.u16",
        k=a.k, frame_start=400_000, frame_end=500_000, crop=tuple(a.crop),
        norm_range=None, extra_noise_dose=0.05, noise_seed=a.noise_seed,
        anscombe=True, anscombe_norm_sample_frames=8,
    )
    base = int(ds.indices[0])
    sigma_tn_norm = float(ds.anscombe_sigma_tn_norm)
    print(f"Calibrated pixel-space sigma_tn_norm (Anscombe, [-1,1] units): {sigma_tn_norm:.5f}")

    gen = torch.Generator(device=dev).manual_seed(0)
    per_frame_stds = []
    with torch.no_grad():
        for f in range(a.n_frames):
            idx = a.frame_start - base + f * 997  # spread across the range, avoid adjacent-frame correlation
            item = ds[idx]
            y = item["clean_target"].to(dev)  # (1,H,W), already in [-1,1]
            latents = []
            for _ in range(a.n_realizations):
                eta = torch.randn(y.shape, device=dev, generator=gen)
                y_k = (y + sigma_tn_norm * eta).clamp(-1, 1)
                y_k_rgb = y_k.expand(3, *y_k.shape[1:]).unsqueeze(0)  # (1,3,H,W)
                latent_dist = vae.encode(y_k_rgb).latent_dist
                latent_mode = latent_dist.mode() * vae.config.scaling_factor
                latents.append(latent_mode.squeeze(0).cpu())
            latents = torch.stack(latents)  # (K, 4, h, w)
            std_per_frame = latents.std(dim=0).mean().item()
            per_frame_stds.append(std_per_frame)
            print(f"  frame {f} (idx={idx}): latent std across {a.n_realizations} realizations = {std_per_frame:.5f}")

    sigma_nature_latent = float(np.mean(per_frame_stds))
    print(f"\nMeasured latent-space sigma_nature (mean over {a.n_frames} frames): {sigma_nature_latent:.5f}")

    timesteps = torch.arange(scheduler.config.num_train_timesteps, device=dev)
    sigmas = torch.sqrt(1.0 - alphas_cumprod[timesteps])
    best_t = int(torch.argmin((sigmas - sigma_nature_latent).abs()).item())
    print(f"Best-matching timestep_nature: {best_t}  (scheduler sigma={sigmas[best_t].item():.5f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(timesteps.cpu(), sigmas.cpu(), label="scheduler sigma(t)")
    ax.axhline(sigma_nature_latent, color="r", linestyle="--", label=f"measured sigma_nature={sigma_nature_latent:.4f}")
    ax.axvline(best_t, color="g", linestyle=":", label=f"best_t={best_t}")
    ax.set_xlabel("timestep")
    ax.set_ylabel("sigma")
    ax.legend()
    fig.tight_layout()
    out_path = OUT / "sdxl_timestep_nature_calibration.png"
    fig.savefig(out_path, dpi=100)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
