#!/usr/bin/env python3
"""Condensed ablation quickcheck: seed-sensitivity + PSNR/SSIM + a short sigma-sweep,
for a single Ambient-Tweedie (Anscombe) checkpoint -- meant to be run once per ablation
variant (consistency_legs = n2c / n2n / dual / both) on a mid-training checkpoint, to
compare them without running the full eval+diagnostic scripts (2 GPU jobs) per variant.

  python scripts/tr_diffusion_ambient_tweedie_ablation_quickcheck.py --ckpt ... --tag n2c
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/BaseTraining")

from sdate.tr_diffusion.ambient_tweedie import _predict, geometric_ladder, sample_below_floor  # noqa: E402
from sdate.tr_diffusion.data import TimeResolvedFrameDataset  # noqa: E402
from sdate.tr_diffusion.load import load_denoiser  # noqa: E402
from sdate.tr_diffusion.noise import inverse_anscombe  # noqa: E402
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

DATA = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2"
OUT = Path("/myhome/data/sdate/shared/time_resolved/tr_recon_cache")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--frame_start", type=int, default=430_000)
    p.add_argument("--n_frames", type=int, default=32)
    p.add_argument("--n_above", type=int, default=4)
    p.add_argument("--n_below_shown", type=int, default=4)
    p.add_argument("--noise_seed", type=int, default=999)
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    a = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)

    model, cfg = load_denoiser(a.ckpt, device=dev)
    lo, hi = float(cfg["norm_min"]), float(cfg["norm_max"])
    DR = hi - lo
    crop = tuple(cfg["crop"])
    sigma_min, sigma_max, sigma_tn_eff = float(cfg["sigma_min"]), float(cfg["sigma_max"]), float(cfg["sigma_tn_eff"])
    extra_noise_dose = float(cfg["extra_noise_dose"])

    ds = TimeResolvedFrameDataset(
        cfg["mov"], memmap_path=f"{DATA}/frames_400k_500k.u16",
        k=int(cfg["k"]), frame_start=400_000, frame_end=500_000, crop=crop, norm_range=(lo, hi),
        extra_noise_dose=extra_noise_dose, noise_seed=a.noise_seed,
        temporal_raw_pairs=bool(cfg.get("temporal_raw_pairs", False)),
        anscombe=True, anscombe_norm_sample_frames=8,
    )
    ds.anscombe_z_min, ds.anscombe_z_max = float(cfg["anscombe_z_min"]), float(cfg["anscombe_z_max"])
    ds.anscombe_sigma_tn_norm = sigma_tn_eff
    base = int(ds.indices[0])

    def counts_native(z_norm: torch.Tensor) -> np.ndarray:
        z_raw = (z_norm.detach().float().cpu() + 1.0) * 0.5 * (ds.anscombe_z_max - ds.anscombe_z_min) + ds.anscombe_z_min
        return (inverse_anscombe(z_raw) / extra_noise_dose).numpy()[:, 0]

    def ref_counts(x_norm: torch.Tensor) -> np.ndarray:
        return ((x_norm.detach().float().cpu().clamp(-1, 1) + 1) * 0.5 * DR + lo).numpy()[:, 0]

    # ---- PSNR/SSIM + seed sensitivity over a frame batch ----
    idx = [a.frame_start - base + i for i in range(a.n_frames)]
    its = [ds[i] for i in idx]
    y = torch.stack([it["clean_target"] for it in its]).to(dev)
    sigma_tn_map = torch.stack([it["sigma_tn_map"] for it in its]).to(dev)
    context = torch.stack([it["context"] for it in its]).to(dev)
    reference_c = ref_counts(torch.stack([it["reference"] for it in its]))
    noisy_c = counts_native(y)

    g1 = torch.Generator(device=dev).manual_seed(a.sample_seed)
    below1 = sample_below_floor(model, y, sigma_tn_map, context, sigma_min, sigma_max, sigma_tn_eff,
                                n_rungs=int(cfg["n_rungs"]), eta=0.0, dps_scale=0.0, generator=g1)
    g2 = torch.Generator(device=dev).manual_seed(a.sample_seed + 1)
    below2 = sample_below_floor(model, y, sigma_tn_map, context, sigma_min, sigma_max, sigma_tn_eff,
                                n_rungs=int(cfg["n_rungs"]), eta=0.0, dps_scale=0.0, generator=g2)
    seed_sensitivity = (below1 - below2).abs().mean().item()
    out_std = below1.std().item()
    below1_c = counts_native(below1)

    def scores(pred_c, ref_c):
        return (float(np.mean([psnr(r, q, data_range=DR) for q, r in zip(pred_c, ref_c)])),
                float(np.mean([ssim(r, q, data_range=DR) for q, r in zip(pred_c, ref_c)])))

    p_noisy, s_noisy = scores(noisy_c, reference_c)
    p_below, s_below = scores(below1_c, reference_c)
    print(f"[{a.tag}] ckpt={a.ckpt}")
    print(f"[{a.tag}] SEED SENSITIVITY: abs={seed_sensitivity:.6f} out_std={out_std:.6f} "
          f"ratio={seed_sensitivity / max(out_std, 1e-8):.4f}")
    print(f"[{a.tag}] noisy_vs_reference       PSNR {p_noisy:6.2f}  SSIM {s_noisy:.3f}")
    print(f"[{a.tag}] below_floor_vs_reference PSNR {p_below:6.2f}  SSIM {s_below:.3f}")

    # ---- short sigma sweep on one frame (structure check, above and below floor) ----
    item = its[0]
    y1 = item["clean_target"].unsqueeze(0).to(dev)
    stn1 = item["sigma_tn_map"].unsqueeze(0).to(dev)
    ctx1 = item["context"].unsqueeze(0).to(dev)
    ref1 = ref_counts(item["reference"].unsqueeze(0))[0]

    rows = []
    above_sigmas = torch.exp(torch.linspace(math.log(sigma_tn_eff), math.log(sigma_max), a.n_above))
    g = torch.Generator(device=dev).manual_seed(0)
    with torch.no_grad():
        for s in above_sigmas:
            var_diff = (s ** 2 - stn1 ** 2).clamp_min(0.0)
            x_t = y1 + var_diff.sqrt().to(dev) * torch.randn(y1.shape, device=dev, generator=g)
            pred = _predict(model, x_t, s.to(dev).expand(1), ctx1, sigma_min, sigma_max)
            rows.append((f"above {s.item():.3f}", counts_native(pred)[0]))

    ladder = geometric_ladder(sigma_min, sigma_tn_eff, int(cfg["n_rungs"]), dev)
    n_rungs = int(cfg["n_rungs"])
    g2b = torch.Generator(device=dev).manual_seed(0)
    with torch.no_grad():
        var_diff = (torch.as_tensor(sigma_tn_eff, device=dev) ** 2 - stn1 ** 2).clamp_min(0.0)
        x = y1 + var_diff.sqrt() * torch.randn(y1.shape, device=dev, generator=g2b)
        capture_every = max(1, n_rungs // a.n_below_shown)
        for m in range(n_rungs, 0, -1):
            sigma_cur = ladder[m]
            pred = _predict(model, x, sigma_cur.expand(1), ctx1, sigma_min, sigma_max)
            if (n_rungs - m) % capture_every == 0:
                rows.append((f"below {sigma_cur.item():.4f}", counts_native(pred)[0]))
            x = pred + ladder[m - 1] * torch.randn_like(x)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vmin, vmax = np.percentile(ref1, [1, 99])
    fig, axes = plt.subplots(len(rows) + 1, 1, figsize=(8, 2.2 * (len(rows) + 1)))
    axes[0].imshow(ref1, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_ylabel("reference", fontsize=8)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for ax, (label, img) in zip(axes[1:], rows):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_ylabel(label, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out_path = OUT / f"ablation_{a.tag}_quickcheck.png"
    fig.savefig(out_path, dpi=100)
    print(f"[{a.tag}] wrote {out_path}")


if __name__ == "__main__":
    main()
