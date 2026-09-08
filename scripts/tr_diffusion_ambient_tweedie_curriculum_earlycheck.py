#!/usr/bin/env python3
"""Early-stage curriculum check: at an intermediate training epoch, only rungs
[floor_idx, n_rungs] have actually been trained -- running the FULL sample_below_floor
chain (which always walks all the way to sigma_min) would unfairly include untrained
territory. This instead walks the SAME chain but stops at the currently-open floor rung,
so the eval only covers what's actually been trained so far.

  python scripts/tr_diffusion_ambient_tweedie_curriculum_earlycheck.py --ckpt ... --tag ... --floor_idx 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/BaseTraining")

from sdate.tr_diffusion.ambient_tweedie import _predict, geometric_ladder  # noqa: E402
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
    p.add_argument("--floor_idx", type=int, required=True, help="lowest currently-open rung index (from the training log's floor_rung=)")
    p.add_argument("--frame_start", type=int, default=430_000)
    p.add_argument("--n_frames", type=int, default=32)
    p.add_argument("--n_below_shown", type=int, default=6)
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
    n_rungs = int(cfg["n_rungs"])
    ladder = geometric_ladder(sigma_min, sigma_tn_eff, n_rungs, dev)
    floor_sigma = ladder[a.floor_idx].item()
    print(f"[{a.tag}] n_rungs={n_rungs} floor_idx={a.floor_idx} floor_sigma={floor_sigma:.5f} "
          f"(walking chain from sigma_tn_eff={sigma_tn_eff:.5f} down to this, NOT to sigma_min={sigma_min})")

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

    # ---- PSNR/SSIM down to the currently-open floor, over a batch of frames ----
    idx = [a.frame_start - base + i for i in range(a.n_frames)]
    its = [ds[i] for i in idx]
    y = torch.stack([it["clean_target"] for it in its]).to(dev)
    sigma_tn_map = torch.stack([it["sigma_tn_map"] for it in its]).to(dev)
    context = torch.stack([it["context"] for it in its]).to(dev)
    reference_c = ref_counts(torch.stack([it["reference"] for it in its]))
    noisy_c = counts_native(y)

    g = torch.Generator(device=dev).manual_seed(a.sample_seed)
    with torch.no_grad():
        var_diff = (torch.as_tensor(sigma_tn_eff, device=dev) ** 2 - sigma_tn_map ** 2).clamp_min(0.0)
        x = y + var_diff.sqrt() * torch.randn(y.shape, device=dev, generator=g)
        for m in range(n_rungs, a.floor_idx, -1):
            sigma_cur, sigma_nxt = ladder[m], ladder[m - 1]
            pred = _predict(model, x, sigma_cur.expand(y.shape[0]), context, sigma_min, sigma_max)
            x = pred + sigma_nxt * torch.randn(y.shape, device=dev, generator=g)
    partial_below = x
    partial_below_c = counts_native(partial_below)

    def scores(pred_c, ref_c):
        return (float(np.mean([psnr(r, q, data_range=DR) for q, r in zip(pred_c, ref_c)])),
                float(np.mean([ssim(r, q, data_range=DR) for q, r in zip(pred_c, ref_c)])))

    p_noisy, s_noisy = scores(noisy_c, reference_c)
    p_partial, s_partial = scores(partial_below_c, reference_c)
    print(f"[{a.tag}] noisy_vs_reference           PSNR {p_noisy:6.2f}  SSIM {s_noisy:.3f}")
    print(f"[{a.tag}] partial_below_vs_reference    PSNR {p_partial:6.2f}  SSIM {s_partial:.3f}  "
          f"(down to sigma={floor_sigma:.5f}, NOT full chain to sigma_min)")

    # ---- sigma-sweep on one frame, from sigma_tn_eff down to the open floor ----
    item = its[0]
    y1 = item["clean_target"].unsqueeze(0).to(dev)
    stn1 = item["sigma_tn_map"].unsqueeze(0).to(dev)
    ctx1 = item["context"].unsqueeze(0).to(dev)
    ref1 = ref_counts(item["reference"].unsqueeze(0))[0]

    rows = []
    g2 = torch.Generator(device=dev).manual_seed(0)
    with torch.no_grad():
        var_diff = (torch.as_tensor(sigma_tn_eff, device=dev) ** 2 - stn1 ** 2).clamp_min(0.0)
        x = y1 + var_diff.sqrt() * torch.randn(y1.shape, device=dev, generator=g2)
        n_open = n_rungs - a.floor_idx
        capture_every = max(1, n_open // a.n_below_shown)
        for m in range(n_rungs, a.floor_idx, -1):
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
    out_path = OUT / f"curriculum_earlycheck_{a.tag}.png"
    fig.savefig(out_path, dpi=100)
    print(f"[{a.tag}] wrote {out_path}")


if __name__ == "__main__":
    main()
