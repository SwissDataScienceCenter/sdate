#!/usr/bin/env python3
"""Single-hop consistency check: reproduce EXACTLY the training-time construction
(sample x_t at sigma_t > sigma_tn_eff -> anchor = h_theta(x_t,sigma_t) -> renoise ONCE
to sigma_t' < sigma_tn_eff -> h_theta(x_t',sigma_t')) and visualize the result, instead
of the multi-rung `sample_below_floor` chain used for the below-floor eval/quickcheck.
Isolates whether the training objective itself produces a sane below-floor estimate in
one hop, independent of whatever the iterative reverse sampler does with compounding error.

  python scripts/tr_diffusion_ambient_tweedie_singlehop_check.py --ckpt ... --tag n2n_capped15_e10
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/BaseTraining")

from sdate.tr_diffusion.ambient_tweedie import _predict, sample_log_uniform  # noqa: E402
from sdate.tr_diffusion.data import TimeResolvedFrameDataset  # noqa: E402
from sdate.tr_diffusion.load import load_denoiser  # noqa: E402
from sdate.tr_diffusion.noise import inverse_anscombe  # noqa: E402
from skimage.metrics import peak_signal_noise_ratio as psnr

DATA = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2"
OUT = Path("/myhome/data/sdate/shared/time_resolved/tr_recon_cache")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--frame_start", type=int, default=430_000)
    p.add_argument("--n_above", type=int, default=3, help="how many sigma_t (above-floor) draws to show as rows")
    p.add_argument("--n_below", type=int, default=4, help="how many sigma_t' (below-floor) targets per row")
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
    adsm_upper_mult = cfg.get("adsm_upper_mult")
    adsm_sigma_t_max = sigma_tn_eff * float(adsm_upper_mult) if adsm_upper_mult else sigma_max
    print(f"sigma_tn_eff={sigma_tn_eff:.5f}  adsm sampling range (matches training)="
          f"[{sigma_tn_eff:.5f}, {adsm_sigma_t_max:.5f}]  (adsm_upper_mult={adsm_upper_mult})")

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

    idx = a.frame_start - base
    item = ds[idx]
    y1 = item["clean_target"].unsqueeze(0).to(dev)
    stn1 = item["sigma_tn_map"].unsqueeze(0).to(dev)
    ctx1 = item["context"].unsqueeze(0).to(dev)
    ref1 = ref_counts(item["reference"].unsqueeze(0))[0]

    g = torch.Generator(device=dev).manual_seed(a.sample_seed)

    above_sigmas = torch.exp(torch.linspace(math.log(sigma_tn_eff), math.log(adsm_sigma_t_max), a.n_above))
    below_sigmas = torch.exp(torch.linspace(math.log(sigma_min), math.log(sigma_tn_eff * 0.999), a.n_below))

    rows = []  # (row_label, [ (col_label, img, psnr) , ... ])
    with torch.no_grad():
        for s_t in above_sigmas:
            var_diff = (s_t ** 2 - stn1 ** 2).clamp_min(0.0)
            x_t = y1 + var_diff.sqrt().to(dev) * torch.randn(y1.shape, device=dev, generator=g)
            anchor = _predict(model, x_t, s_t.to(dev).expand(1), ctx1, sigma_min, sigma_max)
            anchor_img = counts_native(anchor)[0]
            anchor_psnr = psnr(ref1, anchor_img, data_range=DR)
            cols = [("anchor", anchor_img, anchor_psnr)]
            for s_tp in below_sigmas:
                x_tp = anchor + s_tp.to(dev) * torch.randn_like(anchor)
                pred_tp = _predict(model, x_tp, s_tp.to(dev).expand(1), ctx1, sigma_min, sigma_max)
                img = counts_native(pred_tp)[0]
                p = psnr(ref1, img, data_range=DR)
                cols.append((f"sigma'={s_tp.item():.4f}", img, p))
            rows.append((f"sigma_t={s_t.item():.4f}", cols))
            print(f"[{a.tag}] sigma_t={s_t.item():.4f}  anchor PSNR={anchor_psnr:.2f}  "
                  + "  ".join(f"{c[0]}:{c[2]:.2f}dB" for c in cols[1:]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vmin, vmax = np.percentile(ref1, [1, 99])
    ncols = 1 + a.n_below + 1  # reference + anchor + below-floor targets
    fig, axes = plt.subplots(len(rows) + 1, ncols, figsize=(2.2 * ncols, 2.2 * (len(rows) + 1)))
    for r in range(len(rows) + 1):
        for c in range(ncols):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    axes[0, 0].imshow(ref1, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("reference", fontsize=8)
    for c in range(1, ncols):
        axes[0, c].axis("off")
    for r, (row_label, cols) in enumerate(rows, start=1):
        axes[r, 0].set_ylabel(row_label, fontsize=8)
        axes[r, 0].imshow(ref1, cmap="gray", vmin=vmin, vmax=vmax)
        for c, (col_label, img, p) in enumerate(cols, start=1):
            axes[r, c].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            axes[r, c].set_title(f"{col_label}\n{p:.1f}dB", fontsize=7)
    fig.tight_layout()
    out_path = OUT / f"singlehop_{a.tag}.png"
    fig.savefig(out_path, dpi=100)
    print(f"[{a.tag}] wrote {out_path}")


if __name__ == "__main__":
    main()
