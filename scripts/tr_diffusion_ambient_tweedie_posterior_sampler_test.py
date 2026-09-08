#!/usr/bin/env python3
"""Free, inference-only test: compare the OLD naive-renoise sampler (sample_below_floor)
against the NEW corrected posterior sampler (sample_below_floor_posterior) on the SAME
already-trained checkpoint -- no retraining. If the posterior sampler alone is meaningfully
sharper, the blur was (at least partly) a sampling-procedure bug, not something that needs
retraining to fix.

  python scripts/tr_diffusion_ambient_tweedie_posterior_sampler_test.py --ckpt ... --tag n2c
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/BaseTraining")

from sdate.tr_diffusion.ambient_tweedie import sample_below_floor, sample_below_floor_posterior  # noqa: E402
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
    p.add_argument("--noise_seed", type=int, default=999)
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--dps_scale", type=str, default="0.0",
                    help="comma-separated list of DPS guidance step magnitudes to sweep; "
                         "0.0 (default) skips DPS entirely. Note: the guidance step is normalised "
                         "by the WHOLE flattened image's L2 norm, so its effective per-pixel size "
                         "is roughly dps_scale / sqrt(H*W) -- for a 128x512 crop, ~dps_scale/256.")
    p.add_argument("--dps_steps", type=int, default=3,
                    help="number of DPS gradient-refinement steps per rung, only used for scales > 0")
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

    idx = [a.frame_start - base + i for i in range(a.n_frames)]
    its = [ds[i] for i in idx]
    y = torch.stack([it["clean_target"] for it in its]).to(dev)
    sigma_tn_map = torch.stack([it["sigma_tn_map"] for it in its]).to(dev)
    context = torch.stack([it["context"] for it in its]).to(dev)
    reference_c = ref_counts(torch.stack([it["reference"] for it in its]))
    noisy_c = counts_native(y)

    def scores(pred_c, ref_c):
        return (float(np.mean([psnr(r, q, data_range=DR) for q, r in zip(pred_c, ref_c)])),
                float(np.mean([ssim(r, q, data_range=DR) for q, r in zip(pred_c, ref_c)])))

    def laplace(im: np.ndarray) -> np.ndarray:
        # plain-numpy 4-neighbor Laplacian (no scipy dependency), interior pixels only.
        return (-4 * im[1:-1, 1:-1] + im[:-2, 1:-1] + im[2:, 1:-1] + im[1:-1, :-2] + im[1:-1, 2:])

    def sharpness(imgs):
        # mean Laplacian-variance across the batch -- a standard, simple focus/sharpness proxy:
        # higher = more high-frequency content (real detail OR noise; compare against reference's
        # own value, not an absolute threshold).
        vals = [laplace(im.astype(np.float64)).var() for im in imgs]
        return float(np.mean(vals))

    p_noisy, s_noisy = scores(noisy_c, reference_c)
    sharp_ref = sharpness(reference_c)
    sharp_noisy = sharpness(noisy_c)
    print(f"[{a.tag}] noisy_vs_reference  PSNR {p_noisy:6.2f}  SSIM {s_noisy:.3f}  "
          f"sharpness(noisy)={sharp_noisy:.2f}  sharpness(reference)={sharp_ref:.2f}")

    g1 = torch.Generator(device=dev).manual_seed(a.sample_seed)
    old = sample_below_floor(model, y, sigma_tn_map, context, sigma_min, sigma_max, sigma_tn_eff,
                             n_rungs=n_rungs, eta=0.0, dps_scale=0.0, generator=g1)
    old_c = counts_native(old)
    p_old, s_old = scores(old_c, reference_c)
    sharp_old = sharpness(old_c)
    print(f"[{a.tag}] OLD sampler (naive renoise)      PSNR {p_old:6.2f}  SSIM {s_old:.3f}  sharpness={sharp_old:.2f}")

    g2 = torch.Generator(device=dev).manual_seed(a.sample_seed)
    new = sample_below_floor_posterior(model, y, sigma_tn_map, context, sigma_min, sigma_max, sigma_tn_eff,
                                       n_rungs=n_rungs, generator=g2)
    new_c = counts_native(new)
    p_new, s_new = scores(new_c, reference_c)
    sharp_new = sharpness(new_c)
    print(f"[{a.tag}] NEW sampler (posterior renoise)   PSNR {p_new:6.2f}  SSIM {s_new:.3f}  sharpness={sharp_new:.2f}")

    rows = [("reference", reference_c[0]), ("noisy measurement", noisy_c[0]),
            ("OLD sampler", old_c[0]), ("NEW posterior sampler", new_c[0])]

    dps_scales = [float(s) for s in a.dps_scale.split(",") if float(s) > 0]
    for dps_scale in dps_scales:
        g3 = torch.Generator(device=dev).manual_seed(a.sample_seed)
        new_dps = sample_below_floor_posterior(model, y, sigma_tn_map, context, sigma_min, sigma_max, sigma_tn_eff,
                                               n_rungs=n_rungs, dps_scale=dps_scale, dps_steps=a.dps_steps,
                                               generator=g3)
        new_dps_c = counts_native(new_dps)
        p_dps, s_dps = scores(new_dps_c, reference_c)
        sharp_dps = sharpness(new_dps_c)
        print(f"[{a.tag}] NEW+DPS sampler (scale={dps_scale}, steps={a.dps_steps})   "
              f"PSNR {p_dps:6.2f}  SSIM {s_dps:.3f}  sharpness={sharp_dps:.2f}")
        rows.append((f"NEW+DPS (scale={dps_scale}, steps={a.dps_steps})", new_dps_c[0]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vmin, vmax = np.percentile(reference_c[0], [1, 99])
    fig, axes = plt.subplots(len(rows), 1, figsize=(8, 2.2 * len(rows)))
    for ax, (label, img) in zip(axes, rows):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_ylabel(label, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out_path = OUT / f"posterior_sampler_test_{a.tag}.png"
    fig.savefig(out_path, dpi=100)
    print(f"[{a.tag}] wrote {out_path}")

    # Zoomed crop -- fine texture (real detail vs noise) is hard to judge at
    # full-frame thumbnail scale; a magnified sub-region side by side across
    # variants makes the difference much more legible.
    h_full, w_full = reference_c[0].shape
    ch, cw = h_full, min(160, w_full)
    c0 = (w_full - cw) // 2
    fig2, axes2 = plt.subplots(1, len(rows), figsize=(2.6 * len(rows), 2.6 * ch / cw))
    for ax, (label, img) in zip(axes2, rows):
        ax.imshow(img[:, c0:c0 + cw], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(label, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    fig2.tight_layout()
    zoom_path = OUT / f"posterior_sampler_test_{a.tag}_zoom.png"
    fig2.savefig(zoom_path, dpi=150)
    print(f"[{a.tag}] wrote {zoom_path}")


if __name__ == "__main__":
    main()
