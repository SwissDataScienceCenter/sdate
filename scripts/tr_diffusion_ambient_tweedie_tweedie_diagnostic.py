#!/usr/bin/env python3
"""Diagnostic: visualize h_theta(x_t, t) -- the Tweedie/MMSE x0-estimate -- across a
sweep of noise levels straddling sigma_tn_eff, plus multiple below-floor SAMPLES from
different seeds, for one Ambient-Tweedie (Anscombe) checkpoint.

Two structurally different regimes, both shown:

* ABOVE the floor (sigma_t >= sigma_tn_eff): x_t is constructed analytically in ONE step
  from the real measurement y (x_t = y + sqrt(sigma_t^2 - sigma_tn^2) * eta, matching
  exactly how the ADSM loss builds its training input) -- each row is an INDEPENDENT
  single forward pass, no chaining, matching how ADSM actually trains h_theta.

* BELOW the floor (sigma_t < sigma_tn_eff): there is no way to construct x_t analytically
  (the forward formula's sqrt argument goes negative) -- it can only be reached by
  actually walking the model's own reverse jump-and-renoise chain down from the boundary
  at sigma_tn_eff, one rung at a time. So this section traces ONE continuous sampling
  trajectory (not independent draws), recording (x_t, pred) at each captured rung --
  exactly the consistency loss's own bootstrap mechanism, just made visible.

* SAMPLES panel: several full below-floor samples (sigma_min, i.e. the actual
  sample_below_floor deliverable) from different seeds, side by side, to see whether
  there is genuine sample-to-sample diversity (a real distribution) or the samples are
  near-identical (the network-ignores-x_t shortcut, confirmed twice already on earlier
  variants of this module).

  python scripts/tr_diffusion_ambient_tweedie_tweedie_diagnostic.py
"""
from __future__ import annotations

import argparse
import math
import os
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

DATA = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2"
CK = "/myhome/data/sdate/shared/checkpoints"
OUT = Path("/myhome/data/sdate/shared/time_resolved/tr_recon_cache")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=f"{CK}/tr_denoise_ambient_tweedie_anscombe.pt")
    p.add_argument("--frame", type=int, default=430_000, help="single frame index for the sigma sweep")
    p.add_argument("--n_above", type=int, default=6, help="number of above-floor sigma levels")
    p.add_argument("--n_below_shown", type=int, default=6, help="number of below-floor rungs to display")
    p.add_argument("--n_sample_seeds", type=int, default=5)
    p.add_argument("--n_sample_frames", type=int, default=3)
    p.add_argument("--noise_seed", type=int, default=999)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tag", default="ambient_tweedie_anscombe_tweedie_diag")
    return p.parse_args()


def main():
    a = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)

    model, cfg = load_denoiser(a.ckpt, device=dev)
    assert cfg["mode"] == "ambient_tweedie" and cfg.get("anscombe")
    lo, hi = float(cfg["norm_min"]), float(cfg["norm_max"])
    k = int(cfg["k"]); crop = tuple(cfg["crop"])
    sigma_min, sigma_max, sigma_tn_eff = float(cfg["sigma_min"]), float(cfg["sigma_max"]), float(cfg["sigma_tn_eff"])
    n_rungs = int(cfg["n_rungs"])
    extra_noise_dose = float(cfg["extra_noise_dose"])

    ds = TimeResolvedFrameDataset(
        cfg["mov"], memmap_path=f"{DATA}/frames_400k_500k.u16",
        k=k, frame_start=400_000, frame_end=500_000, crop=crop, norm_range=(lo, hi),
        extra_noise_dose=extra_noise_dose, noise_seed=a.noise_seed,
        temporal_raw_pairs=bool(cfg.get("temporal_raw_pairs", False)),
        anscombe=True, anscombe_norm_sample_frames=8,
    )
    ds.anscombe_z_min, ds.anscombe_z_max = float(cfg["anscombe_z_min"]), float(cfg["anscombe_z_max"])
    ds.anscombe_sigma_tn_norm = sigma_tn_eff
    base = int(ds.indices[0])

    def to_counts(z_norm_1chw: torch.Tensor) -> np.ndarray:
        z_raw = (z_norm_1chw.detach().float().cpu() + 1.0) * 0.5 * (ds.anscombe_z_max - ds.anscombe_z_min) + ds.anscombe_z_min
        raw_draw = inverse_anscombe(z_raw)
        return (raw_draw / extra_noise_dose).numpy()

    # ---- single-frame sigma sweep -----------------------------------------------------
    item = ds[a.frame - base]
    y = item["clean_target"].unsqueeze(0).to(dev)
    sigma_tn_map = item["sigma_tn_map"].unsqueeze(0).to(dev)
    context = item["context"].unsqueeze(0).to(dev)
    reference_c = to_counts(item["reference"].unsqueeze(0))[0, 0]

    rows = []  # each: (label, sigma_t, x_t_counts, pred_counts)

    # above the floor: independent single-step draws, log-spaced sigma_tn_eff..sigma_max
    above_sigmas = torch.exp(torch.linspace(math.log(sigma_tn_eff), math.log(sigma_max), a.n_above))
    g = torch.Generator(device=dev).manual_seed(0)
    with torch.no_grad():
        for s in above_sigmas:
            var_diff = (s ** 2 - sigma_tn_map ** 2).clamp_min(0.0)
            eta = torch.randn(y.shape, device=dev, generator=g)
            x_t = y + var_diff.sqrt() * eta
            pred = _predict(model, x_t, s.to(dev).expand(1), context, sigma_min, sigma_max)
            rows.append((f"above sigma={s.item():.4f}", float(s), to_counts(x_t)[0, 0], to_counts(pred)[0, 0]))

    # below the floor: ONE continuous reverse chain from the boundary down to sigma_min
    ladder = geometric_ladder(sigma_min, sigma_tn_eff, n_rungs, dev)  # ascending
    g2 = torch.Generator(device=dev).manual_seed(0)
    with torch.no_grad():
        var_diff = (torch.as_tensor(sigma_tn_eff, device=dev) ** 2 - sigma_tn_map ** 2).clamp_min(0.0)
        x = y + var_diff.sqrt() * torch.randn(y.shape, device=dev, generator=g2)
        below_rows = []
        capture_every = max(1, n_rungs // a.n_below_shown)
        for m in range(n_rungs, 0, -1):
            sigma_cur = ladder[m]
            pred = _predict(model, x, sigma_cur.expand(1), context, sigma_min, sigma_max)
            if (n_rungs - m) % capture_every == 0:
                below_rows.append((f"below sigma={sigma_cur.item():.4f}", float(sigma_cur),
                                   to_counts(x)[0, 0], to_counts(pred)[0, 0]))
            x = pred + ladder[m - 1] * torch.randn_like(x)
        final_sample_c = to_counts(x)[0, 0]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_rows = rows + below_rows
    vmin, vmax = np.percentile(reference_c, [1, 99])
    fig, axes = plt.subplots(len(all_rows), 3, figsize=(9, 3 * len(all_rows)))
    axes = np.atleast_2d(axes)
    for r, (label, sigma, x_c, pred_c) in enumerate(all_rows):
        axes[r, 0].imshow(reference_c, cmap="gray", vmin=vmin, vmax=vmax)
        axes[r, 1].imshow(x_c, cmap="gray", vmin=vmin, vmax=vmax)
        axes[r, 2].imshow(pred_c, cmap="gray", vmin=vmin, vmax=vmax)
        axes[r, 0].set_ylabel(label, fontsize=8)
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        if r == 0:
            axes[r, 0].set_title("reference", fontsize=9)
            axes[r, 1].set_title("x_t (noisy input)", fontsize=9)
            axes[r, 2].set_title("h_theta(x_t,t) [Tweedie est.]", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"{a.tag}_sigma_sweep.png", dpi=110)
    print(f"wrote {OUT / f'{a.tag}_sigma_sweep.png'}  (sigma_tn_eff={sigma_tn_eff:.5f}, sigma_min={sigma_min}, "
          f"sigma_max={sigma_max}, final below-floor sample vs reference mean-abs-diff="
          f"{np.mean(np.abs(final_sample_c - reference_c)):.3f})")

    # ---- multi-seed samples panel (distribution diversity check) ----------------------
    from sdate.tr_diffusion.ambient_tweedie import sample_below_floor
    idx = [a.frame - base + i for i in range(a.n_sample_frames)]
    its = [ds[i] for i in idx]
    y_multi = torch.stack([it["clean_target"] for it in its]).to(dev)
    sigma_tn_map_multi = torch.stack([it["sigma_tn_map"] for it in its]).to(dev)
    context_multi = torch.stack([it["context"] for it in its]).to(dev)
    reference_multi_c = np.stack([to_counts(it["reference"].unsqueeze(0))[0, 0] for it in its])

    samples = []
    for seed in range(a.n_sample_seeds):
        gS = torch.Generator(device=dev).manual_seed(1000 + seed)
        with torch.no_grad():
            s = sample_below_floor(model, y_multi, sigma_tn_map_multi, context_multi,
                                   sigma_min=sigma_min, sigma_max=sigma_max, sigma_tn_eff=sigma_tn_eff,
                                   n_rungs=n_rungs, eta=0.0, dps_scale=0.0, generator=gS)
        samples.append(to_counts(s))

    stacked = np.stack(samples)  # (n_seeds, n_frames, H, W)
    per_frame_seed_std = stacked.std(axis=0).mean()
    per_frame_seed_meanabsdiff = np.mean([np.abs(stacked[i] - stacked[j]).mean()
                                          for i in range(len(samples)) for j in range(i + 1, len(samples))])
    print(f"SAMPLE DIVERSITY -- across {a.n_sample_seeds} seeds, {a.n_sample_frames} frames: "
          f"pixelwise std-across-seeds (mean)={per_frame_seed_std:.4f} counts, "
          f"mean pairwise abs diff={per_frame_seed_meanabsdiff:.4f} counts "
          f"(reference frame dynamic range ~[{np.percentile(reference_multi_c,1):.1f}, "
          f"{np.percentile(reference_multi_c,99):.1f}])")

    n_show_frames = a.n_sample_frames
    fig2, axes2 = plt.subplots(n_show_frames, a.n_sample_seeds + 2, figsize=(3 * (a.n_sample_seeds + 2), 3 * n_show_frames))
    axes2 = np.atleast_2d(axes2)
    for r in range(n_show_frames):
        vmin_r, vmax_r = np.percentile(reference_multi_c[r], [1, 99])
        axes2[r, 0].imshow(reference_multi_c[r], cmap="gray", vmin=vmin_r, vmax=vmax_r)
        axes2[r, 0].set_title("reference" if r == 0 else "", fontsize=9)
        for s_i in range(a.n_sample_seeds):
            axes2[r, s_i + 1].imshow(samples[s_i][r, 0], cmap="gray", vmin=vmin_r, vmax=vmax_r)
            if r == 0:
                axes2[r, s_i + 1].set_title(f"seed {s_i}", fontsize=9)
        diff = np.abs(samples[0][r, 0] - samples[1][r, 0])
        axes2[r, -1].imshow(diff, cmap="magma")
        if r == 0:
            axes2[r, -1].set_title("|seed0-seed1|", fontsize=9)
        for c in range(a.n_sample_seeds + 2):
            axes2[r, c].set_xticks([]); axes2[r, c].set_yticks([])
    fig2.tight_layout()
    fig2.savefig(OUT / f"{a.tag}_samples.png", dpi=110)
    print(f"wrote {OUT / f'{a.tag}_samples.png'}")


if __name__ == "__main__":
    main()
