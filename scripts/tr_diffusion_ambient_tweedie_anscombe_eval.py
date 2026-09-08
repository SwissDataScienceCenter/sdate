#!/usr/bin/env python3
"""Quickcheck eval for the Anscombe-calibrated Ambient-Tweedie diffusion model.

Companion to ``tr_diffusion_ambient_tweedie_eval.py`` (which targets the older
``poisson_head``-heteroscedastic checkpoints, ``clean_target_memmap``/
``var_target_memmap``). This script targets the current recipe instead
(``cfg["anscombe"] = True``): the observation ``y`` is the Anscombe-transformed,
dose-thinned RAW measurement (no ``poisson_head`` dependency at all), with a
single GLOBAL SCALAR noise level ``sigma_tn_eff`` -- see the
``sdate.tr_diffusion.ambient_tweedie`` module docstring and ``data.py``'s
``anscombe=True`` path.

Scores every variant in NATIVE COUNT SPACE, against TWO references:
  - "noisy"     the actual dose-thinned measurement fed to the network as ``y``
                (after inverting Anscombe+dose back to native counts).
  - "reference" the ORIGINAL measured central frame (strictly less noisy than
                "noisy", since it was never further Poisson-thinned) -- the
                best pseudo-GT available, per this project's noise-sweep
                evaluation convention (see noise.py's module docstring).

Also runs the mandatory seed-sensitivity sanity check (this project's
established diagnostic for the "network ignores x_t" shortcut, confirmed
twice already this project on earlier variants of this same module): re-run
the plain sampler with a different seed and report the mean abs diff. Near-
zero (or small-and-non-growing-with-training) means the below-floor result
should not be trusted regardless of how good the PSNR/SSIM numbers look.

  python scripts/tr_diffusion_ambient_tweedie_anscombe_eval.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/BaseTraining")  # ambient_tweedie.py needs pytorch_base.base_loss at import time

from sdate.tr_diffusion.ambient_tweedie import sample_below_floor  # noqa: E402
from sdate.tr_diffusion.data import TimeResolvedFrameDataset  # noqa: E402
from sdate.tr_diffusion.load import load_denoiser  # noqa: E402
from sdate.tr_diffusion.noise import inverse_anscombe  # noqa: E402
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

DATA = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2"
CK = "/myhome/data/sdate/shared/checkpoints"
OUT = Path("/myhome/data/sdate/shared/time_resolved/tr_recon_cache")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=f"{CK}/tr_denoise_ambient_tweedie_anscombe.pt")
    p.add_argument("--frame_start", type=int, default=430_000)
    p.add_argument("--n_frames", type=int, default=16)
    p.add_argument("--eta", type=float, default=0.0, help="sampler stochasticity (0 = deterministic ODE-like)")
    p.add_argument("--dps_scale", type=float, default=0.02,
                   help="DPS guidance strength for the guided variant (0 disables guidance in that variant too).")
    p.add_argument("--noise_seed", type=int, default=999)
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--chunk_size", type=int, default=4, help="sub-batch size for the (gradient-tracked) DPS sampler")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tag", default="ambient_tweedie_anscombe_quickcheck")
    p.add_argument("--n_stills", type=int, default=4)
    return p.parse_args()


def main():
    a = parse_args()
    import sdate.tr_diffusion.ambient_tweedie as _at_mod
    print(f"ambient_tweedie module loaded from: {_at_mod.__file__} "
          f"(mtime={os.path.getmtime(_at_mod.__file__)})")
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)

    model, cfg = load_denoiser(a.ckpt, device=dev)
    assert cfg["mode"] == "ambient_tweedie", f"expected an ambient_tweedie checkpoint, got mode={cfg['mode']}"
    assert cfg.get("anscombe"), f"expected an anscombe-calibrated checkpoint, got anscombe={cfg.get('anscombe')}"
    lo, hi = float(cfg["norm_min"]), float(cfg["norm_max"])
    DR = hi - lo
    k = int(cfg["k"]); crop = tuple(cfg["crop"])
    sigma_min, sigma_max, sigma_tn_eff = float(cfg["sigma_min"]), float(cfg["sigma_max"]), float(cfg["sigma_tn_eff"])
    extra_noise_dose = cfg.get("extra_noise_dose")
    assert extra_noise_dose is not None, "ambient_tweedie checkpoint config is missing extra_noise_dose"

    ds = TimeResolvedFrameDataset(
        cfg["mov"], memmap_path=f"{DATA}/frames_400k_500k.u16",
        k=k, frame_start=400_000, frame_end=500_000, crop=crop, norm_range=(lo, hi),
        extra_noise_dose=extra_noise_dose, noise_seed=a.noise_seed,
        temporal_raw_pairs=bool(cfg.get("temporal_raw_pairs", False)),
        anscombe=True, anscombe_norm_sample_frames=8,  # refit range irrelevant -- overridden below
    )
    # Reuse the EXACT anscombe calibration the checkpoint was trained with, rather than
    # refitting a fresh one on this eval's frame range -- the network's input/output
    # normalization convention is baked into the trained weights.
    ds.anscombe_z_min, ds.anscombe_z_max = float(cfg["anscombe_z_min"]), float(cfg["anscombe_z_max"])
    ds.anscombe_sigma_tn_norm = sigma_tn_eff

    base = int(ds.indices[0])
    idx = [a.frame_start - base + i for i in range(a.n_frames)]
    assert min(idx) >= 0 and max(idx) < len(ds), (
        f"requested block [{a.frame_start}, {a.frame_start + a.n_frames}) falls outside the "
        f"dataset's usable range (dataset indices {base}..{base + len(ds) - 1})"
    )
    its = [ds[i] for i in idx]
    ctx = torch.stack([it["context"] for it in its]).to(dev)
    reference_n = torch.stack([it["reference"] for it in its]).to(dev)  # original (less-noisy) central frame
    y = torch.stack([it["clean_target"] for it in its]).to(dev)  # Anscombe(dose-thinned measurement), z_norm units
    sigma_tn_map = torch.stack([it["sigma_tn_map"] for it in its]).to(dev)

    def counts_native(z_norm: torch.Tensor) -> np.ndarray:
        """z_norm ([-1,1] network units) -> native detector counts."""
        z_raw = (z_norm.detach().float().cpu() + 1.0) * 0.5 * (ds.anscombe_z_max - ds.anscombe_z_min) + ds.anscombe_z_min
        raw_draw = inverse_anscombe(z_raw)  # ~ Poisson(counts * dose), the RAW photon draw
        return (raw_draw / extra_noise_dose).numpy()[:, 0]

    def ref_counts(x_norm: torch.Tensor) -> np.ndarray:
        return ((x_norm.detach().float().cpu().clamp(-1, 1) + 1) * 0.5 * DR + lo).numpy()[:, 0]

    def scores(pred_c, ref_c):
        return (float(np.mean([psnr(r, q, data_range=DR) for q, r in zip(pred_c, ref_c)])),
                float(np.mean([ssim(r, q, data_range=DR) for q, r in zip(pred_c, ref_c)])))

    noisy_c = counts_native(y)          # the actual dose-thinned measurement, in native counts
    reference_c = ref_counts(reference_n)  # the original, less-noisy central frame

    g = torch.Generator(device=dev).manual_seed(a.sample_seed)
    below_plain = sample_below_floor(model, y, sigma_tn_map, ctx, sigma_min=sigma_min, sigma_max=sigma_max,
                                     sigma_tn_eff=sigma_tn_eff, n_rungs=int(cfg["n_rungs"]),
                                     eta=a.eta, dps_scale=0.0, generator=g)
    g2 = torch.Generator(device=dev).manual_seed(a.sample_seed)
    below_dps = sample_below_floor(model, y, sigma_tn_map, ctx, sigma_min=sigma_min, sigma_max=sigma_max,
                                   sigma_tn_eff=sigma_tn_eff, n_rungs=int(cfg["n_rungs"]),
                                   eta=a.eta, dps_scale=a.dps_scale, chunk_size=a.chunk_size, generator=g2)
    for tag, below in (("plain", below_plain), (f"dps{a.dps_scale}", below_dps)):
        print(f"below_floor[{tag}] RAW (pre-clamp, z_norm units) min={below.min().item():.4f} "
              f"max={below.max().item():.4f} mean={below.mean().item():.4f} std={below.std().item():.4f}")

    # MANDATORY sanity check (see module docstring): re-run the plain sampler with a
    # different seed. If the mean abs diff doesn't grow relative to earlier/undertrained
    # checkpoints, the network is still not genuinely using the diffusion trajectory.
    g3 = torch.Generator(device=dev).manual_seed(a.sample_seed + 1)
    below_plain_seed2 = sample_below_floor(model, y, sigma_tn_map, ctx, sigma_min=sigma_min, sigma_max=sigma_max,
                                           sigma_tn_eff=sigma_tn_eff, n_rungs=int(cfg["n_rungs"]),
                                           eta=a.eta, dps_scale=0.0, generator=g3)
    seed_sensitivity = (below_plain - below_plain_seed2).abs().mean().item()
    out_std = below_plain.std().item()
    print(f"SANITY CHECK -- plain sampler, two different seeds, mean abs diff (z_norm units) = "
          f"{seed_sensitivity:.6f}  (output std={out_std:.6f}, ratio={seed_sensitivity / max(out_std, 1e-8):.4f}) "
          f"-- near-zero ratio would mean the network is still ignoring x_t/noise, i.e. NOT genuinely "
          f"using the diffusion trajectory, and the result below should not be trusted")

    variants = {
        "below_floor_plain": counts_native(below_plain),
        f"below_floor_dps{a.dps_scale}": counts_native(below_dps),
    }

    metrics = {
        "seed_sensitivity_abs": seed_sensitivity,
        "seed_sensitivity_ratio": seed_sensitivity / max(out_std, 1e-8),
        "noisy_vs_reference": scores(noisy_c, reference_c),
    }
    for name, arr in variants.items():
        metrics[f"{name}_vs_reference"] = scores(arr, reference_c)
        metrics[f"{name}_vs_noisy"] = scores(arr, noisy_c)

    print(f"eval: ckpt={a.ckpt}  frames=[{a.frame_start},{a.frame_start + a.n_frames})  "
          f"sigma_tn_eff={sigma_tn_eff:.5f}  eta={a.eta}")
    for name, val in metrics.items():
        if isinstance(val, tuple):
            print(f"  {name:34s} PSNR {val[0]:6.2f}  SSIM {val[1]:.3f}")
        else:
            print(f"  {name:34s} {val:.6f}")

    def jsonable(v):
        return {"psnr": v[0], "ssim": v[1]} if isinstance(v, tuple) else v

    (OUT / f"{a.tag}_metrics.json").write_text(json.dumps(
        {"ckpt": a.ckpt, "frame_start": a.frame_start, "n_frames": a.n_frames,
         "sigma_tn_eff": sigma_tn_eff, "sigma_min": sigma_min, "sigma_max": sigma_max,
         "eta": a.eta, "dps_scale": a.dps_scale,
         "metrics": {k_: jsonable(v_) for k_, v_ in metrics.items()}},
        indent=2))

    n_show = min(a.n_stills, a.n_frames)
    cols = ["noisy", "reference", "below_floor_plain", f"below_floor_dps{a.dps_scale}"]
    all_imgs = {"noisy": noisy_c, "reference": reference_c, **variants}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(n_show, len(cols), figsize=(3 * len(cols), 3 * n_show))
    axes = np.atleast_2d(axes)
    vmin, vmax = np.percentile(reference_c, [1, 99])
    for r in range(n_show):
        for c, name in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(all_imgs[name][r], cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"{a.tag}_stills.png", dpi=110)
    print(f"wrote {OUT / f'{a.tag}_metrics.json'} and {OUT / f'{a.tag}_stills.png'}")


if __name__ == "__main__":
    main()
