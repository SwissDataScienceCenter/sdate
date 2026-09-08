#!/usr/bin/env python3
"""Quickcheck eval for the Ambient-Tweedie diffusion model.

Question: does a denoiser trained via the Ambient-Tweedie framework (arXiv:
2404.10177), supervised only at-or-above sigma_tn_eff against poisson_head's
Bayesian posterior mean, produce a BELOW-floor sample (sample_below_floor,
run all the way to sigma_min) that looks cleaner than poisson_head itself --
the thing the earlier N2N clean-target ancestral sampler could not
demonstrate (it stopped at ~poisson_head's own noise level, not below it).

Scores every variant against TWO references (their meaning differs):
  - "noisy"        the actual dose-0.05 measurement (continuity with every
                    earlier ablation's PSNR/SSIM convention).
  - "poisson_head"  the training target itself (mu_hat). A variant scoring
                    HIGH here just reproduced its target (expected, not
                    informative alone); scoring LOWER here while still
                    looking sharp/detailed in the stills would indicate it
                    moved BEYOND poisson_head's own noise floor -- the actual
                    goal. PSNR/SSIM against a noisy reference cannot
                    distinguish "denoised further" from "just wrong", so the
                    stills PNG is the primary evidence for that judgment.

  python scripts/tr_diffusion_ambient_tweedie_eval.py
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
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

DATA = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2"
CK = "/myhome/data/sdate/shared/checkpoints"
OUT = Path("/myhome/data/sdate/shared/time_resolved/tr_recon_cache")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=f"{CK}/tr_denoise_ambient_tweedie_poissonhead.pt")
    p.add_argument("--frame_start", type=int, default=430_000)
    p.add_argument("--n_frames", type=int, default=128)
    p.add_argument("--eta", type=float, default=0.0, help="sampler stochasticity (0 = deterministic ODE-like)")
    p.add_argument("--dps_scale", type=float, default=0.02,
                   help="DPS guidance strength for the guided variant (0 disables guidance in that variant too).")
    p.add_argument("--noise_seed", type=int, default=999)
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--tag", default="ambient_tweedie_poissonhead_quickcheck")
    p.add_argument("--n_stills", type=int, default=4)
    return p.parse_args()


def main():
    a = parse_args()
    import sdate.tr_diffusion.ambient_tweedie as _at_mod
    print(f"ambient_tweedie module loaded from: {_at_mod.__file__} "
          f"(mtime={os.path.getmtime(_at_mod.__file__)})")
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")

    model, cfg = load_denoiser(a.ckpt, device=dev)
    assert cfg["mode"] == "ambient_tweedie", f"expected an ambient_tweedie checkpoint, got mode={cfg['mode']}"
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
        clean_target_memmap=cfg["clean_target_memmap"], var_target_memmap=cfg["var_target_memmap"],
    )
    base = int(ds.indices[0])
    idx = [a.frame_start - base + i for i in range(a.n_frames)]
    assert min(idx) >= 0 and max(idx) < len(ds), (
        f"requested block [{a.frame_start}, {a.frame_start + a.n_frames}) falls outside the "
        f"clean/var-target-covered usable range (dataset indices {base}..{base + len(ds) - 1})"
    )
    its = [ds[i] for i in idx]
    ctx = torch.stack([it["context"] for it in its]).to(dev)
    noisy = torch.stack([it["reference"] for it in its]).to(dev)  # the actual dose-0.05 measurement
    clean = torch.stack([it["clean_target"] for it in its]).to(dev)  # poisson_head's own mu_hat
    sigma_tn_map = torch.stack([it["sigma_tn_map"] for it in its]).to(dev)

    def counts(x):
        return ((x.detach().float().cpu().clamp(-1, 1) + 1) * 0.5 * DR + lo).numpy()[:, 0]

    def scores(pred_c, ref_c):
        return (float(np.mean([psnr(r, q, data_range=DR) for q, r in zip(pred_c, ref_c)])),
                float(np.mean([ssim(r, q, data_range=DR) for q, r in zip(pred_c, ref_c)])))

    noisy_c, clean_c = counts(noisy), counts(clean)

    variants = {"poisson_head": clean_c}

    g = torch.Generator(device=dev).manual_seed(a.sample_seed)
    below_plain = sample_below_floor(model, clean, sigma_tn_map, ctx, sigma_min=sigma_min, sigma_max=sigma_max,
                                     sigma_tn_eff=sigma_tn_eff, n_rungs=int(cfg["n_rungs"]),
                                     eta=a.eta, dps_scale=0.0, generator=g)
    g2 = torch.Generator(device=dev).manual_seed(a.sample_seed)
    below_dps = sample_below_floor(model, clean, sigma_tn_map, ctx, sigma_min=sigma_min, sigma_max=sigma_max,
                                   sigma_tn_eff=sigma_tn_eff, n_rungs=int(cfg["n_rungs"]),
                                   eta=a.eta, dps_scale=a.dps_scale, chunk_size=16, generator=g2)
    for tag, below in (("plain", below_plain), (f"dps{a.dps_scale}", below_dps)):
        print(f"below_floor[{tag}] RAW (pre-clamp, normalized units) min={below.min().item():.4f} "
              f"max={below.max().item():.4f} mean={below.mean().item():.4f} std={below.std().item():.4f} "
              f"frac_outside_[-1,1]={(below.abs() > 1).float().mean().item():.4f}")

    # Sanity check for the EXACT failure mode found earlier (network ignoring x_t/sigma
    # entirely, i.e. output invariant to the actual noise realisation): re-run the PLAIN
    # sampler with a different seed and confirm the output actually changes. If this comes
    # back ~0 again, the network has found some other shortcut and the below-floor result
    # should not be trusted regardless of how good the metrics look.
    g3 = torch.Generator(device=dev).manual_seed(a.sample_seed + 1)
    below_plain_seed2 = sample_below_floor(model, clean, sigma_tn_map, ctx, sigma_min=sigma_min, sigma_max=sigma_max,
                                           sigma_tn_eff=sigma_tn_eff, n_rungs=int(cfg["n_rungs"]),
                                           eta=a.eta, dps_scale=0.0, generator=g3)
    seed_sensitivity = (below_plain - below_plain_seed2).abs().mean().item()
    print(f"SANITY CHECK -- plain sampler, two different seeds, mean abs diff (normalized units) = "
          f"{seed_sensitivity:.6f} (near-zero would mean the network is still ignoring x_t/noise -- "
          f"i.e. NOT genuinely using the diffusion trajectory -- and the result below should not be trusted)")

    variants["below_floor_plain"] = counts(below_plain)
    variants[f"below_floor_dps{a.dps_scale}"] = counts(below_dps)

    metrics = {"noisy_vs_noisy": scores(noisy_c, noisy_c), "poisson_head_vs_noisy": scores(clean_c, noisy_c)}
    for name, arr in variants.items():
        metrics[f"{name}_vs_noisy"] = scores(arr, noisy_c)
        metrics[f"{name}_vs_poisson_head"] = scores(arr, clean_c)

    print(f"eval: ckpt={a.ckpt}  frames=[{a.frame_start},{a.frame_start + a.n_frames})  "
          f"sigma_tn_eff={sigma_tn_eff:.5f}  eta={a.eta}")
    for name, (p_, s_) in metrics.items():
        print(f"  {name:38s} PSNR {p_:6.2f}  SSIM {s_:.3f}")

    (OUT / f"{a.tag}_metrics.json").write_text(json.dumps(
        {"ckpt": a.ckpt, "frame_start": a.frame_start, "n_frames": a.n_frames,
         "sigma_tn_eff": sigma_tn_eff, "sigma_min": sigma_min, "sigma_max": sigma_max,
         "eta": a.eta, "dps_scale": a.dps_scale,
         "metrics": {k_: {"psnr": v_[0], "ssim": v_[1]} for k_, v_ in metrics.items()}},
        indent=2))

    n_show = min(a.n_stills, a.n_frames)
    cols = ["noisy", "poisson_head", "below_floor_plain", f"below_floor_dps{a.dps_scale}"]
    all_imgs = {"noisy": noisy_c, **variants}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(n_show, len(cols), figsize=(3 * len(cols), 3 * n_show))
    axes = np.atleast_2d(axes)
    vmin, vmax = np.percentile(noisy_c, [1, 99])
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
