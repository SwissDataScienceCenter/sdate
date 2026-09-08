#!/usr/bin/env python3
"""Quickcheck eval for the N2N clean-target diffusion model (poisson_head target).

Question: does training the N2N diffusion model to regress toward the
poisson_head Bayesian reconstruction (instead of the other binomial-split raw
half) let ANCESTRAL sampling produce good, below-noise-floor samples -- the
raw-measurement-target recipe's ancestral sampler reconverged to the noisy
input (16.5dB, below the 19dB noisy floor); the later swap-consistency
("x0c") variant improved that to 25.85dB, still ~6dB behind single-shot.

Scores every variant against TWO references (their meaning differs):
  - "noisy"        the actual dose-0.05 measurement (continuity with every
                    earlier ablation's PSNR/SSIM convention).
  - "poisson_head"  the training target itself. A variant scoring HIGH here
                    just reproduced its target (expected, not informative on
                    its own); scoring LOWER here while still looking sharp in
                    the stills would indicate it moved BEYOND poisson_head's
                    own noise floor, i.e. the interesting outcome for
                    ancestral sampling specifically -- PSNR/SSIM against a
                    noisy reference cannot distinguish "denoised further" from
                    "just wrong", so the stills PNG is the primary evidence
                    for that judgment, these two metrics are secondary.

  python scripts/tr_diffusion_n2n_cleantarget_eval.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")

from sdate.tr_diffusion.data import TimeResolvedFrameDataset  # noqa: E402
from sdate.tr_diffusion.load import load_denoiser  # noqa: E402
from sdate.tr_diffusion.pipeline import (  # noqa: E402
    denoise_frames_baseline, partial_diffusion_n2n, pred_x0_n2n_ensemble, pred_x0_n2n_swap_ensemble,
)
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

DATA = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2"
CK = "/myhome/data/sdate/shared/checkpoints"
OUT = Path("/myhome/data/sdate/shared/time_resolved/tr_recon_cache")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=f"{CK}/tr_denoise_diffusion_n2n_cleantarget_poissonhead.pt")
    p.add_argument("--baseline_ckpt", default=f"{CK}/tr_denoise_baseline_k1_dose005.pt")
    p.add_argument("--frame_start", type=int, default=430_000)
    p.add_argument("--n_frames", type=int, default=128)
    p.add_argument("--ts", type=int, nargs="+", default=[300, 400, 500, 600],
                   help="single-shot timestep sweep; best is reported/used for stills")
    p.add_argument("--B", type=int, default=8, help="posterior-mean draws for single-shot/swap")
    p.add_argument("--ancestral_steps", type=int, default=50)
    p.add_argument("--noise_seed", type=int, default=999)
    p.add_argument("--tag", default="n2n_cleantarget_poissonhead_quickcheck")
    p.add_argument("--n_stills", type=int, default=4)
    return p.parse_args()


def main():
    a = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")

    model, cfg = load_denoiser(a.ckpt, device=dev)
    baseline, bcfg = load_denoiser(a.baseline_ckpt, device=dev)
    assert cfg["norm_min"] == bcfg["norm_min"] and cfg["norm_max"] == bcfg["norm_max"], \
        "checkpoints must share a norm range for the comparison to be meaningful"
    lo, hi = float(cfg["norm_min"]), float(cfg["norm_max"])
    DR = hi - lo
    k = int(cfg["k"]); crop = tuple(cfg["crop"])
    p_bins = int(cfg.get("p_bins", 100))
    pred_type = cfg.get("n2n_prediction_type", "epsilon")

    ds = TimeResolvedFrameDataset(
        f"{DATA}/212_Wunderkerze2.mov", memmap_path=f"{DATA}/frames_400k_500k.u16",
        k=k, frame_start=400_000, frame_end=500_000, crop=crop, norm_range=(lo, hi),
        extra_noise_dose=0.05, noise_seed=a.noise_seed,
        n2n=True, p_bins=p_bins, temporal_raw_pairs=bool(cfg.get("temporal_raw_pairs", False)),
        clean_target_memmap=cfg["clean_target_memmap"],
    )
    base = int(ds.indices[0])
    idx = [a.frame_start - base + i for i in range(a.n_frames)]
    assert min(idx) >= 0 and max(idx) < len(ds), (
        f"requested block [{a.frame_start}, {a.frame_start + a.n_frames}) falls outside the "
        f"clean-target-covered usable range (dataset indices {base}..{base + len(ds) - 1})"
    )
    its = [ds[i] for i in idx]
    ctx = torch.stack([it["context"] for it in its]).to(dev)
    noisy = torch.stack([it["reference"] for it in its]).to(dev)  # the actual dose-0.05 measurement
    clean = torch.stack([it["clean_target"] for it in its]).to(dev)  # poisson_head's own output

    def counts(x):
        return ((x.detach().float().cpu().clamp(-1, 1) + 1) * 0.5 * DR + lo).numpy()[:, 0]

    def scores(pred_c, ref_c):
        return (float(np.mean([psnr(r, q, data_range=DR) for q, r in zip(pred_c, ref_c)])),
                float(np.mean([ssim(r, q, data_range=DR) for q, r in zip(pred_c, ref_c)])))

    noisy_c, clean_c = counts(noisy), counts(clean)

    variants = {}  # name -> counts array (n_frames, H, W)

    variants["poisson_head"] = clean_c  # the training target's own point, for reference

    with torch.no_grad():
        bl = denoise_frames_baseline(baseline, noisy, ctx, present=True)
    variants["n2v_baseline"] = counts(bl)

    g = torch.Generator(device=dev).manual_seed(0)
    best_t, best_psnr, best_ss1, best_pmB = None, -1e9, None, None
    for t in a.ts:
        _, samples = pred_x0_n2n_ensemble(model, noisy, ctx, q=1.0, timestep=t, num_samples=a.B,
                                          p_bins=p_bins, prediction_type=pred_type, chunk_size=32,
                                          generator=g)
        ss1_c = counts(samples[:, 0])
        p_ss1 = scores(ss1_c, noisy_c)[0]
        if p_ss1 > best_psnr:
            best_psnr, best_t, best_ss1, best_pmB = p_ss1, t, ss1_c, counts(samples.mean(dim=1))
    variants[f"single_shot_t{best_t}"] = best_ss1
    variants[f"posterior_mean_B{a.B}_t{best_t}"] = best_pmB

    _, swap_samples = pred_x0_n2n_swap_ensemble(model, noisy, ctx, norm_min=lo, norm_max=hi,
                                                timestep=best_t, p_bins=p_bins,
                                                prediction_type=pred_type, num_samples=a.B,
                                                chunk_size=32, generator=g)
    variants[f"swap_pm_B{a.B}_t{best_t}"] = counts(swap_samples.mean(dim=1))

    anc_mean, _ = partial_diffusion_n2n(model, noisy, ctx, q=1.0, t_start=500, t_end=0,
                                        num_steps=a.ancestral_steps, p_bins=p_bins,
                                        prediction_type=pred_type, num_samples=a.B,
                                        chunk_size=16, generator=g)
    variants[f"ancestral_pm_B{a.B}"] = counts(anc_mean)

    metrics = {"noisy_vs_noisy": scores(noisy_c, noisy_c), "poisson_head_vs_noisy": scores(clean_c, noisy_c)}
    for name, arr in variants.items():
        metrics[f"{name}_vs_noisy"] = scores(arr, noisy_c)
        metrics[f"{name}_vs_poisson_head"] = scores(arr, clean_c)

    print(f"eval: ckpt={a.ckpt}  frames=[{a.frame_start},{a.frame_start + a.n_frames})  "
          f"pred_type={pred_type}  best single-shot t={best_t}")
    for name, (p_, s_) in metrics.items():
        print(f"  {name:38s} PSNR {p_:6.2f}  SSIM {s_:.3f}")

    (OUT / f"{a.tag}_metrics.json").write_text(json.dumps(
        {"ckpt": a.ckpt, "frame_start": a.frame_start, "n_frames": a.n_frames, "best_t": best_t,
         "ancestral_steps": a.ancestral_steps, "B": a.B,
         "metrics": {k_: {"psnr": v_[0], "ssim": v_[1]} for k_, v_ in metrics.items()}},
        indent=2))

    n_show = min(a.n_stills, a.n_frames)
    cols = ["noisy", "poisson_head", f"single_shot_t{best_t}", f"posterior_mean_B{a.B}_t{best_t}",
            f"swap_pm_B{a.B}_t{best_t}", f"ancestral_pm_B{a.B}"]
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
