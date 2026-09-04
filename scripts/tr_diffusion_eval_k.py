#!/usr/bin/env python3
"""Evaluate a diffusion denoiser checkpoint (any k) on a fixed held-out frame set.

Projection-space PSNR/SSIM vs the native (low-noise) reference, at dose 0.05, for
single-shot pred_x0 (t) and posterior-mean over B samples. The frame set is the
same contiguous block for every k, so results are directly comparable across the
k-ablation. Prints one line per metric.

  python scripts/tr_diffusion_eval_k.py --ckpt checkpoints/tr_denoise_diffusion_k1_dose005.pt
"""
from __future__ import annotations
import argparse, sys
import numpy as np, torch
sys.path.insert(0, "/myhome/sdate")
from sdate.tr_diffusion.data import TimeResolvedFrameDataset
from sdate.tr_diffusion.load import load_denoiser
from sdate.tr_diffusion.pipeline import pred_x0_ensemble
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

DATA = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--frame_start", type=int, default=470_000)
    p.add_argument("--n_frames", type=int, default=128)
    p.add_argument("--t", type=int, default=500)
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--noise_seed", type=int, default=999)
    a = p.parse_args()
    dev = torch.device("cuda")

    model, cfg = load_denoiser(a.ckpt, device=dev)
    k = int(cfg["k"]); crop = tuple(cfg["crop"]); lo, hi = cfg["norm_min"], cfg["norm_max"]; DR = hi - lo
    ds = TimeResolvedFrameDataset(f"{DATA}/212_Wunderkerze2.mov", memmap_path=f"{DATA}/frames_400k_500k.u16",
                                  k=k, frame_start=400_000, frame_end=500_000, crop=crop,
                                  norm_range=(lo, hi), extra_noise_dose=0.05, noise_seed=a.noise_seed)
    base = int(ds.indices[0])
    idx = [a.frame_start - base + i for i in range(a.n_frames)]
    its = [ds[i] for i in idx]
    cen = torch.stack([it["central"] for it in its]).to(dev)
    ctx = torch.stack([it["context"] for it in its]).to(dev)
    ref = torch.stack([it["reference"] for it in its]).to(dev)

    def counts(x):
        return ((x.detach().float().cpu().clamp(-1, 1) + 1) * 0.5 * DR + lo).numpy()[:, 0]

    def scores(pred_c, ref_c):
        return (float(np.mean([psnr(r, q, data_range=DR) for q, r in zip(pred_c, ref_c)])),
                float(np.mean([ssim(r, q, data_range=DR) for q, r in zip(pred_c, ref_c)])))

    ref_c, in_c = counts(ref), counts(cen)
    g = torch.Generator(device=dev).manual_seed(0)
    _, samples = pred_x0_ensemble(model, cen, ctx, timestep=a.t, num_samples=a.B,
                                  ratio=cfg["n2v_ratio"], window=cfg["n2v_window"], chunk_size=32, generator=g)
    ss1 = counts(samples[:, 0])                 # single-shot
    pmB = counts(samples.mean(dim=1))           # posterior mean B

    pin = scores(in_c, ref_c)
    p1 = scores(ss1, ref_c)
    pB = scores(pmB, ref_c)
    print(f"k={k}  ckpt={a.ckpt.split('/')[-1]}  ({a.n_frames} frames @ {a.frame_start}, t={a.t}, B={a.B})")
    print(f"  noisy input       PSNR {pin[0]:6.2f}  SSIM {pin[1]:.3f}")
    print(f"  single-shot (B=1) PSNR {p1[0]:6.2f}  SSIM {p1[1]:.3f}")
    print(f"  posterior mean B={a.B:<2d} PSNR {pB[0]:6.2f}  SSIM {pB[1]:.3f}")


if __name__ == "__main__":
    main()
