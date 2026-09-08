"""VAE-only sanity check: does the frozen pretrained SDXL VAE (madebyollin/sdxl-vae-fp16-fix)
faithfully round-trip our grayscale-replicated-to-RGB CT projection content at all, with NO
diffusion/noise/training involved?

Motivation: every below-floor SDXL-LoRA sample we've produced so far (checkpoints 4000/5000/
24000) shows a pronounced regular honeycomb/hexagonal tiling artifact that doesn't match the
reference's actual mottled structure, and that artifact got WORSE (not better) with more
training. Before concluding anything about the diffusion/LoRA training itself, rule out (or
confirm) that the artifact's root cause is the VAE encode/decode round-trip alone -- i.e. that
the pretrained VAE, trained on natural RGB photos, simply cannot represent this content well
when fed a single grayscale channel replicated to 3 channels.

Takes a handful of REFERENCE (native, full-dose, "clean" as this dataset defines it -- no
extra synthetic noise) frames, replicates to 3ch, VAE round-trips them (no diffusion at all),
and reports PSNR/SSIM of decoded-vs-original plus a visual comparison, isolating the VAE from
everything else in the pipeline.
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/BaseTraining")

import numpy as np
import torch
from diffusers import AutoencoderKL
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from sdate.tr_diffusion.data import TimeResolvedFrameDataset

VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
DATA = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2"


def laplace(im: np.ndarray) -> np.ndarray:
    return (-4 * im[1:-1, 1:-1] + im[:-2, 1:-1] + im[2:, 1:-1] + im[1:-1, :-2] + im[1:-1, 2:])


def sharpness(imgs):
    return float(np.mean([laplace(im.astype(np.float64)).var() for im in imgs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_start", type=int, default=430000)
    ap.add_argument("--n_frames", type=int, default=4)
    ap.add_argument("--crop_h", type=int, default=128)
    ap.add_argument("--crop_w", type=int, default=512)
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--sample", action="store_true",
                     help="use vae.encode(...).sample() (stochastic posterior) instead of .mode()")
    ap.add_argument("--out", default="/myhome/data/sdate/shared/time_resolved/tr_recon_cache/sdxl_vae_sanity.png")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if a.dtype == "fp16" else torch.float32
    print(f"device={dev} dtype={dtype}")

    vae = AutoencoderKL.from_pretrained(VAE_MODEL, torch_dtype=dtype).to(dev)
    vae.eval()

    ds = TimeResolvedFrameDataset(
        f"{DATA}/212_Wunderkerze2.mov", memmap_path=f"{DATA}/frames_400k_500k.u16",
        k=0, frame_start=400_000, frame_end=500_000, crop=(a.crop_h, a.crop_w),
        norm_range=None, anscombe=False,
    )
    base = int(ds.indices[0])
    lo, hi = float(ds.norm_min), float(ds.norm_max)
    DR = hi - lo

    idx = [a.frame_start - base + i for i in range(a.n_frames)]
    items = [ds[i] for i in idx]
    ref = torch.stack([it["central"] for it in items])  # (B,1,H,W), [-1,1], native full-dose
    bsz = ref.shape[0]

    def to_counts(x_norm: torch.Tensor) -> np.ndarray:
        return ((x_norm.detach().float().cpu().clamp(-1, 1) + 1) * 0.5 * DR + lo).numpy()[:, 0]

    ref_c = to_counts(ref)

    ref_rgb = ref.to(dtype=dtype, device=dev).expand(bsz, 3, a.crop_h, a.crop_w)
    with torch.no_grad():
        posterior = vae.encode(ref_rgb).latent_dist
        latent = posterior.sample() if a.sample else posterior.mode()
        latent = latent * vae.config.scaling_factor
        decoded = vae.decode(latent / vae.config.scaling_factor, return_dict=False)[0]
    decoded_1ch = decoded.mean(dim=1, keepdim=True)
    decoded_c = to_counts(decoded_1ch)

    p = float(np.mean([psnr(r, q, data_range=DR) for q, r in zip(decoded_c, ref_c)]))
    s = float(np.mean([ssim(r, q, data_range=DR) for q, r in zip(decoded_c, ref_c)]))
    print(f"[vae_sanity dtype={a.dtype} sample={a.sample}] "
          f"reference_vs_vae_roundtrip  PSNR {p:6.2f}  SSIM {s:.3f}  "
          f"sharpness(reference)={sharpness(ref_c):.2f}  sharpness(vae_roundtrip)={sharpness(decoded_c):.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vmin, vmax = np.percentile(ref_c[0], [1, 99])
    diff = decoded_c[0] - ref_c[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
    axes[0].imshow(ref_c[0], cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("reference (native, full-dose)", fontsize=9)
    axes[1].imshow(decoded_c[0], cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"VAE round-trip (PSNR {p:.2f}/SSIM {s:.3f})", fontsize=9)
    dv = np.percentile(np.abs(diff), 99)
    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-dv, vmax=dv)
    axes[2].set_title("decoded - reference", fontsize=9)
    plt.colorbar(im2, ax=axes[2], fraction=0.03)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
