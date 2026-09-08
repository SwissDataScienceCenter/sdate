#!/usr/bin/env python3
"""Below-floor sampling + eval for the NATIVE-CLEAN SDXL-LoRA ambient-Tweedie
checkpoint (see `sdate/tr_diffusion/ambient_tweedie_sdxl/data.py`'s
`SDXLAmbientNativeCleanDataset` and `_sdxl_ambient_train_native_clean_wrapper.sh`).

This is a DIFFERENT litmus test from `tr_diffusion_ambient_tweedie_sdxl_eval.py`
(the earlier real-noisy-measurement eval): that model was trained treating our
actual ~15dB-PSNR noisy measurement as "the one ambient observation", which
regressed with more training. This checkpoint was instead trained on genuinely
CLEAN native reference frames (their literal upstream recipe -- see
`train_low_level_laion10k.yaml`'s `timestep_nature: 100` on real, clean LAION
photos), so testing it on our real noisy measurement would be a mismatched
setting (its assumed noise floor, sigma(100), is far below our real measurement's
actual noise level).

Instead, we build a FULLY CONTROLLED test where we know the ground truth exactly:
take a real clean reference frame, VAE-encode it (the TRUE clean latent x0),
inject REAL synthetic Gaussian noise onto it at exactly sigma(timestep_nature)
using the standard VP forward process (`noise_scheduler.add_noise`, which we
confirmed matches ambient_utils' sigma convention: sigma_t = sqrt(1-alphas_cumprod[t]),
noisy = sqrt(alphas_cumprod[t])*x0 + sigma_t*noise -- exactly DDPM's add_noise),
then run the model's below-floor sampler (same `move_one_step` ladder-walk as the
other eval script) starting from that KNOWN corruption, walking down to t=0.

This directly tests the paper's core claim in a setting where "ground truth" is
not in question: does the below-floor sample end up CLOSER to the true clean
image than the corrupted observation itself?

Usage:
  python scripts/tr_diffusion_ambient_tweedie_sdxl_nativeclean_eval.py \
      --ckpt_dir /myhome/data/sdate/shared/checkpoints/tr_diff_ambient_tweedie_sdxl_lora_nativeclean/checkpoint-6000 \
      --n_steps 20 --n_frames 8 --tag nc_ckpt6000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/BaseTraining")

import ambient_utils  # noqa: E402
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel  # noqa: E402
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim  # noqa: E402

from sdate.tr_diffusion.ambient_tweedie_sdxl.train_lora_sdxl import encode_prompt, tokenize_prompt  # noqa: E402
from sdate.tr_diffusion.data import TimeResolvedFrameDataset  # noqa: E402

DATA = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2"
OUT = Path("/myhome/data/sdate/shared/time_resolved/tr_recon_cache")
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
PLACEHOLDER_CAPTION = "a grayscale x-ray computed tomography projection image"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--timestep_nature", type=int, default=100)
    p.add_argument("--n_steps", type=int, default=20)
    p.add_argument("--crop_h", type=int, default=128)
    p.add_argument("--crop_w", type=int, default=512)
    p.add_argument("--frame_start", type=int, default=430_000)
    p.add_argument("--n_frames", type=int, default=8)
    p.add_argument("--noise_seed", type=int, default=999)
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def move_one_step(xt, timesteps_xt, timesteps_xs, unet, prompt_embeds, unet_added_conditions,
                  alphas_cumprod, generator=None):
    """Verbatim port of train_lora_sdxl.py's `move_one_step` closure -- see the sibling
    eval script for the dtype-cast note (fp16 UNet vs fp32 alphas_cumprod)."""
    orig_dtype = xt.dtype
    sigma_t = ambient_utils.diffusers_utils.timesteps_to_sigma(timesteps_xt, alphas_cumprod)
    var_t = sigma_t ** 2
    noise_pred_xt = unet(xt, timesteps_xt, prompt_embeds, added_cond_kwargs=unet_added_conditions,
                        return_dict=False)[0]
    x0_pred = ambient_utils.from_noise_pred_to_x0_pred_vp(xt, noise_pred_xt, sigma_t)

    sigma_s = ambient_utils.diffusers_utils.timesteps_to_sigma(timesteps_xs, alphas_cumprod)
    var_s = sigma_s ** 2
    alpha_s = torch.sqrt(1 - var_s)[:, None, None, None]

    fresh_noise_coeff = ((var_s / var_t).sqrt() * (1 - (1 - var_t) / (1 - var_s)).sqrt())[:, None, None, None]
    old_noise_coeff = torch.max(var_s[:, None, None, None] - fresh_noise_coeff ** 2,
                                torch.zeros_like(fresh_noise_coeff)).sqrt()
    z = torch.randn(x0_pred.shape, device=x0_pred.device, generator=generator)
    xs = alpha_s * x0_pred + old_noise_coeff * noise_pred_xt + fresh_noise_coeff * z
    return xs.to(orig_dtype), x0_pred


def laplace(im: np.ndarray) -> np.ndarray:
    return (-4 * im[1:-1, 1:-1] + im[:-2, 1:-1] + im[2:, 1:-1] + im[1:-1, :-2] + im[1:-1, 2:])


def sharpness(imgs):
    return float(np.mean([laplace(im.astype(np.float64)).var() for im in imgs]))


def main():
    a = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)

    print(f"Loading base model + LoRA from {a.ckpt_dir} ...")
    vae = AutoencoderKL.from_pretrained(VAE_MODEL, torch_dtype=torch.float32).to(dev)
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet", torch_dtype=torch.float16).to(dev)
    scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
    alphas_cumprod = scheduler.alphas_cumprod.to(dev)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL, vae=vae, unet=unet, torch_dtype=torch.float16, variant=None)
    pipe.load_lora_weights(a.ckpt_dir)
    pipe = pipe.to(dev)
    unet = pipe.unet
    tokenizer_one, tokenizer_two = pipe.tokenizer, pipe.tokenizer_2
    text_encoder_one, text_encoder_two = pipe.text_encoder, pipe.text_encoder_2

    tokens_one = tokenize_prompt(tokenizer_one, [PLACEHOLDER_CAPTION])
    tokens_two = tokenize_prompt(tokenizer_two, [PLACEHOLDER_CAPTION])
    prompt_embeds, pooled_prompt_embeds = encode_prompt(
        text_encoders=[text_encoder_one, text_encoder_two], tokenizers=None, prompt=None,
        text_input_ids_list=[tokens_one.to(dev), tokens_two.to(dev)])

    ds = TimeResolvedFrameDataset(
        f"{DATA}/212_Wunderkerze2.mov", memmap_path=f"{DATA}/frames_400k_500k.u16",
        k=0, frame_start=400_000, frame_end=500_000, crop=(a.crop_h, a.crop_w),
        norm_range=None,
    )
    base = int(ds.indices[0])
    lo, hi = float(ds.norm_min), float(ds.norm_max)
    DR = hi - lo

    idx = [a.frame_start - base + i for i in range(a.n_frames)]
    items = [ds[i] for i in idx]
    ref = torch.stack([it["central"] for it in items]).to(dev)  # (B,1,H,W), [-1,1], TRUE clean
    bsz = ref.shape[0]

    def to_counts(x_norm: torch.Tensor) -> np.ndarray:
        return ((x_norm.detach().float().cpu().clamp(-1, 1) + 1) * 0.5 * DR + lo).numpy()[:, 0]

    def latent_to_counts(latent: torch.Tensor) -> np.ndarray:
        decoded = vae.decode(latent.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
        decoded_1ch = decoded.mean(dim=1, keepdim=True)
        return to_counts(decoded_1ch)

    reference_c = to_counts(ref)

    def scores(pred_c, ref_c):
        return (float(np.mean([psnr(r, q, data_range=DR) for q, r in zip(pred_c, ref_c)])),
                float(np.mean([ssim(r, q, data_range=DR) for q, r in zip(pred_c, ref_c)])))

    ref_rgb = ref.to(torch.float32).expand(bsz, 3, a.crop_h, a.crop_w)
    gen_corrupt = torch.Generator(device=dev).manual_seed(a.noise_seed)
    with torch.no_grad():
        x0_true = vae.encode(ref_rgb.to(vae.dtype)).latent_dist.mode() * vae.config.scaling_factor
        # KNOWN synthetic corruption at exactly sigma(timestep_nature), same VP forward
        # process ambient_utils' sigma convention corresponds to (verified against
        # ambient_utils/diffusers.py: sigma_t = sqrt(1-alphas_cumprod[t]), noisy =
        # sqrt(alphas_cumprod[t])*x0 + sigma_t*noise -- exactly DDPMScheduler.add_noise).
        noise = torch.randn(x0_true.shape, device=dev, generator=gen_corrupt)
        t_nature = torch.full((bsz,), a.timestep_nature, device=dev, dtype=torch.long)
        x_corrupted = scheduler.add_noise(x0_true, noise, t_nature)

    corrupted_c = latent_to_counts(x_corrupted)
    p_c, s_c = scores(corrupted_c, reference_c)
    print(f"[{a.tag}] corrupted_vs_reference (KNOWN synthetic noise @ t={a.timestep_nature})  "
          f"PSNR {p_c:6.2f}  SSIM {s_c:.3f}  "
          f"sharpness(corrupted)={sharpness(corrupted_c):.2f}  sharpness(reference)={sharpness(reference_c):.2f}")

    def compute_time_ids(bsz):
        target_size = (a.crop_h, a.crop_w)
        add_time_ids = torch.tensor([list(target_size + (0, 0) + target_size)] * bsz, device=dev, dtype=unet.dtype)
        return add_time_ids

    unet_added_conditions = {
        "time_ids": compute_time_ids(bsz),
        "text_embeds": pooled_prompt_embeds.expand(bsz, -1).to(unet.dtype),
    }
    prompt_embeds_b = prompt_embeds.expand(bsz, -1, -1).to(unet.dtype)

    ladder = torch.from_numpy(np.linspace(0, a.timestep_nature, a.n_steps + 1)).round().long().to(dev)
    gen = torch.Generator(device=dev).manual_seed(a.sample_seed)
    x = x_corrupted.to(unet.dtype)
    with torch.no_grad():
        for m in range(len(ladder) - 1, 0, -1):
            t_cur = ladder[m].expand(bsz)
            t_next = ladder[m - 1].expand(bsz)
            x, x0_pred = move_one_step(x, t_cur, t_next, unet, prompt_embeds_b, unet_added_conditions,
                                       alphas_cumprod, generator=gen)

    sample_c = latent_to_counts(x)
    p_s, s_s = scores(sample_c, reference_c)
    sharp_s = sharpness(sample_c)
    print(f"[{a.tag}] SDXL-LoRA below-floor sample   PSNR {p_s:6.2f}  SSIM {s_s:.3f}  sharpness={sharp_s:.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vmin, vmax = np.percentile(reference_c[0], [1, 99])
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.6))
    for ax, (label, img) in zip(axes, [("reference (true clean)", reference_c[0]),
                                       ("corrupted (known synth. noise)", corrupted_c[0]),
                                       ("SDXL-LoRA below-floor", sample_c[0])]):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_ylabel(label, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out_path = OUT / f"sdxl_ambient_nativeclean_eval_{a.tag}.png"
    fig.savefig(out_path, dpi=100)
    print(f"[{a.tag}] wrote {out_path}")


if __name__ == "__main__":
    main()
