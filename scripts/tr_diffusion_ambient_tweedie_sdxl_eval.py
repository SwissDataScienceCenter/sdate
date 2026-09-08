#!/usr/bin/env python3
"""Below-floor sampling + eval for the SDXL-LoRA ambient-Tweedie checkpoint.

Their own `eval_scripts/generate.py` is NOT what we need here: it evaluates
TEXT-TO-IMAGE GENERATION quality (a prompt -> a novel sample, stopped early at
sigma_nature, for FID-style comparisons) -- it never starts from a real observation.
We need the opposite: start from OUR specific real measurement `y`, and denoise it
BELOW the observed noise floor -- a reconstruction task, not a generation task.

The sampler here reuses their training script's own `move_one_step` renoise formula
verbatim (see `train_lora_sdxl.py`'s consistency-loss branch) -- the model was
trained to be self-consistent under exactly this VP posterior step, so inference
should use the same one, just walking a full ladder from `timestep_nature` down to
~0 instead of the training loss's 1-2 random hops.

Usage:
  python scripts/tr_diffusion_ambient_tweedie_sdxl_eval.py \
      --ckpt_dir /myhome/data/sdate/shared/checkpoints/tr_diff_ambient_tweedie_sdxl_lora/checkpoint-1000 \
      --n_steps 20 --n_frames 8 --tag ckpt1000
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
    p.add_argument("--ckpt_dir", required=True, help="checkpoint-N directory (has pytorch_lora_weights.safetensors)")
    p.add_argument("--tag", required=True)
    p.add_argument("--timestep_nature", type=int, default=102)
    p.add_argument("--n_steps", type=int, default=20, help="ladder resolution from timestep_nature down to 0")
    p.add_argument("--crop_h", type=int, default=128)
    p.add_argument("--crop_w", type=int, default=512)
    p.add_argument("--frame_start", type=int, default=430_000)
    p.add_argument("--n_frames", type=int, default=8)
    p.add_argument("--extra_noise_dose", type=float, default=0.05)
    p.add_argument("--noise_seed", type=int, default=999)
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def move_one_step(xt, timesteps_xt, timesteps_xs, unet, prompt_embeds, unet_added_conditions,
                  alphas_cumprod, generator=None):
    """Verbatim port of train_lora_sdxl.py's `move_one_step` closure (its consistency-loss
    branch), generalised to accept an explicit target timestep instead of a random offset.

    `alphas_cumprod` is float32 (the scheduler's own buffer) -- mixing it with the fp16 UNet
    output upcasts every downstream tensor to float32 via normal broadcasting, so by the
    SECOND call `xt` would silently be float32 while the UNet's weights are fp16 -> crash
    inside `unet(...)`. Do the sigma/alpha arithmetic in fp32 (desirable for numerical
    stability, matches this project's established `ve_posterior_step` pattern) but cast the
    returned state back to `xt`'s original dtype before returning."""
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
        norm_range=None, extra_noise_dose=a.extra_noise_dose, noise_seed=a.noise_seed,
        anscombe=True, anscombe_norm_sample_frames=8,
    )
    base = int(ds.indices[0])
    lo, hi = float(ds.norm_min), float(ds.norm_max)
    DR = hi - lo

    idx = [a.frame_start - base + i for i in range(a.n_frames)]
    items = [ds[i] for i in idx]
    y = torch.stack([it["clean_target"] for it in items]).to(dev)  # (B,1,H,W), [-1,1]
    reference = torch.stack([it["reference"] for it in items])
    bsz = y.shape[0]

    def ref_counts(x_norm: torch.Tensor) -> np.ndarray:
        return ((x_norm.detach().float().cpu().clamp(-1, 1) + 1) * 0.5 * DR + lo).numpy()[:, 0]

    def latent_to_counts(latent: torch.Tensor) -> np.ndarray:
        decoded = vae.decode(latent.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
        decoded_1ch = decoded.mean(dim=1, keepdim=True)  # 3-channel replicated -> average back to 1
        return ref_counts(decoded_1ch)

    reference_c = ref_counts(reference)
    noisy_c = ref_counts(y)

    def scores(pred_c, ref_c):
        return (float(np.mean([psnr(r, q, data_range=DR) for q, r in zip(pred_c, ref_c)])),
                float(np.mean([ssim(r, q, data_range=DR) for q, r in zip(pred_c, ref_c)])))

    def laplace(im: np.ndarray) -> np.ndarray:
        return (-4 * im[1:-1, 1:-1] + im[:-2, 1:-1] + im[2:, 1:-1] + im[1:-1, :-2] + im[1:-1, 2:])

    def sharpness(imgs):
        return float(np.mean([laplace(im.astype(np.float64)).var() for im in imgs]))

    p_noisy, s_noisy = scores(noisy_c, reference_c)
    print(f"[{a.tag}] noisy_vs_reference  PSNR {p_noisy:6.2f}  SSIM {s_noisy:.3f}  "
          f"sharpness(noisy)={sharpness(noisy_c):.2f}  sharpness(reference)={sharpness(reference_c):.2f}")

    y_rgb = y.to(torch.float32).expand(bsz, 3, a.crop_h, a.crop_w)
    with torch.no_grad():
        model_input = vae.encode(y_rgb.to(vae.dtype)).latent_dist.sample() * vae.config.scaling_factor

    def compute_time_ids(bsz):
        target_size = (a.crop_h, a.crop_w)
        add_time_ids = torch.tensor([list(target_size + (0, 0) + target_size)] * bsz, device=dev, dtype=unet.dtype)
        return add_time_ids

    unet_added_conditions = {
        "time_ids": compute_time_ids(bsz),
        "text_embeds": pooled_prompt_embeds.expand(bsz, -1).to(unet.dtype),
    }
    prompt_embeds_b = prompt_embeds.expand(bsz, -1, -1).to(unet.dtype)

    # ASCENDING (ladder[0]=0 ... ladder[-1]=timestep_nature) -- the loop below walks it from
    # high index (timestep_nature, the boundary) down to low index (0), matching this
    # project's established VE-sampler convention. Building this DESCENDING instead (as an
    # earlier version of this script did) makes m=len(ladder)-1 the FIRST iteration start at
    # t_cur=0 while xt is actually the high-noise boundary latent -- a severe train/inference
    # distribution mismatch that produced all-NaN output under fp16 (confirmed empirically).
    ladder = torch.from_numpy(np.linspace(0, a.timestep_nature, a.n_steps + 1)).round().long().to(dev)
    gen = torch.Generator(device=dev).manual_seed(a.sample_seed)
    x = model_input.to(unet.dtype)
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
    for ax, (label, img) in zip(axes, [("reference", reference_c[0]), ("noisy measurement", noisy_c[0]),
                                       ("SDXL-LoRA below-floor", sample_c[0])]):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_ylabel(label, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out_path = OUT / f"sdxl_ambient_eval_{a.tag}.png"
    fig.savefig(out_path, dpi=100)
    print(f"[{a.tag}] wrote {out_path}")


if __name__ == "__main__":
    main()
