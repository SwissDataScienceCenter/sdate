"""recon_utils.py – Shared reconstruction helpers for the iterative LA-Fourier pipeline.

Used by both the notebook and ``scripts/reconstruct_ladiff_slices.py`` to avoid
code duplication.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch

from ladiff.fourier_wedge import (
    apply_missing_wedge,
    inpaint_fourier_wedge,
    apply_circle_mask,
)
from ladiff.schedulers.scheduling_ddim import GuidedDDIMScheduler
from ladiff.schedulers.pipeline_ddim import DDIMPipeline
from diffusers.models import UNet2DModel


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def make_norm_fns(
    norm_min: float,
    norm_max: float,
) -> Tuple[Callable, Callable]:
    """Return ``(normalize_fn, denormalize_fn)`` closures for the given range."""

    def normalize_fn(x: torch.Tensor) -> torch.Tensor:
        return apply_circle_mask((x - norm_min) / (norm_max - norm_min))

    def denormalize_fn(x: torch.Tensor) -> torch.Tensor:
        return apply_circle_mask(x * (norm_max - norm_min) + norm_min)

    return normalize_fn, denormalize_fn


def load_norm_json(norm_json_path: Path) -> Tuple[float, float]:
    """Load ``norm_min`` / ``norm_max`` from a sidecar JSON produced by the dataset notebook."""
    with open(norm_json_path) as f:
        cfg = json.load(f)
    return float(cfg["norm_min"]), float(cfg["norm_max"])


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_unet(
    checkpoint_path: Path,
    slice_size: int,
    device: torch.device,
    tiny: bool = False,
) -> UNet2DModel:
    """Instantiate and load a UNet2DModel from a checkpoint."""
    channels = (32, 32, 32, 32, 64, 64) if tiny else (64, 64, 128, 128, 256, 256)
    model = UNet2DModel(
        sample_size=slice_size,
        in_channels=2,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=channels,
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D",
            "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
            "UpBlock2D", "UpBlock2D", "UpBlock2D",
        ),
        class_embed_type="timestep",
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inpainting guidance
# ---------------------------------------------------------------------------

def inpaint_guidance(
    source_volume: torch.Tensor,
    vol_init: torch.Tensor,
    angular_range_deg: int,
    start_angle: int,
    tilt_axis: int,
    device: torch.device,
    timestep: int = 0,
    inpaint_steps: int = 1,
) -> torch.Tensor:
    """Apply Fourier-inpainting guidance using limited-angle data.

    Parameters
    ----------
    source_volume:
        Full-angle ground-truth or oracle volume on any device, shape ``(D, H, W)``.
    vol_init:
        Current denoised estimate, shape ``(D, H, W)``.
    angular_range_deg:
        Angular range *kept* (i.e. the complement of the missing wedge).
    start_angle:
        Start angle of the kept region in degrees.
    tilt_axis:
        Tilt axis index (0, 1, or 2).
    device:
        Computation device.
    timestep:
        Current diffusion timestep (used to apply guidance only every
        ``inpaint_steps`` intervals of 20 steps).
    inpaint_steps:
        Apply inpainting every this many 20-step intervals (1 = every step).

    Returns
    -------
    Reconstructed volume on *device*, shape ``(D, H, W)``.
    """
    if (timestep // 20) % inpaint_steps == 0:
        recon = inpaint_fourier_wedge(
            target_volume=vol_init.to(device),
            source_volume=source_volume.to(device),
            angular_range_deg=180 - angular_range_deg,
            start_angle_deg=start_angle - 80,
            tilt_axis=tilt_axis,
        )
        return recon
    return vol_init


# ---------------------------------------------------------------------------
# LA-FBP computation
# ---------------------------------------------------------------------------

def compute_la_fbp(
    gt_volume: torch.Tensor,
    angular_range_deg: int,
    start_angle: int,
    tilt_axis: int,
    device: torch.device,
) -> torch.Tensor:
    """Apply the missing-wedge forward operator and return the LA-FBP volume."""
    la = apply_missing_wedge(
        gt_volume.to(device), angular_range_deg, start_angle, tilt_axis
    ).cpu()
    la = apply_circle_mask(la)
    return la


# ---------------------------------------------------------------------------
# Single-slice iterative refinement
# ---------------------------------------------------------------------------

def iterative_refine_slice(
    slice_idx: int,
    gt_fbp: torch.Tensor,
    la_fbp: torch.Tensor,
    model: UNet2DModel,
    normalize_fn: Callable,
    denormalize_fn: Callable,
    angular_range_deg: int,
    start_angle: int,
    tilt_axis: int,
    device: torch.device,
    num_slices_batch: int = 32,
    num_outer_iters: int = 30,
    start_step_frac: float = 0.8,
    num_inference_steps: int = 40,
    average_gradient_steps: bool = True,
) -> Tuple[np.ndarray, list]:
    """Run iterative DDIM refinement for a single input slice.

    The slice at ``slice_idx`` is repeated ``num_slices_batch`` times so that the
    pipeline can average over multiple stochastic draws to reduce noise.

    Parameters
    ----------
    slice_idx:
        Index into the *original* gt_fbp volume (used for GT reference metrics).
    gt_fbp:
        Full GT volume on CPU, shape ``(N, H, W)``.
    la_fbp:
        Limited-angle volume on CPU, shape ``(N, H, W)``.
    model:
        Loaded UNet2DModel on *device*.
    normalize_fn / denormalize_fn:
        Normalisation callables.
    angular_range_deg / start_angle / tilt_axis:
        Missing-wedge geometry parameters.
    device:
        Torch device.
    num_slices_batch:
        Number of times the slice is repeated in the batch (Monte-Carlo average).
    num_outer_iters:
        Number of outer refinement iterations.
    start_step_frac:
        Fraction of inference steps at which to inject noise (0 = full denoising,
        1 = almost no denoising).
    num_inference_steps:
        Total DDIM steps per pass.
    average_gradient_steps:
        Whether to average gradient steps over the batch.

    Returns
    -------
    best_recon_np:
        Best reconstruction as a ``(H, W)`` numpy array.
    metrics_history:
        List of dicts with per-iteration metrics.
    """
    from skimage.metrics import peak_signal_noise_ratio as compute_psnr
    from skimage.metrics import structural_similarity as compute_ssim

    gt_slice = gt_fbp[slice_idx]                    # (H, W)
    la_slice = la_fbp[slice_idx]                    # (H, W)

    # Build a synthetic batch: repeat the single slice N times
    gt_batch = gt_slice.unsqueeze(0).repeat(num_slices_batch, 1, 1)  # (B, H, W)
    la_batch = la_slice.unsqueeze(0).repeat(num_slices_batch, 1, 1)  # (B, H, W)
    la_batch = apply_circle_mask(la_batch)
    gt_batch = apply_circle_mask(gt_batch)

    # Metric reference values (use full gt_fbp for percentile range)
    gt_np = gt_fbp.numpy()
    metric_min = float(np.percentile(gt_np, 1))
    metric_max = float(np.percentile(gt_np, 99))
    dr = metric_max - metric_min

    gt_slice_np = gt_slice.numpy()

    def _guidance(vol_init: torch.Tensor, t: int) -> torch.Tensor:
        x = inpaint_guidance(
            source_volume=gt_batch.to(device),
            vol_init=vol_init.to(device),
            angular_range_deg=angular_range_deg,
            start_angle=start_angle,
            tilt_axis=tilt_axis,
            device=device,
            timestep=int(t),
            inpaint_steps=1,
        )
        if average_gradient_steps:
            x = torch.mean(x, dim=0).repeat(x.shape[0], 1, 1)  # average over slices reduce noise
        return x

    iterative_recon = la_batch.clone().to(device)
    metrics_history = []

    best_ssim = -1.0
    best_recon_np = la_slice.numpy().copy()

    for outer_iter in range(num_outer_iters):
        torch.cuda.empty_cache()

        sched = GuidedDDIMScheduler(
            num_train_timesteps=1000,
            guidance_function=_guidance,
        )
        pipe = DDIMPipeline(
            unet=model,
            scheduler=sched,
            fdk_prior=la_batch.clone().to(device),
            normalize_fn=normalize_fn,
            denormalize_fn=denormalize_fn,
            slice_batch_size=num_slices_batch,
        )

        result = pipe.truncated_pipeline(
            initial_guess=iterative_recon.to(device),
            start_step=int(start_step_frac * num_inference_steps),
            num_inference_steps=num_inference_steps,
            use_clipped_model_output=True,
        )
        iterative_recon = result.images  # (B, H, W)

        # Mean over the batch → collapse stochastic variance
        mean_recon = iterative_recon.mean(dim=0).repeat(num_slices_batch, 1, 1)
        
        mean_recon = apply_circle_mask(inpaint_guidance(
            source_volume=gt_batch.to(device),
            vol_init=mean_recon.to(device),
            angular_range_deg=angular_range_deg,
            start_angle=start_angle,
            tilt_axis=tilt_axis,
            device=device,
        ))

        mid = num_slices_batch // 2
        full_mid_np = iterative_recon[mid].cpu().numpy()
        mean_mid_np = mean_recon[mid].cpu().numpy()

        psnr_full = compute_psnr(gt_slice_np, full_mid_np, data_range=dr)
        ssim_full = compute_ssim(gt_slice_np, full_mid_np, data_range=dr)
        psnr_mean = compute_psnr(gt_slice_np, mean_mid_np, data_range=dr)
        ssim_mean = compute_ssim(gt_slice_np, mean_mid_np, data_range=dr)

        metrics_history.append({
            "outer_iter": outer_iter,
            "psnr_full": psnr_full,
            "ssim_full": ssim_full,
            "psnr_mean": psnr_mean,
            "ssim_mean": ssim_mean,
        })

        if ssim_mean > best_ssim:
            best_ssim = ssim_mean
            best_recon_np = mean_mid_np.copy()

        # Next iteration starts from the mean reconstruction
        iterative_recon = mean_recon.to(device)

    return best_recon_np, metrics_history
