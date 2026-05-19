"""Patch-based 3D iso-diffusion reconstruction helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from diffusers.models import UNet2DModel, UNet3DConditionModel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from isodiffusion.fourier_wedge import enforce_known_fourier, load_norm_json, make_norm_fns
from isodiffusion.schedulers.scheduling_ddim import GuidedDDIMScheduler


_TRAIN_CONDITIONAL_3D_DEFAULT_CHANNELS = (64, 128, 256)
_TRAIN_CONDITIONAL_3D_DOWN_BLOCK_TYPES = (
    "DownBlock3D",
    "DownBlock3D",
    "CrossAttnDownBlock3D",
)
_TRAIN_CONDITIONAL_3D_UP_BLOCK_TYPES = (
    "CrossAttnUpBlock3D",
    "UpBlock3D",
    "UpBlock3D",
)
_TRAIN_CONDITIONAL_2D_DEFAULT_CHANNELS = (64, 64, 128, 128, 256, 256)
_TRAIN_CONDITIONAL_2D_DOWN_BLOCK_TYPES = (
    "DownBlock2D",
    "DownBlock2D",
    "DownBlock2D",
    "DownBlock2D",
    "AttnDownBlock2D",
    "DownBlock2D",
)
_TRAIN_CONDITIONAL_2D_UP_BLOCK_TYPES = (
    "UpBlock2D",
    "AttnUpBlock2D",
    "UpBlock2D",
    "UpBlock2D",
    "UpBlock2D",
    "UpBlock2D",
)


def _parse_channels(value) -> Tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(v.strip()) for v in value.split(",") if v.strip())
    return tuple(int(v) for v in value)


def model_config_from_norm_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def create_unet3d_from_config(config: Dict) -> UNet3DConditionModel:
    """Create the same 3D conditional UNet used by train_conditional_3d."""
    channels = _parse_channels(config.get("channels", _TRAIN_CONDITIONAL_3D_DEFAULT_CHANNELS))
    expected_blocks = len(_TRAIN_CONDITIONAL_3D_DOWN_BLOCK_TYPES)
    if len(channels) != expected_blocks:
        raise ValueError(
            "train_conditional_3d uses exactly "
            f"{expected_blocks} block_out_channels, got {len(channels)}: {channels}"
        )

    return UNet3DConditionModel(
        sample_size=int(config.get("volume_size", 64)),
        in_channels=2,
        out_channels=1,
        down_block_types=_TRAIN_CONDITIONAL_3D_DOWN_BLOCK_TYPES,
        up_block_types=_TRAIN_CONDITIONAL_3D_UP_BLOCK_TYPES,
        block_out_channels=channels,
        layers_per_block=int(config.get("layers_per_block", 2)),
        cross_attention_dim=int(config.get("cross_attention_dim", 128)),
        attention_head_dim=int(config.get("attention_head_dim", 4)),
        norm_num_groups=int(config.get("norm_num_groups", 16)),
    )


def create_unet2d_from_config(config: Dict) -> UNet2DModel:
    """Create the same 2D conditional UNet used by train_conditional_2d."""
    channels = _parse_channels(config.get("channels", _TRAIN_CONDITIONAL_2D_DEFAULT_CHANNELS))
    expected_blocks = len(_TRAIN_CONDITIONAL_2D_DOWN_BLOCK_TYPES)
    if len(channels) != expected_blocks:
        raise ValueError(
            "train_conditional_2d uses exactly "
            f"{expected_blocks} block_out_channels, got {len(channels)}: {channels}"
        )

    return UNet2DModel(
        sample_size=int(config.get("volume_size", 64)),
        in_channels=2,
        out_channels=1,
        layers_per_block=int(config.get("layers_per_block", 2)),
        block_out_channels=channels,
        down_block_types=_TRAIN_CONDITIONAL_2D_DOWN_BLOCK_TYPES,
        up_block_types=_TRAIN_CONDITIONAL_2D_UP_BLOCK_TYPES,
        class_embed_type="timestep",
    )


def load_unet3d(
    checkpoint_path: Path,
    device: torch.device,
    config_path: Optional[Path] = None,
) -> Tuple[UNet3DConditionModel, Dict]:
    """Load a 3D conditional UNet and its sidecar configuration."""
    checkpoint_path = Path(checkpoint_path)
    if config_path is None:
        config_path = checkpoint_path.with_name(checkpoint_path.stem + "_norm.json")
    config = model_config_from_norm_json(Path(config_path))
    model = create_unet3d_from_config(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


def load_unet2d(
    checkpoint_path: Path,
    device: torch.device,
    config_path: Optional[Path] = None,
) -> Tuple[UNet2DModel, Dict]:
    """Load a 2D conditional UNet and its sidecar configuration."""
    checkpoint_path = Path(checkpoint_path)
    if config_path is None:
        config_path = checkpoint_path.with_name(checkpoint_path.stem + "_norm.json")
    config = model_config_from_norm_json(Path(config_path))
    model = create_unet2d_from_config(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


def patch_starts(length: int, patch_size: int, overlap: int) -> List[int]:
    """Return starts that cover an axis, aligning the last patch to the boundary."""
    if patch_size > length:
        raise ValueError(f"patch_size={patch_size} is larger than axis length {length}")
    if overlap < 0 or overlap >= patch_size:
        raise ValueError("overlap must satisfy 0 <= overlap < patch_size")

    stride = patch_size - overlap
    starts = list(range(0, max(length - patch_size + 1, 1), stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def iter_patch_slices(
    shape: Sequence[int],
    patch_size: int,
    overlap: int,
) -> Iterable[Tuple[slice, slice, slice]]:
    z_starts = patch_starts(int(shape[0]), patch_size, overlap)
    y_starts = patch_starts(int(shape[1]), patch_size, overlap)
    x_starts = patch_starts(int(shape[2]), patch_size, overlap)
    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                yield (
                    slice(z0, z0 + patch_size),
                    slice(y0, y0 + patch_size),
                    slice(x0, x0 + patch_size),
                )


def _default_encoder_hidden_states(
    batch_size: int,
    cross_attention_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.zeros(batch_size, 1, cross_attention_dim, device=device, dtype=dtype)


def _apply_known_fourier_batch(
    estimate_batch: torch.Tensor,
    measured_batch: torch.Tensor,
    angular_range_deg: float,
    start_angle_deg: float,
    tilt_axis: int,
) -> torch.Tensor:
    outputs = []
    for estimate, measured in zip(estimate_batch, measured_batch):
        outputs.append(
            enforce_known_fourier(
                estimate_volume=estimate.squeeze(0),
                measured_volume=measured.squeeze(0),
                angular_range_deg=angular_range_deg,
                start_angle_deg=start_angle_deg,
                tilt_axis=tilt_axis,
            ).unsqueeze(0)
        )
    return torch.stack(outputs, dim=0)


@torch.no_grad()
def denoise_patch_batch(
    initial_patches: torch.Tensor,
    condition_patches: torch.Tensor,
    ground_truth_patches: torch.Tensor,
    model: UNet3DConditionModel,
    normalize_fn: Callable[[torch.Tensor], torch.Tensor],
    denormalize_fn: Callable[[torch.Tensor], torch.Tensor],
    angular_range_deg: float,
    start_angle_deg: float,
    tilt_axis: int,
    device: torch.device,
    num_inference_steps: int = 40,
    start_step_frac: float = 0.8,
    eta: float = 0.0,
    use_clipped_model_output: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Denoise a batch of raw-valued ``(B, D, H, W)`` patches."""
    if initial_patches.shape != condition_patches.shape:
        raise ValueError("initial_patches and condition_patches must have the same shape")

    model_dtype = next(model.parameters()).dtype
    initial = initial_patches.to(device=device, dtype=model_dtype)
    condition_raw = condition_patches.to(device=device, dtype=model_dtype).unsqueeze(1)
    ground_truth_raw = ground_truth_patches.to(device=device, dtype=model_dtype).unsqueeze(1)
    condition = normalize_fn(condition_raw).to(dtype=model_dtype)
    image = normalize_fn(initial).unsqueeze(1).to(dtype=model_dtype)

    scheduler = GuidedDDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    start_step = int(start_step_frac * num_inference_steps)
    start_step = min(max(start_step, 0), len(timesteps) - 1)

    noise = torch.randn(image.shape, device=device, dtype=image.dtype, generator=generator)
    start_timestep = timesteps[start_step]
    start_t = torch.full((image.shape[0],), int(start_timestep.item()), device=device, dtype=torch.long)
    image = scheduler.add_noise(image, noise, start_t)

    def guidance_fn(pred_original_raw: torch.Tensor, timestep: int) -> torch.Tensor:
        return _apply_known_fourier_batch(
            estimate_batch=pred_original_raw,
            measured_batch=ground_truth_raw,
            angular_range_deg=angular_range_deg,
            start_angle_deg=start_angle_deg,
            tilt_axis=tilt_axis,
        )

    scheduler.guidance_function = guidance_fn
    encoder_hidden_states = _default_encoder_hidden_states(
        image.shape[0],
        int(model.config.cross_attention_dim),
        device=device,
        dtype=image.dtype,
    )

    for timestep in timesteps[start_step:]:
        model_input = torch.cat([image, condition], dim=1)
        noise_pred = model(
            model_input,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]
        image = scheduler.step(
            noise_pred,
            timestep,
            image,
            eta=eta,
            use_clipped_model_output=use_clipped_model_output,
            generator=generator,
            normalize_fn=normalize_fn,
            denormalize_fn=denormalize_fn,
        ).prev_sample

    return denormalize_fn(image).squeeze(1).float().cpu()


@torch.no_grad()
def reconstruct_volume_patches(
    initial_volume: torch.Tensor,
    condition_volume: torch.Tensor,
    ground_truth: torch.Tensor,
    model: UNet3DConditionModel,
    normalize_fn: Callable[[torch.Tensor], torch.Tensor],
    denormalize_fn: Callable[[torch.Tensor], torch.Tensor],
    angular_range_deg: float,
    start_angle_deg: float,
    tilt_axis: int,
    patch_size: int = 64,
    overlap: int = 5,
    batch_size: int = 1,
    num_inference_steps: int = 40,
    start_step_frac: float = 0.8,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Run 3D diffusion over overlapping patches and average overlaps uniformly."""
    if initial_volume.shape != condition_volume.shape:
        raise ValueError("initial_volume and condition_volume must have the same shape")
    if initial_volume.ndim != 3:
        raise ValueError(f"Expected 3D volumes, got shape {tuple(initial_volume.shape)}")

    if device is None:
        device = next(model.parameters()).device

    initial_volume = initial_volume.float().cpu()
    condition_volume = condition_volume.float().cpu()
    output = torch.zeros_like(initial_volume)
    weights = torch.zeros_like(initial_volume)

    patch_slices = list(iter_patch_slices(initial_volume.shape, patch_size, overlap))
    for start in range(0, len(patch_slices), batch_size):
        chunk_slices = patch_slices[start : start + batch_size]
        initial_batch = torch.stack([initial_volume[s] for s in chunk_slices], dim=0)
        condition_batch = torch.stack([condition_volume[s] for s in chunk_slices], dim=0)
        ground_truth_batch = torch.stack([ground_truth[s] for s in chunk_slices], dim=0)
        pred_batch = denoise_patch_batch(
            initial_patches=initial_batch,
            condition_patches=condition_batch,
            ground_truth_patches=ground_truth_batch,
            model=model,
            normalize_fn=normalize_fn,
            denormalize_fn=denormalize_fn,
            angular_range_deg=angular_range_deg,
            start_angle_deg=start_angle_deg,
            tilt_axis=tilt_axis,
            device=device,
            num_inference_steps=num_inference_steps,
            start_step_frac=start_step_frac,
        )
        for slc, pred in zip(chunk_slices, pred_batch):
            output[slc] += pred
            weights[slc] += 1.0

    return output / weights.clamp_min(1.0)


def load_norm_fns_from_checkpoint_sidecar(checkpoint_path: Path) -> Tuple[Callable, Callable, Dict]:
    config_path = Path(checkpoint_path).with_name(Path(checkpoint_path).stem + "_norm.json")
    norm_min, norm_max = load_norm_json(config_path)
    normalize_fn, denormalize_fn = make_norm_fns(norm_min, norm_max)
    return normalize_fn, denormalize_fn, model_config_from_norm_json(config_path)


def load_npy_volume(path: Path) -> torch.Tensor:
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D npy volume at {path}, got shape {arr.shape}")
    return torch.from_numpy(np.ascontiguousarray(arr))
