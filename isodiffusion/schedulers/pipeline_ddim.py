# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
from diffusers.models import UNet3DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.import_utils import is_torch_xla_available
from diffusers.utils.torch_utils import randn_tensor

from isodiffusion.schedulers.scheduling_ddim import GuidedDDIMScheduler


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


def _as_3tuple(value: Union[int, Sequence[int]], name: str) -> Tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError(f"{name} must be an int or a length-3 sequence, got {value}")
    return tuple(int(v) for v in value)


def _patch_starts(size: int, patch_size: int, stride: int) -> List[int]:
    """Sliding-window start positions; the last patch always snaps to the edge."""
    if patch_size > size:
        raise ValueError(f"patch_size={patch_size} is larger than axis size {size}")
    if stride <= 0:
        raise ValueError("stride must be positive; reduce overlap or patch size")

    starts = list(range(0, max(size - patch_size + 1, 1), stride))
    last = size - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


class DDIMPipeline(DiffusionPipeline):
    r"""
    DDIM pipeline for batched 3D iso-diffusion reconstruction.

    The pipeline keeps the scheduler state as full normalized volumes of shape
    ``(B, D, H, W)``. At each timestep only the UNet call is partitioned into
    overlapping 3D subvolumes, then all subvolumes are packed as
    ``(B * num_subvolumes, C, patch_d, patch_h, patch_w)`` before inference.
    Predicted noise patches are averaged back into ``(B, D, H, W)`` before the
    scheduler step, so ``GuidedDDIMScheduler.guidance_function`` can operate on
    full raw volumes.
    """

    model_cpu_offload_seq = "unet"

    def __init__(
        self,
        unet: UNet3DConditionModel,
        scheduler: GuidedDDIMScheduler,
        conditioning: Optional[torch.Tensor] = None,
        normalize_fn: Optional[callable] = lambda x: x,
        denormalize_fn: Optional[callable] = lambda x: x,
        subvolume_batch_size: int = 1,
        overlap: int = 0,
        slice_batch_size: Optional[int] = None,
    ):
        super().__init__()

        guidance_function = getattr(scheduler, "guidance_function", None)
        scheduler = GuidedDDIMScheduler.from_config(scheduler.config)
        scheduler.guidance_function = guidance_function

        if conditioning is not None:
            if conditioning.dim() not in (3, 4):
                raise ValueError("conditioning must have shape (D, H, W) or (B, D, H, W)")
            conditioning = normalize_fn(conditioning)

        if slice_batch_size is not None:
            subvolume_batch_size = slice_batch_size

        if subvolume_batch_size < 1:
            raise ValueError("subvolume_batch_size must be >= 1")

        self.conditioning = conditioning
        self.normalize_fn = normalize_fn
        self.denormalize_fn = denormalize_fn
        self.subvolume_batch_size = int(subvolume_batch_size)
        self.slice_batch_size = self.subvolume_batch_size
        self.overlap = int(overlap)

        self.register_modules(unet=unet, scheduler=scheduler)

    @property
    def _device(self) -> torch.device:
        try:
            return self._execution_device
        except AttributeError:
            return next(self.unet.parameters()).device

    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        if self.conditioning is not None:
            if self.conditioning.dim() == 3:
                volume_shape = tuple(self.conditioning.shape)
            else:
                conditioning_batch_size = int(self.conditioning.shape[0])
                if batch_size not in (1, conditioning_batch_size):
                    raise ValueError(
                        "batch_size must be 1 or match conditioning.shape[0] when conditioning is batched"
                    )
                batch_size = conditioning_batch_size
                volume_shape = tuple(self.conditioning.shape[-3:])
        else:
            volume_shape = _as_3tuple(self.unet.config.sample_size, "unet.config.sample_size")

        initial_guess = torch.zeros((batch_size, *volume_shape), device=self._device, dtype=self.unet.dtype)
        return self.truncated_pipeline(
            initial_guess=initial_guess,
            start_step=0,
            generator=generator,
            eta=eta,
            num_inference_steps=num_inference_steps,
            use_clipped_model_output=use_clipped_model_output,
            output_type=output_type,
            return_dict=return_dict,
        )

    @staticmethod
    def _as_volume_batch(volume: torch.Tensor, name: str) -> Tuple[torch.Tensor, bool]:
        if volume.dim() == 3:
            return volume.unsqueeze(0), True
        if volume.dim() == 4:
            return volume, False
        raise ValueError(f"{name} must have shape (D, H, W) or (B, D, H, W), got {tuple(volume.shape)}")

    @staticmethod
    def _prepare_conditioning(
        conditioning: Optional[torch.Tensor],
        batch_size: int,
        volume_shape: Tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if conditioning is None:
            return None

        if conditioning.dim() == 3:
            conditioning = conditioning.unsqueeze(0)
        elif conditioning.dim() != 4:
            raise ValueError(
                "conditioning must have shape (D, H, W) or (B, D, H, W), "
                f"got {tuple(conditioning.shape)}"
            )

        if tuple(conditioning.shape[-3:]) != volume_shape:
            raise ValueError(
                f"conditioning volume shape {tuple(conditioning.shape[-3:])} "
                f"must match initial_guess volume shape {volume_shape}"
            )

        if conditioning.shape[0] == 1 and batch_size != 1:
            conditioning = conditioning.expand(batch_size, -1, -1, -1)
        elif conditioning.shape[0] != batch_size:
            raise ValueError(
                f"conditioning batch size {conditioning.shape[0]} must be 1 or match "
                f"initial_guess batch size {batch_size}"
            )

        return conditioning.to(device=device, dtype=dtype)

    def _extract_model_input_patches(
        self,
        model_input: torch.Tensor,
        patch_shape: Tuple[int, int, int],
        starts: Tuple[List[int], List[int], List[int]],
    ) -> torch.Tensor:
        d_starts, h_starts, w_starts = starts
        patch_d, patch_h, patch_w = patch_shape
        patches = [
            model_input[
                batch_idx,
                :,
                d0 : d0 + patch_d,
                h0 : h0 + patch_h,
                w0 : w0 + patch_w,
            ]
            for batch_idx in range(model_input.shape[0])
            for d0 in d_starts
            for h0 in h_starts
            for w0 in w_starts
        ]
        return torch.stack(patches, dim=0)

    def _aggregate_noise_patches(
        self,
        noise_pred_patches: torch.Tensor,
        batch_size: int,
        volume_shape: Tuple[int, int, int],
        patch_shape: Tuple[int, int, int],
        starts: Tuple[List[int], List[int], List[int]],
    ) -> torch.Tensor:
        d_starts, h_starts, w_starts = starts
        patch_d, patch_h, patch_w = patch_shape
        d, h, w = volume_shape

        accum = torch.zeros(batch_size, d, h, w, device=noise_pred_patches.device, dtype=noise_pred_patches.dtype)
        count = torch.zeros_like(accum)

        patch_idx = 0
        for batch_idx in range(batch_size):
            for d0 in d_starts:
                for h0 in h_starts:
                    for w0 in w_starts:
                        patch = noise_pred_patches[patch_idx, 0]
                        accum[
                            batch_idx,
                            d0 : d0 + patch_d,
                            h0 : h0 + patch_h,
                            w0 : w0 + patch_w,
                        ] += patch
                        count[
                            batch_idx,
                            d0 : d0 + patch_d,
                            h0 : h0 + patch_h,
                            w0 : w0 + patch_w,
                        ] += 1.0
                        patch_idx += 1

        return accum / count.clamp_min(1.0)

    def truncated_pipeline(
        self,
        initial_guess: torch.Tensor,
        start_step: int = 10,
        p_use_conditioning: float = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Run truncated guided DDIM from raw full-volume initial guesses.

        Args:
            initial_guess: Raw ``(D, H, W)`` volume or batched ``(B, D, H, W)`` volumes.
            start_step: Index into the DDIM timestep schedule where denoising starts.
            p_use_conditioning: Probability of keeping the conditioning channel for
                each model subvolume. ``1.0`` keeps conditioning for all subvolumes.
        """
        if not 0.0 <= p_use_conditioning <= 1.0:
            raise ValueError("p_use_conditioning must be between 0.0 and 1.0")

        device = self._device
        dtype = self.unet.dtype
        initial_guess, squeeze_output = self._as_volume_batch(initial_guess, "initial_guess")
        image = self.normalize_fn(initial_guess.to(device=device, dtype=dtype))
        batch_size, d, h, w = image.shape

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        if start_step < 0 or start_step >= len(timesteps):
            raise ValueError(
                f"start_step={start_step} is out of range for a schedule with "
                f"{len(timesteps)} steps (valid: 0 to {len(timesteps) - 1})."
            )

        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        start_timestep = timesteps[start_step]
        start_t = torch.full((batch_size,), int(start_timestep.item()), device=device, dtype=torch.long)
        image = self.scheduler.add_noise(image, noise, start_t)

        conditioning = self._prepare_conditioning(
            self.conditioning,
            batch_size=batch_size,
            volume_shape=(d, h, w),
            device=device,
            dtype=dtype,
        )

        patch_shape = _as_3tuple(self.unet.config.sample_size, "unet.config.sample_size")
        patch_d, patch_h, patch_w = patch_shape
        if self.overlap < 0 or self.overlap >= min(patch_shape):
            raise ValueError("overlap must satisfy 0 <= overlap < min(patch_shape)")

        stride = tuple(axis - self.overlap for axis in patch_shape)
        starts = (
            _patch_starts(d, patch_d, stride[0]),
            _patch_starts(h, patch_h, stride[1]),
            _patch_starts(w, patch_w, stride[2]),
        )
        n_patches_per_volume = len(starts[0]) * len(starts[1]) * len(starts[2])
        n_patches = batch_size * n_patches_per_volume

        encoder_hidden_states = torch.zeros(
            self.subvolume_batch_size,
            1,
            int(self.unet.config.cross_attention_dim),
            device=device,
            dtype=dtype,
        )

        for t in self.progress_bar(timesteps[start_step:]):
            if conditioning is not None:
                model_input = torch.stack([image, conditioning], dim=1)
            else:
                model_input = image.unsqueeze(1)

            model_input_patches = self._extract_model_input_patches(model_input, patch_shape, starts)

            if p_use_conditioning < 1.0 and model_input_patches.shape[1] > 1:
                use_conditioning = torch.rand(
                    model_input_patches.shape[0],
                    device=device,
                    generator=generator if isinstance(generator, torch.Generator) else None,
                ) < p_use_conditioning
                model_input_patches[:, 1] *= use_conditioning[:, None, None, None]

            noise_pred_patches = torch.empty(
                n_patches,
                1,
                patch_d,
                patch_h,
                patch_w,
                device=device,
                dtype=model_input_patches.dtype,
            )

            with torch.no_grad():
                for batch_start in range(0, n_patches, self.subvolume_batch_size):
                    batch_end = min(batch_start + self.subvolume_batch_size, n_patches)
                    chunk_input = model_input_patches[batch_start:batch_end]
                    chunk_encoder_states = encoder_hidden_states[: chunk_input.shape[0]]
                    pred_chunk = self.unet(
                        chunk_input,
                        t,
                        encoder_hidden_states=chunk_encoder_states,
                        return_dict=False,
                    )[0]
                    noise_pred_patches[batch_start:batch_end] = pred_chunk
            
            model_output = self._aggregate_noise_patches(
                noise_pred_patches=noise_pred_patches,
                batch_size=batch_size,
                volume_shape=(d, h, w),
                patch_shape=patch_shape,
                starts=starts,
            )

            step_kwargs = {}
            if use_clipped_model_output is not None:
                step_kwargs["use_clipped_model_output"] = use_clipped_model_output

            image = self.scheduler.step(
                model_output,
                int(t.item()),
                image,
                eta=eta,
                generator=generator,
                normalize_fn=self.normalize_fn,
                denormalize_fn=self.denormalize_fn,
                **step_kwargs,
            ).prev_sample

            if XLA_AVAILABLE:
                xm.mark_step()

        image = self.denormalize_fn(image).float().cpu()
        if squeeze_output:
            image = image[0]
        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
