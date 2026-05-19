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

from typing import List, Optional, Tuple, Union

import torch 

from diffusers import UNet2DModel
from ladiff.schedulers.scheduling_ddim import GuidedDDIMScheduler
from diffusers.utils.import_utils import is_torch_xla_available
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class DDIMPipeline(DiffusionPipeline):
    r""" 
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`GuidedDDIMScheduler`].
        fdk_prior (`torch.Tensor`, *optional*):
            An optional FDK prior of shape (D,H,W) to be concatenated as an additional channel to the model input.
        normalize_fn (`callable`, *optional*):
            An optional function to normalize the input volume and FDK prior.
        denormalize_fn (`callable`, *optional*):
            An optional function to denormalize the output volume.
    """

    model_cpu_offload_seq = "unet"

    def __init__(
            self, 
            unet: UNet2DModel, 
            scheduler: GuidedDDIMScheduler,
            fdk_prior: Optional[torch.Tensor] = None,
            normalize_fn: Optional[callable] = lambda x:x,
            denormalize_fn: Optional[callable] = lambda x:x,
            slice_batch_size: int = 2,
        ):
        super().__init__()

        # make sure scheduler can always be converted to DDIM
        scheduler = GuidedDDIMScheduler.from_config(scheduler.config)

        if fdk_prior is not None:
            if fdk_prior.dim() != 3:
                raise ValueError("fdk_prior must be (D,H,W) for single-volume mode")
            # normalize fdk prior
            fdk_prior = normalize_fn(fdk_prior)
        self.fdk_prior = fdk_prior

        self.normalize_fn = normalize_fn
        self.denormalize_fn = denormalize_fn
        self.slice_batch_size = slice_batch_size

        self.register_modules(unet=unet, scheduler=scheduler)

    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://huggingface.co/papers/2010.02502) paper. Only
                applies to the [`~schedulers.GuidedDDIMScheduler`], and is ignored in other schedulers. A value of `0`
                corresponds to DDIM and `1` corresponds to DDPM.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            use_clipped_model_output (`bool`, *optional*, defaults to `None`):
                If `True` or `False`, see documentation for [`GuidedDDIMScheduler.step`]. If `None`, nothing is passed
                downstream to the scheduler (use `None` for schedulers which don't support this argument).
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDIMPipeline
        >>> import PIL.Image
        >>> import numpy as np

        >>> # load model and scheduler
        >>> pipe = DDIMPipeline.from_pretrained("fusing/ddim-lsun-bedroom")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe(eta=0.0, num_inference_steps=50)

        >>> # process image to PIL
        >>> image_processed = image.cpu().permute(0, 2, 3, 1)
        >>> image_processed = (image_processed + 1.0) * 127.5
        >>> image_processed = image_processed.numpy().astype(np.uint8)
        >>> image_pil = PIL.Image.fromarray(image_processed[0])

        >>> # save image
        >>> image_pil.save("test.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # Determine image shape
        if self.fdk_prior is not None:
            image_shape = self.fdk_prior.shape  # (D,H,W)
        elif isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, *self.unet.config.sample_size)

        # Start from zeros — at start_step=0 the noise schedule fully noises
        # the input (alpha_cumprod ≈ 0), so this is equivalent to starting
        # from pure random noise.
        initial_guess = torch.zeros(image_shape, device=self._execution_device, dtype=self.unet.dtype)

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

    def truncated_pipeline(
        self,
        initial_guess: torch.Tensor,
        start_step: int = 10,
        p_use_conditioning: float = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Truncated diffusion pipeline that starts from a noised version of an initial guess
        rather than pure random noise, preserving features of the input.

        Args:
            initial_guess (`torch.Tensor`):
                A noiseless initial reconstruction of shape (D, H, W). Will be normalized,
                noised to the starting timestep, and then denoised from there.
            start_step (`int`, *optional*, defaults to 10):
                The diffusion step index (into the timestep schedule) to start denoising from.
                E.g. if `num_inference_steps=50` and `start_step=10`, denoising runs from
                timestep index 10 to 49 (the last 40 steps). Lower values add more noise and
                run more steps; higher values preserve more of the initial guess.
            p_use_conditioning (`float`, *optional*, defaults to 0.5):
                Probability of using conditioning on each slice during denoising.
            generator (`torch.Generator`, *optional*):
                A generator for deterministic noise sampling.
            eta (`float`, *optional*, defaults to 0.0):
                DDIM eta parameter. 0 = deterministic DDIM, 1 = DDPM.
            num_inference_steps (`int`, *optional*, defaults to 50):
                Total number of steps in the full schedule (used to build the timestep grid).
            use_clipped_model_output (`bool`, *optional*, defaults to `None`):
                Passed to the scheduler step.
            output_type (`str`, *optional*, defaults to `"pil"`):
                Output format.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return an `ImagePipelineOutput`.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`
        """
        device = self._execution_device

        # Normalize the initial guess
        image = self.normalize_fn(initial_guess.to(device, dtype=self.unet.dtype))
        D, H, W = image.shape[0], image.shape[1], image.shape[2]

        # Build the full timestep schedule
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps  # descending, e.g. [999, 979, ...]

        if start_step < 0 or start_step >= len(timesteps):
            raise ValueError(
                f"start_step={start_step} is out of range for a schedule with "
                f"{len(timesteps)} steps (valid: 0 to {len(timesteps) - 1})."
            )

        if p_use_conditioning < 0.0 or p_use_conditioning > 1.0:
            raise ValueError("p_use_conditioning must be between 0.0 and 1.0.")

        # Add noise at the starting timestep
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        start_timestep = timesteps[start_step]
        image = self.scheduler.add_noise(image, noise, start_timestep)

        # Denoise from start_step onward
        fdk_prior = self.fdk_prior # already normalized in __init__
        truncated_timesteps = timesteps[start_step:]

        for t in self.progress_bar(truncated_timesteps):
            noisy_slices = image  # (D,H,W)

            if fdk_prior is not None:
                if fdk_prior.dim() != 3:
                    raise ValueError("fdk_prior must be (D,H,W) for single-volume mode")
                model_input = torch.stack([noisy_slices, fdk_prior.to(device)], dim=1)  # (D,2,H,W)
            else:
                model_input = noisy_slices.unsqueeze(1)  # (D,1,H,W)

            if hasattr(self.unet.config, 'class_embed_type') and self.unet.config.class_embed_type is not None:
                use_conditioning = torch.rand((D,), device=device) < p_use_conditioning  # randomly decide if we do conditioning or not on this slice
                model_input[:, 1] *= use_conditioning[:, None, None]

            slice_idx = torch.arange(D, device=device)
            with torch.no_grad():
                noise_pred_slices = torch.empty((D, 1, H, W), device=device, dtype=model_input.dtype)
                for start in range(0, D, self.slice_batch_size):
                    end = min(start + self.slice_batch_size, D)
                    chunk_input = model_input[start:end]
                    chunk_slice_idx = slice_idx[start:end]
                

                    if hasattr(self.unet.config, 'class_embed_type') and self.unet.config.class_embed_type is not None:
                        conditional_labels = use_conditioning[start:end]
                        pred_chunk = self.unet(
                            chunk_input, t, class_labels=conditional_labels, return_dict=False
                        )[0]
                    else:
                        pred_chunk = self.unet(
                            chunk_input, t, return_dict=False
                        )[0]

                    noise_pred_slices[start:end] = pred_chunk

            model_output = noise_pred_slices.permute(1, 0, 2, 3)[0].contiguous()  # (D,H,W)

            image = self.scheduler.step(
                model_output, t, image, eta=eta,
                use_clipped_model_output=use_clipped_model_output,
                generator=generator,
                normalize_fn=self.normalize_fn,
                denormalize_fn=self.denormalize_fn
            ).prev_sample

            if XLA_AVAILABLE:
                xm.mark_step()

        image = self.denormalize_fn(image)
        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
