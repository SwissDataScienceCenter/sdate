"""Dataset adapter: swaps the official ambient-tweedie SDXL-LoRA training script's
HF-`datasets`/imagefolder loading (built for captioned photos, see
`train_lora_sdxl.py`'s original `preprocess_train`/`compute_vae_encodings`) for this
project's real time-resolved CT projection data, while still producing the exact same
downstream artifact their (otherwise UNCHANGED) training loop expects: a dataset of
dicts with `model_input` (a VAE-encoded latent), `input_ids_one`/`input_ids_two`
(tokenized captions), `original_sizes`/`crop_top_lefts` (SDXL's size-conditioning
inputs).

Two differences from their `compute_vae_encodings`, both load-bearing -- see
`scripts/tr_diffusion_ambient_tweedie_sdxl_calibrate.py`'s docstring for the full
reasoning:

  1. No synthetic noise injection. Their pipeline VAE-encodes a CLEAN image and
     injects a synthetic Gaussian draw at `timestep_nature` to manufacture an ambient
     example. We already HAVE a real noisy measurement (`y` = poisson_head's `mu_hat`,
     Anscombe-normalized to [-1,1], replicated to 3 channels to satisfy their
     `.convert("RGB")` assumption) -- `model_input = vae.encode(y).sample() *
     scaling_factor` directly, no `add_noise` call. `timestep_nature` is instead
     calibrated empirically (the calibration script) to match how far this real
     `model_input` actually sits from the true clean latent.
  2. Every example is already exactly `crop`-shaped (no resize/random-crop
     augmentation) -- `original_sizes`/`crop_top_lefts` are the SAME constant tuple
     for the whole dataset, not read per-image.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sdate.tr_diffusion.data import TimeResolvedFrameDataset


class SDXLAmbientRealDataset(Dataset):
    """Yields the RAW (not-yet-VAE-encoded) real measurement, replicated to 3 channels.
    VAE encoding happens once, up front, via :func:`precompute_model_inputs` --
    mirroring their own one-time `dataset.map(compute_vae_encodings, ...)` step."""

    def __init__(self, mov_path: str, memmap_path: str, frame_start: int, frame_end: int,
                crop, extra_noise_dose: float, noise_seed: Optional[int] = None,
                max_samples: Optional[int] = None):
        self.crop = (int(crop[0]), int(crop[1]))
        self.ds = TimeResolvedFrameDataset(
            mov_path, memmap_path=memmap_path, k=0, frame_start=frame_start, frame_end=frame_end,
            crop=self.crop, norm_range=None, extra_noise_dose=extra_noise_dose,
            noise_seed=noise_seed, anscombe=True, anscombe_norm_sample_frames=8,
            max_samples=max_samples,
        )

    @property
    def sigma_tn_norm(self) -> float:
        return float(self.ds.anscombe_sigma_tn_norm)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        y = self.ds[idx]["clean_target"]  # (1, H, W), already Anscombe-normalized to [-1, 1]
        pixel_values = y.clamp(-1, 1).expand(3, *self.crop).contiguous()
        return {"pixel_values": pixel_values}


@torch.no_grad()
def precompute_model_inputs(dataset: SDXLAmbientRealDataset, vae, device, batch_size: int = 16,
                            num_workers: int = 4, cache_path: Optional[str] = None,
                            log_every: int = 500) -> torch.Tensor:
    """Mirrors `compute_vae_encodings` in `train_lora_sdxl.py` minus the synthetic
    `add_noise` call (see module docstring). Returns a single CPU fp16 tensor of shape
    (N, 4, h, w) in dataset order. If `cache_path` is given and already exists (with a
    matching `.meta.npz` recording N/shape), loads it directly instead of recomputing --
    this precompute is a real, non-trivial GPU cost (one VAE forward pass per training
    example), worth persisting across resumed training runs the same way every other
    cached artifact in this project is (see e.g. `reconstruct.denoise_sequence`'s
    `first_index`/`num_frames`/`crop` sidecar convention).
    """
    n = len(dataset)
    if cache_path is not None:
        meta_path = Path(str(cache_path) + ".meta.npz")
        if Path(cache_path).exists() and meta_path.exists():
            meta = np.load(meta_path)
            if int(meta["n"]) == n:
                shape = (n, int(meta["c"]), int(meta["h"]), int(meta["w"]))
                arr = np.memmap(cache_path, dtype=np.float16, mode="r", shape=shape)
                print(f"[precompute_model_inputs] loaded cached latents from {cache_path} (shape={shape})")
                return torch.from_numpy(np.ascontiguousarray(arr)).float()
            print(f"[precompute_model_inputs] cache at {cache_path} has n={int(meta['n'])} != {n}, recomputing")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    vae.eval()
    chunks = []
    seen = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, dtype=vae.dtype)
        latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
        chunks.append(latents.to(torch.float16).cpu())
        seen += pixel_values.shape[0]
        if seen % log_every < batch_size:
            print(f"[precompute_model_inputs] encoded {seen}/{n}")
    model_inputs = torch.cat(chunks, dim=0)

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        arr = np.memmap(cache_path, dtype=np.float16, mode="w+", shape=tuple(model_inputs.shape))
        arr[:] = model_inputs.numpy()
        arr.flush()
        np.savez(str(cache_path) + ".meta.npz", n=model_inputs.shape[0], c=model_inputs.shape[1],
                h=model_inputs.shape[2], w=model_inputs.shape[3])
        print(f"[precompute_model_inputs] wrote cache to {cache_path} (shape={tuple(model_inputs.shape)})")

    return model_inputs.float()


class SDXLAmbientNativeCleanDataset(Dataset):
    """Literal reproduction of the upstream recipe (see their shipped
    `configs/train_low_level_laion10k.yaml`: real LAION photos, `timestep_nature: 100`,
    no dataset-level noise injection of any kind). Yields the NATIVE (full-dose,
    un-thinned, no Anscombe transform) reference frame, treated as genuinely clean --
    exactly like their real photos -- replicated to 3 channels. Their own training
    loop's `timestep_nature`-based synthetic corruption is the ONLY noise the model
    ever sees; we inject nothing ourselves. This deliberately does NOT match the real
    problem we care about (denoising the actual noisy measurement) -- it isolates
    whether their pipeline can recover a clean image below an assumed noise floor AT
    ALL on our grayscale content, before attempting to bridge back to the real-noisy-
    measurement setting."""

    def __init__(self, mov_path: str, memmap_path: str, frame_start: int, frame_end: int,
                crop, max_samples: Optional[int] = None):
        self.crop = (int(crop[0]), int(crop[1]))
        self.ds = TimeResolvedFrameDataset(
            mov_path, memmap_path=memmap_path, k=0, frame_start=frame_start, frame_end=frame_end,
            crop=self.crop, norm_range=None, max_samples=max_samples,
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        y = self.ds[idx]["central"]  # (1, H, W), already normalized to [-1, 1], native full-dose
        pixel_values = y.clamp(-1, 1).expand(3, *self.crop).contiguous()
        return {"pixel_values": pixel_values}


class PrecomputedSDXLAmbientDataset(Dataset):
    """Final dataset fed to the (otherwise unchanged) training loop's DataLoader:
    precomputed `model_input` latents + a FIXED placeholder caption's tokenized ids
    (we have no real captions -- SDXL's UNet still needs text-conditioning inputs, so
    every example uses the same placeholder, tokenized once) + the constant
    `original_sizes`/`crop_top_lefts` every example shares."""

    def __init__(self, model_inputs: torch.Tensor, original_size, crop_top_left,
                input_ids_one: torch.Tensor, input_ids_two: torch.Tensor):
        self.model_inputs = model_inputs
        self.original_size = tuple(original_size)
        self.crop_top_left = tuple(crop_top_left)
        self.input_ids_one = input_ids_one[0]  # (L,) -- same ids reused for every example
        self.input_ids_two = input_ids_two[0]

    def __len__(self):
        return self.model_inputs.shape[0]

    def __getitem__(self, idx):
        return {
            "model_input": self.model_inputs[idx],
            "original_sizes": self.original_size,
            "crop_top_lefts": self.crop_top_left,
            "input_ids_one": self.input_ids_one,
            "input_ids_two": self.input_ids_two,
        }
