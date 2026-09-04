"""Load a trained denoiser + its normalization from a training checkpoint.

``train.py`` writes ``<ckpt>.pt`` (state dict under ``model_state_dict``, saved by
``pytorch_base.PyTorchExperiment``) and a sibling ``<ckpt>_config.json`` holding
the architecture (mode / k / crop / include_mirror) and the ``(norm_min,
norm_max)`` used for training.  This module rebuilds the exact model and the
normalize / denormalize functions from those files.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch

from .geometry import ANGLE_TIME_COND_CHANNELS
from .model import create_baseline_unet, create_diffusion_unet


def config_path(checkpoint_path) -> Path:
    return Path(str(checkpoint_path).replace(".pt", "_config.json"))


def load_config(checkpoint_path) -> Dict:
    p = config_path(checkpoint_path)
    if not p.exists():
        raise FileNotFoundError(f"no config sidecar at {p}")
    with open(p) as f:
        return json.load(f)


def build_model(config: Dict):
    """Build the (untrained) UNet described by a training ``config`` dict."""
    mode = config.get("mode", "diffusion")
    k = int(config.get("k", 1))
    crop = tuple(config.get("crop", (128, 512)))
    include_mirror = bool(config.get("include_mirror", False))
    neighborhoods = config.get("neighborhoods", "both")
    extra_cond_channels = ANGLE_TIME_COND_CHANNELS if config.get("cond_angle_time", False) else 0
    temporal_raw_pairs = bool(config.get("temporal_raw_pairs", False))
    fn = create_diffusion_unet if mode == "diffusion" else create_baseline_unet
    return fn(k=k, sample_size=crop, include_mirror=include_mirror, neighborhoods=neighborhoods,
             extra_cond_channels=extra_cond_channels, temporal_raw_pairs=temporal_raw_pairs)


def _load_checkpoint_with_retry(checkpoint_path, retries: int = 3, delay: float = 2.0):
    """``torch.load`` with retries for a checkpoint that may be mid-write.

    ``PyTorchExperiment`` (pytorch_base) saves via a direct, non-atomic
    ``torch.save(checkpoint, self.checkpoint_path)`` every epoch when
    ``save_always=True``. Reading exactly during that write (or after a save was
    interrupted, e.g. by contention on a shared NFS mount) raises a zip/stream
    error; a short retry covers the in-progress-write case. A checkpoint that is
    genuinely corrupted (interrupted past save) will still fail after retries —
    that requires the next successful epoch save to overwrite it.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            return torch.load(checkpoint_path, map_location="cpu")
        except RuntimeError as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(delay)
    raise RuntimeError(
        f"failed to load checkpoint {checkpoint_path} after {retries} attempts "
        f"(possibly mid-write or corrupted from an interrupted save): {last_exc}"
    ) from last_exc


def load_denoiser(
    checkpoint_path, device: Optional[torch.device] = None, strict: bool = True
) -> Tuple[torch.nn.Module, Dict]:
    """Return ``(model.eval() on device, config)`` for a trained checkpoint."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(checkpoint_path)
    model = build_model(config)
    ckpt = _load_checkpoint_with_retry(checkpoint_path)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=strict)
    return model.to(device).eval(), config


def make_norm_fns(config: Dict) -> Tuple[Callable, Callable]:
    """Return ``(normalize, denormalize)`` matching the training config's range."""
    lo, hi = float(config["norm_min"]), float(config["norm_max"])
    span = hi - lo
    return (lambda c: 2.0 * (c - lo) / span - 1.0, lambda x: (x + 1.0) * 0.5 * span + lo)
