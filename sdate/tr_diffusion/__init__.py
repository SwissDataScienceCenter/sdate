"""Time-resolved conditional diffusion / baseline frame denoiser.

Self-supervised denoising of individual time-resolved CT projection frames from
the ``212_Wunderkerze2`` continuous-rotation acquisition.  A central frame is
denoised from conditioning context — rotation-adjacent frames and same-viewing-
angle frames one or more turns away (via the calibrated rotation period) — while
the central frame itself is only ever seen through a Noise2Void blind-spot
corruption (masked loss), so denoising is genuinely learned rather than copied.

Two architecture-matched models are trained and compared: a conditional DDIM
diffusion denoiser and a single-pass regression baseline.

See ``README.md``.  Independent of ``isodiffusion`` except for the shared
``pytorch_base`` training harness and the diffusers UNet2DModel.
"""

from __future__ import annotations

from .data import TimeResolvedFrameDataset
from .geometry import (
    DEG_PER_FRAME,
    PERIOD_180,
    PERIOD_360,
    ROT_AXIS_COL,
    build_context_layout,
    context_channels,
    usable_frame_range,
)
from .load import load_denoiser, make_norm_fns
from .model import create_baseline_unet, create_diffusion_unet
from .n2v import blind_spot_corrupt
from .noise import add_poisson_noise, binomial_complementary_split, binomial_split, binomial_thin
from .pipeline import (
    denoise_frames, denoise_frames_baseline, denoise_frames_ensemble,
    denoise_frames_n2n_baseline, partial_diffusion, partial_diffusion_n2n,
    pred_x0_ensemble, pred_x0_n2n_ensemble, pred_x0_n2n_swap_ensemble,
)

__all__ = [
    "TimeResolvedFrameDataset",
    "build_context_layout",
    "context_channels",
    "usable_frame_range",
    "create_diffusion_unet",
    "create_baseline_unet",
    "load_denoiser",
    "make_norm_fns",
    "denoise_frames",
    "denoise_frames_baseline",
    "denoise_frames_ensemble",
    "denoise_frames_n2n_baseline",
    "pred_x0_ensemble",
    "pred_x0_n2n_ensemble",
    "pred_x0_n2n_swap_ensemble",
    "partial_diffusion",
    "partial_diffusion_n2n",
    "blind_spot_corrupt",
    "add_poisson_noise",
    "binomial_split",
    "binomial_thin",
    "binomial_complementary_split",
    "DEG_PER_FRAME",
    "PERIOD_360",
    "PERIOD_180",
    "ROT_AXIS_COL",
]
