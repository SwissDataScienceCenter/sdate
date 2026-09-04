"""UNet builders for the conditional diffusion denoiser and its baseline.

Both use the diffusers :class:`UNet2DModel` (as in
``isodiffusion/train_conditional_2d.py``).  ``class_embed_type="timestep"`` lets
us pass a 0/1 ``class_labels`` flag marking whether the corrupted central frame
is present — this is the "with / without central" conditioning-dropout signal,
embedded and added to the time embedding.

Channel contract (see :func:`sdate.tr_diffusion.geometry.build_context_layout`):

* **Diffusion** — ``in_channels = 2 + 4k``:
  ``[x_t, corrupted_central, <4k context>]``; predicts the ε (noise) map.
* **Baseline** — ``in_channels = 1 + 4k``:
  ``[corrupted_central, <4k context>]``; a single-pass regressor that predicts
  the denoised central ``x_0`` directly (timestep is fixed to 0).

Same block layout / capacity for both, so the diffusion-vs-regression comparison
is architecture-controlled — "the math is the same, only the training differs".
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

from diffusers.models import UNet2DModel

from .geometry import context_channels

# Six stages, one attention block near the bottleneck (mirrors train_conditional_2d).
_DOWN = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
_UP = ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
_CHANNELS = (64, 64, 128, 128, 256, 256)


def _build(
    in_channels: int,
    sample_size: Union[int, Tuple[int, int]],
    block_out_channels: Sequence[int],
    layers_per_block: int,
) -> UNet2DModel:
    return UNet2DModel(
        sample_size=sample_size,
        in_channels=int(in_channels),
        out_channels=1,
        layers_per_block=int(layers_per_block),
        block_out_channels=tuple(block_out_channels),
        down_block_types=_DOWN,
        up_block_types=_UP,
        class_embed_type="timestep",
    )


def create_diffusion_unet(
    k: int = 1,
    sample_size: Union[int, Tuple[int, int]] = (128, 512),
    block_out_channels: Sequence[int] = _CHANNELS,
    layers_per_block: int = 2,
    include_mirror: bool = False,
    neighborhoods: str = "both",
    extra_cond_channels: int = 0,
    temporal_raw_pairs: bool = False,
) -> UNet2DModel:
    """ε-prediction UNet; ``in_channels = 2 + <context channels> + extra_cond_channels``
    (x_t + corrupted + context [+ angle/time conditioning planes])."""
    return _build(2 + context_channels(k, include_mirror, neighborhoods, temporal_raw_pairs) + int(extra_cond_channels),
                  sample_size, block_out_channels, layers_per_block)


def create_baseline_unet(
    k: int = 1,
    sample_size: Union[int, Tuple[int, int]] = (128, 512),
    block_out_channels: Sequence[int] = _CHANNELS,
    layers_per_block: int = 2,
    include_mirror: bool = False,
    neighborhoods: str = "both",
    extra_cond_channels: int = 0,
    temporal_raw_pairs: bool = False,
) -> UNet2DModel:
    """Single-pass x_0-regression UNet; ``in_channels = 1 + <context channels> + extra_cond_channels``
    (no x_t channel; ``extra_cond_channels`` e.g. the angle/time conditioning planes -- see
    :func:`sdate.tr_diffusion.geometry.angle_time_cond_array`; ``temporal_raw_pairs`` doubles the
    temporal context taps -- see :func:`sdate.tr_diffusion.geometry.build_context_layout`)."""
    return _build(1 + context_channels(k, include_mirror, neighborhoods, temporal_raw_pairs) + int(extra_cond_channels),
                  sample_size, block_out_channels, layers_per_block)
