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

**Standard defaults (see README "Key experimental findings" + project memory
project-tr-diffusion):** ``k=3`` (monotonic, ~free gain over k=1/k=2 -- the only
cost is data-loading I/O, no capacity/inference-cost penalty) and, for the
baseline, ``poisson_head=True`` (the two-head Gamma-Poisson NB-NLL head -- see
:mod:`sdate.tr_diffusion.nb_head` -- supersedes both the plain Huber/MAE/MSE
point estimate and the single-channel ``loss_type="poisson"`` ablation, since
its own ``mu`` output is empirically the same context-conditional point
estimate (measured pixelwise correlation 0.999) -- the two-head model is a
strict superset: read off ``mu`` alone for the old single-head behaviour, or
the full posterior-mean combination for the sharper, observation-aware one;
see ``poisson_posterior`` in :func:`sdate.tr_diffusion.pipeline.denoise_frames_baseline`).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
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
    out_channels: int = 1,
) -> UNet2DModel:
    return UNet2DModel(
        sample_size=sample_size,
        in_channels=int(in_channels),
        out_channels=int(out_channels),
        layers_per_block=int(layers_per_block),
        block_out_channels=tuple(block_out_channels),
        down_block_types=_DOWN,
        up_block_types=_UP,
        class_embed_type="timestep",
    )


def create_diffusion_unet(
    k: int = 3,
    sample_size: Union[int, Tuple[int, int]] = (128, 512),
    block_out_channels: Sequence[int] = _CHANNELS,
    layers_per_block: int = 2,
    include_mirror: bool = False,
    neighborhoods: str = "both",
    extra_cond_channels: int = 0,
    temporal_raw_pairs: bool = False,
    condition_on_measurement: bool = True,
) -> UNet2DModel:
    """ε-prediction UNet; ``in_channels = 2 + <context channels> + extra_cond_channels``
    (x_t + corrupted + context [+ angle/time conditioning planes]).

    ``condition_on_measurement=False`` (Ambient-Tweedie only, see
    :mod:`sdate.tr_diffusion.ambient_tweedie`) drops the second base channel
    (``in_channels = 1 + ...``, ``x_t`` alone): that "corrupted"/measurement
    channel is the noisy label ``y`` the network is trained to predict, so
    feeding it in directly as an input gives the network a trivial shortcut
    that ignores ``x_t``/the noise level entirely -- confirmed by an ablation
    where two very different reverse-sampling trajectories produced
    pixel-identical output. ``y`` stays available for the LOSS (and for
    DPS-style measurement guidance at inference) -- only the network's own
    input tensor excludes it."""
    base = 2 if condition_on_measurement else 1
    return _build(base + context_channels(k, include_mirror, neighborhoods, temporal_raw_pairs) + int(extra_cond_channels),
                  sample_size, block_out_channels, layers_per_block)


def create_baseline_unet(
    k: int = 3,
    sample_size: Union[int, Tuple[int, int]] = (128, 512),
    block_out_channels: Sequence[int] = _CHANNELS,
    layers_per_block: int = 2,
    include_mirror: bool = False,
    neighborhoods: str = "both",
    extra_cond_channels: int = 0,
    temporal_raw_pairs: bool = False,
    poisson_head: bool = True,
    poisson_init_mean: Optional[float] = None,
    poisson_init_var: Optional[float] = None,
) -> UNet2DModel:
    """Single-pass x_0-regression UNet; ``in_channels = 1 + <context channels> + extra_cond_channels``
    (no x_t channel; ``extra_cond_channels`` e.g. the angle/time conditioning planes -- see
    :func:`sdate.tr_diffusion.geometry.angle_time_cond_array`; ``temporal_raw_pairs`` doubles the
    temporal context taps -- see :func:`sdate.tr_diffusion.geometry.build_context_layout`).
    ``poisson_head=True`` gives the head 2 output channels (a Gamma-Poisson ``(mu, var)``
    belief trained with the exact Negative-Binomial NLL and combined with the real
    observation at inference -- see :mod:`sdate.tr_diffusion.nb_head`) instead of the
    default single-channel point estimate.

    ``poisson_init_mean``/``poisson_init_var`` (meaningful whenever channel 0 is a
    softplus-constrained Poisson mean -- ``poisson_head=True`` OR the single-channel
    ``loss_type="poisson"`` ablation in :class:`sdate.tr_diffusion.losses.BaselineN2VLoss`;
    ``poisson_init_var`` only applies to channel 1, so it's a no-op unless
    ``poisson_head=True``): initialise the head's output bias so training starts
    with ``mu``/``var`` already near the REAL raw-count magnitude (e.g. the
    dataset's ``(norm_min+norm_max)/2``), instead of near the default random-init
    value (``softplus(~0) ~ 0.7``). Input features stay ``[-1, 1]``-normalised
    throughout, so this bridges a large, otherwise near-unlearnable scale gap:
    Adam's step size is roughly bounded by the learning rate regardless of
    gradient magnitude, so an untrained bias initialised near 0 needs on the
    order of ``(true_count / lr)`` steps to drift to a true count magnitude of a
    few hundred -- far more than a short training run provides, which silently
    strands ``mu``/``var`` at ~O(1) and produces a denoised value that is a
    fixed, scale-wrong blend of the observation rather than a real denoising.
    Since ``softplus(x) ~= x`` for ``x`` this large, the bias is set directly to
    the target value (no explicit inverse-softplus needed)."""
    model = _build(1 + context_channels(k, include_mirror, neighborhoods, temporal_raw_pairs) + int(extra_cond_channels),
                   sample_size, block_out_channels, layers_per_block, out_channels=2 if poisson_head else 1)
    with torch.no_grad():
        if poisson_init_mean is not None:
            model.conv_out.bias[0] = float(poisson_init_mean)
        if poisson_head and poisson_init_var is not None:
            model.conv_out.bias[1] = float(poisson_init_var)
    return model
