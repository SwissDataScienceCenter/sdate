"""Experiment configuration for tr_fbp_codec.

Kept deliberately small: this package is a single prototype experiment
(Wunderkerze2, native full-dose data), not a general-purpose framework, so
config only exposes what's actually varied during prototyping (context
length, mapping mode, frame range) rather than every architectural knob.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

QuantMode = Literal["truncate", "rescale"]


@dataclass
class QuantConfig:
    """Int16 (or float counts) -> 12-bit (4096-level) mapping.

    ``truncate``: clip(round(x), 0, 4095) -- valid only when the data's real
    dynamic range already fits under 4096 (checked per-dataset, see
    quantization.py:evaluate_quantization_loss; true for Wunderkerze2 and
    Sewellia at current dose levels, NOT a safe universal assumption).

    ``rescale``: linear rescale of [data_min, data_max] -> [0, 4095], for
    datasets whose native range exceeds 12 bits.
    """

    mode: QuantMode = "truncate"
    data_min: Optional[float] = None  # required for "rescale"
    data_max: Optional[float] = None  # required for "rescale"
    n_levels: int = 4096


@dataclass
class DataConfig:
    """Wunderkerze2 native-data slice used for prototyping.

    ``frame_start``/``frame_end`` are ABSOLUTE frame indices (same
    numbering as ``profiles.REGISTRY["wunderkerze2"]`` and
    ``MemmapFrameSource.get(idx)`` -- note the registry dict is called
    REGISTRY, not PROFILES).

    Defaults deliberately fall inside the existing, already-built native
    joint-FBP-context tap cache (see DEPENDENCIES.md ``FBP_TAP_CACHE_*``) so
    prototyping needs zero new GPU/ASTRA reconstruction compute -- picking a
    range outside [412800, 467200) means building a fresh cache first (a
    real RunAI/GPU job, not a config change).
    """

    profile_name: str = "wunderkerze2"
    frame_start: int = 413_000
    frame_end: int = 415_000
    holdout_frame_start: Optional[int] = 414_800  # last 200 frames held out
    holdout_frame_end: Optional[int] = 415_000


@dataclass
class CodecConfig:
    """Context length / architecture-facing knobs.

    ``k``: number of preceding raw context frames f[n..n+k-1] used to
    predict f[n+k]. The paper's own choice (k=3) is a rule of thumb, not a
    constraint -- see README.md.

    ``use_fbp_prior``: toggles the FBP-reprojected tap on/off, so the same
    training/eval code path can produce both the "paper reproduction"
    baseline (False) and "ours" (True) without duplicating logic.

    ``block_size``: spatial crop size for training samples -- 64 (settled
    2026-09-07), bigger than the paper's 32x32 since the classification
    head's memory cost is trivial even at 64x64 (~134MB/sample for the
    logits + CE backward, in fp32 -- see README.md); chosen for a smaller
    block-edge bpp tax, not memory pressure. ``None`` means full-frame
    prediction instead (also affordable, ~17-35 samples/GPU at 40-80GB --
    left available as an option, not the default).

    ``block_margin``: extra context pixels cropped around the supervised
    ``block_size`` region (loss is only computed on the interior), so the
    3D convs never see a fake zero-padded boundary where real neighbouring
    pixels actually exist in the full frame. Set to cover the model's own
    spatial receptive field: with ``ModelConfig.n_blocks=6`` (2 conv3d
    layers/block, kernel_size=3 each) plus the stem conv, radius is
    ``(2*6+1)*1 = 13`` pixels -- 16 leaves a small safety margin. Revisit
    if ``n_blocks`` changes.
    """

    k: int = 5
    use_fbp_prior: bool = True
    block_size: Optional[int] = 64
    block_margin: int = 16
    n_classes: int = 4096
