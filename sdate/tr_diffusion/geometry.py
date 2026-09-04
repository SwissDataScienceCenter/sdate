"""Rotation geometry of the ``212_Wunderkerze2`` time-resolved projection stream.

Every video frame is one CT projection of a sample rotating continuously about a
vertical axis.  The rotation *rate* was calibrated in
``notebooks/wunderkerze_rotation_calibration.ipynb`` (see project memory
``project-wunderkerze2-rotation``):

* ``DEG_PER_FRAME`` ≈ 1.801402 °/frame, constant to < 0.05 % over frames 400k-600k.
* One full turn (360°, *genuinely identical* viewing geometry) = ``PERIOD_360``
  ≈ 199.844 frames.
* A half turn (180°) = ``PERIOD_180`` ≈ 99.922 frames; a θ+180° view is the
  left-right mirror of θ about the rotation-axis detector column ``ROT_AXIS_COL``
  ≈ 269.85 (of 528) — *not* the image centre.

Only the *rate* is calibrated, not the absolute angle of frame 0, so all angles
here are relative.  The period is deliberately kept as a float: it is **not** an
integer number of frames, so "same angle N turns away" lands between two integer
frames and must be linearly interpolated (see :mod:`sdate.tr_diffusion.frames`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# --- Calibrated constants (frames 400000-600000, 212_Wunderkerze2) -----------
DEG_PER_FRAME: float = 1.801402
PERIOD_360: float = 360.0 / DEG_PER_FRAME          # ≈ 199.844 frames / turn
PERIOD_180: float = 180.0 / DEG_PER_FRAME          # ≈ 99.922 frames / half turn
ROT_AXIS_COL: float = 269.85                        # detector column of the axis
FRAME_H: int = 128
FRAME_W: int = 528

ANGLE_TIME_COND_CHANNELS: int = 3  # sin(angle), cos(angle), normalised time


def angle_time_cond_array(frame_indices: np.ndarray, deg_per_frame: float,
                          frame_start: int, frame_end: int) -> np.ndarray:
    """Per-frame ``(sin(angle), cos(angle), normalised_time)`` conditioning, shape ``(N, 3)``.

    ``angle`` is the frame's rotation angle mod 360 (sin/cos avoids the 0/360
    wrap discontinuity a raw linear angle channel would have). ``normalised_time``
    is the frame index linearly mapped to ``[-1, 1]`` over ``[frame_start, frame_end)``
    -- the SAME range a checkpoint's config persists, so training and inference
    apply an identical mapping.
    """
    fi = np.asarray(frame_indices, dtype=np.float64)
    angle_rad = np.deg2rad((fi * float(deg_per_frame)) % 360.0)
    span = max(frame_end - frame_start, 1)
    t_norm = 2.0 * (fi - frame_start) / span - 1.0
    return np.stack([np.sin(angle_rad), np.cos(angle_rad), t_norm], axis=-1).astype(np.float32)


@dataclass(frozen=True)
class ContextTap:
    """A single conditioning frame relative to a central integer frame index.

    Attributes
    ----------
    kind:
        ``"rotation"`` — an adjacent frame (small angular offset, highly
        correlated view at a nearby *time*); ``"temporal"`` — the *same viewing
        angle* an integer number of turns away (different time, identical
        geometry).
    step:
        Signed offset.  For ``rotation`` it is an integer frame offset; for
        ``temporal`` it is a signed number of full turns.
    frame_offset:
        Signed offset in (possibly fractional) frames from the central index.
    mirror:
        Whether the tap must be left-right flipped about :data:`ROT_AXIS_COL`
        (used only for 180° half-turn taps; unused by the default layout).
    interp:
        Whether this tap must be read via linear interpolation between the two
        bracketing integer frames (``frame_offset`` is fractional) rather than a
        single integer frame access. True for the default temporal taps (see
        ``temporal_raw_pairs=False`` below); always False for rotation taps and
        for the raw-pair temporal taps (``temporal_raw_pairs=True``), which read
        the two bracketing frames directly instead of blending them.
    pair:
        For raw-pair temporal taps only: ``"lo"``/``"hi"`` marking which of the
        two bracketing integer frames this tap is (``None`` otherwise).
    """

    kind: str
    step: int
    frame_offset: float
    mirror: bool = False
    interp: bool = False
    pair: Optional[str] = None

    @property
    def name(self) -> str:
        tag = {"rotation": "rot", "temporal": "tmp"}[self.kind]
        suffix = ("m" if self.mirror else "") + (f"_{self.pair}" if self.pair else "")
        return f"{tag}{self.step:+d}{suffix}"


NEIGHBORHOODS = ("both", "rotation", "temporal")


def build_context_layout(k: int, include_mirror: bool = False,
                         period_360: float = PERIOD_360,
                         period_180: Optional[float] = None,
                         neighborhoods: str = "both",
                         temporal_raw_pairs: bool = False) -> List[ContextTap]:
    """Return the ordered conditioning taps for context radius ``k``.

    Order (fixed contract shared by the dataset and the model input assembly)::

        rotation  i-k … i-1, i+1 … i+k          (2k taps)
        temporal  i-k·P … i-1·P, i+1·P … i+k·P   (2k taps, or 4k -- see below)

    giving ``4k`` context channels by default. ``in_channels`` for the diffusion
    model is ``2 + 4k`` (x_t + corrupted-central + context); for the baseline it
    is ``1 + 4k`` (no x_t). With ``include_mirror`` two extra half-turn mirror
    taps (``i ± P/2``, flipped) are appended — off by default.

    ``temporal_raw_pairs`` (opt-in, default False -- existing checkpoints rely
    on the default layout and would silently misalign if it changed): since
    ``period_360`` is never an integer number of frames, each default temporal
    tap linearly interpolates between the two bracketing integer frames (a
    FIXED blend, e.g. ~0.844/0.156 for wunderkerze2) -- confirmed to introduce
    visible ghosting at moving edges (see project memory / temporal_interp_blur
    diagnostic). ``temporal_raw_pairs=True`` instead gives the model BOTH
    bracketing frames as separate, unblurred channels and lets it learn to
    combine them itself: each of the ``2k`` temporal taps becomes 2 raw taps
    (``pair="lo"``/``"hi"``), so the temporal channel count doubles ``2k -> 4k``
    and the total context doubles ``4k -> 6k`` (``k`` rotation-side taps stay
    the same; the *temporal* side goes from "k before, k after" to "2k before,
    2k after" -- twice as many raw temporal taps per side, one bracketing pair
    each instead of one blended tap each).

    ``neighborhoods`` selects which tap kind(s) to keep — ``"both"`` (default,
    2k rotation + 2k temporal), ``"rotation"`` (angular neighbours only, 2k
    taps, no same-angle-across-turns taps), or ``"temporal"`` (same-angle
    across turns only, 2k taps, no angular neighbours). Used to ablate the two
    conditioning neighbourhoods independently. ``include_mirror`` taps are
    themselves ``"temporal"``-kind, so they are only appended when
    ``neighborhoods`` is ``"both"`` or ``"temporal"``; they are unaffected by
    ``temporal_raw_pairs`` (still interpolated at the half-turn offset).

    ``period_360`` is the rotation period in frames (from the dataset profile;
    defaults to the wunderkerze value). ``period_180`` defaults to half of it.
    """
    if k < 0:
        raise ValueError(f"context radius k must be >= 0, got {k}")
    if neighborhoods not in NEIGHBORHOODS:
        raise ValueError(f"neighborhoods must be one of {NEIGHBORHOODS}, got {neighborhoods!r}")
    if period_180 is None:
        period_180 = period_360 / 2.0

    # k=0 -> no context taps at all (pure single-frame N2V ablation; both loops
    # below are naturally empty since range(-0,0) and range(1,1) are both empty).
    taps: List[ContextTap] = []
    if neighborhoods in ("both", "rotation"):
        for step in list(range(-k, 0)) + list(range(1, k + 1)):
            taps.append(ContextTap("rotation", step, float(step)))
    if neighborhoods in ("both", "temporal"):
        for step in list(range(-k, 0)) + list(range(1, k + 1)):
            offset = step * period_360
            if temporal_raw_pairs:
                lo = math.floor(offset)
                taps.append(ContextTap("temporal", step, float(lo), pair="lo"))
                taps.append(ContextTap("temporal", step, float(lo + 1), pair="hi"))
            else:
                taps.append(ContextTap("temporal", step, offset, interp=True))
        if include_mirror:
            taps.append(ContextTap("temporal", -1, -period_180, mirror=True, interp=True))
            taps.append(ContextTap("temporal", 1, period_180, mirror=True, interp=True))
    return taps


def context_channels(k: int, include_mirror: bool = False, neighborhoods: str = "both",
                     temporal_raw_pairs: bool = False) -> int:
    """Number of context channels for radius ``k``."""
    return len(build_context_layout(k, include_mirror, neighborhoods=neighborhoods,
                                    temporal_raw_pairs=temporal_raw_pairs))


def usable_frame_range(
    start: int, end: int, k: int, include_mirror: bool = False,
    period_360: float = PERIOD_360, neighborhoods: str = "both",
    temporal_raw_pairs: bool = False,
) -> Tuple[int, int]:
    """Sub-range of ``[start, end)`` whose taps all stay inside ``[start, end)``.

    The interpolated temporal taps reach the furthest, so the margin is the
    largest absolute (fractional) frame offset rounded up plus one guard frame.
    """
    taps = build_context_layout(k, include_mirror, period_360=period_360, neighborhoods=neighborhoods,
                                temporal_raw_pairs=temporal_raw_pairs)
    # k=0 -> no taps -> nominal 1-frame margin (no context reach to bound).
    margin = int(max((abs(t.frame_offset) for t in taps), default=0.0)) + 1
    lo = start + margin
    hi = end - margin
    if hi <= lo:
        raise ValueError(
            f"frame range [{start}, {end}) too small for k={k} "
            f"(needs > {2 * margin} frames)"
        )
    return lo, hi
