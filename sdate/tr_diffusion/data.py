"""Dataset of central-frame + conditioning-context samples from the ``.mov``.

Each item is one central integer frame ``i`` together with its ordered context
(see :func:`sdate.tr_diffusion.geometry.build_context_layout`):

* ``central``   ``(1, H, W)``   — the diffusion target ``x_0`` (the measured,
  noisy central frame, optionally with extra Poisson noise added).
* ``context``   ``(4k, H, W)`` (or ``6k`` with ``temporal_raw_pairs=True``) —
  rotation-adjacent + same-angle temporal frames in the fixed channel order;
  the model's conditioning. The temporal taps land at a non-integer frame
  offset (``PERIOD_360`` is never an integer number of frames); by default
  these are linearly interpolated between the two bracketing frames, which
  introduces a small but consistent ghosting artifact at moving edges (the
  blend ratio is fixed, e.g. ~0.844/0.156 for wunderkerze2 -- confirmed
  visually, see the ``temporal_interp_blur`` diagnostic). ``temporal_raw_pairs=
  True`` gives the model both bracketing frames as separate, un-blurred
  channels instead and lets it learn to combine them itself.
* ``reference`` ``(1, H, W)``   — only in the extra-noise regime: the original
  measured central frame (strictly less noisy than ``central``), the pseudo-GT
  for evaluation.
* ``cond_channels`` ``(3, H, W)`` — only if ``cond_angle_time=True``: the central
  frame's rotation angle (as sin/cos, avoiding the 0/360 wrap) and its normalised
  position in ``[frame_start, frame_end)``, each broadcast to a constant-valued
  plane and given to the model as extra input channels (see
  :func:`sdate.tr_diffusion.geometry.angle_time_cond_array`) -- lets the model
  learn angle/time-dependent structure directly instead of only inferring it
  from the neighbour frames' pixel content.

All frames are denormalised to a common count space, cropped to ``crop`` around
the rotation axis, then affinely normalised to ``[-1, 1]`` with a single
``(norm_min, norm_max)`` fit over a sample of frames (saved to the checkpoint
sidecar so inference can invert it).  The N2V blind-spot corruption and the
diffusion noising are applied later, inside the loss, so a fresh mask/noise is
drawn every step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .frames import FrameSource, open_frame_source
from .geometry import (
    ANGLE_TIME_COND_CHANNELS, DEG_PER_FRAME, ROT_AXIS_COL, angle_time_cond_array,
    build_context_layout, usable_frame_range,
)
from .noise import add_poisson_noise, binomial_split


def _center_crop(frame: np.ndarray, out_h: int, out_w: int, axis_col: float) -> np.ndarray:
    """Crop ``(H, W)`` to ``(out_h, out_w)``: height-centred, width around ``axis_col``."""
    h, w = frame.shape[-2:]
    if out_h > h or out_w > w:
        raise ValueError(f"crop {(out_h, out_w)} exceeds frame {(h, w)}")
    top = (h - out_h) // 2
    left = int(round(axis_col - out_w / 2.0))
    left = max(0, min(left, w - out_w))
    return frame[..., top : top + out_h, left : left + out_w]


def _mirror_cols(frame: np.ndarray, axis_col: float) -> np.ndarray:
    """Flip columns left-right about ``axis_col`` (a θ+180° mirror view)."""
    w = frame.shape[-1]
    src = np.clip(np.round(2.0 * axis_col - np.arange(w)).astype(np.int64), 0, w - 1)
    return frame[..., src]


class TimeResolvedFrameDataset(Dataset):
    def __init__(
        self,
        mov_path: Union[str, Path],
        k: int = 1,
        frame_start: int = 400_000,
        frame_end: int = 600_000,
        crop: Tuple[int, int] = (128, 512),
        memmap_path: Optional[Union[str, Path]] = None,
        include_mirror: bool = False,
        neighborhoods: str = "both",
        norm_range: Optional[Tuple[float, float]] = None,
        norm_percentiles: Tuple[float, float] = (0.5, 99.5),
        norm_sample_frames: int = 64,
        extra_noise_dose: Optional[float] = None,
        noise_seed: Optional[int] = 0,
        max_samples: Optional[int] = None,
        seed: int = 0,
        axis_col: float = ROT_AXIS_COL,
        deg_per_frame: float = DEG_PER_FRAME,
        n2n: bool = False,
        p_range: Tuple[float, float] = (0.1, 0.9),
        p_bins: int = 100,
        cond_angle_time: bool = False,
        cond_frame_start: Optional[int] = None,
        cond_frame_end: Optional[int] = None,
        temporal_raw_pairs: bool = False,
    ):
        self.mov_path = str(mov_path)
        # Range the normalised-time conditioning maps to [-1, 1] over. Defaults to
        # this call's own frame_start/frame_end (matches training, where "the
        # dataset's range" and "the model's calibrated time range" are the same
        # thing) but MUST be overridden at inference to the checkpoint's own
        # SAVED training frame_start/frame_end when evaluating over a different
        # range -- otherwise the model sees a differently-scaled time value than
        # it was calibrated on.
        self._cond_frame_start = int(cond_frame_start) if cond_frame_start is not None else int(frame_start)
        self._cond_frame_end = int(cond_frame_end) if cond_frame_end is not None else int(frame_end)
        self.cond_angle_time = bool(cond_angle_time)
        self.memmap_path = str(memmap_path) if memmap_path is not None else None
        self.k = int(k)
        self.crop = (int(crop[0]), int(crop[1]))
        self.include_mirror = bool(include_mirror)
        self.neighborhoods = str(neighborhoods)
        self.extra_noise_dose = extra_noise_dose
        # int -> reproducible per-frame noise (deterministic eval); None -> fresh
        # noise on every access (stochastic; sample from the dose distribution when training).
        self.noise_seed = None if noise_seed is None else int(noise_seed)
        self.n2n = bool(n2n)
        self.p_range = (float(p_range[0]), float(p_range[1]))
        self.p_bins = int(p_bins)
        if self.n2n and self.extra_noise_dose is None:
            raise ValueError("n2n mode requires extra_noise_dose (the fixed measurement dose to split)")
        self.axis_col = float(axis_col)
        self.deg_per_frame = float(deg_per_frame)
        self.temporal_raw_pairs = bool(temporal_raw_pairs)
        self._period_360 = 360.0 / self.deg_per_frame
        self.layout = build_context_layout(self.k, self.include_mirror, period_360=self._period_360,
                                           neighborhoods=self.neighborhoods,
                                           temporal_raw_pairs=self.temporal_raw_pairs)

        self._source: Optional[FrameSource] = None  # built lazily (per worker)
        src = self._get_source()
        # Clamp the requested [start, end) to the source's valid global index range
        # (a memmap slice may cover only part of the stream).
        frame_start = max(int(frame_start), src.first_index)
        frame_end = min(int(frame_end), src.last_index)
        lo, hi = usable_frame_range(frame_start, frame_end, self.k, self.include_mirror,
                                    period_360=self._period_360, neighborhoods=self.neighborhoods,
                                    temporal_raw_pairs=self.temporal_raw_pairs)
        indices = np.arange(lo, hi, dtype=np.int64)
        if max_samples is not None and max_samples < len(indices):
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(indices, size=int(max_samples), replace=False))
        self.indices = indices

        if norm_range is None:
            self.norm_min, self.norm_max = self._fit_norm(src, norm_percentiles, norm_sample_frames, seed)
        else:
            self.norm_min, self.norm_max = float(norm_range[0]), float(norm_range[1])

    # --- source / normalisation -------------------------------------------
    def _get_source(self) -> FrameSource:
        if self._source is None:
            self._source = open_frame_source(self.mov_path, self.memmap_path)
        return self._source

    def _fit_norm(self, src, percentiles, n_sample, seed) -> Tuple[float, float]:
        rng = np.random.default_rng(seed + 1)
        picks = rng.choice(self.indices, size=min(n_sample, len(self.indices)), replace=False)
        vals = np.concatenate([
            _center_crop(src.get(int(i)), *self.crop, self.axis_col).ravel() for i in picks
        ])
        lo, hi = np.percentile(vals, percentiles)
        if hi - lo < 1e-6:
            hi = lo + 1.0
        return float(lo), float(hi)

    def normalize(self, counts: torch.Tensor) -> torch.Tensor:
        return 2.0 * (counts - self.norm_min) / (self.norm_max - self.norm_min) - 1.0

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 1.0) * 0.5 * (self.norm_max - self.norm_min) + self.norm_min

    @property
    def _cond_channels(self) -> int:
        return ANGLE_TIME_COND_CHANNELS if self.cond_angle_time else 0

    @property
    def in_channels_diffusion(self) -> int:
        return 2 + len(self.layout) + self._cond_channels

    @property
    def in_channels_baseline(self) -> int:
        return 1 + len(self.layout) + self._cond_channels

    # --- item ---------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.indices)

    def _read_tap(self, src: FrameSource, ci: int, tap) -> np.ndarray:
        frame = src.get_interp(ci + tap.frame_offset) if tap.interp else src.get(ci + int(tap.frame_offset))
        if tap.mirror:
            frame = _mirror_cols(frame, self.axis_col)
        return _center_crop(frame, *self.crop, self.axis_col)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src = self._get_source()
        ci = int(self.indices[idx])

        central = _center_crop(src.get(ci), *self.crop, self.axis_col)
        if self.layout:
            context = np.stack([self._read_tap(src, ci, tap) for tap in self.layout], axis=0)
        else:
            # k=0 (pure N2V ablation): no context taps at all.
            context = np.empty((0, *self.crop), dtype=np.float32)

        central = torch.from_numpy(np.ascontiguousarray(central)).float().unsqueeze(0)
        context = torch.from_numpy(np.ascontiguousarray(context)).float()

        out: Dict[str, torch.Tensor] = {"frame_index": torch.tensor(ci, dtype=torch.long)}

        if self.cond_angle_time:
            cond = angle_time_cond_array(np.array([ci]), self.deg_per_frame,
                                         self._cond_frame_start, self._cond_frame_end)[0]
            out["cond_channels"] = torch.from_numpy(cond).view(-1, 1, 1).expand(-1, *self.crop).contiguous()

        if self.n2n:
            # Noise2Noise: split the fixed dose measurement into two independent views
            # (input fraction p, target 1-p); full-dose neighbours as context; condition on p.
            gen = None if self.noise_seed is None else torch.Generator().manual_seed(self.noise_seed + ci)
            if self.noise_seed is None:
                p = float(torch.empty(1).uniform_(self.p_range[0], self.p_range[1]).item())
            else:
                pg = torch.Generator().manual_seed(self.noise_seed + ci + 777)
                p = float(torch.empty(1).uniform_(self.p_range[0], self.p_range[1], generator=pg).item())
            inp, tgt = binomial_split(central, self.extra_noise_dose, p, generator=gen)
            context = add_poisson_noise(context, self.extra_noise_dose, generator=gen)
            out["reference"] = self.normalize(central)          # native (eval GT)
            out["central_input"] = self.normalize(inp)
            out["central_target"] = self.normalize(tgt)
            out["context"] = self.normalize(context)
            out["p"] = torch.tensor(p, dtype=torch.float32)
            out["p_bin"] = torch.tensor(int(round(p * self.p_bins)), dtype=torch.long)
            return out

        if self.extra_noise_dose is not None:
            # Independent extra Poisson noise per frame; original central is the ref.
            # noise_seed=None -> fresh noise each call (training draws a new realisation
            # from the dose distribution every epoch); int -> reproducible (eval).
            gen = None if self.noise_seed is None else torch.Generator().manual_seed(self.noise_seed + ci)
            out["reference"] = self.normalize(central)
            central = add_poisson_noise(central, self.extra_noise_dose, generator=gen)
            context = add_poisson_noise(context, self.extra_noise_dose, generator=gen)

        out["central"] = self.normalize(central)
        out["context"] = self.normalize(context)
        return out
