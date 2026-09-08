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
* ``aux_channel`` ``(T, H, W)`` — only if ``aux_channel_memmap`` is set: T>=1
  cached extra input channels from an external source (e.g. a frozen
  cross-domain model's own prediction on this dataset's frames, or the T
  same-angle joint-FBP reprojection taps), loaded the
  same way as ``clean_target_memmap`` (float16 counts + a
  ``first_index``/``num_frames``/``crop`` sidecar) but renormalised with
  THIS dataset's own ``(norm_min, norm_max)`` and exposed as an input, not a
  training target.
* ``context_warped`` ``(4k, H, W)`` (or ``6k``) — only if
  ``warped_temporal_memmaps`` is set: a COPY of ``context`` with the
  "temporal" taps named in that dict overridden by their cached
  motion-compensated value (see the angular-resolution-gap warped-context
  experiment) -- rotation taps, and any temporal tap not named in the dict,
  are identical to ``context``. ``context`` itself is ALWAYS the raw,
  live-read value, unchanged, so code that needs to see exactly what a frozen
  checkpoint was originally trained on (e.g. bootstrap self-distillation's
  own belief computation) still can.

All frames are denormalised to a common count space, cropped to ``crop`` around
the rotation axis, then affinely normalised to ``[-1, 1]`` with a single
``(norm_min, norm_max)`` fit over a sample of frames (saved to the checkpoint
sidecar so inference can invert it).  The N2V blind-spot corruption and the
diffusion noising are applied later, inside the loss, so a fresh mask/noise is
drawn every step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .frames import FrameSource, open_frame_source
from .geometry import (
    ANGLE_TIME_COND_CHANNELS, DEG_PER_FRAME, ROT_AXIS_COL, angle_time_cond_array,
    build_context_layout, usable_frame_range,
)
from .noise import add_poisson_noise, anscombe_transform_thinned, binomial_split


def resolve_warped_context_dir(warped_context_dir, k: int, include_mirror: bool,
                               neighborhoods: str, temporal_raw_pairs: bool,
                               deg_per_frame: float) -> Dict[str, str]:
    """``--warped_context_dir`` -> ``{tap.name: path}`` for this exact layout.

    Single source of truth for the ``warped_ctx_{tap.name}.f16`` naming
    convention (see ``scripts/tr_diffusion_warped_context_precompute.py``),
    shared by ``train.py`` (resolving the flag before dataset construction)
    and ``reconstruct.py`` (at inference) -- computing the tap-name set two
    different ways in two files is exactly the kind of transposition risk
    that would silently apply the wrong cache to the wrong tap.
    """
    layout = build_context_layout(k, include_mirror, period_360=360.0 / deg_per_frame,
                                  neighborhoods=neighborhoods, temporal_raw_pairs=temporal_raw_pairs)
    temporal_names = [t.name for t in layout if t.kind == "temporal"]
    out = {}
    for name in temporal_names:
        path = Path(warped_context_dir) / f"warped_ctx_{name}.f16"
        if not path.exists():
            raise FileNotFoundError(f"{warped_context_dir} is missing warped_ctx_{name}.f16 for this "
                                    f"k={k}/neighborhoods={neighborhoods!r}/temporal_raw_pairs="
                                    f"{temporal_raw_pairs} layout (expected temporal taps: {temporal_names})")
        out[name] = str(path)
    return out


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
        k: int = 3,
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
        clean_target_memmap: Optional[Union[str, Path]] = None,
        var_target_memmap: Optional[Union[str, Path]] = None,
        aux_channel_memmap: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
        warped_temporal_memmaps: Optional[Dict[str, Union[str, Path]]] = None,
        anscombe: bool = False,
        anscombe_norm_sample_frames: int = 64,
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

        # Optional external "clean" regression target (e.g. a Bayesian
        # two-head poisson_head reconstruction), read from a cached
        # denoised-sequence memmap (see reconstruct.denoise_sequence's
        # output format: float16 counts, shape (num_frames, *crop), with a
        # first_index/num_frames/crop sidecar at "<path>.meta.npz"). Only
        # meaningful together with n2n=True (see __getitem__).
        self._clean_mm = None
        self._clean_first = None
        if clean_target_memmap is not None:
            meta = np.load(str(clean_target_memmap) + ".meta.npz")
            clean_crop = tuple(int(c) for c in meta["crop"])
            if clean_crop != self.crop:
                raise ValueError(f"clean_target_memmap crop {clean_crop} != dataset crop {self.crop}")
            self._clean_first = int(meta["first_index"])
            clean_n = int(meta["num_frames"])
            self._clean_mm = np.memmap(str(clean_target_memmap), dtype=np.float16, mode="r",
                                       shape=(clean_n, *self.crop))
            # Training centres must have a clean reference -> intersect the
            # requested range with what the clean-target cache actually covers.
            frame_start = max(int(frame_start), self._clean_first)
            frame_end = min(int(frame_end), self._clean_first + clean_n)

        # Companion per-pixel posterior-VARIANCE cache (see
        # reconstruct.denoise_sequence's var_out_path) -- the heteroscedastic
        # "sigma_tn(x)^2" an Ambient-Tweedie-style consumer needs alongside the
        # mean above. Always paired with clean_target_memmap (same caching run,
        # same first_index/num_frames/crop), so no separate range intersection.
        self._var_mm = None
        self._var_first = None
        if var_target_memmap is not None:
            if clean_target_memmap is None:
                raise ValueError("var_target_memmap requires clean_target_memmap (its companion mean cache)")
            vmeta = np.load(str(var_target_memmap) + ".meta.npz")
            var_crop = tuple(int(c) for c in vmeta["crop"])
            if var_crop != self.crop:
                raise ValueError(f"var_target_memmap crop {var_crop} != dataset crop {self.crop}")
            self._var_first = int(vmeta["first_index"])
            var_n = int(vmeta["num_frames"])
            if self._var_first != self._clean_first or var_n != clean_n:
                raise ValueError("var_target_memmap must cover the exact same frame range as clean_target_memmap "
                                 f"(clean: first={self._clean_first} n={clean_n}, var: first={self._var_first} n={var_n})")
            self._var_mm = np.memmap(str(var_target_memmap), dtype=np.float16, mode="r",
                                     shape=(var_n, *self.crop))

        # Optional cached EXTRA INPUT channel (e.g. a frozen cross-domain
        # model's own prediction on this dataset's frames -- see the
        # "noise2clean auxiliary channel" experiment). Same cache format/
        # loading convention as clean_target_memmap above (float16 counts,
        # first_index-relative indexing, range intersection) but exposed as
        # an INPUT (out["aux_channel"]), not a training target, and
        # renormalised with THIS dataset's own norm_min/max (the cache is
        # stored in raw counts, so whichever dataset loads it applies its
        # own normalisation -- consistent with how clean_target/sigma_tn_map
        # are handled above).
        # `aux_channel_memmap` accepts either a single path (legacy: one extra
        # channel, e.g. the noise2clean cross-domain aux channel) or a
        # list/tuple of paths (e.g. the T same-angle joint-FBP reprojection
        # taps) -- stacked into a single `(T, H, W)` `out["aux_channel"]` at
        # read time either way. All memmaps must share first_index/num_frames/
        # crop (mirrors the warped_temporal_memmaps consistency check above).
        self._aux_mms: List[np.memmap] = []
        self._aux_first = None
        if aux_channel_memmap is not None:
            aux_paths = ([aux_channel_memmap] if isinstance(aux_channel_memmap, (str, Path))
                        else list(aux_channel_memmap))
            first_index = num_frames = None
            for path in aux_paths:
                ameta = np.load(str(path) + ".meta.npz")
                aux_crop = tuple(int(c) for c in ameta["crop"])
                if aux_crop != self.crop:
                    raise ValueError(f"aux_channel_memmap crop {aux_crop} != dataset crop {self.crop}")
                a_first, a_n = int(ameta["first_index"]), int(ameta["num_frames"])
                if first_index is None:
                    first_index, num_frames = a_first, a_n
                elif (a_first, a_n) != (first_index, num_frames):
                    raise ValueError(
                        f"aux_channel_memmap {path!r} (first_index={a_first}, num_frames={a_n}) "
                        f"disagrees with the others (first_index={first_index}, num_frames={num_frames}) "
                        "-- all aux-channel caches must have been built over the identical frame range")
                self._aux_mms.append(np.memmap(str(path), dtype=np.float16, mode="r",
                                               shape=(a_n, *self.crop)))
            self._aux_first = first_index
            frame_start = max(int(frame_start), first_index)
            frame_end = min(int(frame_end), first_index + num_frames)

        self.layout = build_context_layout(self.k, self.include_mirror, period_360=self._period_360,
                                           neighborhoods=self.neighborhoods,
                                           temporal_raw_pairs=self.temporal_raw_pairs)

        # Additive warped-context override (see the angular-resolution-gap experiment):
        # each entry replaces ONE named "temporal" tap's live-read value with a
        # precomputed motion-compensated one when context_warped is built in
        # __getitem__. Keyed by ContextTap.name (not position) so a mismatched file
        # can't silently apply to the wrong tap. All memmaps must share first_index/
        # num_frames/crop -- mirrors the clean_target_memmap/var_target_memmap
        # companion check below.
        self._warped_mms: Dict[str, np.memmap] = {}
        self._warped_first: Optional[int] = None
        if warped_temporal_memmaps:
            temporal_names = {t.name for t in self.layout if t.kind == "temporal"}
            unknown = set(warped_temporal_memmaps) - temporal_names
            if unknown:
                raise ValueError(f"warped_temporal_memmaps has keys not in this layout's temporal "
                                 f"taps ({sorted(temporal_names)}): {sorted(unknown)}")
            first_index = None
            num_frames = None
            for name, path in warped_temporal_memmaps.items():
                wmeta = np.load(str(path) + ".meta.npz")
                w_crop = tuple(int(c) for c in wmeta["crop"])
                if w_crop != self.crop:
                    raise ValueError(f"warped_temporal_memmaps[{name!r}] crop {w_crop} != dataset crop {self.crop}")
                w_first, w_n = int(wmeta["first_index"]), int(wmeta["num_frames"])
                if first_index is None:
                    first_index, num_frames = w_first, w_n
                elif (w_first, w_n) != (first_index, num_frames):
                    raise ValueError(
                        f"warped_temporal_memmaps[{name!r}] (first_index={w_first}, num_frames={w_n}) "
                        f"disagrees with the others (first_index={first_index}, num_frames={num_frames}) "
                        "-- all warped-context caches must have been built over the identical frame range")
                self._warped_mms[name] = np.memmap(str(path), dtype=np.float16, mode="r",
                                                   shape=(w_n, *self.crop))
            self._warped_first = first_index
            frame_start = max(int(frame_start), first_index)
            frame_end = min(int(frame_end), first_index + num_frames)

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

        # Anscombe-Gaussianized Ambient-Tweedie target: replicate "Consistent
        # Diffusion Meets Tweedie" (arXiv:2404.10177) LITERALLY -- a single GLOBAL
        # SCALAR noise level -- by variance-stabilizing the dose-thinned Poisson
        # measurement itself (z = 2*sqrt(counts+3/8), Var[z] ~= 1 REGARDLESS of the
        # local count rate) instead of using poisson_head's heteroscedastic
        # per-pixel posterior. See sdate.tr_diffusion.noise.anscombe_transform and
        # the ambient_tweedie.py module docstring.
        self.anscombe = bool(anscombe)
        self.anscombe_z_min: Optional[float] = None
        self.anscombe_z_max: Optional[float] = None
        self.anscombe_sigma_tn_norm: Optional[float] = None
        if self.anscombe:
            if self.extra_noise_dose is None:
                raise ValueError("anscombe=True requires extra_noise_dose (the dose fraction to "
                                 "Gaussianize -- there is no raw Poisson measurement to transform otherwise)")
            if self._var_mm is not None:
                raise ValueError("anscombe=True and var_target_memmap are alternative ways to get a "
                                 "homoscedastic-ish y for Ambient-Tweedie -- pass only one")
            self.anscombe_z_min, self.anscombe_z_max = self._fit_anscombe_norm(
                src, norm_percentiles, anscombe_norm_sample_frames, seed)
            # Var[z] ~= 1 in RAW (pre-normalization) z-units -- validated with a Monte
            # Carlo check (Poisson draws at lambda in [1,1000], forward-transformed,
            # empirical variance) rather than trusted purely from the asymptotic
            # theory: Var[z] in [0.72 (lambda=1), 0.92 (lambda=2), 0.999-1.001
            # (lambda>=5)]. This project's dose-thinned counts run in the hundreds,
            # safely inside the accurate regime, so the constant 1.0 is used as-is
            # rather than re-deriving a per-dataset empirical correction.
            self.anscombe_sigma_tn_norm = 2.0 / (self.anscombe_z_max - self.anscombe_z_min)

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

    def _fit_anscombe_norm(self, src, percentiles, n_sample, seed) -> Tuple[float, float]:
        rng = np.random.default_rng(seed + 3)
        picks = rng.choice(self.indices, size=min(n_sample, len(self.indices)), replace=False)
        vals = []
        for i in picks:
            frame = torch.from_numpy(np.ascontiguousarray(
                _center_crop(src.get(int(i)), *self.crop, self.axis_col))).float()
            noisy = add_poisson_noise(frame, self.extra_noise_dose)  # fresh draw; one-off range fit, no seed needed
            vals.append(anscombe_transform_thinned(noisy, self.extra_noise_dose).numpy().ravel())
        lo, hi = np.percentile(np.concatenate(vals), percentiles)
        if hi - lo < 1e-6:
            hi = lo + 1.0
        return float(lo), float(hi)

    def normalize_z(self, z: torch.Tensor) -> torch.Tensor:
        return 2.0 * (z - self.anscombe_z_min) / (self.anscombe_z_max - self.anscombe_z_min) - 1.0

    def denormalize_z(self, z_norm: torch.Tensor) -> torch.Tensor:
        return (z_norm + 1.0) * 0.5 * (self.anscombe_z_max - self.anscombe_z_min) + self.anscombe_z_min

    @property
    def _cond_channels(self) -> int:
        return ANGLE_TIME_COND_CHANNELS if self.cond_angle_time else 0

    @property
    def in_channels_diffusion(self) -> int:
        return 2 + len(self.layout) + self._cond_channels + (1 if self._var_mm is not None else 0)

    @property
    def in_channels_ambient_tweedie(self) -> int:
        """Ambient-Tweedie's network sees ONLY ``x_t`` (+ context, if any) -- never
        ``y``/``clean_target`` nor ``sigma_tn_map``, both of which stay label/loss-only
        (see the ``ambient_tweedie.py`` module docstring: two earlier versions fed one
        or the other in as a side channel, and both let the network learn a shortcut
        that ignores ``x_t``). No ``+1`` for ``sigma_tn_map`` here even when
        ``var_target_memmap``/``anscombe`` is set, unlike ``in_channels_diffusion``."""
        return 1 + len(self.layout) + self._cond_channels

    @property
    def in_channels_baseline(self) -> int:
        return 1 + len(self.layout) + self._cond_channels + len(self._aux_mms)

    # --- item ---------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.indices)

    def _read_tap(self, src: FrameSource, ci: int, tap) -> np.ndarray:
        frame = src.get_interp(ci + tap.frame_offset) if tap.interp else src.get(ci + int(tap.frame_offset))
        if tap.mirror:
            frame = _mirror_cols(frame, self.axis_col)
        return _center_crop(frame, *self.crop, self.axis_col)

    def _warped_context(self, context_raw: torch.Tensor, ci: int) -> Optional[torch.Tensor]:
        """``context_raw`` (RAW COUNTS, post-thinning, pre-normalise) with the named
        temporal taps overridden by their cached motion-compensated value. Returns
        ``None`` if no ``warped_temporal_memmaps`` were configured."""
        if not self._warped_mms:
            return None
        warped = context_raw.clone()
        for i, tap in enumerate(self.layout):
            mm = self._warped_mms.get(tap.name)
            if mm is not None:
                val = np.asarray(mm[ci - self._warped_first]).astype(np.float32)
                warped[i] = torch.from_numpy(val)
        return warped

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

        if self._aux_mms:
            aux = np.stack([np.asarray(mm[ci - self._aux_first]) for mm in self._aux_mms]).astype(np.float32)
            out["aux_channel"] = self.normalize(torch.from_numpy(aux))

        if self.anscombe:
            # Anscombe-Gaussianized Ambient-Tweedie: y = Anscombe(dose-thinned central
            # measurement) has an approximately GLOBAL CONSTANT noise variance (~1 in
            # raw z-units), replicating the paper's literal single-scalar-sigma_tn
            # setting directly -- no poisson_head, no per-pixel heteroscedastic map.
            # context is thinned to the SAME dose/draw, same rationale as the
            # var_mm branch below (kept even though the current recipe forces k=0 for
            # this mode -- see train.py -- so this is a no-op on an empty tensor).
            gen = None if self.noise_seed is None else torch.Generator().manual_seed(self.noise_seed + ci)
            central_noisy = add_poisson_noise(central, self.extra_noise_dose, generator=gen)
            context_noisy = add_poisson_noise(context, self.extra_noise_dose, generator=gen)
            z = anscombe_transform_thinned(central_noisy, self.extra_noise_dose)
            out["reference"] = self.normalize(central)
            out["clean_target"] = self.normalize_z(z)
            out["sigma_tn_map"] = torch.full_like(z, self.anscombe_sigma_tn_norm)
            out["context"] = self.normalize(context_noisy)
            return out

        if self._var_mm is not None:
            # Ambient-Tweedie: y = clean_target (poisson_head posterior mean),
            # sigma_tn_map = its posterior std (Gaussian approximation), both from
            # the SAME cached run -- which denoised a dose-thinned CENTRAL frame
            # using dose-thinned CONTEXT neighbours (see reconstruct.denoise_sequence
            # -> data.py's plain extra_noise_dose branch below, same generator for
            # both). The model's own context tap must match that SAME thinned
            # scenario, or it's conditioning on strictly cleaner side-information
            # than what the cached posterior actually saw -- silently leaking
            # detail the ADSM/consistency math doesn't account for.
            gen = None if self.noise_seed is None else torch.Generator().manual_seed(self.noise_seed + ci)
            if self.extra_noise_dose is not None:
                context = add_poisson_noise(context, self.extra_noise_dose, generator=gen)
            clean = np.asarray(self._clean_mm[ci - self._clean_first]).astype(np.float32)
            var_counts = np.asarray(self._var_mm[ci - self._var_first]).astype(np.float32)
            sigma_tn_norm = np.sqrt(var_counts) * (2.0 / (self.norm_max - self.norm_min))
            out["reference"] = self.normalize(central)
            out["clean_target"] = self.normalize(torch.from_numpy(clean).unsqueeze(0))
            out["sigma_tn_map"] = torch.from_numpy(sigma_tn_norm).unsqueeze(0)
            out["context"] = self.normalize(context)
            return out

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
            if self._clean_mm is not None:
                clean = np.asarray(self._clean_mm[ci - self._clean_first]).astype(np.float32)
                out["clean_target"] = self.normalize(torch.from_numpy(clean).unsqueeze(0))
            return out

        if self.extra_noise_dose is not None:
            # Independent extra Poisson noise per frame; original central is the ref.
            # noise_seed=None -> fresh noise each call (training draws a new realisation
            # from the dose distribution every epoch); int -> reproducible (eval).
            gen = None if self.noise_seed is None else torch.Generator().manual_seed(self.noise_seed + ci)
            out["reference"] = self.normalize(central)
            central = add_poisson_noise(central, self.extra_noise_dose, generator=gen)
            context = add_poisson_noise(context, self.extra_noise_dose, generator=gen)

        warped = self._warped_context(context, ci)
        if warped is not None:
            out["context_warped"] = self.normalize(warped)

        out["central"] = self.normalize(central)
        out["context"] = self.normalize(context)
        return out
