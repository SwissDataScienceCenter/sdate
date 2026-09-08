"""Dataset for tr_fbp_codec training/eval.

Combines native raw context+target frames (via ``MemmapFrameSource``) with
the FBP-reprojected prior tap.

**Training tap source**: the existing cached joint-FBP-context ``tap0``
file (see DEPENDENCIES.md ``FBP_TAP_CACHE_*``) -- built from
revolution-aligned windows that can include frames at/after the target
(the accepted target-leakage simplification, see README.md). NOT causal.

**Inference tap source**: NOT YET IMPLEMENTED here -- the genuinely causal
rolling reconstruction (build FBP from the ~200 frames strictly preceding
the target, reproject to the target's own angle) still needs to be written
using ``reconstruct.reconstruct`` + ``phi_context.reproject_at_angles``/
``reproject_to_counts`` (see README.md "Causality / decodability"). Do not
reuse ``TrFbpCodecDataset`` for a real inference/decode benchmark yet.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import CodecConfig, DataConfig, QuantConfig
from .quantization import quantize_to_12bit

# See DEPENDENCIES.md "Reused data artifact" -- built for a different
# (T5-native Gaussian-floor denoiser) project, reused here as-is. Path is
# RunAI-sandbox-specific (/myhome/...); override via TR_FBP_CODEC_TAP_CACHE_DIR
# on environments with a different filesystem layout (e.g. CSCS Clariden,
# where /myhome doesn't exist at all) -- see DEPENDENCIES.md "Portability".
_TAP_CACHE_DIR = Path(os.environ.get(
    "TR_FBP_CODEC_TAP_CACHE_DIR", "/myhome/data/sdate/shared/time_resolved/jointfbp_context"
))
_TAP_CACHE_PREFIX = os.environ.get(
    "TR_FBP_CODEC_TAP_CACHE_PREFIX", "212_Wunderkerze2_jointfbpctx_T5_native_tap0"
)


def _load_tap0_cache() -> Tuple[np.memmap, int, int]:
    meta = np.load(_TAP_CACHE_DIR / f"{_TAP_CACHE_PREFIX}.f16.meta.npz")
    first_index = int(meta["first_index"])
    num_frames = int(meta["num_frames"])
    crop = tuple(int(c) for c in meta["crop"])
    mm = np.memmap(
        _TAP_CACHE_DIR / f"{_TAP_CACHE_PREFIX}.f16",
        dtype=np.float16, mode="r", shape=(num_frames, *crop),
    )
    return mm, first_index, num_frames


class TrFbpCodecDataset(Dataset):
    """Random block-crop samples: k context frames (+ FBP-prior tap) -> target.

    ``split``: "train" or "holdout" (ranges from ``DataConfig``). Held-out
    targets still draw real preceding context frames even when those frames
    fall in the training range -- this is NOT leakage. A real decoder
    always has genuinely already-decoded prior frames available; by the
    time decoding reaches the held-out portion of the stream, the training
    portion IS already-known/decoded data, exactly as it would be for a
    real deployed codec (see README.md / chat discussion, 2026-09-07).
    """

    def __init__(
        self,
        data_cfg: DataConfig,
        codec_cfg: CodecConfig,
        quant_cfg: QuantConfig,
        split: str = "train",
        crops_per_frame: int = 4,
    ):
        from sdate.tr_diffusion import reconstruct as R
        from sdate.tr_diffusion.frames import MemmapFrameSource
        from sdate.tr_diffusion.profiles import REGISTRY

        self._R = R

        assert split in ("train", "holdout")
        self.codec_cfg = codec_cfg
        self.quant_cfg = quant_cfg
        self.crops_per_frame = crops_per_frame

        profile = REGISTRY[data_cfg.profile_name]
        self.profile = profile
        # Override for environments with a different filesystem layout than
        # the RunAI sandbox's /myhome/... (e.g. Clariden) -- see
        # DEPENDENCIES.md "Portability". mov_path need not physically exist
        # (only used to locate the .norm.npz sidecar next to it).
        memmap_path = os.environ.get("TR_FBP_CODEC_MEMMAP_PATH", profile.memmap_path)
        mov_path = os.environ.get("TR_FBP_CODEC_MOV_PATH", profile.mov_path)
        self.src = MemmapFrameSource(memmap_path, mov_path)

        if codec_cfg.use_fbp_prior:
            self.tap_mm, self.tap_first, self.tap_n = _load_tap0_cache()
        else:
            self.tap_mm = self.tap_first = self.tap_n = None

        if split == "train":
            lo = data_cfg.frame_start
            hi = data_cfg.holdout_frame_start or data_cfg.frame_end
        else:
            if data_cfg.holdout_frame_start is None:
                raise ValueError("split='holdout' requested but DataConfig has no holdout range")
            lo, hi = data_cfg.holdout_frame_start, data_cfg.holdout_frame_end

        k = codec_cfg.k
        earliest_target = max(lo, data_cfg.frame_start + k)
        latest_target = hi
        if codec_cfg.use_fbp_prior:
            earliest_target = max(earliest_target, self.tap_first)
            latest_target = min(latest_target, self.tap_first + self.tap_n)
        if earliest_target >= latest_target:
            raise ValueError(
                f"empty valid target range [{earliest_target}, {latest_target}) for split={split!r} "
                "-- check DataConfig against the tap cache's coverage (see DEPENDENCIES.md)"
            )
        self.targets = np.arange(earliest_target, latest_target)

        bs, m = codec_cfg.block_size, codec_cfg.block_margin
        self.crop_extent = (bs + 2 * m) if bs is not None else None
        self.block_size = bs
        self.block_margin = m

    def __len__(self) -> int:
        return len(self.targets) * self.crops_per_frame

    def _sample_crop_origin(self, h: int, w: int) -> Tuple[int, int]:
        ext = self.crop_extent
        top = int(np.random.randint(0, h - ext + 1))
        left = int(np.random.randint(0, w - ext + 1))
        return top, left

    def __getitem__(self, idx: int):
        t = int(self.targets[idx % len(self.targets)])
        k = self.codec_cfg.k
        ctx_idx = np.arange(t - k, t)

        # NOTE: must center-crop via native_window (profile.crop/rot_axis_col),
        # NOT raw MemmapFrameSource.get() -- the stored memmap is (128,528) but
        # the tap cache (and profile.crop) is (128,512); using the uncropped
        # frame here silently mismatches width against the prior tap (caught
        # via a real shape-mismatch bug during smoke-testing, see git history).
        context = self._R.native_window(
            self.src, ctx_idx, self.profile.crop, self.profile.rot_axis_col
        )  # (k, H, W) float32
        target_full = self._R.native_window(
            self.src, np.array([t]), self.profile.crop, self.profile.rot_axis_col
        )[0]  # (H, W) float32

        prior_full: Optional[torch.Tensor] = None
        if self.codec_cfg.use_fbp_prior:
            local = t - self.tap_first
            prior_full = torch.from_numpy(np.asarray(self.tap_mm[local]).astype(np.float32))

        h, w = target_full.shape
        if self.crop_extent is not None:
            top, left = self._sample_crop_origin(h, w)
            ext = self.crop_extent
            context = context[:, top:top + ext, left:left + ext]
            target_full = target_full[top:top + ext, left:left + ext]
            if prior_full is not None:
                prior_full = prior_full[top:top + ext, left:left + ext]
            bm = self.block_margin
            target_crop = target_full[bm:bm + self.block_size, bm:bm + self.block_size]
        else:
            target_crop = target_full

        target_q = torch.from_numpy(quantize_to_12bit(target_crop.numpy(), self.quant_cfg))

        item = {"context": context, "target_q": target_q, "frame_idx": t}
        if prior_full is not None:
            item["prior"] = prior_full
        return item


def collate(batch):
    """Default collate -- handles the "no prior" (paper-reproduction baseline) case
    where ``"prior"`` is absent from every item, which torch's default collate
    can't do cleanly (it would error trying to collate a missing key)."""
    out = {
        "context": torch.stack([b["context"] for b in batch]),
        "target_q": torch.stack([b["target_q"] for b in batch]),
        "frame_idx": torch.tensor([b["frame_idx"] for b in batch]),
    }
    if "prior" in batch[0]:
        out["prior"] = torch.stack([b["prior"] for b in batch])
    return out
