"""
Efficient storage of Gaussian parameters and compressed frames.

Handles:
- Redundant-parameter stripping (14 stored → 9 effective per Gaussian)
- FP16/FP32 serialisation
- Complete sequence save/load (keyframe + delta frames)
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Union

import numpy as np
import torch

from gsplat_compress.compression import (
    CompressedFrame,
    D_EFF,
    D_FULL,
    PARAMS_PER_GAUSSIAN_EFF,
    PARAMS_PER_GAUSSIAN_FULL,
)
from gsplat_compress.initialize import GaussianParams


# ═══════════════════════════════════════════════════════════════════════════
# Efficient keyframe serialisation
# ═══════════════════════════════════════════════════════════════════════════

def _strip_redundant(params: GaussianParams) -> dict[str, torch.Tensor]:
    """Strip redundant parameters for efficient storage.

    Keeps only the 9 non-redundant parameters per Gaussian:
      means_x, means_y, means_z  (3)
      log_scales_x, log_scales_y (2)  — drop z
      quat_w, quat_z             (2)  — drop x, y (out-of-plane)
      rgb_r                      (1)  — drop g, b (grayscale copies)
      opacity                    (1)
    """
    return {
        "means": params.means.detach().cpu(),              # [N, 3]
        "log_scales_xy": params.log_scales[:, :2].detach().cpu(),  # [N, 2]
        "quats_wz": params.quats[:, [0, 3]].detach().cpu(),       # [N, 2]
        "rgb_scalar": params.rgbs[:, 0:1].detach().cpu(),         # [N, 1]
        "opacity": params.opacities.detach().cpu(),               # [N]
    }


def _expand_redundant(
    stripped: dict[str, torch.Tensor], device: torch.device,
) -> GaussianParams:
    """Reconstruct full parameter tensors from the 9-param representation."""
    N = stripped["means"].shape[0]

    means = stripped["means"].to(device)

    # Reconstruct log_scales: [sx, sy, log(0.5)]
    ls_xy = stripped["log_scales_xy"].to(device)
    ls_z = torch.full((N, 1), math.log(0.5), device=device)
    log_scales = torch.cat([ls_xy, ls_z], dim=1)

    # Reconstruct quats: [w, 0, 0, z]
    wz = stripped["quats_wz"].to(device)
    quats = torch.zeros(N, 4, device=device)
    quats[:, 0] = wz[:, 0]
    quats[:, 3] = wz[:, 1]

    # Reconstruct rgbs: [r, r, r]
    r = stripped["rgb_scalar"].to(device)
    rgbs = r.expand(N, 3).contiguous()

    opacities = stripped["opacity"].to(device)

    return GaussianParams(means, log_scales, quats, rgbs, opacities)


def _strip_codebook_redundant(codebook: torch.Tensor) -> torch.Tensor:
    """Keep only the D_EFF=8 effective columns from a [K, 14] codebook.

    Columns to keep:
      0: Δmean_x, 1: Δmean_y,  (drop 2: Δmean_z)
      3: Δscale_x, 4: Δscale_y, (drop 5: Δscale_z)
      6: Δquat_w,               (drop 7: Δquat_x, 8: Δquat_y), 9: Δquat_z
      10: Δrgb_r,               (drop 11: Δrgb_g, 12: Δrgb_b)
      13: Δopacity
    """
    keep_cols = [0, 1, 3, 4, 6, 9, 10, 13]
    return codebook[:, keep_cols]


def _expand_codebook_redundant(cb_eff: torch.Tensor) -> torch.Tensor:
    """Expand an [K, 8] efficient codebook back to [K, 14].

    Maps: [Δmx, Δmy, Δsx, Δsy, Δqw, Δqz, Δr, Δo] → full 14 columns.
    """
    K = cb_eff.shape[0]
    full = torch.zeros(K, D_FULL, dtype=cb_eff.dtype, device=cb_eff.device)
    #       eff idx → full idx
    full[:, 0] = cb_eff[:, 0]    # Δmean_x
    full[:, 1] = cb_eff[:, 1]    # Δmean_y
    # col 2: Δmean_z = 0
    full[:, 3] = cb_eff[:, 2]    # Δscale_x
    full[:, 4] = cb_eff[:, 3]    # Δscale_y
    # col 5: Δscale_z = 0
    full[:, 6] = cb_eff[:, 4]    # Δquat_w
    # col 7, 8: Δquat_x, Δquat_y = 0
    full[:, 9] = cb_eff[:, 5]    # Δquat_z
    full[:, 10] = cb_eff[:, 6]   # Δrgb_r
    full[:, 11] = cb_eff[:, 6]   # Δrgb_g = Δrgb_r
    full[:, 12] = cb_eff[:, 6]   # Δrgb_b = Δrgb_r
    full[:, 13] = cb_eff[:, 7]   # Δopacity
    return full


# ═══════════════════════════════════════════════════════════════════════════
# Save / Load
# ═══════════════════════════════════════════════════════════════════════════

def save_sequence(
    path: Union[str, Path],
    keyframe_params: GaussianParams,
    compressed_frames: list[CompressedFrame],
    use_fp16: bool = True,
    metadata: dict | None = None,
) -> Path:
    """Save a compressed video sequence to a single ``.pt`` file.

    Parameters
    ----------
    path : str | Path
        Output file path.
    keyframe_params : GaussianParams
        Base-frame Gaussians (stored with redundancy stripping).
    compressed_frames : list[CompressedFrame]
        One entry per delta frame.
    use_fp16 : bool
        Cast float parameters to FP16 for storage.
    metadata : dict | None
        Optional metadata (image size, quality, etc.).

    Returns
    -------
    Path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if use_fp16 else torch.float32

    # Keyframe: strip redundant + cast
    stripped = _strip_redundant(keyframe_params)
    kf_data = {k: v.to(dtype) for k, v in stripped.items()}

    # Delta frames: strip codebook + pack labels efficiently
    delta_data = []
    for cf in compressed_frames:
        cb_eff = _strip_codebook_redundant(cf.codebook).to(dtype)
        # Pack labels as int16 if n_clusters <= 32767, else int32
        lbl_dtype = torch.int16 if cf.n_clusters <= 32767 else torch.int32
        delta_data.append({
            "codebook": cb_eff,
            "labels": cf.labels.to(lbl_dtype),
            "scaler_mean": cf.scaler_mean,
            "scaler_scale": cf.scaler_scale,
        })

    payload = {
        "keyframe": kf_data,
        "deltas": delta_data,
        "use_fp16": use_fp16,
        "n_frames": 1 + len(compressed_frames),
    }
    if metadata is not None:
        payload["metadata"] = metadata

    torch.save(payload, path)
    return path


def load_sequence(
    path: Union[str, Path],
    device: torch.device,
) -> tuple[GaussianParams, list[CompressedFrame], dict]:
    """Load a compressed video sequence from a ``.pt`` file.

    Returns
    -------
    keyframe_params : GaussianParams — on *device*
    compressed_frames : list[CompressedFrame]
    metadata : dict
    """
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    # Keyframe: expand to full params
    kf_data = {k: v.float() for k, v in payload["keyframe"].items()}
    keyframe = _expand_redundant(kf_data, device)

    # Delta frames: expand codebooks
    compressed_frames = []
    for dd in payload["deltas"]:
        cb_eff = dd["codebook"].float()
        cb_full = _expand_codebook_redundant(cb_eff)
        compressed_frames.append(CompressedFrame(
            codebook=cb_full,
            labels=dd["labels"].long(),
            scaler_mean=dd["scaler_mean"],
            scaler_scale=dd["scaler_scale"],
        ))

    metadata = payload.get("metadata", {})
    return keyframe, compressed_frames, metadata
