"""HEVC (libx265) and FFV1 lossless baselines for 12-bit-quantized frames.

Uses the persistent ffmpeg build at ``/myhome/tools/ffmpeg`` (see repo
memory ``reference-persistent-ffmpeg``) -- its ``bin/ffmpeg`` is already a
self-contained wrapper that sets ``LD_LIBRARY_PATH`` internally, so no env
setup is needed beyond pointing at it.

**Not reused from `sdate.video_compression`/`sdate.stream_hvec`** (see
DEPENDENCIES.md): both are hardcoded to 10-bit via a float-normalized
``gray16le`` -> ``format=gray10le`` rescale path. Our ground truth is
already-quantized integer values in [0, 4095] -- feeding those through that
rescale path would silently corrupt them (verified during the design
session's HEVC-feasibility check). This module declares the raw input
directly as ``gray12le`` (HEVC) or ``gray16le`` (FFV1), no rescale filter.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

# RunAI-sandbox-specific default; override via TR_FBP_CODEC_FFMPEG on
# environments with a different filesystem layout (e.g. CSCS Clariden) --
# see DEPENDENCIES.md "Portability".
FFMPEG = os.environ.get("TR_FBP_CODEC_FFMPEG", "/myhome/tools/ffmpeg/bin/ffmpeg")


def _run(cmd: list) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg command failed ({' '.join(cmd)}):\n{proc.stderr.decode(errors='replace')}"
        )


def _frames_to_raw_bytes(frames: np.ndarray, max_val: int = 4095) -> bytes:
    """frames: (T, H, W) uint16 in [0, max_val]. ``max_val`` is 4095 for raw
    12-bit frames but wider (see ``_DIFF_BIAS``) for a biased temporal-diff
    stream, which still fits the same uint16/gray16le container."""
    if frames.dtype != np.uint16:
        raise ValueError(f"expected uint16, got {frames.dtype}")
    if frames.max() > max_val:
        raise ValueError(f"expected values in [0,{max_val}], got max={frames.max()}")
    return np.ascontiguousarray(frames).tobytes()


def encode_hevc12_lossless(frames: np.ndarray, fps: float = 30.0) -> bytes:
    """frames: (T, H, W) uint16 in [0, 4095]. Returns the encoded .mkv bytes."""
    t, h, w = frames.shape
    with tempfile.TemporaryDirectory() as td:
        raw_path = Path(td) / "in.raw"
        out_path = Path(td) / "out.mkv"
        raw_path.write_bytes(_frames_to_raw_bytes(frames))
        _run([
            FFMPEG, "-y",
            "-f", "rawvideo", "-pix_fmt", "gray12le", "-s", f"{w}x{h}", "-r", str(fps),
            "-i", str(raw_path),
            "-c:v", "libx265", "-pix_fmt", "gray12le", "-x265-params", "lossless=1",
            str(out_path),
        ])
        return out_path.read_bytes()


def decode_hevc12_lossless(encoded: bytes, h: int, w: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "in.mkv"
        out_path = Path(td) / "out.raw"
        in_path.write_bytes(encoded)
        _run([FFMPEG, "-y", "-i", str(in_path), "-pix_fmt", "gray12le", "-f", "rawvideo", str(out_path)])
        raw = out_path.read_bytes()
    arr = np.frombuffer(raw, dtype=np.uint16)
    return arr.reshape(-1, h, w)


def encode_ffv1_lossless(frames: np.ndarray, fps: float = 30.0, max_val: int = 4095) -> bytes:
    """frames: (T, H, W) uint16 in [0, max_val]. Returns the encoded .mkv bytes.

    ``max_val`` defaults to 4095 (raw 12-bit frames) but is wider for a
    biased temporal-diff stream (see ``encode_ffv1_lossless_diff``) -- FFV1
    itself is agnostic, gray16le already covers the full range either way.
    """
    t, h, w = frames.shape
    with tempfile.TemporaryDirectory() as td:
        raw_path = Path(td) / "in.raw"
        out_path = Path(td) / "out.mkv"
        raw_path.write_bytes(_frames_to_raw_bytes(frames, max_val=max_val))
        _run([
            FFMPEG, "-y",
            "-f", "rawvideo", "-pix_fmt", "gray16le", "-s", f"{w}x{h}", "-r", str(fps),
            "-i", str(raw_path),
            "-c:v", "ffv1", "-pix_fmt", "gray16le",
            str(out_path),
        ])
        return out_path.read_bytes()


def decode_ffv1_lossless(encoded: bytes, h: int, w: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "in.mkv"
        out_path = Path(td) / "out.raw"
        in_path.write_bytes(encoded)
        _run([FFMPEG, "-y", "-i", str(in_path), "-pix_fmt", "gray16le", "-f", "rawvideo", str(out_path)])
        raw = out_path.read_bytes()
    arr = np.frombuffer(raw, dtype=np.uint16)
    return arr.reshape(-1, h, w)


def bits_per_pixel(encoded: bytes, n_frames: int, h: int, w: int) -> float:
    return len(encoded) * 8 / (n_frames * h * w)


# Max magnitude of a 12-bit-to-12-bit signed diff (P_i - P_{i-1} in
# [-4095, 4095]) -- biasing by this keeps the diff stream unsigned (fits
# gray16le, same container FFV1 already uses) without touching entropy: a
# constant per-pixel offset doesn't change FFV1's own (median-predictor)
# residuals, it just shifts everything into a representable range.
_DIFF_BIAS = 4095


def frames_to_temporal_diff(frames: np.ndarray) -> np.ndarray:
    """(T, H, W) uint16 in [0, 4095] -> temporal-diff stream, same dtype/shape.

    ``out[0] = frames[0]`` (first frame unchanged), ``out[i] = frames[i] -
    frames[i-1] + _DIFF_BIAS`` for i>0. Exactly invertible by
    :func:`temporal_diff_to_frames` -- this is a lossless reparametrization,
    not a new compression step by itself.
    """
    diffs = frames.astype(np.int32)
    out = diffs.copy()
    out[1:] = diffs[1:] - diffs[:-1] + _DIFF_BIAS
    return out.astype(np.uint16)


def temporal_diff_to_frames(diffs: np.ndarray) -> np.ndarray:
    """Exact inverse of :func:`frames_to_temporal_diff`."""
    d = diffs.astype(np.int32)
    d[1:] -= _DIFF_BIAS
    return np.cumsum(d, axis=0).astype(np.uint16)


def encode_ffv1_lossless_diff(frames: np.ndarray, fps: float = 30.0) -> bytes:
    """FFV1 applied to the temporal-diff stream instead of raw frames."""
    return encode_ffv1_lossless(frames_to_temporal_diff(frames), fps=fps, max_val=2 * _DIFF_BIAS)


def decode_ffv1_lossless_diff(encoded: bytes, h: int, w: int) -> np.ndarray:
    """Inverse of :func:`encode_ffv1_lossless_diff` -- returns raw frames."""
    return temporal_diff_to_frames(decode_ffv1_lossless(encoded, h, w))
