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
from typing import Sequence, Tuple

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


def frames_to_boxavg3(frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(T, H, W) uint16 in [0, 4095] -> (avg_stream, rem_stream), both same shape.

    Frames 0, 1 pass through unchanged (no 3-frame window yet -- they seed
    the recurrence). For i >= 2: ``S_i = P_i + P_{i-1} + P_{i-2}`` (integer,
    range [0, 12285]), split via floor-div/mod into ``avg[i] = S_i // 3``
    (range [0, 4095] -- SAME range as a raw frame, and its noise variance is
    ~1/3rd of a single frame's since it's a genuine 3-frame average, not a
    diff: the shared slowly-varying signal is kept, only independent noise
    gets averaged down) and ``rem[i] = S_i % 3`` (in {0,1,2}, a cheap side
    channel needed only for exact reconstruction -- ``S_i = 3*avg[i] +
    rem[i]`` identically, by the floor/mod identity, no rounding loss).
    ``rem[0], rem[1]`` are unused (always 0).

    Exactly invertible by :func:`boxavg3_to_frames`.
    """
    frames_i32 = frames.astype(np.int32)
    avg = frames_i32.copy()
    rem = np.zeros_like(frames_i32)
    if frames.shape[0] > 2:
        s = frames_i32[2:] + frames_i32[1:-1] + frames_i32[:-2]
        avg[2:] = s // 3
        rem[2:] = s % 3
    return avg.astype(np.uint16), rem.astype(np.uint16)


def boxavg3_to_frames(avg: np.ndarray, rem: np.ndarray) -> np.ndarray:
    """Exact inverse of :func:`frames_to_boxavg3`.

    Sequential by construction (each recovered ``P_i`` needs the two
    previously-recovered frames), so this is a plain Python loop -- fine
    for offline verification, not meant as a fast decoder.
    """
    t = avg.shape[0]
    frames = avg.astype(np.int32).copy()  # frames[0], frames[1] already exact
    for i in range(2, t):
        s_i = 3 * avg[i].astype(np.int32) + rem[i].astype(np.int32)
        frames[i] = s_i - frames[i - 1] - frames[i - 2]
    return frames.astype(np.uint16)


def encode_ffv1_lossless_boxavg3(frames: np.ndarray, fps: float = 30.0) -> Tuple[bytes, bytes]:
    """FFV1 on the box-average-3 reparametrization: (avg_bytes, rem_bytes).

    Both streams must be kept (and their sizes both counted) to reconstruct
    losslessly -- see :func:`bits_per_pixel_combined`.
    """
    avg, rem = frames_to_boxavg3(frames)
    avg_bytes = encode_ffv1_lossless(avg, fps=fps, max_val=4095)
    rem_bytes = encode_ffv1_lossless(rem, fps=fps, max_val=2)
    return avg_bytes, rem_bytes


def decode_ffv1_lossless_boxavg3(avg_bytes: bytes, rem_bytes: bytes, h: int, w: int) -> np.ndarray:
    """Inverse of :func:`encode_ffv1_lossless_boxavg3` -- returns raw frames."""
    avg = decode_ffv1_lossless(avg_bytes, h, w)
    rem = decode_ffv1_lossless(rem_bytes, h, w)
    return boxavg3_to_frames(avg, rem)


def bits_per_pixel_combined(streams: Sequence[bytes], n_frames: int, h: int, w: int) -> float:
    """Like :func:`bits_per_pixel` but for a codec split across multiple
    byte streams that must ALL be kept to decode (e.g. boxavg3's avg+rem)."""
    return sum(len(s) for s in streams) * 8 / (n_frames * h * w)


def decode_ffv1_lossless_diff(encoded: bytes, h: int, w: int) -> np.ndarray:
    """Inverse of :func:`encode_ffv1_lossless_diff` -- returns raw frames."""
    return temporal_diff_to_frames(decode_ffv1_lossless(encoded, h, w))
