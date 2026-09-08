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


def _frames_to_raw_bytes(frames: np.ndarray) -> bytes:
    """frames: (T, H, W) uint16 in [0, 4095]."""
    if frames.dtype != np.uint16:
        raise ValueError(f"expected uint16, got {frames.dtype}")
    if frames.max() > 4095:
        raise ValueError(f"expected values in [0,4095], got max={frames.max()}")
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


def encode_ffv1_lossless(frames: np.ndarray, fps: float = 30.0) -> bytes:
    """frames: (T, H, W) uint16 in [0, 4095]. Returns the encoded .mkv bytes."""
    t, h, w = frames.shape
    with tempfile.TemporaryDirectory() as td:
        raw_path = Path(td) / "in.raw"
        out_path = Path(td) / "out.mkv"
        raw_path.write_bytes(_frames_to_raw_bytes(frames))
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
