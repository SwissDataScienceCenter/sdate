"""Frame access for the ``212_Wunderkerze2`` HEVC projection stream.

The ``.mov`` is per-frame min-max normalised 10-bit HEVC: decoding a frame with
``ffmpeg -pix_fmt gray16le`` yields ``uint16`` in ``[0, 65535]`` that represents
``[0, 1]`` for *that frame only*.  The true detector **counts** are recovered
with the ``.norm.npz`` sidecar (arrays ``per_frame_min`` / ``per_frame_max``)::

    counts_k = per_frame_min[k] + decoded_k / 65535 * (per_frame_max[k] - per_frame_min[k])

Denormalising to a common count space is mandatory: only then are a frame and
its temporal / rotation neighbours on the same intensity scale, and only then is
Poisson noise (the extra-noise regime) well defined.

Two sources implement the same interface:

* :class:`FfmpegFrameSource` — random access straight from the 55 GB ``.mov`` via
  windowed ffmpeg seeks + an LRU frame cache.  Zero setup; slower.  Good for
  notebooks and small runs.
* :class:`MemmapFrameSource` — a pre-extracted ``uint16`` memmap of the decoded
  frames (built by ``extract_frames.py``).  Instant random access; the right
  choice for real training.

Both return **counts** as ``float32`` ``(H, W)`` and support sub-frame linear
interpolation via :meth:`FrameSource.get_interp`.
"""

from __future__ import annotations

import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from .geometry import FRAME_H, FRAME_W

_U16_MAX = 65535.0
_DEFAULT_FFMPEG = "/myhome/bin/ffmpeg"


def load_norm_sidecar(mov_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load the ``.norm.npz`` sidecar next to ``mov_path``.

    Returns a dict with float32 ``per_frame_min`` / ``per_frame_max`` (length =
    number of frames) and the int ``start_frame`` / ``end_frame`` of the source.
    """
    mov_path = Path(mov_path)
    side = mov_path.with_suffix(".norm.npz")
    if not side.exists():
        # e.g. ``foo.mov`` -> ``foo.norm.npz`` when suffix replacement is ambiguous
        side = mov_path.with_name(mov_path.stem + ".norm.npz")
    if not side.exists():
        raise FileNotFoundError(f"normalization sidecar not found for {mov_path}")
    d = np.load(side)
    return {
        "per_frame_min": d["per_frame_min"].astype(np.float32),
        "per_frame_max": d["per_frame_max"].astype(np.float32),
        "start_frame": int(d["start_frame"]),
        "end_frame": int(d["end_frame"]),
    }


def denormalize(decoded_u16: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """Map one decoded ``uint16`` frame back to detector counts (float32)."""
    span = max(float(fmax) - float(fmin), 1e-6)
    return (decoded_u16.astype(np.float32) / _U16_MAX) * span + float(fmin)


class FrameSource:
    """Common interface: integer / interpolated access to count frames.

    Frame indices are always **global** (source frame index in the original
    stream).  Valid indices are ``[first_index, first_index + num_frames)``:
    a full ``.mov`` starts at 0, a pre-extracted memmap slice starts at its
    ``start_frame``.
    """

    height: int = FRAME_H
    width: int = FRAME_W
    num_frames: int = 0
    first_index: int = 0

    @property
    def last_index(self) -> int:
        """Exclusive upper bound of valid global indices."""
        return self.first_index + self.num_frames

    def get(self, idx: int) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def get_interp(self, fidx: float) -> np.ndarray:
        """Linearly interpolate counts at a fractional frame index."""
        lo = int(np.floor(fidx))
        frac = float(fidx - lo)
        if frac <= 1e-6:
            return self.get(lo)
        hi = lo + 1
        return (1.0 - frac) * self.get(lo) + frac * self.get(hi)


class FfmpegFrameSource(FrameSource):
    """Windowed ffmpeg decoding of the ``.mov`` with an LRU count-frame cache."""

    def __init__(
        self,
        mov_path: Union[str, Path],
        ffmpeg: str = _DEFAULT_FFMPEG,
        fps: float = 30.0,
        window: int = 8,
        cache_size: int = 4096,
    ):
        self.mov_path = str(mov_path)
        self.ffmpeg = ffmpeg if Path(ffmpeg).exists() else "ffmpeg"
        self.fps = float(fps)
        self.window = int(window)
        self.cache_size = int(cache_size)
        side = load_norm_sidecar(mov_path)
        self.per_frame_min = side["per_frame_min"]
        self.per_frame_max = side["per_frame_max"]
        self.num_frames = int(self.per_frame_min.shape[0])
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._frame_bytes = 2 * self.height * self.width

    def _decode_window(self, start: int, n: int) -> np.ndarray:
        t = (start + 0.5) / self.fps
        proc = subprocess.run(
            [self.ffmpeg, "-v", "error", "-ss", f"{t:.6f}", "-i", self.mov_path,
             "-frames:v", str(n), "-pix_fmt", "gray16le", "-f", "rawvideo", "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        )
        buf = proc.stdout
        got = len(buf) // self._frame_bytes
        arr = np.frombuffer(buf[: got * self._frame_bytes], np.uint16)
        return arr.reshape(got, self.height, self.width)

    def _cache_put(self, idx: int, counts: np.ndarray) -> None:
        self._cache[idx] = counts
        self._cache.move_to_end(idx)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def get(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self.num_frames:
            raise IndexError(f"frame {idx} out of range [0, {self.num_frames})")
        cached = self._cache.get(idx)
        if cached is not None:
            self._cache.move_to_end(idx)
            return cached
        # Decode a small window so nearby taps (i±1, interpolation brackets) hit.
        start = max(0, idx - self.window // 2)
        n = min(self.window, self.num_frames - start)
        raw = self._decode_window(start, n)
        for j in range(raw.shape[0]):
            fk = start + j
            counts = denormalize(raw[j], self.per_frame_min[fk], self.per_frame_max[fk])
            self._cache_put(fk, counts)
        out = self._cache.get(idx)
        if out is None:  # short read at the tail; decode exactly this frame
            raw1 = self._decode_window(idx, 1)
            out = denormalize(raw1[0], self.per_frame_min[idx], self.per_frame_max[idx])
            self._cache_put(idx, out)
        return out


class MemmapFrameSource(FrameSource):
    """Fast random access from a pre-extracted decoded-``uint16`` memmap.

    The memmap holds the *decoded* (per-frame-normalised) ``uint16`` frames of a
    contiguous ``[start_frame, start_frame + num_frames)`` slice, shape
    ``(num_frames, H, W)``; counts are recovered per access using the sidecar.
    See :mod:`sdate.tr_diffusion.extract_frames`.
    """

    def __init__(self, memmap_path: Union[str, Path], mov_path: Union[str, Path]):
        meta = np.load(Path(memmap_path).with_suffix(".meta.npz"))
        self.start_frame = int(meta["start_frame"])
        self.num_frames = int(meta["num_frames"])
        self.height = int(meta["height"])
        self.width = int(meta["width"])
        self.first_index = self.start_frame
        self._mm = np.memmap(
            memmap_path, dtype=np.uint16, mode="r",
            shape=(self.num_frames, self.height, self.width),
        )
        side = load_norm_sidecar(mov_path)
        self.per_frame_min = side["per_frame_min"]
        self.per_frame_max = side["per_frame_max"]

    def get(self, idx: int) -> np.ndarray:
        local = idx - self.start_frame
        if local < 0 or local >= self.num_frames:
            raise IndexError(
                f"frame {idx} outside extracted range "
                f"[{self.start_frame}, {self.start_frame + self.num_frames})"
            )
        return denormalize(
            np.asarray(self._mm[local]), self.per_frame_min[idx], self.per_frame_max[idx]
        )


def open_frame_source(
    mov_path: Union[str, Path], memmap_path: Optional[Union[str, Path]] = None,
) -> FrameSource:
    """Return a :class:`MemmapFrameSource` if ``memmap_path`` is given, else ffmpeg."""
    if memmap_path is not None:
        return MemmapFrameSource(memmap_path, mov_path)
    return FfmpegFrameSource(mov_path)
