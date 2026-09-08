"""Frame-window sources for streaming anomaly detection.

A ``WindowSource`` is the one thing the detector needs from wherever the
projection stream actually lives: a contiguous block of raw detector counts
for a window of frame indices, denormalised and cropped, as a GPU tensor
(plus an I0 estimate for the counts->attenuation step). Two sources:

* :class:`MovWindowSource` -- a pre-extracted ``uint16`` memmap (instant
  random access, but needs a one-off extraction step and its own disk space).
* :class:`FfmpegWindowSource` -- reads straight from the raw ``.mov`` via one
  bulk ffmpeg seek+decode PER WINDOW, no extraction/copy at all. Simple, but
  wasteful for a sliding-window scan: since windows overlap heavily (stride
  << window length), a given frame gets independently re-decoded once for
  every overlapping window that needs it (for the default T=21/11/5, up to
  ~38x over: 1 (single-rev) + 5 + 11 + 21).
* :class:`SequentialFfmpegWindowSource` -- also reads straight from the raw
  ``.mov``, but via ONE persistent ffmpeg process reading forward through the
  file exactly once, with a rolling in-memory buffer -- every frame is
  decoded once for the whole run, no matter how many overlapping windows
  need it. Requires windows to be requested in non-decreasing order (true of
  :func:`sw_anomaly_det.detector.stream_anomaly_scores`'s own scan) and
  ``trim_before`` to be called once per timestep to bound buffer memory.

An h5-backed or live-acquisition source can be added later behind this same
interface without touching the detector.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import torch


class WindowSource(ABC):
    """Contiguous-window counts access, denormalised + cropped, GPU-resident."""

    first_index: int
    last_index: int
    crop: Tuple[int, int]
    axis_col: float

    @abstractmethod
    def read_window(self, idx: np.ndarray, device: torch.device) -> torch.Tensor:
        """``idx``: contiguous ascending frame indices. Returns ``(V, H, W)`` float32 on ``device``."""
        raise NotImplementedError

    @abstractmethod
    def estimate_I0(self, idx_sample: np.ndarray, pct: float = 99.5) -> float:
        """Flat/air level estimate (high percentile of counts) over a sample of frames."""
        raise NotImplementedError

    def trim_before(self, keep_from: int) -> None:
        """Optional: release any resources/buffer content strictly before ``keep_from``.

        A no-op for sources with no internal buffer (memmap/per-call decode);
        :class:`SequentialFfmpegWindowSource` overrides this to bound its
        rolling buffer's memory. Safe to call unconditionally on any source.
        """
        pass


class MovWindowSource(WindowSource):
    """Reads from a pre-extracted memmap of a cached ``.mov`` (project convention).

    Same on-disk layout every ``sdate.tr_diffusion`` pipeline already uses
    (``DatasetProfile.memmap_path`` + ``.mov_path`` + its ``.norm.npz``
    sidecar) -- no new extraction step needed for datasets already set up for
    the existing tr_diffusion pipelines.
    """

    def __init__(self, memmap_path: str, mov_path: str, crop: Tuple[int, int], axis_col: float):
        from sdate.tr_diffusion.frames import MemmapFrameSource

        self._src = MemmapFrameSource(memmap_path, mov_path)
        self.crop = tuple(crop)
        self.axis_col = float(axis_col)
        self.first_index = self._src.first_index
        self.last_index = self._src.last_index

    def read_window(self, idx: np.ndarray, device: torch.device) -> torch.Tensor:
        from sdate.tr_diffusion.reconstruct import native_window_gpu

        return native_window_gpu(self._src, idx, self.crop, self.axis_col, device)

    def estimate_I0(self, idx_sample: np.ndarray, pct: float = 99.5) -> float:
        from sdate.tr_diffusion.reconstruct import estimate_I0

        return estimate_I0(self._src, idx_sample, self.crop, self.axis_col, pct)


class SequentialFfmpegWindowSource(WindowSource):
    """One persistent ffmpeg process, read forward through the ``.mov`` exactly
    once, with a rolling buffer -- see the module docstring for why this
    matters (a sliding-window scan re-requests heavily overlapping ranges,
    and :class:`FfmpegWindowSource`'s independent per-call decode pays for
    that overlap ~38x over for the default T=21/11/5).

    Requires ``read_window`` to be called with non-decreasing window starts
    (true of the detector's own scan order) -- a window starting before the
    current buffer's start raises, since the persistent stream can't seek
    backward. Call :meth:`trim_before` once per evaluated timestep (the
    detector's ``stream_anomaly_scores`` does this automatically) to drop
    buffered frames no future window can still need; without it the buffer
    grows unboundedly for the length of the run.
    """

    def __init__(self, mov_path: str, crop: Tuple[int, int], axis_col: float,
                height: int, width: int, frame_start: int,
                fps: float = 30.0, ffmpeg: str = "/myhome/bin/ffmpeg"):
        import subprocess
        from pathlib import Path

        from sdate.tr_diffusion.frames import load_norm_sidecar

        self.mov_path = str(mov_path)
        self.crop = tuple(crop)
        self.axis_col = float(axis_col)
        self.height, self.width = int(height), int(width)
        self._frame_bytes = 2 * self.height * self.width
        side = load_norm_sidecar(mov_path)
        self.per_frame_min = side["per_frame_min"]
        self.per_frame_max = side["per_frame_max"]
        self.first_index = 0
        self.last_index = int(self.per_frame_min.shape[0])
        self.fps = float(fps)

        ffmpeg = ffmpeg if Path(ffmpeg).exists() else "ffmpeg"
        t = (int(frame_start) + 0.5) / self.fps
        # No `-frames:v` cap and no `check=True` (long-lived Popen, not a
        # one-shot `run`) -- read forward for as long as the caller keeps
        # asking, matching extract_frames.py's own streaming-decode pattern.
        # stderr=DEVNULL: closing the pipe before ffmpeg reaches EOF (the
        # normal case -- a run almost never needs every frame up to the true
        # end of the file) makes it log a harmless "Broken pipe" muxer error
        # on exit; nothing our own error handling here depends on.
        self._proc = subprocess.Popen(
            [ffmpeg, "-v", "error", "-ss", f"{t:.6f}", "-i", self.mov_path,
             "-pix_fmt", "gray16le", "-f", "rawvideo", "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        self._decoded_up_to = int(frame_start)   # exclusive upper bound of what's been pulled off stdout
        self._chunks: List[Tuple[int, np.ndarray]] = []  # [(global_start, (n,H,W) uint16), ...]

    def _advance_to(self, hi: int) -> None:
        need = hi - self._decoded_up_to
        if need <= 0:
            return
        raw = self._proc.stdout.read(self._frame_bytes * need)
        got = len(raw) // self._frame_bytes
        if got < need:
            raise EOFError(f"{self.mov_path}: ffmpeg stream ended at frame "
                           f"{self._decoded_up_to + got} (wanted up to {hi})")
        arr = np.frombuffer(raw, np.uint16).reshape(need, self.height, self.width)
        self._chunks.append((self._decoded_up_to, arr))
        self._decoded_up_to += need

    def _slice(self, lo: int, hi: int) -> np.ndarray:
        parts = []
        for cstart, arr in self._chunks:
            cend = cstart + arr.shape[0]
            if cend <= lo or cstart >= hi:
                continue
            parts.append(arr[max(lo, cstart) - cstart: min(hi, cend) - cstart])
        return np.concatenate(parts, axis=0)

    def read_window(self, idx: np.ndarray, device: torch.device) -> torch.Tensor:
        from sdate.tr_diffusion.reconstruct import _crop_gpu

        lo, hi = int(idx[0]), int(idx[-1]) + 1
        buf_start = self._chunks[0][0] if self._chunks else self._decoded_up_to
        if lo < buf_start:
            raise ValueError(f"window start {lo} precedes the buffered range (starts at {buf_start}) "
                             "-- SequentialFfmpegWindowSource requires non-decreasing window starts")
        self._advance_to(hi)
        raw = self._slice(lo, hi)
        x = torch.from_numpy(raw.copy()).to(device).float()  # .copy(): buffer slices aren't writable
        fmin = torch.as_tensor(self.per_frame_min[lo:hi], device=device).view(-1, 1, 1)
        fmax = torch.as_tensor(self.per_frame_max[lo:hi], device=device).view(-1, 1, 1)
        x = x / 65535.0 * (fmax - fmin).clamp_min(1e-6) + fmin
        return _crop_gpu(x, self.crop[0], self.crop[1], self.axis_col)

    def trim_before(self, keep_from: int) -> None:
        new_chunks = []
        for cstart, arr in self._chunks:
            cend = cstart + arr.shape[0]
            if cend <= keep_from:
                continue
            if cstart < keep_from:
                arr = arr[keep_from - cstart:]
                cstart = keep_from
            new_chunks.append((cstart, arr))
        self._chunks = new_chunks

    def estimate_I0(self, idx_sample: np.ndarray, pct: float = 99.5) -> float:
        # A sparse, spread-out sample is fundamentally at odds with a single
        # forward pass -- use a handful of independent one-off seeks instead
        # (this runs ONCE, before the persistent stream/buffer exists yet).
        from sdate.tr_diffusion.frames import FfmpegFrameSource
        from sdate.tr_diffusion.reconstruct import estimate_I0

        probe = FfmpegFrameSource(self.mov_path, fps=self.fps, window=1, cache_size=len(idx_sample) + 8,
                                  height=self.height, width=self.width)
        return estimate_I0(probe, idx_sample, self.crop, self.axis_col, pct)

    def close(self) -> None:
        self._proc.stdout.close()
        self._proc.wait()


class FfmpegWindowSource(WindowSource):
    """Reads directly from the raw ``.mov``, one bulk ffmpeg decode per window.

    No pre-extraction/memmap, no extra disk. Each ``read_window`` call does a
    single ffmpeg subprocess (``-ss <seek> -i <mov> -frames:v <n> ... pipe:1``,
    via :class:`sdate.tr_diffusion.frames.FfmpegFrameSource`'s own bulk-decode
    path -- NOT its default per-frame ``.get()``/LRU-cache path, which would
    mean hundreds of separate ffmpeg subprocess calls for one multi-thousand-
    frame window) -- exactly the same decode this project already uses for
    calibration-average reads, just windowed instead of whole-file.

    This is CPU/software-decode bound (see the module's own README note on
    this project's ffmpeg build -- no hardware video decode compiled in), so
    every window pays real wall-clock proportional to its frame count, unlike
    :class:`MovWindowSource`'s near-free memmap reads. Use
    :meth:`sw_anomaly_det.detector` 's per-window ``read_seconds``/
    ``recon_seconds`` split to see how much of the budget this is actually
    costing on a given dataset before committing to a long run.
    """

    def __init__(self, mov_path: str, crop: Tuple[int, int], axis_col: float,
                height: int, width: int, fps: float = 30.0, ffmpeg: str = "/myhome/bin/ffmpeg"):
        from sdate.tr_diffusion.frames import FfmpegFrameSource, load_norm_sidecar

        # window=1/cache_size=1: this instance is only ever used for its
        # bulk `_decode_window` and for `estimate_I0`'s sparse `.get()` calls,
        # never for the per-frame LRU-cache convention -- no reason to hold a
        # bigger cache resident against a big multi-window run's memory budget.
        # height/width MUST be the dataset's real native frame size (e.g. the
        # DatasetProfile's own height/width) -- FfmpegFrameSource defaults to
        # wunderkerze2's size and silently decodes zero frames otherwise (see
        # its own docstring/comment).
        self._src = FfmpegFrameSource(mov_path, ffmpeg=ffmpeg, fps=fps, window=1, cache_size=1,
                                      height=height, width=width)
        self.mov_path = str(mov_path)
        self.crop = tuple(crop)
        self.axis_col = float(axis_col)
        self.first_index = 0
        self.last_index = self._src.num_frames
        side = load_norm_sidecar(mov_path)
        self.per_frame_min = side["per_frame_min"]
        self.per_frame_max = side["per_frame_max"]

    def read_window(self, idx: np.ndarray, device: torch.device) -> torch.Tensor:
        from sdate.tr_diffusion.reconstruct import _crop_gpu

        lo, hi = int(idx[0]), int(idx[-1]) + 1
        raw = self._src._decode_window(lo, hi - lo)  # (n, H, W) uint16, one ffmpeg call
        x = torch.from_numpy(raw.copy()).to(device).float()  # .copy(): decode buffer isn't writable
        fmin = torch.as_tensor(self.per_frame_min[lo:hi], device=device).view(-1, 1, 1)
        fmax = torch.as_tensor(self.per_frame_max[lo:hi], device=device).view(-1, 1, 1)
        x = x / 65535.0 * (fmax - fmin).clamp_min(1e-6) + fmin
        return _crop_gpu(x, self.crop[0], self.crop[1], self.axis_col)

    def estimate_I0(self, idx_sample: np.ndarray, pct: float = 99.5) -> float:
        from sdate.tr_diffusion.reconstruct import estimate_I0

        return estimate_I0(self._src, idx_sample, self.crop, self.axis_col, pct)
