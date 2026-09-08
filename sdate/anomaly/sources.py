"""Example orchestrators that feed frames into :class:`AnomalyDetector`.

These live *outside* the detector core on purpose — the library never decodes
anything itself.  Swap this for an HDF5 reader or a live detector socket; all it
must do is yield ``(frame_idx, frame_HxW)`` pairs in acquisition order.

``mov_frame_stream`` streams the ``212_Wunderkerze2`` HEVC ``.mov`` via a single
ffmpeg pipe and recovers original photon counts from the per-frame min/max
sidecar (same recovery as ``make_movies.py``).
"""

from __future__ import annotations

import subprocess
from typing import Iterator, Optional, Tuple

import numpy as np

FFMPEG = "/myhome/bin/ffmpeg"
MOV = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2/212_Wunderkerze2.mov"
NORM = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2/212_Wunderkerze2.norm.npz"
W, H, FPS = 528, 128, 30.0


def mov_frame_stream(start: int, n: int, mov: str = MOV, norm: Optional[str] = NORM,
                     ffmpeg: str = FFMPEG) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(global_frame_idx, frame)`` for ``n`` frames from ``start``.

    If ``norm`` is given, frames are rescaled from the per-frame-normalised video
    back to original counts using the ``.norm.npz`` sidecar.
    """
    frame_bytes = 2 * W * H
    pmin = pmax = None
    if norm is not None:
        sc = np.load(norm)
        pmin = sc["per_frame_min"].astype(np.float64)
        pmax = sc["per_frame_max"].astype(np.float64)

    t = (start + 0.5) / FPS
    proc = subprocess.Popen(
        [ffmpeg, "-v", "error", "-ss", f"{t:.6f}", "-i", mov,
         "-frames:v", str(n), "-pix_fmt", "gray16le", "-f", "rawvideo", "pipe:1"],
        stdout=subprocess.PIPE)
    k = start
    try:
        while k < start + n:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            g = np.frombuffer(buf, np.uint16).reshape(H, W).astype(np.float64)
            if pmin is not None:
                g = g / 65535.0 * (pmax[k] - pmin[k]) + pmin[k]   # original counts
            yield k, g
            k += 1
    finally:
        proc.stdout.close()
        proc.wait()
