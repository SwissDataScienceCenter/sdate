#!/usr/bin/env python3
"""Pre-extract a contiguous frame range of the ``.mov`` to a uint16 memmap.

Random ffmpeg seeks over the 55 GB stream are slow; for real training extract
the working range once to a flat ``uint16`` file that
:class:`sdate.tr_diffusion.frames.MemmapFrameSource` memmaps for instant access.
The *decoded* (per-frame-normalised) values are stored as-is; counts are
recovered on access via the ``.norm.npz`` sidecar, so this file stays exact and
half the size of float32.

    python -m sdate.tr_diffusion.extract_frames \
        --mov /myhome/data/sdate/shared/time_resolved/212_Wunderkerze2/212_Wunderkerze2.mov \
        --out /myhome/data/sdate/shared/time_resolved/212_Wunderkerze2/frames_400k_600k.u16 \
        --frame_start 400000 --frame_end 600000
"""

from __future__ import annotations

import subprocess
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from .frames import _DEFAULT_FFMPEG, load_norm_sidecar
from .geometry import FRAME_H, FRAME_W


def main() -> None:
    p = ArgumentParser(description="Extract a .mov frame range to a uint16 memmap.")
    p.add_argument("--mov", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--frame_start", type=int, default=400_000)
    p.add_argument("--frame_end", type=int, default=600_000)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--height", type=int, default=FRAME_H)
    p.add_argument("--width", type=int, default=FRAME_W)
    p.add_argument("--ffmpeg", type=str, default=_DEFAULT_FFMPEG)
    args = p.parse_args()

    side = load_norm_sidecar(args.mov)
    end = min(args.frame_end, int(side["per_frame_min"].shape[0]))
    n = end - args.frame_start
    if n <= 0:
        raise ValueError("empty frame range")
    h, w = args.height, args.width
    frame_bytes = 2 * h * w

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(out, dtype=np.uint16, mode="w+", shape=(n, h, w))

    ffmpeg = args.ffmpeg if Path(args.ffmpeg).exists() else "ffmpeg"
    t = (args.frame_start + 0.5) / args.fps
    proc = subprocess.Popen(
        [ffmpeg, "-v", "error", "-ss", f"{t:.6f}", "-i", args.mov,
         "-frames:v", str(n), "-pix_fmt", "gray16le", "-f", "rawvideo", "pipe:1"],
        stdout=subprocess.PIPE,
    )
    got = 0
    while got < n:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        mm[got] = np.frombuffer(buf, np.uint16).reshape(h, w)
        got += 1
        if got % 5000 == 0:
            print(f"  extracted {got}/{n}")
    proc.stdout.close()
    proc.wait()
    mm.flush()

    if got != n:
        print(f"WARNING: decoded {got} frames, expected {n}")
    np.savez(
        out.with_suffix(".meta.npz"),
        start_frame=args.frame_start, num_frames=got, height=h, width=w,
        mov=str(args.mov),
    )
    print(f"Wrote {got} frames -> {out}  (+ {out.with_suffix('.meta.npz').name})")


if __name__ == "__main__":
    main()
