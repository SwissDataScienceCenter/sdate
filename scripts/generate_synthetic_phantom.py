#!/usr/bin/env python3
"""Generate the ``synthetic_v1`` time-resolved phantom dataset.

Renders :func:`sdate.tr_diffusion.phantom.default_scene` with the exact
geometry of the ``synthetic_v1`` profile (see ``profiles.py``) and writes it
in the SAME on-disk layout ``sdate.tr_diffusion.frames.MemmapFrameSource``
expects for a real acquisition: a decoded ``uint16`` memmap (+ ``.meta.npz``)
and a ``.norm.npz`` sidecar next to a (non-existent, never decoded) ``mov_path``
-- so the rest of the tr_diffusion pipeline (data.py/train.py/reconstruct.py)
loads this dataset with zero code changes, exactly like a calibrated real one.

Unlike a real acquisition, the "native" frames written here are the CLEAN
(noise-free) analytic projections -- the whole point of a synthetic dataset is
exact ground truth. ``--extra_noise_dose`` at train/eval time then controls
noise the same way it does for the real datasets.

    python -m scripts.generate_synthetic_phantom --frame_end 100000
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sdate.tr_diffusion.phantom import attenuation_to_counts, default_scene, render_projections
from sdate.tr_diffusion.profiles import REGISTRY


def main() -> None:
    p = ArgumentParser(description=__doc__)
    p.add_argument("--profile", type=str, required=True, help="name registered in profiles.REGISTRY")
    p.add_argument("--frame_start", type=int, default=None, help="default: profile's frame_start")
    p.add_argument("--frame_end", type=int, default=None, help="default: profile's frame_end")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    prof = REGISTRY[args.profile]
    frame_start = args.frame_start if args.frame_start is not None else prof.frame_start
    frame_end = args.frame_end if args.frame_end is not None else prof.frame_end
    n = frame_end - frame_start
    H, W = prof.height, prof.width
    I0 = prof.norm_range[1]
    device = torch.device(args.device)

    caps = default_scene(frame_start=frame_start, frame_end=frame_end, height=H, width=W, seed=args.seed)
    print(f"{len(caps)} capsules, {n} frames [{frame_start},{frame_end}), {H}x{W}, device={device}")

    mov_path = Path(prof.mov_path)
    mov_path.parent.mkdir(parents=True, exist_ok=True)
    memmap_path = Path(prof.memmap_path)

    mm = np.memmap(memmap_path, dtype=np.uint16, mode="w+", shape=(n, H, W))
    fmin, fmax = prof.norm_range
    span = fmax - fmin

    for start in range(0, n, args.chunk):
        end = min(start + args.chunk, n)
        frame_idx = np.arange(frame_start + start, frame_start + end)
        with torch.no_grad():
            pline = render_projections(caps, frame_idx, H, W, prof.deg_per_frame, device=device)
            counts = attenuation_to_counts(pline, I0=I0)
            decoded = ((counts.clamp(fmin, fmax) - fmin) / span * 65535.0).round().to(torch.int32)
            decoded = decoded.clamp(0, 65535).to(torch.uint16).cpu().numpy()
        mm[start:end] = decoded
        if (start // args.chunk) % 10 == 0:
            print(f"  {end}/{n}")
    mm.flush()
    del mm

    np.savez(
        mov_path.with_suffix(".norm.npz"),
        per_frame_min=np.full(n, fmin, dtype=np.float32),
        per_frame_max=np.full(n, fmax, dtype=np.float32),
        start_frame=frame_start, end_frame=frame_end,
    )
    np.savez(
        memmap_path.with_suffix(".meta.npz"),
        start_frame=frame_start, num_frames=n, height=H, width=W, mov=str(mov_path),
    )
    print(f"wrote {memmap_path} ({memmap_path.stat().st_size / 1e9:.2f} GB) "
          f"+ {mov_path.with_suffix('.norm.npz').name} + {memmap_path.with_suffix('.meta.npz').name}")


if __name__ == "__main__":
    main()
