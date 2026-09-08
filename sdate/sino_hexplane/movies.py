"""Movie export for sinogram-domain HexPlane projections.

Sweeps over the projection (angle) axis and writes a side-by-side panel movie
(e.g. ``noisy | clean | predicted``), mirroring the HEVC 10-bit-grey style of
:mod:`sdate.tr_naf.movies` (``HevcGray10Streamer``, needs ffmpeg+libx265 on
PATH).  If HEVC encoding is unavailable, falls back to a plain GIF so an output
is always produced.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch


def _panel_frame(sinos: Sequence[torch.Tensor], a: int, vmax: float,
                 sep: int) -> torch.Tensor:
    """Side-by-side ``(R, sum C)`` frame at angle ``a`` in ``[0, 1]``, bright dividers."""
    cols: List[torch.Tensor] = []
    for j, s in enumerate(sinos):
        f = (s[a].float() / vmax).clamp(0.0, 1.0)
        cols.append(f)
        if sep and j < len(sinos) - 1:
            cols.append(torch.ones(f.shape[0], sep, device=f.device))
    return torch.cat(cols, dim=1).cpu()


def generate_projection_movie(
    out_base,
    panels: Sequence[Tuple[str, torch.Tensor]],
    fps: int = 5,
    vmax: Optional[float] = None,
    sep: int = 2,
    crf: int = 18,
    verbose: bool = True,
) -> Path:
    """Write a projection sweep movie: one video frame per angle.

    Parameters
    ----------
    out_base : path stem (directory + basename, no extension); parent is created.
    panels : list of ``(label, sino)`` in left-to-right order, each ``sino`` an
        ``(A, R, C)`` tensor sharing the same angle count ``A``.
    fps, crf : encoder settings.  vmax : shared intensity divisor (default = max
        over all panels).  sep : bright divider width between panels.

    Returns the written path (``.mov`` if HEVC succeeded, else a ``.gif``).
    """
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    labels = [l for l, _ in panels]
    sinos = [s for _, s in panels]
    A = sinos[0].shape[0]
    if any(s.shape[0] != A for s in sinos):
        raise ValueError("all panels must share the same angle count A")
    if vmax is None:
        vmax = max(float(s.max()) for s in sinos)
    vmax = max(float(vmax), 1e-8)
    tag = "-".join(labels)

    # Primary: HEVC 10-bit grey, matching sdate.tr_naf.movies.
    try:
        from sdate.stream_hvec import HevcGray10Streamer
        from sdate.stream_hvec.stream_gray10 import EncoderParams

        name = f"{out_base.name}_{tag}.mov"
        params = EncoderParams(fps=fps, crf_sw=crf, preset_sw="medium", force_software=True)
        streamer = HevcGray10Streamer(base_path=out_base.parent,
                                      segment_prefix=out_base.name, params=params)
        with streamer.start_segment(outfile=name):
            for a in range(A):
                streamer.append_frame(_panel_frame(sinos, a, vmax, sep))
        path = out_base.parent / name
        if verbose:
            print(f"wrote HEVC movie {path}  ({A} frames, panels [{tag}])")
        return path
    except Exception as e:  # ffmpeg/libx265 missing, etc.
        if verbose:
            print(f"[HEVC unavailable] {type(e).__name__}: {e} -> writing GIF fallback")

    # Fallback: GIF (needs only Pillow).
    import imageio
    frames_np = [(_panel_frame(sinos, a, vmax, sep).numpy() * 255).astype(np.uint8)
                 for a in range(A)]
    path = out_base.parent / f"{out_base.name}_{tag}.gif"
    imageio.mimsave(path, frames_np, duration=1.0 / max(fps, 1))
    if verbose:
        print(f"wrote GIF {path}  ({A} frames, panels [{tag}])")
    return path
