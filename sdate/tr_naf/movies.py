"""HEVC movie export for time-resolved NAF reconstructions.

Mirrors the movie generation in ``scripts/inference_time_resolved.py``: for a few
evenly-spaced slices, write one HEVC (10-bit grayscale) ``.mov`` per slice that
shows that slice **evolving over scan time**.  Unlike the DDIM script (one saved
volume per frame), the NAF field is evaluated at each time to produce the frame.

Typical labels:
* ``recon``  — TR-NAF volume evaluated at each time ``t``.
* ``gt``     — ground-truth volume at each time.
* ``sw_fbp`` — the (static) sliding-window FBP baseline, repeated for every time
  so it plays alongside the others and visibly does *not* move.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


def _slice_indices(depth: int, k: int) -> List[int]:
    if k >= depth:
        return list(range(depth))
    return np.linspace(0, depth - 1, k, dtype=int).tolist()


def generate_slice_movies(
    movie_dir,
    times: Sequence[float],
    recon_volumes: Optional[List[torch.Tensor]] = None,
    gt_volumes: Optional[List[torch.Tensor]] = None,
    sw_fbp=None,
    num_movie_slices: int = 5,
    fps: int = 5,
    crf: int = 18,
    vmax: Optional[float] = None,
    slice_axis: int = 0,
    prefix: str = "",
    verbose: bool = True,
) -> List[Path]:
    """Write per-slice HEVC movies over time for the provided volume series.

    Parameters
    ----------
    movie_dir : output directory (created if needed).
    times : the ``T`` scan times (only its length is used for static series).
    recon_volumes, gt_volumes : lists of ``(X, Y, Z)`` volumes, one per time.
    sw_fbp : the sliding-window FBP baseline — either a list of ``(X, Y, Z)``
        volumes (one per time, dynamic) or a single ``(X, Y, Z)`` volume
        (static, repeated across all times).
    num_movie_slices : number of evenly-spaced slices along ``slice_axis``.
    vmax : intensity divisor before clamping to ``[0, 1]``.  If ``None``, taken as
        the max over the ground-truth (else recon) volumes so all labels share
        one scale for fair side-by-side playback.
    Returns the list of written ``.mov`` paths.
    """
    from sdate.stream_hvec import HevcGray10Streamer
    from sdate.stream_hvec.stream_gray10 import EncoderParams

    movie_dir = Path(movie_dir)
    movie_dir.mkdir(parents=True, exist_ok=True)
    T = len(times)

    series: Dict[str, List[torch.Tensor]] = {}
    if recon_volumes is not None:
        series["recon"] = list(recon_volumes)
    if gt_volumes is not None:
        series["gt"] = list(gt_volumes)
    if sw_fbp is not None:
        # Accept a per-frame list (dynamic) or a single volume (static, repeated).
        if isinstance(sw_fbp, torch.Tensor):
            series["sw_fbp"] = [sw_fbp] * T
        else:
            series["sw_fbp"] = list(sw_fbp)

    if not series:
        raise ValueError("Provide at least one of recon_volumes / gt_volumes / sw_fbp.")

    # Shared intensity scale.
    if vmax is None:
        ref = gt_volumes if gt_volumes is not None else recon_volumes
        if ref is None:
            ref = [sw_fbp]
        vmax = float(torch.stack([v.float() for v in ref]).max())
    vmax = max(vmax, 1e-8)

    depth = next(iter(series.values()))[0].shape[slice_axis]
    sidx = _slice_indices(depth, num_movie_slices)
    params = EncoderParams(fps=fps, crf_sw=crf, preset_sw="medium", force_software=True)
    if verbose:
        print(f"Movies: labels={list(series)} slices={sidx} times={T} -> {movie_dir}")

    written: List[Path] = []
    for label, vols in series.items():
        for s in sidx:
            name = f"{prefix}{label}_slice_{s:03d}.mov"
            streamer = HevcGray10Streamer(
                base_path=movie_dir, segment_prefix=f"{prefix}{label}_s{s:03d}", params=params
            )
            with streamer.start_segment(outfile=name):
                for vol in vols:
                    frame = vol.select(slice_axis, s).float()
                    frame = (frame / vmax).clamp(0.0, 1.0).cpu()
                    streamer.append_frame(frame)
            written.append(movie_dir / name)
            if verbose:
                print(f"  wrote {name}")
    return written


def generate_comparison_movie(movie_dir, times: Sequence[float],
                              panels: "list[tuple]", slice_indices=None,
                              num_movie_slices: int = 3, fps: int = 5, crf: int = 18,
                              vmax: Optional[float] = None, slice_axis: int = 0,
                              sep: int = 2, prefix: str = "cmp_",
                              verbose: bool = True) -> List[Path]:
    """Write side-by-side movies: each frame stacks several panels horizontally.

    ``panels`` : list of ``(label, volumes)`` in left-to-right order, where
    ``volumes`` is a list of ``(X,Y,Z)`` per time (dynamic) or a single ``(X,Y,Z)``
    (static, repeated).  Panels share one intensity scale.  Label order is encoded
    in the filename (``prefix + labels joined by '-'``); a bright separator column
    of width ``sep`` divides panels.
    """
    from sdate.stream_hvec import HevcGray10Streamer
    from sdate.stream_hvec.stream_gray10 import EncoderParams

    movie_dir = Path(movie_dir); movie_dir.mkdir(parents=True, exist_ok=True)
    T = len(times)
    series = []
    for label, vols in panels:
        series.append((label, [vols] * T if isinstance(vols, torch.Tensor) else list(vols)))

    if vmax is None:
        ref = next((v for lab, v in series if lab.lower() in ("gt", "ground truth")), series[0][1])
        vmax = float(torch.stack([v.float() for v in ref]).max())
    vmax = max(vmax, 1e-8)

    depth = series[0][1][0].shape[slice_axis]
    sidx = slice_indices or _slice_indices(depth, num_movie_slices)
    params = EncoderParams(fps=fps, crf_sw=crf, preset_sw="medium", force_software=True)
    labels = "-".join(lab for lab, _ in series)
    if verbose:
        print(f"Comparison movie: panels [{labels}] slices={sidx} times={T} -> {movie_dir}")

    written: List[Path] = []
    for s in sidx:
        name = f"{prefix}{labels}_slice_{s:03d}.mov"
        streamer = HevcGray10Streamer(base_path=movie_dir, segment_prefix=f"{prefix}s{s:03d}", params=params)
        with streamer.start_segment(outfile=name):
            for t in range(T):
                cols = []
                for j, (_, vols) in enumerate(series):
                    frame = (vols[t].select(slice_axis, s).float() / vmax).clamp(0.0, 1.0)
                    cols.append(frame)
                    if sep and j < len(series) - 1:
                        cols.append(torch.ones(frame.shape[0], sep, device=frame.device))  # divider
                streamer.append_frame(torch.cat(cols, dim=1).cpu())
        written.append(movie_dir / name)
        if verbose:
            print(f"  wrote {name}")
    return written


def movies_from_result(result: Dict, times: Sequence[float], movie_dir,
                       gt_volumes: Optional[List[torch.Tensor]] = None,
                       sw_fbp: Optional[torch.Tensor] = None, **kwargs) -> List[Path]:
    """Convenience: evaluate a TR-NAF ``result`` at each time, then write movies."""
    from .reconstruct import reconstruct_volume_at

    recon = [reconstruct_volume_at(result, float(t)) for t in times]
    return generate_slice_movies(
        movie_dir, times, recon_volumes=recon, gt_volumes=gt_volumes,
        sw_fbp=sw_fbp, **kwargs
    )
