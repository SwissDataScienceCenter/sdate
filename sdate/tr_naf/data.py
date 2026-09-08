"""Build a synthetic multi-sweep time-resolved limited-angle acquisition.

Each tif timestep is a full 3-D volume of the object frozen at one instant.  We
sample ``num_frames`` timesteps, assign each a normalised time ``t in [0, 1]``
and a **rotating angular wedge**; the wedges advance by ``angle_range_deg`` per
frame so that over the whole scan they sweep 0-180 several times (parallel-beam
projections are 180-periodic, so wrapping past 180 is just repeated coverage at
a later time).  Each frame's volume is forward-projected over its own wedge to
give that frame's limited-angle sinogram.

This mirrors the geometry in ``notebooks/test_time_resolved_ddim.ipynb`` but
returns a flat per-frame list carrying ``(angles, t)`` so the reconstruction is
agnostic to whether the data is binned (one ``t`` per frame, as here) or truly
continuous (one ``t`` per projection) later on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Frame:
    """One acquired frame.

    Two measurement representations, one populated per regime:
    * ``sino``   — line integrals (limited-angle / clean regime; MSE loss).
    * ``counts`` — raw detector counts (noise regime; count-space Poisson/Anscombe
      loss).  The flat (I0), dark, gain and read-noise live in ``meta`` since they
      are shared across frames (same beam/detector).
    """
    t_norm: float               # normalised scan time in [0, 1]
    angles_deg: np.ndarray      # (V,) projection angles for this frame
    true_volume: torch.Tensor   # (X, Y, Z) ground-truth object at this time (for eval)
    sino: Optional[torch.Tensor] = None      # (V, R, C) line integrals
    counts: Optional[torch.Tensor] = None    # (V, R, C) raw counts


def resample_to_cube(volume: torch.Tensor, cube: int) -> torch.Tensor:
    """Trilinearly resample a ``(D, H, W)`` volume to ``(cube, cube, cube)``."""
    v = volume.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    v = F.interpolate(v, size=(cube, cube, cube), mode="trilinear", align_corners=False)
    return v.squeeze(0).squeeze(0)


def build_acquisition(
    data_path: Union[str, Path],
    num_frames: int = 25,
    angle_range_deg: float = 36.0,
    angle_start_deg: float = 0.0,
    num_full_projs: int = 1000,
    full_angle_span_deg: float = 180.0,
    timestep_skip: int = 0,
    cube_size: int = 128,
    det_spacing_mm: float = 1.0,
    lamino_angle_deg: float = 0.0,
    full_boundary_frames: int = 0,
    normalize_range: Optional[Tuple[float, float]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[List[Frame], dict]:
    """Return ``(frames, meta)``.

    ``num_frames * angle_range_deg / full_angle_span_deg`` = number of 180-sweeps.
    Projection density matches the notebook: ``num_full_projs`` over the full span,
    so each wedge gets ``round(angle_range_deg / full_angle_span_deg * num_full_projs)``
    projections.

    ``full_boundary_frames`` : the first and last this-many frames are acquired over
    the **full** ``[0, full_angle_span_deg]`` range (``num_full_projs`` projections)
    instead of their limited-angle wedge.  Physically justified when the object is
    static before the dynamics begin and after they end, so a complete scan of those
    boundary time points is available; this anchors the temporal-spline endpoints
    and removes limited-angle boundary artifacts.
    """
    from ladiff.datasets import TifVolumeSliceDataset
    from astra_torch.lamino import build_lamino_projector

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TifVolumeSliceDataset(
        data_path=data_path,
        file_range=num_frames,
        resize=cube_size,
        normalize_range=normalize_range,
        augment=False,
        skip=timestep_skip,
    )
    if dataset.num_files < num_frames:
        raise ValueError(
            f"Requested {num_frames} frames but only {dataset.num_files} tif files resolved."
        )

    num_la_projs = max(1, round(angle_range_deg / full_angle_span_deg * num_full_projs))
    vol_shape = (cube_size, cube_size, cube_size)
    det_shape = (cube_size, cube_size)  # (R, C) = (D, W)

    frames: List[Frame] = []
    for i in range(num_frames):
        vol = resample_to_cube(dataset.get_volume(i).to(device), cube_size)
        is_boundary = (i < full_boundary_frames) or (i >= num_frames - full_boundary_frames)
        if is_boundary:
            # Static object at this endpoint -> full-angle scan available.
            angles = np.linspace(0.0, full_angle_span_deg, num_full_projs)
        else:
            a_start = angle_start_deg + i * angle_range_deg
            angles = np.linspace(a_start, a_start + angle_range_deg, num_la_projs)
        projector = build_lamino_projector(
            vol_shape=vol_shape, det_shape=det_shape, angles_deg=angles,
            lamino_angle_deg=lamino_angle_deg, det_spacing_mm=det_spacing_mm, device=device,
        )
        with torch.no_grad():
            sino = projector(vol.unsqueeze(0).unsqueeze(0)).squeeze(0)  # (V, R, C)
        t_norm = i / max(num_frames - 1, 1)
        frames.append(Frame(t_norm=t_norm, angles_deg=angles, sino=sino, true_volume=vol))

    meta = dict(
        vol_shape=vol_shape, det_shape=det_shape, cube_size=cube_size,
        det_spacing_mm=det_spacing_mm, lamino_angle_deg=lamino_angle_deg,
        num_frames=num_frames, angle_range_deg=angle_range_deg,
        num_la_projs=num_la_projs, num_sweeps=num_frames * angle_range_deg / full_angle_span_deg,
        full_boundary_frames=full_boundary_frames,
        norm_min=dataset.norm_min, norm_max=dataset.norm_max,
    )
    return frames, meta


def sliding_window_fbp(frames: List[Frame], meta: dict,
                       window: Optional[int] = None,
                       full_angle_span_deg: float = 180.0,
                       filter_type: str = "hann",
                       device: Optional[torch.device] = None) -> List[torch.Tensor]:
    """Dynamic SW-FBP baseline: **one FBP volume per time frame**.

    For frame ``i`` the reconstruction uses a *sliding window* of ``window``
    consecutive frames centred on ``i`` whose wedges together span
    ``full_angle_span_deg`` (~180°), so the baseline tracks motion within the
    window's temporal width (with the usual limited-angle-per-window artifacts).
    ``window`` defaults to ``round(full_angle_span_deg / angle_range_deg)``.
    Near the ends the window is shifted inward to keep its size (and angular
    coverage) constant.  Returns a list of ``(X, Y, Z)`` volumes, one per frame.
    """
    from astra_torch.lamino import fbp_reconstruction_masked

    device = device or frames[0].sino.device
    n = len(frames)
    if window is None:
        window = max(1, round(full_angle_span_deg / meta["angle_range_deg"]))
    window = min(window, n)
    half = window // 2

    vols: List[torch.Tensor] = []
    for i in range(n):
        lo = min(max(0, i - half), n - window)   # keep a full-size window inside [0, n)
        sub = frames[lo:lo + window]
        sino = torch.cat([f.sino for f in sub], dim=0)
        angles = np.concatenate([f.angles_deg for f in sub], axis=0)
        vol = fbp_reconstruction_masked(
            projs_vrc=sino, angles_deg=angles, lamino_angle_deg=meta["lamino_angle_deg"],
            vol_shape=meta["vol_shape"], det_spacing_mm=meta["det_spacing_mm"],
            filter_type=filter_type, device=device,
        ).squeeze(0)
        vols.append(vol)
    return vols


def build_noise_acquisition(
    data_path: Union[str, Path],
    num_frames: int = 25,
    angle_range_deg: float = 180.0,      # full angular range per frame by default
    angle_start_deg: float = 0.0,
    num_full_projs: int = 360,
    full_angle_span_deg: float = 180.0,
    timestep_skip: int = 0,
    cube_size: int = 128,
    det_spacing_mm: float = 1.0,
    lamino_angle_deg: float = 0.0,
    photons: float = 1e4,                # flat-field expected photons I0 (noise/dose knob)
    p_max: float = 2.5,                  # max line integral (sets contrast/transmission range)
    gain: float = 1.0,
    read_noise: float = 0.0,
    dark: float = 0.0,
    normalize_range: Optional[Tuple[float, float]] = None,
    noise_seed: Optional[int] = 0,
    device: Optional[torch.device] = None,
) -> Tuple[List[Frame], dict]:
    """Build a **count-space** acquisition (noise regime).

    Each frame is forward-projected (default full 0-180 range) and turned into raw
    detector **counts** via the Poisson-Gaussian model, keeping the flat (I0), dark,
    gain and read-noise explicit in ``meta``.  This is the exact interface a real
    GigaFrost loader will fill (counts + flat + dark) — the model/loss are agnostic.

    ``photons`` (I0) is the dose knob: large -> clean, small -> photon-starved.
    Line integrals are scaled so the global max = ``p_max`` (keeps ``exp(-p)`` in a
    realistic transmission range); ``meta['scale']`` records the factor so the
    reconstructed mu can be mapped back to normalized units for metrics.
    """
    from ladiff.datasets import TifVolumeSliceDataset
    from astra_torch.lamino import build_lamino_projector
    from .noise import simulate_counts

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device)
    if noise_seed is not None:
        gen.manual_seed(int(noise_seed))

    dataset = TifVolumeSliceDataset(
        data_path=data_path, file_range=num_frames, resize=cube_size,
        normalize_range=normalize_range, augment=False, skip=timestep_skip,
    )
    if dataset.num_files < num_frames:
        raise ValueError(f"Requested {num_frames} frames but only {dataset.num_files} tif files resolved.")

    num_projs = max(1, round(angle_range_deg / full_angle_span_deg * num_full_projs))
    vol_shape = (cube_size, cube_size, cube_size)
    det_shape = (cube_size, cube_size)

    # Pass 1: project every frame, keep clean line integrals + global max for scaling.
    vols, clean_sinos, angles_all = [], [], []
    for i in range(num_frames):
        vol = resample_to_cube(dataset.get_volume(i).to(device), cube_size)
        angles = np.linspace(angle_start_deg, angle_start_deg + angle_range_deg, num_projs)
        projector = build_lamino_projector(
            vol_shape=vol_shape, det_shape=det_shape, angles_deg=angles,
            lamino_angle_deg=lamino_angle_deg, det_spacing_mm=det_spacing_mm, device=device,
        )
        with torch.no_grad():
            sino = projector(vol.unsqueeze(0).unsqueeze(0)).squeeze(0)  # (V, R, C)
        vols.append(vol); clean_sinos.append(sino); angles_all.append(angles)

    global_max = max(float(s.max()) for s in clean_sinos)
    scale = p_max / max(global_max, 1e-8)
    flat = torch.full(det_shape, float(photons), device=device)   # uniform I0 (per-pixel-ready)

    # Pass 2: scale to p_max range and draw counts.
    frames: List[Frame] = []
    for i in range(num_frames):
        p = clean_sinos[i] * scale                                  # (V, R, C) in [0, p_max]
        counts = simulate_counts(p, flat, gain=gain, read_noise=read_noise, dark=dark, generator=gen)
        t_norm = i / max(num_frames - 1, 1)
        frames.append(Frame(t_norm=t_norm, angles_deg=angles_all[i],
                            true_volume=vols[i], counts=counts))

    meta = dict(
        regime="noise", vol_shape=vol_shape, det_shape=det_shape, cube_size=cube_size,
        det_spacing_mm=det_spacing_mm, lamino_angle_deg=lamino_angle_deg,
        num_frames=num_frames, angle_range_deg=angle_range_deg, num_la_projs=num_projs,
        num_sweeps=num_frames * angle_range_deg / full_angle_span_deg,
        photons=photons, p_max=p_max, scale=scale, gain=gain, read_noise=read_noise, dark=dark,
        flat=flat.cpu(), norm_min=dataset.norm_min, norm_max=dataset.norm_max,
    )
    return frames, meta


def per_frame_fbp(frames: List[Frame], meta: dict, filter_type: str = "hann",
                  device: Optional[torch.device] = None) -> List[torch.Tensor]:
    """Baseline for the noise regime: full-angle FBP of each frame's log-corrected counts.

    Classical reconstruction cannot use counts directly, so it log-corrects first
    (``p = -ln((N-dark)/I0)``) then filters — the honest "log-domain FBP" contrast
    to the count-space NAF.  Returns one ``(X, Y, Z)`` volume per frame (in
    normalized mu units, i.e. divided by ``meta['scale']``).
    """
    from astra_torch.lamino import fbp_reconstruction_masked
    from .noise import counts_to_line_integral

    device = device or frames[0].counts.device
    flat = meta["flat"].to(device)
    vols = []
    for f in frames:
        p_meas = counts_to_line_integral(f.counts.to(device), flat, dark=meta["dark"])
        p_meas = p_meas / meta["scale"]                      # back to normalized mu units
        vol = fbp_reconstruction_masked(
            projs_vrc=p_meas, angles_deg=f.angles_deg, lamino_angle_deg=meta["lamino_angle_deg"],
            vol_shape=meta["vol_shape"], det_spacing_mm=meta["det_spacing_mm"],
            filter_type=filter_type, device=device,
        ).squeeze(0)
        vols.append(vol.clamp_min(0.0))
    return vols
