"""Phi-domain joint-FBP context taps for periodic (theta, phi) time-resolved CT.

Direct analogue of the time-domain joint-FBP context-tap mechanism
(``scripts/tr_diffusion_jointfbp_context_cache.py``): instead of T fixed-width
windows staggered in time, this gates on the periodic motion phase ``phi``
with T GROWING-RADIUS windows sharing one center ``phi_t`` (theta is left
completely free -- gating on phi alone already gives near-complete theta
coverage even at the smallest radius, because phi cycles far faster than
theta sweeps: see project memory ``project-tr-diffusion-sewellia-phi-context``).

Design (confirmed with the user):
- ``phi_t`` is snapped to the nearest of ``n_bins`` uniform circular bins over
  ``[0, 2*pi)`` (default 360, i.e. ~1 deg/bin) -- every real projection reuses
  its nearest bin's precomputed T taps rather than getting its own exact-phi
  reconstruction.
- Radius ``r_i = pct_i * 2*pi`` directly (``pct_i`` given as a fraction, not a
  percent). IMPORTANT: circular distance saturates at half the period (pi),
  so any ``pct_i >= 0.5`` selects the ENTIRE dataset regardless of
  ``phi_t`` -- identical, redundant reconstructions. The default 11-value
  progression therefore spans ``(0, 0.5]`` with a single deliberate "global"
  tap at exactly 50% (same spirit as the project's k=21 pseudo-GT: a
  phi-independent, whole-dataset reference channel), rather than wasting
  slots on values above the saturation point.
- Each bin's T reconstructions are reprojected only at the real target
  projections' own exact theta (their nearest-bin group) -- zero angular
  mismatch, matching how the time-domain taps avoid angular mismatch.

Always go through the ``dark``/``flat`` round trip (:func:`reconstruct_phi_gated`
+ :func:`reproject_to_counts`) -- raw counts -> attenuation via
:func:`sdate.tr_diffusion.reconstruct.counts_to_attenuation_flatdark` (real
``-log((count-dark)/(flat-dark))``) -> FBP -> reproject -> back to the input
domain via ``attenuation_to_counts_flatdark`` (``dark + (flat-dark)*exp(-p)``),
so context taps land in the SAME domain as the raw noisy target -- exactly
matching ``scripts/tr_diffusion_jointfbp_context_cache.py`` for wunderkerze2.

For data with no real per-pixel darks/flats (e.g. the Sewellia preview h5
file, which ships an already flat/dark-corrected transmission sinogram --
the ratio step already applied, but NOT yet ``-log``'d into attenuation),
pass trivial identity calibration ``dark=0, flat=1``: the formulas above
reduce to plain ``-log(x)`` going in and plain ``exp(-x)`` coming back out,
which is exactly the missing half of the round trip needed to keep the
context tap in the same (transmission) domain as the target -- there is no
separate "no darks/flats" code path, just a trivial choice of ``dark``/``flat``.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch

TWO_PI = 2 * np.pi

DEFAULT_PCTS: Tuple[float, ...] = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.17, 0.23, 0.30, 0.40, 0.50)


def circular_dist(a, b, period: float = TWO_PI):
    """Elementwise circular distance between ``a`` (array) and ``b`` (scalar or array)."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % period
    return np.minimum(d, period - d)


def phi_bin_centers(n_bins: int = 360, period: float = TWO_PI) -> np.ndarray:
    return (np.arange(n_bins) + 0.5) * period / n_bins


def snap_to_bins(phi: np.ndarray, n_bins: int = 360, period: float = TWO_PI) -> np.ndarray:
    """Nearest-bin index (no interpolation) for every value in ``phi``, circular."""
    phi = np.asarray(phi) % period
    bin_width = period / n_bins
    return (np.floor(phi / bin_width + 0.5).astype(np.int64)) % n_bins


def phi_gate_mask(phase: np.ndarray, phi_center: float, radius: float, period: float = TWO_PI) -> np.ndarray:
    return circular_dist(phase, phi_center, period) <= radius


def radii_from_pcts(pcts: Sequence[float] = DEFAULT_PCTS, period: float = TWO_PI) -> np.ndarray:
    return np.asarray(pcts, dtype=float) * period


def reconstruct_phi_gated(
    sinogram: np.ndarray,
    theta_deg: np.ndarray,
    phase: np.ndarray,
    phi_center: float,
    radius: float,
    dark: torch.Tensor,
    flat: torch.Tensor,
    device: torch.device,
    det_bin: int = 1,
    vol_shape: Optional[Tuple[int, int, int]] = None,
    period: float = TWO_PI,
    max_views: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], int]:
    """FBP-reconstruct the volume from all projections within ``radius`` of ``phi_center``.

    Returns ``(volume, n_selected)``; ``volume`` is ``None`` if no projections fall
    in the gate. ``theta_deg`` (all of it, not just the gate's own angular span)
    is used as-is for the selected subset -- no gap-filling/interpolation.

    ``dark``/``flat``: per-pixel calibration maps (or trivial scalars ``0``/``1``
    for already-corrected data -- see module docstring). ``sinogram`` is
    converted to attenuation via
    :func:`sdate.tr_diffusion.reconstruct.counts_to_attenuation_flatdark`
    before FBP. Pair with :func:`reproject_to_counts` to bring the reprojected
    context back into the same domain as the raw target.

    ``max_views``: ASTRA's CUDA backend has a hard per-reconstruction view-count
    ceiling -- confirmed empirically on this project's detector/volume shape to
    sit between 15999 views (succeeds) and 20000 (``AstraError: ... Failed to
    allocate 556x20000x6 GPU array``, a CUDA 3D-array allocation failure, not a
    timing/memory-budget issue -- reconstruction time scales ~linearly with view
    count and stays well under 1s even near this ceiling). When ``n_sel`` exceeds
    ``max_views``, deterministically subsample via a fixed stride (preserves
    angular spread rather than a random subset) down to ``max_views`` views.
    """
    from sdate.tr_diffusion import reconstruct as R

    mask = phi_gate_mask(phase, phi_center, radius, period)
    idx = np.flatnonzero(mask)
    n_sel = int(idx.size)
    if n_sel == 0:
        return None, 0
    if max_views is not None and n_sel > max_views:
        stride = n_sel / max_views
        pick = np.floor(np.arange(max_views) * stride).astype(np.int64)
        idx = idx[pick]

    p = torch.from_numpy(np.ascontiguousarray(sinogram[idx])).to(device=device, dtype=torch.float32)
    p = R.counts_to_attenuation_flatdark(p, dark, flat)
    angles = np.asarray(theta_deg)[idx].astype(np.float64)
    vol = R.reconstruct(p, angles, det_bin=det_bin, method="fbp", vol_shape=vol_shape, device=device)
    return vol, n_sel


def reconstruct_phi_gated_groups(
    sinogram: np.ndarray,
    theta_deg: np.ndarray,
    phase: np.ndarray,
    phi_center: float,
    radius: float,
    dark: torch.Tensor,
    flat: torch.Tensor,
    device: torch.device,
    det_bin: int = 1,
    vol_shape: Optional[Tuple[int, int, int]] = None,
    period: float = TWO_PI,
    group_size: int = 900,
    max_groups: int = 4,
    clamp: bool = True,
    exclude_idx: Optional[np.ndarray] = None,
) -> Tuple[list, int]:
    """Like :func:`reconstruct_phi_gated`, but exploits ALL available redundancy
    within the gate by running up to ``max_groups`` DISJOINT full reconstructions
    of exactly ``group_size`` views each, instead of one reconstruction capped
    down to a single view-count ceiling (which throws away everything beyond
    that ceiling).

    ``exclude_idx``: optional raw projection indices to remove from the gate
    before grouping (e.g. the views already consumed by an independent
    Noise2Noise anchor pair built from the SAME phi center) -- since anchor
    and context gates here are nested disks around one center, without this
    the context would silently contain the anchor's own photon draws,
    breaking the independence a Noise2Noise-style setup depends on. Default
    ``None`` preserves the original behaviour exactly (no exclusion).

    ``group_size=900`` is the empirically-confirmed safe per-reconstruction view
    ceiling at ``det_bin=1`` (native full resolution) on an 80GB A100 -- 900
    views succeeds using ~36GB, 1000 views OOMs wanting 70GB+. The falloff is
    SHARP, not gradual (confirmed via a systematic probe), so ``group_size``
    must never be exceeded by any single reconstruction call -- this is why
    groups are capped at exactly ``group_size`` via a global stride-then-split
    rather than dividing the available views evenly across ``m`` groups (which
    would push every group's size above the ceiling once ``n_sel`` isn't an
    exact multiple of ``group_size``).

    When the gate has fewer than ``group_size`` real projections (the narrow
    end of a T-level progression), falls back to a single reconstruction using
    all of them -- matching :func:`reconstruct_phi_gated`'s behaviour there.

    Returns ``(volumes, n_sel)``: ``volumes`` is a list of 1..``max_groups``
    independently-reconstructed tensors (each from a disjoint, stride-sampled
    subset spanning the full gate, so each group keeps good angular/phase
    coverage rather than being a contiguous time-block); ``n_sel`` is the TOTAL
    number of real projections available in the gate (diagnostic -- not all of
    it is necessarily used if it exceeds ``max_groups * group_size``).
    """
    from sdate.tr_diffusion import reconstruct as R

    mask = phi_gate_mask(phase, phi_center, radius, period)
    idx = np.flatnonzero(mask)
    if exclude_idx is not None and exclude_idx.size > 0:
        idx = np.setdiff1d(idx, exclude_idx, assume_unique=False)
    n_sel = int(idx.size)
    if n_sel == 0:
        return [], 0

    if n_sel < group_size:
        groups_idx = [idx]
    else:
        m = max(1, min(max_groups, n_sel // group_size))
        total_needed = m * group_size
        if n_sel > total_needed:
            stride = n_sel / total_needed
            pick = np.floor(np.arange(total_needed) * stride).astype(np.int64)
            idx_sel = idx[pick]
        else:
            idx_sel = idx
        groups_idx = [idx_sel[k::m] for k in range(m)]

    import cupy as cp
    cp_pool = cp.get_default_memory_pool()
    volumes = []
    for g_idx in groups_idx:
        p = torch.from_numpy(np.ascontiguousarray(sinogram[g_idx])).to(device=device, dtype=torch.float32)
        p = R.counts_to_attenuation_flatdark(p, dark, flat)
        angles = np.asarray(theta_deg)[g_idx].astype(np.float64)
        vol = R.reconstruct(p, angles, det_bin=det_bin, method="fbp", vol_shape=vol_shape, device=device,
                            clamp=clamp)
        volumes.append(vol)
        del p
        torch.cuda.empty_cache()
        cp_pool.free_all_blocks()
    return volumes, n_sel


def reproject_at_angles(
    vol: torch.Tensor,
    angles_deg: np.ndarray,
    det_shape: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Forward-project ``vol`` at exactly ``angles_deg`` (no gap between context and target).

    Returns values in attenuation domain (``vol`` from :func:`reconstruct_phi_gated`
    is always built via the ``dark``/``flat`` round trip). See
    :func:`reproject_to_counts` to convert
    back to counts when real darks/flats are available.
    """
    from astra_torch.lamino import build_lamino_projector

    vol_shape = tuple(vol.shape)
    proj_layer = build_lamino_projector(
        vol_shape=vol_shape, det_shape=det_shape, angles_deg=np.asarray(angles_deg, dtype=np.float64),
        lamino_angle_deg=0.0, voxel_size_mm=1.0, det_spacing_mm=1.0, device=device,
    )
    with torch.no_grad():
        out = proj_layer(vol.unsqueeze(0).unsqueeze(0))[0]
    return out


def reproject_to_counts(
    vol: torch.Tensor,
    angles_deg: np.ndarray,
    det_shape: Tuple[int, int],
    device: torch.device,
    dark: torch.Tensor,
    flat: torch.Tensor,
) -> torch.Tensor:
    """Reproject ``vol`` (built from attenuation, i.e. via ``dark``/``flat`` in
    :func:`reconstruct_phi_gated`) and convert back to the counts domain --
    ``count = dark + (flat-dark)*exp(-p)`` -- so the resulting context tap
    lives in the same counts-like domain as the raw noisy target, exactly
    mirroring ``scripts/tr_diffusion_jointfbp_context_cache.py``.
    """
    from sdate.tr_diffusion import reconstruct as R

    atten = reproject_at_angles(vol, angles_deg, det_shape, device)
    return R.attenuation_to_counts_flatdark(atten, dark, flat)
