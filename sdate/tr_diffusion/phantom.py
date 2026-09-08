"""Analytic synthetic time-resolved CT phantom.

A closed-form parallel-beam forward model, built so every projection is exact
at the *true continuous* rotation angle and time of its frame -- unlike a
frame-sampled 4D dataset (e.g. TomoBank's dynamic category), which only has
independently-reconstructed volumes at discrete time steps and therefore
shows a discontinuity whenever a synthesized projection stream crosses from
one time-step's volume to the next. Here every object pose is a closed-form
function of continuous time, so there is nothing to interpolate or jump
between: frame ``f`` (integer or fractional) is projected at exactly its own
angle ``theta(f)`` and its own object state at time ``f``.

Geometry convention (must match ``astra_torch.lamino`` exactly, since that is
what reconstructs these projections downstream): with ``lamino_angle_deg=0``,
``_create_lamino_geometry`` gives ray direction ``(cos theta, sin theta, 0)``,
detector u-direction ``(-sin theta, cos theta, 0)``, v-direction ``(0,0,1)``,
so a world point ``(x,y,z)`` lands at detector coordinate::

    u = -x * sin(theta) + y * cos(theta)
    v = z

Pixel index <-> world coordinate uses ASTRA's centered-window convention
(``WindowMin/Max = -/+ n/2`` at voxel size 1): index ``i`` in ``[0, n)`` is
world coordinate ``i - (n - 1) / 2``. Both the volume grid and the detector
grid use this same convention here, so a primitive placed at world (x, y, z)
reconstructs back to the same voxel location -- verified in
``scripts/validate_synthetic_phantom.py``.

Every primitive is a "capsule": a solid of revolution about the z-axis whose
in-plane radius envelope ``r_eff(z)`` is either a smooth spheroid taper
(``shape="round"``) or a flat-topped cylinder (``shape="flat"``, genuine step
edges across detector rows -- the "sharp objects" of the design). Both share
one projection kernel: a circular disc of radius ``r_eff(z)`` at in-plane
centre ``(cx, cy)`` has the classic chord-length projection
``2 * mu * sqrt(max(0, r_eff(z)**2 - (u - u0(theta))**2))``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Literal, Sequence

import numpy as np
import torch

TimeFn = Callable[[np.ndarray], np.ndarray]  # t (frames, float64) -> value(s), same shape


def const(v: float) -> TimeFn:
    return lambda t: np.full_like(t, float(v), dtype=np.float64)


@dataclass
class Capsule:
    """A time-varying solid of revolution about z (sphere/spheroid or flat cylinder).

    ``cx, cy, cz``: in-plane/height centre, ``radius``: in-plane radius,
    ``z_half``: half-height of the z-extent, ``mu``: attenuation coefficient.
    All are ``TimeFn`` evaluated at the frame indices being rendered.
    ``shape="round"`` tapers ``radius`` to 0 at ``z_half`` (spheroid); ``"flat"``
    holds the full ``radius`` out to ``z_half`` then steps to 0 (cylinder).
    """

    cx: TimeFn
    cy: TimeFn
    cz: TimeFn
    radius: TimeFn
    z_half: TimeFn
    mu: TimeFn
    shape: Literal["round", "flat"] = "round"
    name: str = ""


def render_projections(
    capsules: Sequence[Capsule],
    frame_indices: np.ndarray,
    height: int,
    width: int,
    deg_per_frame: float,
    device: torch.device = torch.device("cpu"),
    angle0_deg: float = 0.0,
) -> torch.Tensor:
    """Clean attenuation line integrals ``p`` for a batch of frames.

    Returns ``(N, height, width)`` float32: ``p[n, row, col]`` is the parallel-beam
    attenuation line integral at frame ``frame_indices[n]``, detector row ``row``
    (world ``z = row - (height-1)/2``), detector column ``col`` (world
    ``u = col - (width-1)/2``). Superposition over capsules (Radon transform is
    linear), each capsule's own pose/size/density evaluated at the SAME
    continuous frame index used for its angle -- so a fractional frame index
    gets an exact fractional-time object state, not an interpolated one.
    """
    t = frame_indices.astype(np.float64)
    n = t.shape[0]
    theta = np.deg2rad(angle0_deg + t * float(deg_per_frame))  # (N,)
    theta_t = torch.as_tensor(theta, dtype=torch.float64, device=device).view(n, 1, 1)

    z = (torch.arange(height, dtype=torch.float64, device=device) - (height - 1) / 2.0).view(1, height, 1)
    u = (torch.arange(width, dtype=torch.float64, device=device) - (width - 1) / 2.0).view(1, 1, width)

    p = torch.zeros((n, height, width), dtype=torch.float64, device=device)
    for cap in capsules:
        cx = torch.as_tensor(cap.cx(t), dtype=torch.float64, device=device).view(n, 1, 1)
        cy = torch.as_tensor(cap.cy(t), dtype=torch.float64, device=device).view(n, 1, 1)
        cz = torch.as_tensor(cap.cz(t), dtype=torch.float64, device=device).view(n, 1, 1)
        radius = torch.as_tensor(cap.radius(t), dtype=torch.float64, device=device).view(n, 1, 1).clamp_min(0.0)
        z_half = torch.as_tensor(cap.z_half(t), dtype=torch.float64, device=device).view(n, 1, 1).clamp_min(1e-6)
        mu = torch.as_tensor(cap.mu(t), dtype=torch.float64, device=device).view(n, 1, 1)

        dz = z - cz  # (N, H, 1)
        if cap.shape == "round":
            r_eff = radius * torch.sqrt((1.0 - (dz / z_half) ** 2).clamp_min(0.0))
        elif cap.shape == "flat":
            r_eff = radius * (dz.abs() < z_half).to(radius.dtype)
        else:
            raise ValueError(f"unknown capsule shape {cap.shape!r}")

        u0 = -cx * torch.sin(theta_t) + cy * torch.cos(theta_t)  # (N, 1, 1)
        chord2 = (r_eff ** 2 - (u - u0) ** 2).clamp_min(0.0)  # (N, H, W)
        p = p + mu * 2.0 * torch.sqrt(chord2)

    # mu is allowed to be negative (a "void"/lower-attenuation inclusion in a
    # denser background, e.g. a gas bubble in a melt) -- physically the total
    # line-integrated attenuation still can't go negative, so clamp the sum,
    # not each capsule.
    return p.clamp_min(0.0).to(torch.float32)


def attenuation_to_counts(p: torch.Tensor, I0: float) -> torch.Tensor:
    """Beer-Lambert: clean detector counts from attenuation line integrals."""
    return I0 * torch.exp(-p)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _bump(t: np.ndarray, t0: float, t1: float, rise: float, fall: float) -> np.ndarray:
    """Smooth 0->1->0 window on ``[t0, t1]``, easing in/out over ``rise``/``fall`` frames.

    C1-continuous everywhere (zero slope at both ends) -- used for
    nucleation/dissolution so a transient capsule's radius never jumps.
    """
    up = _smoothstep((t - t0) / max(rise, 1e-6))
    down = 1.0 - _smoothstep((t - (t1 - fall)) / max(fall, 1e-6))
    return np.minimum(up, down).clip(0.0, 1.0)


def _signed_mu(rng: np.random.Generator, lo: float, hi: float) -> float:
    """A random attenuation contrast of magnitude in ``[lo, hi]``, either sign.

    Positive = denser inclusion (grain/particle); negative = a void/bubble in
    a denser surrounding medium. Excludes a dead zone near 0 so every capsule
    stays visible.
    """
    return float(rng.choice([-1.0, 1.0]) * rng.uniform(lo, hi))


def default_scene(
    frame_start: int = 0,
    frame_end: int = 100_000,
    height: int = 128,
    width: int = 512,
    seed: int = 0,
    n_grains: int = 1800,
    dynamic_fraction: float = 0.2,
    n_large: int = 6,
    n_mid: int = 30,
) -> List[Capsule]:
    """A densely-packed granular/foam phantom -- one uniform grain population.

    Earlier versions used a sparse population of "moving objects" (10-25px,
    later 5-11px) against an empty or near-empty background. That design could
    never reproduce the dramatic "unusable noisy recon -> perfect denoised"
    story real datasets show: reconstruction noise for a feature shrinks with
    its size, so ANY larger feature (a single bigger object, or just a chance
    cluster of a few same-sign objects overlapping) survives the noise as a
    visible anchor, and a viewer/model can track the scene through that alone.
    Real materials (foams, slurries, granular media) don't offer such an
    anchor: they're a densely-packed, irregular texture with no standout
    simple shape, so noise degrades everything roughly together.

    This scene reproduces that instead: ``n_grains`` capsules with a
    *narrow, bounded* size range (radius/z_half both 4-12px, no large
    outliers), placed densely enough to fill most of the field of view
    (~90%+ areal coverage, grains touching/overlapping like a real granular
    cross-section) with alternating-sign attenuation contrast (denser grains
    and voids/bubbles both occur). There is no separate large "background"
    capsule (its own moiré artifact isn't worth it once grains already fill
    the frame) and no special "texture" tier -- the grains ARE the texture.

    Most grains (``1 - dynamic_fraction``) are static -- a real jammed
    granular pack mostly doesn't move except at localized rearrangement
    events. The remaining ``dynamic_fraction`` nucleate/grow/dissolve via
    :func:`_bump` (staggered lifecycles across the whole sequence, same
    few-hundred-to-few-thousand-frame timescale as before) with a small
    bounded drift on top -- but critically, their radius never exceeds the
    SAME 4-12px bound as the static grains, so a dynamic grain never becomes
    a size/contrast outlier relative to its neighbours.

    Every grain's radial extent from the z-axis, ``sqrt(cx(t)**2+cy(t)**2) +
    radius(t)``, is kept ``<= width/2 - 20`` for ALL ``t`` (see the discussion
    in earlier versions of this docstring in project memory for why this
    matters for parallel-beam geometry specifically).

    On top of the grain pack, two more tiers give the scene its main driving
    dynamics -- a breathing ("5x bigger and back") tier was tried and didn't
    read as real motion; this replaces it with genuine size diversity plus
    genuine large-scale translation instead:

    * ``n_large`` "large" objects, roughly 6-10x a grain's radius (45-80px vs
      grains' 4-12px), mixed in among the grains at fixed (static) positions
      and sizes -- a real granular/foam cross-section often has a few much
      bigger inclusions sitting among the fine texture, not just uniform
      grains. Best-of-many placement (see below) so they read as several
      distinct big features rather than fusing into one blob. Contrast
      (0.015-0.025), close to a grain's own (0.025-0.05).
    * ``n_mid`` "mid" objects, 3-6x a grain's radius (20-45px), each sweeping
      back and forth through a substantial fraction of the field of view
      (50-110px amplitude) over a *slow* period (6000-15000 frames) -- this is
      the scene's clearly-visible moving population, distinct from the
      grains' deliberately subtle in-place jitter. Contrast (0.018-0.028).

    All three tiers' contrast floors were raised together (grains
    0.02->0.025, large 0.010->0.015, mid 0.012->0.018) after an early pass
    reconstructed with a wide 0-to-peak intensity spread -- a lot of
    individual objects sat close to 0 (near-invisible) with only rare
    same-sign overlap clusters reaching the top of the range. Narrowing each
    tier's own floor-to-ceiling ratio, combined with a less extreme display
    percentile (``vmax_pctile=99`` instead of ``99.9`` -- the ``.9`` was
    itself stretching the scale to accommodate rare overlap brightness
    spikes), puts most individual objects' reconstructed peak at roughly
    30-100% of the display ceiling instead of 0-100%.

    Both tiers are a handful of objects on top of ~1800 grains, not the
    primary population, so the fine-grain noise-defeat property (see above)
    is untouched.
    """
    rng = np.random.default_rng(seed)
    r_lim = width / 2.0 - 20.0
    z_lim = height / 2.0 - 20.0
    span = float(frame_end - frame_start)
    caps: List[Capsule] = []

    n_dynamic = int(round(n_grains * dynamic_fraction))
    dynamic_idx = set(rng.choice(n_grains, size=n_dynamic, replace=False).tolist())
    orbit_max = r_lim - 22.0  # leaves room for drift amp (<=8) + max radius (12) + margin

    for i in range(n_grains):
        orbit = rng.uniform(0.0, orbit_max)
        ang = rng.uniform(0.0, 2 * np.pi)
        cx0, cy0 = orbit * np.cos(ang), orbit * np.sin(ang)
        cz0 = rng.uniform(-z_lim, z_lim)
        radius0 = rng.uniform(4.0, 12.0)
        z_half0 = rng.uniform(4.0, 12.0)
        mu0 = _signed_mu(rng, 0.025, 0.05)
        shape = "flat" if rng.random() < 0.25 else "round"

        if i in dynamic_idx:
            t0 = frame_start + span * rng.uniform(0.0, 0.95)
            life = rng.uniform(800.0, 3000.0)
            rise = life * rng.uniform(0.2, 0.35)
            fall = life * rng.uniform(0.2, 0.35)
            t1 = t0 + life
            amp = rng.uniform(3.0, 8.0)
            period = rng.uniform(2000.0, 8000.0)
            phase = rng.uniform(0.0, 2 * np.pi)
            drift_ang = rng.uniform(0.0, 2 * np.pi)

            def radius(t, radius0=radius0, t0=t0, t1=t1, rise=rise, fall=fall):
                return radius0 * _bump(t, t0, t1, rise, fall)

            def cx(t, cx0=cx0, amp=amp, period=period, phase=phase, drift_ang=drift_ang):
                return cx0 + amp * np.cos(drift_ang) * np.sin(2 * np.pi * t / period + phase)

            def cy(t, cy0=cy0, amp=amp, period=period, phase=phase, drift_ang=drift_ang):
                return cy0 + amp * np.sin(drift_ang) * np.sin(2 * np.pi * t / period + phase)

            caps.append(Capsule(cx, cy, const(cz0), radius, const(z_half0), const(mu0),
                                shape=shape, name=f"grain{i}_dyn"))
        else:
            caps.append(Capsule(const(cx0), const(cy0), const(cz0), const(radius0), const(z_half0),
                                const(mu0), shape=shape, name=f"grain{i}_static"))

    # -- "large" objects: static size diversity, ~10x a grain's radius ------
    # Best-of-many placement (maximise the minimum separation margin to
    # already-placed large objects, out of many random candidates) so they
    # read as several distinct big features against the grain pack -- plain
    # uniform placement, and even a fixed-threshold rejection sampler, in the
    # (necessarily tight, given their own size) orbit range mostly put them
    # on top of each other, fusing into one big blob rather than "large
    # objects mixed with the small ones".
    large_radius_max, large_margin = 80.0, 10.0
    large_orbit_max = max(r_lim - (large_radius_max + large_margin), 0.0)
    placed: List[tuple] = []  # (cx0, cy0, radius0)
    for k in range(n_large):
        radius0 = rng.uniform(45.0, large_radius_max)
        z_half0 = radius0 * rng.uniform(0.8, 1.2)
        best_xy, best_margin = None, -np.inf
        for _candidate in range(500):
            # Uniform in AREA (not radius) -- uniform-radius sampling piles
            # candidates up near the centre (equal weight per radius bin, but
            # the centre bins cover far less area), starving the packing.
            orbit = large_orbit_max * np.sqrt(rng.uniform(0.0, 1.0))
            ang = rng.uniform(0.0, 2 * np.pi)
            cx0, cy0 = orbit * np.cos(ang), orbit * np.sin(ang)
            margin = (np.inf if not placed else
                     min(np.hypot(cx0 - px, cy0 - py) - (radius0 + pr) for px, py, pr in placed))
            if margin > best_margin:
                best_xy, best_margin = (cx0, cy0), margin
        cx0, cy0 = best_xy
        placed.append((cx0, cy0, radius0))
        cz0 = rng.uniform(-z_lim, z_lim)
        mu0 = _signed_mu(rng, 0.015, 0.025)
        shape = "flat" if rng.random() < 0.3 else "round"
        caps.append(Capsule(const(cx0), const(cy0), const(cz0), const(radius0), const(z_half0),
                            const(mu0), shape=shape, name=f"large{k}"))

    # -- "mid" objects: the scene's main driving dynamics --------------------
    # Sweep back and forth through a substantial fraction of the FOV over a
    # slow period -- distinct from the grains' subtle in-place jitter. FOV
    # bound must cover the worst case (base offset + sweep amplitude both at
    # their max) even though random phases make that combination rare.
    mid_radius_max, mid_amp_max, mid_margin = 45.0, 110.0, 10.0
    mid_orbit_max = max(r_lim - (mid_radius_max + mid_amp_max + mid_margin), 0.0)
    for k in range(n_mid):
        radius0 = rng.uniform(20.0, mid_radius_max)
        z_half0 = radius0 * rng.uniform(0.8, 1.2)
        orbit = rng.uniform(0.0, mid_orbit_max)
        ang = rng.uniform(0.0, 2 * np.pi)
        cx0, cy0 = orbit * np.cos(ang), orbit * np.sin(ang)
        cz0 = rng.uniform(-z_lim, z_lim)
        mu0 = _signed_mu(rng, 0.018, 0.028)
        shape = "flat" if rng.random() < 0.25 else "round"

        drift_amp = rng.uniform(50.0, mid_amp_max)
        drift_period = rng.uniform(6000.0, 15000.0)  # slow
        drift_phase = rng.uniform(0.0, 2 * np.pi)
        drift_ang = rng.uniform(0.0, 2 * np.pi)

        def cx(t, cx0=cx0, drift_amp=drift_amp, drift_period=drift_period,
              drift_phase=drift_phase, drift_ang=drift_ang):
            return cx0 + drift_amp * np.cos(drift_ang) * np.sin(2 * np.pi * t / drift_period + drift_phase)

        def cy(t, cy0=cy0, drift_amp=drift_amp, drift_period=drift_period,
              drift_phase=drift_phase, drift_ang=drift_ang):
            return cy0 + drift_amp * np.sin(drift_ang) * np.sin(2 * np.pi * t / drift_period + drift_phase)

        caps.append(Capsule(cx, cy, const(cz0), const(radius0), const(z_half0), const(mu0),
                            shape=shape, name=f"mid{k}"))

    return caps
