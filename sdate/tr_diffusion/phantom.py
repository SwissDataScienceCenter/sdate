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

    return p.to(torch.float32)


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


def default_scene(
    frame_start: int = 0,
    frame_end: int = 100_000,
    height: int = 128,
    width: int = 512,
    seed: int = 0,
) -> List[Capsule]:
    """A hand-tuned population of capsules for the synthetic time-resolved dataset.

    Every capsule's radial extent from the z-axis, ``sqrt(cx(t)**2+cy(t)**2) +
    radius(t)``, is kept ``<= width/2 - 20`` for ALL ``t`` -- staying within the
    detector footprint at every rotation angle. This matters specifically for
    parallel-beam geometry: an object that drifts outside this radius does not
    just leave the frame, it periodically flickers back into view once per
    rotation (whenever the rotation angle happens to align it back within the
    finite detector width), which would look like a spurious per-turn artifact
    rather than a real trajectory.

    Two kinds of activity, both timed to change visibly rotation-to-rotation but
    stay ~constant within any single rotation (``period_360`` = 180 frames at
    ``deg_per_frame=2.0``):
    - persistent capsules that drift (bounded radial oscillation) and pulse in
      size slowly over thousands of frames;
    - transient capsules that nucleate, grow, and dissolve over a
      few-hundred-to-few-thousand-frame window via :func:`_bump`, staggered
      across the whole sequence so there is always something changing.
    Two pairs of capsules follow bounded (tanh-saturating, never unbounded)
    crossing trajectories so they visibly pass through the same region
    mid-sequence -- an "encounter" that reads as a merge/split, for free from
    linear superposition of the Radon transform, no special-case topology code.
    """
    rng = np.random.default_rng(seed)
    r_lim = width / 2.0 - 20.0
    z_lim = height / 2.0 - 20.0
    span = float(frame_end - frame_start)
    caps: List[Capsule] = []

    # -- persistent, radially-bounded drifting/pulsing capsules --------------
    n_persist = 10
    for i in range(n_persist):
        orbit_r = rng.uniform(20.0, 175.0)
        osc_amp = rng.uniform(5.0, 12.0)
        base_ang = rng.uniform(0.0, 2 * np.pi)
        period = rng.uniform(3000.0, 15000.0)
        phase = rng.uniform(0.0, 2 * np.pi)
        radius0 = rng.uniform(12.0, 22.0)
        rpulse = rng.uniform(0.1, 0.2) * radius0
        rperiod = rng.uniform(2000.0, 8000.0)
        cz0 = rng.uniform(-z_lim, z_lim)
        z_half0 = rng.uniform(10.0, 22.0)
        mu0 = rng.uniform(0.03, 0.09)
        shape = "flat" if i < 3 else "round"

        def cx(t, orbit_r=orbit_r, osc_amp=osc_amp, base_ang=base_ang, period=period, phase=phase):
            return (orbit_r + osc_amp * np.sin(2 * np.pi * t / period + phase)) * np.cos(base_ang)

        def cy(t, orbit_r=orbit_r, osc_amp=osc_amp, base_ang=base_ang, period=period, phase=phase):
            return (orbit_r + osc_amp * np.sin(2 * np.pi * t / period + phase)) * np.sin(base_ang)

        def radius(t, radius0=radius0, rpulse=rpulse, rperiod=rperiod, phase=phase):
            return radius0 + rpulse * np.sin(2 * np.pi * t / rperiod + phase)

        caps.append(Capsule(cx, cy, const(cz0), radius, const(z_half0), const(mu0),
                            shape=shape, name=f"persist{i}"))

    # -- two bounded crossing pairs ("encounter" events) ---------------------
    for j in range(2):
        cz0 = rng.uniform(-z_lim, z_lim)
        y0 = rng.uniform(-40.0, 40.0)
        t_cross = frame_start + span * rng.uniform(0.25, 0.75)
        width_frames = rng.uniform(3000.0, 8000.0)
        amp = rng.uniform(60.0, min(150.0, np.sqrt(max(r_lim ** 2 - y0 ** 2, 0.0)) - 25.0))
        radius0 = rng.uniform(14.0, 20.0)
        z_half0 = rng.uniform(14.0, 20.0)
        mu0 = rng.uniform(0.05, 0.08)
        for sign in (+1.0, -1.0):
            def cx(t, sign=sign, amp=amp, t_cross=t_cross, width_frames=width_frames):
                return sign * amp * np.tanh((t - t_cross) / width_frames)

            caps.append(Capsule(cx, const(y0), const(cz0), const(radius0), const(z_half0), const(mu0),
                                shape="round", name=f"cross{j}_{int(sign)}"))

    # -- transient nucleation/dissolution events, staggered across the sequence
    n_events = 24
    for k in range(n_events):
        t0 = frame_start + span * rng.uniform(0.0, 0.95)
        life = rng.uniform(800.0, 3000.0)
        rise = life * rng.uniform(0.2, 0.35)
        fall = life * rng.uniform(0.2, 0.35)
        t1 = t0 + life
        orbit = rng.uniform(0.0, r_lim - 30.0)
        ang = rng.uniform(0.0, 2 * np.pi)
        cx0, cy0 = orbit * np.cos(ang), orbit * np.sin(ang)
        cz0 = rng.uniform(-z_lim, z_lim)
        radius0 = rng.uniform(10.0, 22.0)
        z_half0 = rng.uniform(8.0, 20.0)
        mu0 = rng.uniform(0.04, 0.10)
        shape = "flat" if k % 4 == 0 else "round"

        def radius(t, radius0=radius0, t0=t0, t1=t1, rise=rise, fall=fall):
            return radius0 * _bump(t, t0, t1, rise, fall)

        caps.append(Capsule(const(cx0), const(cy0), const(cz0), radius, const(z_half0), const(mu0),
                            shape=shape, name=f"event{k}"))

    return caps
