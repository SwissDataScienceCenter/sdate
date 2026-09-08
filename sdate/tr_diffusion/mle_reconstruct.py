"""Pure Poisson-MLE reconstruction of the attenuation volume from raw sinogram
counts -- gradient descent and ADMM+TV variants, both via the ASTRA
laminography forward/adjoint operators (:mod:`astra_torch.lamino`).

Unlike :mod:`sdate.tr_diffusion.map_reconstruct` (MAP with a denoiser-derived
Gamma prior baked in), this module has NO prior term at all: the objective is
the textbook Poisson transmission negative log-likelihood,

    l_i(mu)      = [A mu]_i                    # line integral, ASTRA forward projector
    lambda_i(mu) = I0_i * exp(-l_i(mu)) + d_i   # expected raw detector count rate
    y_i | mu ~ Poisson(lambda_i(mu))

``y_vrc`` is the dose-rescaled measurement this project's synthetic thinning
produces (``y = Poisson(counts * dose) / dose``, see
:func:`sdate.tr_diffusion.noise.add_poisson_noise`), so the actual observed
Poisson count is ``z = dose * y`` with rate ``dose * lambda_i(mu)`` -- the same
``dose`` convention already used (and its failure mode already documented) in
``map_reconstruct.py``. Up to an additive, mu-independent constant, the
per-pixel Poisson NLL is::

    L(mu) = sum_i [ dose * lambda_i(mu) - dose * y_i * log(lambda_i(mu)) ]

Two solvers are provided:

* :func:`gd_mle_reconstruct` -- plain Adam gradient descent on ``L(mu)``,
  optionally plus an anisotropic TV penalty (reuses
  :func:`sdate.tr_diffusion.map_reconstruct._tv_loss`, the exact same
  regulariser used by the ADMM variant below, so the two solvers are directly
  comparable -- only the optimisation method differs, not the objective).
* :func:`admm_tv_mle_reconstruct` -- scaled-dual ADMM-TV. The Poisson data
  term is nonlinear in ``mu`` (through ``exp(-A mu)``), so there is no
  closed-form mu-update; it is solved inexactly with a handful of inner Adam
  steps per outer iteration (a standard, practical choice for ADMM with a
  non-quadratic data term). The z/dual updates for the TV term ARE closed
  form (anisotropic soft-threshold).

Both are known (from ``map_reconstruct.py``'s own documented findings on this
exact dataset) to be capable of diverging via generic GD/SIRT-style
semi-convergence without TV regularisation -- see :func:`auto_tv_weight` for
how the TV weight grid is scaled to something meaningful without hand-guessed
constants (the data term is a raw SUM over every ray in the sinogram, which
can be ~1e10 for a several-thousand-view joint reconstruction, while TV is a
raw SUM over volume voxels, ~1e5 -- wildly different scales).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from .map_reconstruct import _tv_loss

__all__ = [
    "poisson_data_loss",
    "auto_tv_weight",
    "auto_admm_rho",
    "gd_mle_reconstruct",
    "admm_tv_mle_reconstruct",
]


def poisson_data_loss(l: torch.Tensor, y: torch.Tensor, I0: torch.Tensor, dark: torch.Tensor,
                      dose: float, l_clip: float = 20.0, eps: float = 1e-6) -> torch.Tensor:
    """Poisson transmission NLL (summed over all rays) given line integrals ``l``.

    ``l`` = ``A @ mu`` (pre-``exp``), clamped to ``[-l_clip, l_clip]`` purely as
    a numerical-safety guard against ``exp`` overflow (mirrors
    ``map_reconstruct.map_reconstruct``'s ``l_clip``). ``y``, ``I0``, ``dark``
    all in raw count units, ``I0``/``dark`` broadcastable against ``y``'s
    ``(V, R, C)`` shape.
    """
    lc = l.clamp(-l_clip, l_clip)
    lam = I0 * torch.exp(-lc) + dark
    return (dose * lam - dose * y * torch.log(lam.clamp_min(eps))).sum()


def _finite_diffs(mu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward finite differences along each of the 3 volume axes (unpadded).

    Matches :func:`sdate.tr_diffusion.map_reconstruct._tv_loss`'s own dz/dy/dx
    slicing convention exactly, so ``dz.abs().sum() + dy.abs().sum() +
    dx.abs().sum()`` reproduces ``_tv_loss(mu)``.
    """
    dz = mu[1:, :, :] - mu[:-1, :, :]
    dy = mu[:, 1:, :] - mu[:, :-1, :]
    dx = mu[:, :, 1:] - mu[:, :, :-1]
    return dz, dy, dx


def _soft_threshold(v: torch.Tensor, t: float) -> torch.Tensor:
    """Elementwise soft-threshold (proximal operator of ``t * |.|``)."""
    return torch.sign(v) * (v.abs() - t).clamp_min(0.0)


def auto_tv_weight(mu0: torch.Tensor, proj_layer, y: torch.Tensor, I0: torch.Tensor,
                   dark: torch.Tensor, dose: float, l_clip: float = 20.0,
                   eps: float = 1e-6) -> float:
    """Gradient-norm-balanced reference TV weight at the warm-start volume ``mu0``.

    Returns ``||d(data_loss)/d(mu)|| / ||d(tv_loss)/d(mu)||`` evaluated once at
    ``mu0`` -- the ``tv_weight`` at which the two gradients have comparable
    magnitude, i.e. a sensible CENTRE for a log-scale grid search. Needed
    because the data term's raw-sum scale grows with the number of views (very
    different between e.g. k=5 and k=40), so no single hand-picked constant
    works across the whole k sweep -- see module docstring.
    """
    mu = mu0.detach().clone().requires_grad_(True)
    l = proj_layer(mu.unsqueeze(0).unsqueeze(0))[0]
    data_loss = poisson_data_loss(l, y, I0, dark, dose, l_clip=l_clip, eps=eps)
    (g_data,) = torch.autograd.grad(data_loss, mu, retain_graph=True)
    tv = _tv_loss(mu)
    (g_tv,) = torch.autograd.grad(tv, mu)
    return float(g_data.norm() / g_tv.norm().clamp_min(1e-12))


def auto_admm_rho(mu0: torch.Tensor, tv0: float, obj_thresh: float = 1e-6,
                  eps: float = 1e-8) -> float:
    """ADMM's ``rho``, auto-scaled from ``tv0`` (the SAME gradient-norm-balanced
    weight :func:`auto_tv_weight` computes for GD+TV) and the RECONSTRUCTED
    OBJECT's own typical finite-difference magnitude.

    Empirically necessary: a naively fixed ``rho=1.0`` makes the z-update's
    soft-threshold ``tv_weight / rho`` many orders of magnitude larger than
    the volume's actual per-voxel differences (data-loss gradients are a raw
    SUM over every ray, so ``tv0`` -- balanced against THAT -- is huge, while
    real attenuation differences between neighbouring voxels are tiny) --
    confirmed via a smoke test where every ``tv_weight`` in the grid produced
    an IDENTICAL reconstruction (z saturated to all-zero regardless of the
    grid point). Scaling ``rho = tv0 / typical(|D mu0|)`` makes the z-update's
    threshold land at ``mult * typical(|D mu0|)`` for a grid multiplier
    ``mult`` -- i.e. meaningfully close to the real finite-difference scale --
    while also making the mu-subproblem's constraint-quadratic-penalty
    gradient land at order ``tv0`` after the first z-update (once the
    residual is of typical magnitude), matching GD+TV's own regularisation
    pull for the same ``tv_weight``.

    ``typical(|D mu0|)`` is the MEDIAN restricted to voxel pairs that are BOTH
    inside the object (``|mu0| > obj_thresh``) -- a real reconstruction volume
    is mostly empty background (``mu0 == 0`` outside the sample, clamped), so
    the plain unrestricted median is dominated by trivially-zero background
    differences and collapses to ~0 (confirmed empirically: on the real
    warm-start volume the unrestricted median rounded to the ``eps`` floor,
    giving an absurdly large ``rho``) -- excluding background pairs recovers
    a scale that actually reflects the object's own texture/edges.
    """
    dz, dy, dx = _finite_diffs(mu0)

    def _active(diff, a, b):
        active = (a.abs() > obj_thresh) & (b.abs() > obj_thresh)
        return diff.abs()[active]

    vals = torch.cat([
        _active(dz, mu0[1:, :, :], mu0[:-1, :, :]),
        _active(dy, mu0[:, 1:, :], mu0[:, :-1, :]),
        _active(dx, mu0[:, :, 1:], mu0[:, :, :-1]),
    ])
    dscale = vals.median() if vals.numel() > 0 else torch.tensor(eps, device=mu0.device)
    return float(tv0 / dscale.clamp_min(eps))


def gd_mle_reconstruct(
    y_vrc: torch.Tensor,
    angles_deg: np.ndarray,
    I0: torch.Tensor,
    dark: torch.Tensor,
    dose: float,
    vol_shape: Optional[Tuple[int, int, int]] = None,
    lamino_angle_deg: float = 0.0,
    tilt_angle_deg: float = 0.0,
    voxel_size_mm: float = 1.0,
    det_spacing_mm: float = 1.0,
    tv_weight: float = 0.0,
    mu_init: Optional[torch.Tensor] = None,
    n_iters: int = 400,
    lr: float = 1e-1,
    optimizer_type: str = "adam",
    l_clip: float = 20.0,
    eps: float = 1e-6,
    clamp_min: float = 0.0,
    disk_mask: Optional[torch.Tensor] = None,
    log_every: int = 0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Adam/SGD gradient descent on the pure Poisson-MLE objective.

    ``tv_weight=0`` (default) is plain, unregularised MLE ("pure GD"); any
    ``tv_weight>0`` adds :func:`sdate.tr_diffusion.map_reconstruct._tv_loss`,
    giving "GD+TV". ``mu_init`` is required in practice for this dataset (see
    module docstring on divergence) -- pass the baseline-denoiser-derived warm
    start, not zeros/FBP-of-noisy-data.

    ``disk_mask``, if given, is a ``(H, W)`` boolean/float mask over the
    volume's last two (reconstruction-plane) axes -- broadcasts over the
    slice axis -- hard-zeroed into ``mu`` after every step.

    **Do not actually use this for the data-fitting solvers.** It was added to
    try to eliminate a halo/streak artifact visible outside the inscribed
    reconstruction circle (a truncation / interior-tomography symptom) and
    empirically made things dramatically WORSE, not better: on wunderkerze2
    k=20/dose=0.05, PSNR dropped from 18.3->11.5 dB (plain GD), 21.9->12.5 dB
    (GD+TV best), ~22.8->15.9 dB (ADMM+TV best), and a large bright/dark
    streak bias appeared INSIDE the reconstruction circle (visible directly
    in the diff maps) -- the artifact was relocated, not removed. This is the
    textbook ROI/interior-tomography pitfall: if any real attenuating
    material (sample, mount, ...) sits outside the assumed disk at some
    projection angles, forcing the Poisson data-consistency term to explain
    ALL measured counts using ONLY interior voxels dumps that truncation
    mismatch into the interior as bias, since the optimizer has nowhere else
    to put it. FBP-based reconstructions are unaffected precisely because
    they don't iterate against the data term.

    The correct fix for the visual halo is to mask AFTER reconstruction only
    -- for scoring (already how ``masked_psnr``/``masked_ssim`` work) and for
    display (crop diff maps / images to the circle) -- never inside the
    optimization loop. This parameter is kept (default ``None``, i.e. unused)
    only so this finding has somewhere concrete to live; do not pass it for
    real reconstructions.
    """
    from astra_torch.lamino import build_lamino_projector

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = y_vrc.to(device=device, dtype=torch.float32)
    v, r, c = y.shape
    I0_b = I0.to(device=device, dtype=torch.float32)
    dark_b = dark.to(device=device, dtype=torch.float32)
    if I0_b.ndim == 2:
        I0_b = I0_b.unsqueeze(0)
    if dark_b.ndim == 2:
        dark_b = dark_b.unsqueeze(0)

    vol_shape = vol_shape or (r, c, c)
    if mu_init is None:
        raise ValueError("mu_init is required (warm-start from the baseline denoiser's own "
                         "reconstruction) -- plain-MLE GD is documented to diverge from zeros "
                         "on this dataset (see module docstring)")
    mask_b = disk_mask.to(device=device, dtype=torch.float32) if disk_mask is not None else None
    mu = mu_init.detach().clone().to(device=device, dtype=torch.float32).clamp_min(clamp_min)
    if mask_b is not None:
        mu.mul_(mask_b)
    mu.requires_grad_(True)

    proj_layer = build_lamino_projector(
        vol_shape=vol_shape, det_shape=(r, c), angles_deg=np.asarray(angles_deg),
        lamino_angle_deg=lamino_angle_deg, tilt_angle_deg=tilt_angle_deg,
        voxel_size_mm=voxel_size_mm, det_spacing_mm=det_spacing_mm, device=device,
    )

    if optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam([mu], lr=lr)
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD([mu], lr=lr)
    else:
        raise ValueError("optimizer_type must be 'adam' or 'sgd'")

    for step in range(n_iters):
        optimizer.zero_grad(set_to_none=True)
        l = proj_layer(mu.unsqueeze(0).unsqueeze(0))[0]
        data_loss = poisson_data_loss(l, y, I0_b, dark_b, dose, l_clip=l_clip, eps=eps)
        tv = _tv_loss(mu) if tv_weight > 0 else None
        loss = data_loss + tv_weight * tv if tv is not None else data_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            mu.clamp_(min=clamp_min)
            if mask_b is not None:
                mu.mul_(mask_b)
        if log_every and step % log_every == 0:
            tv_str = f"  tv={tv.item():.6g}" if tv is not None else ""
            print(f"  [gd_mle] step {step}/{n_iters}  loss={loss.item():.6g}  "
                 f"data_loss={data_loss.item():.6g}{tv_str}  "
                 f"mu[min={mu.min().item():.4g},max={mu.max().item():.4g}]", flush=True)

    return mu.detach()


def admm_tv_mle_reconstruct(
    y_vrc: torch.Tensor,
    angles_deg: np.ndarray,
    I0: torch.Tensor,
    dark: torch.Tensor,
    dose: float,
    vol_shape: Optional[Tuple[int, int, int]] = None,
    lamino_angle_deg: float = 0.0,
    tilt_angle_deg: float = 0.0,
    voxel_size_mm: float = 1.0,
    det_spacing_mm: float = 1.0,
    tv_weight: float = 0.05,
    rho: float = 1.0,
    mu_init: Optional[torch.Tensor] = None,
    n_outer: int = 25,
    inner_iters: int = 8,
    inner_lr: float = 1e-1,
    l_clip: float = 20.0,
    eps: float = 1e-6,
    clamp_min: float = 0.0,
    disk_mask: Optional[torch.Tensor] = None,
    log_every: int = 0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Scaled-dual ADMM-TV Poisson-MLE reconstruction.

    ``disk_mask``, if given, is a ``(H, W)`` mask over the reconstruction
    plane, hard-zeroed into ``mu`` after every inner step -- **do not use it**,
    see :func:`gd_mle_reconstruct`'s docstring for the empirical finding
    (it relocates truncation bias into the reconstruction interior instead of
    removing it). Kept only as a documented, unused-by-default option.

    Splits via one auxiliary variable per finite-difference axis (``z_z, z_y,
    z_x``, matching :func:`_finite_diffs`/``_tv_loss``'s own dz/dy/dx
    decomposition) so the TV proximal step is a closed-form anisotropic
    soft-threshold:

        mu-subproblem  (no closed form -- data term is nonlinear in mu):
            data_loss(mu) + (rho/2) * sum_axis || D_axis(mu) - z_axis + u_axis ||^2
            solved INEXACTLY with `inner_iters` Adam steps.
        z-subproblem (closed form):
            z_axis = shrink(D_axis(mu) + u_axis, tv_weight / rho)
        dual update:
            u_axis += D_axis(mu) - z_axis

    ``rho`` is a single fixed scalar for this first pass -- Adam's
    per-parameter adaptive scaling makes the inexact mu-subproblem fairly
    robust to its absolute value; a follow-up ``rho`` sweep is worth doing
    only if this undersells ADMM relative to plain GD+TV.
    """
    from astra_torch.lamino import build_lamino_projector

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = y_vrc.to(device=device, dtype=torch.float32)
    v, r, c = y.shape
    I0_b = I0.to(device=device, dtype=torch.float32)
    dark_b = dark.to(device=device, dtype=torch.float32)
    if I0_b.ndim == 2:
        I0_b = I0_b.unsqueeze(0)
    if dark_b.ndim == 2:
        dark_b = dark_b.unsqueeze(0)

    vol_shape = vol_shape or (r, c, c)
    if mu_init is None:
        raise ValueError("mu_init is required (warm-start from the baseline denoiser's own "
                         "reconstruction) -- see module docstring")
    mask_b = disk_mask.to(device=device, dtype=torch.float32) if disk_mask is not None else None
    mu = mu_init.detach().clone().to(device=device, dtype=torch.float32).clamp_min(clamp_min)
    if mask_b is not None:
        mu.mul_(mask_b)
    mu.requires_grad_(True)

    proj_layer = build_lamino_projector(
        vol_shape=vol_shape, det_shape=(r, c), angles_deg=np.asarray(angles_deg),
        lamino_angle_deg=lamino_angle_deg, tilt_angle_deg=tilt_angle_deg,
        voxel_size_mm=voxel_size_mm, det_spacing_mm=det_spacing_mm, device=device,
    )

    with torch.no_grad():
        dz0, dy0, dx0 = _finite_diffs(mu)
        z_z, z_y, z_x = dz0.clone(), dy0.clone(), dx0.clone()
        u_z, u_y, u_x = torch.zeros_like(z_z), torch.zeros_like(z_y), torch.zeros_like(z_x)

    optimizer = torch.optim.Adam([mu], lr=inner_lr)

    for outer in range(n_outer):
        # --- mu-subproblem: inexact (few Adam steps) ---
        for _ in range(inner_iters):
            optimizer.zero_grad(set_to_none=True)
            l = proj_layer(mu.unsqueeze(0).unsqueeze(0))[0]
            data_loss = poisson_data_loss(l, y, I0_b, dark_b, dose, l_clip=l_clip, eps=eps)
            dz, dy, dx = _finite_diffs(mu)
            admm_pen = (
                (dz - z_z + u_z).pow(2).sum()
                + (dy - z_y + u_y).pow(2).sum()
                + (dx - z_x + u_x).pow(2).sum()
            )
            loss = data_loss + 0.5 * rho * admm_pen
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                mu.clamp_(min=clamp_min)
                if mask_b is not None:
                    mu.mul_(mask_b)

        # --- z-subproblem: closed-form anisotropic soft-threshold ---
        with torch.no_grad():
            dz, dy, dx = _finite_diffs(mu)
            z_z = _soft_threshold(dz + u_z, tv_weight / rho)
            z_y = _soft_threshold(dy + u_y, tv_weight / rho)
            z_x = _soft_threshold(dx + u_x, tv_weight / rho)
            # --- dual update ---
            u_z = u_z + dz - z_z
            u_y = u_y + dy - z_y
            u_x = u_x + dx - z_x

        if log_every and outer % log_every == 0:
            with torch.no_grad():
                l = proj_layer(mu.unsqueeze(0).unsqueeze(0))[0]
                data_loss = poisson_data_loss(l, y, I0_b, dark_b, dose, l_clip=l_clip, eps=eps)
                tv = _tv_loss(mu)
            print(f"  [admm_tv] outer {outer}/{n_outer}  data_loss={data_loss.item():.6g}  "
                 f"tv={tv.item():.6g}  mu[min={mu.min().item():.4g},max={mu.max().item():.4g}]",
                 flush=True)

    return mu.detach()
