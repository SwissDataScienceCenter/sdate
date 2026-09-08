"""MAP reconstruction of the attenuation volume from raw sinogram counts,
combining the Poisson transmission likelihood with the denoiser's per-ray
Gamma prior belief -- gradient descent via the ASTRA laminography forward/
adjoint operators (:mod:`astra_torch.lamino`).

Forward model (transmission with dark/flat correction)::

    l_i(mu)      = [A mu]_i                    # line integral, ASTRA forward projector
    lambda_i(mu) = I0_i * exp(-l_i(mu)) + d_i   # expected raw detector count rate

The REAL generative process has no prior -- it is just
``y_i | mu ~ Poisson(lambda_i(mu))``. The denoiser's Gamma(alpha_i, beta_i)
belief (moment-matched from its two-head ``(mu_net_i, var_net_i)`` output,
exactly as in :func:`sdate.tr_diffusion.nb_head.posterior_mean`) is not part of
that physical process -- it enters only as a prior ``p(lambda_i)``, an
inductive bias. The MAP objective combines both terms; as derived in the
loss-correction spec, up to a mu-independent constant this is EXACTLY the sum
of the per-pixel posterior Gamma(alpha_i + y_i, beta_i + 1) log-densities,
evaluated at the physically-implied lambda_i(mu) instead of collapsed to its
mean::

    L(mu) = sum_i [ beta_post_i * lambda_i(mu) - (alpha_post_i - 1) * log(lambda_i(mu)) ]
    alpha_post_i = alpha_i + y_i
    beta_post_i  = beta_i + 1

This module does NOT touch the denoiser network, its training loop, or
:func:`sdate.tr_diffusion.nb_head.posterior_mean` -- it reuses only the
moment-matched Gamma shape/rate parameters that function is built from
(recomputed here from ``(mu_net, var_net)`` with the same two-line relation,
inlined rather than imported, to keep this reconstruction step decoupled from
the training-loss module), applied directly against the physically-implied
``lambda_i(mu)``, never against a denoised point estimate. In particular the
volume is warm-started from a plain FBP of the raw counts alone (never from
the denoiser's point estimate) to avoid quietly reintroducing the same
context-correlated bias this design is meant to avoid.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def gamma_shape_rate(mu_net: torch.Tensor, var_net: torch.Tensor,
                     eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Moment-matched ``Gamma(alpha, beta)`` parameters from the denoiser's ``(mu, var)`` belief.

    Same two-line relation used (inline) in
    :func:`sdate.tr_diffusion.nb_head.nb_nll`/``posterior_mean`` --
    ``alpha = mu^2/var``, ``beta = mu/var`` -- exposed here as its own function
    since this module needs it independently of those training-loss-only
    functions (see module docstring for why it isn't imported from there).
    """
    var = var_net.clamp_min(eps)
    alpha = mu_net.pow(2) / var
    beta = mu_net / var
    return alpha, beta


def _tv_loss(mu: torch.Tensor) -> torch.Tensor:
    """Anisotropic (L1) total variation of a 3D volume: summed per-axis
    absolute finite difference (SUM, not mean, to stay on the same scale
    convention as the data term's ray SUM in :func:`map_reconstruct` -- a
    mean here would be many orders of magnitude smaller and need an
    impractically huge ``tv_weight`` to have any effect). Piecewise-constant-
    favoring rather than a quadratic smoothness penalty, so it damps
    noise-driven streaks without indiscriminately blurring real edges.
    """
    dz = (mu[1:, :, :] - mu[:-1, :, :]).abs().sum()
    dy = (mu[:, 1:, :] - mu[:, :-1, :]).abs().sum()
    dx = (mu[:, :, 1:] - mu[:, :, :-1]).abs().sum()
    return dz + dy + dx


def map_reconstruct(
    y_vrc: torch.Tensor,
    mu_net_vrc: torch.Tensor,
    var_net_vrc: torch.Tensor,
    angles_deg: np.ndarray,
    I0: torch.Tensor,
    dark: torch.Tensor,
    vol_shape: Optional[Tuple[int, int, int]] = None,
    lamino_angle_deg: float = 0.0,
    tilt_angle_deg: float = 0.0,
    voxel_size_mm: float = 1.0,
    det_spacing_mm: float = 1.0,
    n_iters: int = 300,
    lr: float = 1e-1,
    optimizer_type: str = "adam",
    clamp_min: float = 0.0,
    warm_start: str = "fbp",
    fbp_filter_type: str = "hann",
    l_clip: float = 20.0,
    eps: float = 1e-6,
    dose: float = 1.0,
    tv_weight: float = 0.0,
    log_every: int = 0,
    device: Optional[torch.device] = None,
    mu_init: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gradient-descent MAP reconstruction of the attenuation volume ``mu``.

    Parameters
    ----------
    y_vrc, mu_net_vrc, var_net_vrc :
        ``(V, R, C)`` raw per-ray transmission counts and the denoiser's
        two-head ``(mu, var)`` belief for the SAME rays, all in raw count
        units (see :func:`sdate.tr_diffusion.pipeline.denoise_frames_baseline`
        with ``poisson_head=True`` for how these are produced). Fixed inputs;
        never modified or optimised here.
    dose :
        KNOWN thinning fraction ``y_vrc`` was actually generated at -- same
        role and same failure mode as
        :func:`sdate.tr_diffusion.nb_head.nb_nll`'s ``dose`` parameter: if
        ``y_vrc`` comes from this project's synthetic
        ``add_poisson_noise(..., dose=...)`` ablation (``y = Poisson(lambda *
        dose) / dose``, same mean as native but ``1/dose`` the variance)
        rather than a real native (``dose=1``) acquisition, passing the wrong
        ``dose`` here silently mis-weights the data term against the prior by
        that same factor -- the exact bug found and fixed in ``nb_nll``/
        ``posterior_mean`` applies identically here. Default ``1.0`` (native).
    I0, dark :
        ``(R, C)`` (or broadcastable) calibrated flat-field / dark-field
        RATES, in the SAME count units as ``y_vrc`` -- confirm this before
        calling; rescale upstream if flat/dark frames were normalised
        differently (see :func:`sdate.tr_diffusion.reconstruct.load_calibration_average`
        for how this project's own flat/dark maps are produced).
    vol_shape :
        Defaults to ``(R, C, C)`` -- this project's own convention (see
        :func:`sdate.tr_diffusion.reconstruct.reconstruct`), NOT the
        differently-ordered convention some of astra_torch.lamino's docstrings
        claim; :func:`build_lamino_projector`/``gd_reconstruction_masked``
        both take this tuple positionally with no remapping, so this default
        is directly consistent with the rest of this codebase's FBP/GD calls.
    warm_start :
        ``"fbp"`` (default) -- FBP of the raw-counts-only attenuation estimate
        ``-log((y - dark) / (I0 - dark))`` (see
        :func:`sdate.tr_diffusion.reconstruct.counts_to_attenuation_flatdark`);
        deliberately NOT the denoiser's point estimate, so the data term stays
        built from ``y, alpha, beta`` only (see module docstring). ``"zeros"``
        skips FBP entirely.
    l_clip :
        Clip line integrals to ``[-l_clip, l_clip]`` before ``exp(-l)`` --
        purely a numerical-safety guard against ``exp`` overflow if ``mu``
        strays far from the true attenuation values (mainly relevant early on
        with ``warm_start="zeros"``); has no effect once ``l`` is within range.
    tv_weight :
        Weight of an anisotropic (L1) total-variation penalty on ``mu``,
        added to the per-ray data objective each step (see :func:`_tv_loss`).
        ``0.0`` (default) -- no spatial regularisation, matching the original
        spec scope exactly. Real CT data was found to make the per-ray-only
        objective diverge via generic GD/SIRT-style semi-convergence (noise
        amplified into streaks that grow without bound over iterations,
        reproduced even with NO prior at all via plain least-squares GD on
        the existing baseline arm) -- this knob exists to damp exactly that,
        without touching the data term, prior, or warm start.
    log_every :
        If ``>0``, print the objective every this many iterations. ``0``
        (default) disables logging.
    mu_init :
        Diagnostic/experimental override: if given, used as the starting
        volume verbatim instead of computing ``warm_start``. ``None``
        (default) -- no change to the documented ``warm_start`` behaviour.
        Exists to test whether GD divergence on real data is a function of
        the starting point (e.g. starting from a much better-than-raw-FBP
        volume, such as an FBP of the per-ray closed-form posterior-mode
        target) rather than the objective/optimizer themselves. Passing
        anything derived from ``mu_net``/``var_net`` here is the exact
        leakage the module docstring's warm-start policy exists to avoid --
        fine for a one-off diagnostic, not for a result meant to demonstrate
        this function fusing prior and likelihood correctly.

    Returns
    -------
    ``mu`` -- the reconstructed attenuation volume, ``vol_shape``, detached,
    clamped to ``>= clamp_min``.

    Notes
    -----
    Concavity of this objective is NOT guaranteed (the classical concavity
    result for Poisson transmission log-likelihood holds only when there is
    no additive background; ``dark > 0`` breaks it -- see Fessler/Erdogan on
    paraboloidal surrogates for the classical treatment of this exact issue).
    Adam (the default) is used rather than plain SGD for this reason. This
    MAP objective carries persistent regularisation via ``beta_post`` at every
    iteration, so the semi-convergence noise re-amplification seen in
    unregularised SIRT/OSEM is not expected here BY CONSTRUCTION -- but that
    depends on the network's uncertainty calibration being reasonable, so it
    is worth confirming empirically (e.g. a held-out count-split Poisson risk
    curve vs iteration) rather than assuming it, especially before trusting a
    fixed ``n_iters`` across very different acquisitions.

    This regularises each RAY's implied ``lambda_i`` toward the network's
    belief. By default (``tv_weight=0``) it applies NO spatial regularisation
    across voxels of ``mu`` (matches spec scope: "only mu is optimised", no
    TV/smoothness prior) -- ``tv_weight>0`` opts into one (see its docstring
    above). Confirmed on a synthetic phantom: with a well-sampled angle set, a
    confident and CORRECT per-ray prior beats plain FBP, and an uninformative
    prior degrades toward the well-known noise-amplification failure mode of
    unregularised Poisson-transmission MLE (both as expected). But with a
    SPARSE/limited angle set, the per-voxel inverse problem is genuinely
    underdetermined without spatial regularisation, so even a perfect,
    maximally-confident per-ray prior does not guarantee convergence close to
    the true volume -- gradient descent can settle on a different volume that
    still satisfies the (non-spatial) per-ray objective about as well.

    On real wunderkerze2 data (dose=0.05), the SAME divergence shows up even
    with tv_weight=0 and a well-sampled 180-view angle set: mu grows without
    bound over iterations regardless of learning rate, and was shown (via a
    plain, prior-free least-squares GD reconstruction on the raw noisy AND
    the already-good baseline arm, both with tv_weight-equivalent=0) to be
    generic GD/SIRT semi-convergence, not something specific to this
    function's Poisson/Gamma data term -- i.e. exactly the failure mode
    ``tv_weight>0`` is for.
    """
    from astra_torch.lamino import build_lamino_projector, fbp_reconstruction_masked

    from .reconstruct import counts_to_attenuation_flatdark

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = y_vrc.to(device=device, dtype=torch.float32)
    mu_net = mu_net_vrc.to(device=device, dtype=torch.float32)
    var_net = var_net_vrc.to(device=device, dtype=torch.float32)
    v, r, c = y.shape

    I0_b = I0.to(device=device, dtype=torch.float32)
    dark_b = dark.to(device=device, dtype=torch.float32)
    if I0_b.ndim == 2:
        I0_b = I0_b.unsqueeze(0)
    if dark_b.ndim == 2:
        dark_b = dark_b.unsqueeze(0)

    alpha, beta = gamma_shape_rate(mu_net, var_net, eps=eps)
    n = float(dose) * y
    alpha_post = alpha + n
    beta_post = beta + float(dose)

    if log_every:
        def _s(t):
            return f"min={t.min().item():.4g} max={t.max().item():.4g} mean={t.mean().item():.4g}"
        print(f"  [map_reconstruct] y: {_s(y)}", flush=True)
        print(f"  [map_reconstruct] mu_net: {_s(mu_net)}  var_net: {_s(var_net)}", flush=True)
        print(f"  [map_reconstruct] alpha: {_s(alpha)}  beta: {_s(beta)}", flush=True)
        print(f"  [map_reconstruct] alpha_post: {_s(alpha_post)}  beta_post: {_s(beta_post)}", flush=True)
        print(f"  [map_reconstruct] I0: {_s(I0_b)}  dark: {_s(dark_b)}", flush=True)

    vol_shape = vol_shape or (r, c, c)

    if mu_init is not None:
        mu0 = mu_init.to(device=device, dtype=torch.float32).clamp_min(0.0)
    elif warm_start == "fbp":
        atten0 = counts_to_attenuation_flatdark(y, dark_b, I0_b, eps=eps)
        mu0 = fbp_reconstruction_masked(
            atten0, angles_deg, lamino_angle_deg=lamino_angle_deg, tilt_angle_deg=tilt_angle_deg,
            voxel_size_mm=voxel_size_mm, vol_shape=vol_shape, det_spacing_mm=det_spacing_mm,
            filter_type=fbp_filter_type, device=device,
        ).clamp_min(0.0)
    elif warm_start == "zeros":
        mu0 = torch.zeros(vol_shape, dtype=torch.float32, device=device)
    else:
        raise ValueError("warm_start must be 'fbp' or 'zeros'")

    if log_every:
        print(f"  [map_reconstruct] mu0 (warm start): min={mu0.min().item():.4g} "
             f"max={mu0.max().item():.4g} mean={mu0.mean().item():.4g}", flush=True)

    mu = mu0.detach().clone().to(device).requires_grad_(True)

    proj_layer = build_lamino_projector(
        vol_shape=vol_shape, det_shape=(r, c), angles_deg=angles_deg,
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
        l = proj_layer(mu.unsqueeze(0).unsqueeze(0))[0]  # (V, R, C), A * mu
        l = l.clamp(-l_clip, l_clip)
        lam = I0_b * torch.exp(-l) + dark_b
        data_loss = (beta_post * lam - (alpha_post - 1.0) * torch.log(lam.clamp_min(eps))).sum()
        tv = _tv_loss(mu) if tv_weight > 0 else None
        loss = data_loss + tv_weight * tv if tv is not None else data_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            mu.clamp_(min=clamp_min)
        if log_every and step % log_every == 0:
            tv_str = f"  tv={tv.item():.6g}" if tv is not None else ""
            print(f"  [map_reconstruct] step {step}/{n_iters}  loss={loss.item():.6g}  "
                 f"data_loss={data_loss.item():.6g}{tv_str}  "
                 f"mu[min={mu.min().item():.4g},max={mu.max().item():.4g}]  "
                 f"l[min={l.min().item():.4g},max={l.max().item():.4g}]  "
                 f"lam[min={lam.min().item():.4g},max={lam.max().item():.4g}]", flush=True)

    return mu.detach()
