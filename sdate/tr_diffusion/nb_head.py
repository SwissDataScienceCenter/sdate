"""Two-head Gamma-Poisson belief: exact Negative-Binomial NLL + posterior-mean combination.

Context: the single-channel baseline head (:mod:`sdate.tr_diffusion.losses`
``BaselineN2VLoss`` with ``loss_type="huber"``/``mae``/``mse``) trains a point
estimate against the wrong (homoscedastic) noise model, and -- independent of
loss choice -- the blind-spot receptive field means that point estimate is a
function of context only (it never sees the observed pixel), so any such loss
converges to ``E[x | context]`` and cannot get sharper than that. This module
fixes both: the head outputs a per-pixel Gamma-Poisson prior belief
``(mu, var)`` trained with the exact Negative-Binomial marginal likelihood, and
at inference that belief is updated with the REAL observed count via the exact
conjugate posterior -- so the value ultimately written out is a genuine
function of the observation, not purely of context.

Everything here operates in RAW DETECTOR-COUNT units (non-negative,
denormalised -- see :mod:`sdate.tr_diffusion.noise`), NOT the ``[-1, 1]``
normalised space the network's input channels live in: the Gamma-Poisson
conjugacy is exact only for the true (non-negative, unshifted) count variable.
Callers are responsible for denormalising targets/observations and
renormalising results (see ``poisson_head=True`` in
:class:`~sdate.tr_diffusion.losses.BaselineN2VLoss` and
:func:`~sdate.tr_diffusion.pipeline.denoise_frames_baseline`).

This is opt-in and does not touch the existing single-channel path.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def split_mu_var(raw: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """``raw`` = the 2-channel head output ``(B, 2, H, W)`` -> ``(mu, var)``, each ``(B, 1, H, W)``.

    ``mu`` = context-predicted mean count (rate); ``var`` = the head's
    self-reported uncertainty about that mean. Both strictly positive.
    """
    a, b = raw[:, 0:1], raw[:, 1:2]
    mu = F.softplus(a) + eps
    var = F.softplus(b) + eps
    return mu, var


def poisson_nll(y: torch.Tensor, mu: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Plain Poisson NLL ``mu - y*log(mu)`` (drops the ``y``-only ``log(y!)`` constant), per-pixel.

    Used for the ``mu``-only warm start: with ``var`` unused (hence receiving no
    gradient) this fits ``mu`` alone, before switching on the full
    heteroscedastic :func:`nb_nll` below -- see the training-stability note in
    the loss-correction spec (heteroscedastic losses can collapse by cheaply
    inflating ``var`` instead of improving ``mu`` if trained jointly from
    scratch).
    """
    return mu - y * torch.log(mu.clamp_min(eps))


def nb_nll(y: torch.Tensor, mu: torch.Tensor, var: torch.Tensor, dose: float = 1.0,
          eps: float = 1e-6) -> torch.Tensor:
    """Exact Negative-Binomial NLL, per-pixel (no reduction).

    Moment-matches a ``Gamma(alpha, beta)`` prior over the clean rate
    ``lambda`` to ``(mu, var)`` (``alpha = mu^2/var``, ``beta = mu/var``) and
    marginalises against ``Poisson(lambda * dose)``, giving the exact NB
    marginal for the observed count ``y``. Drops the ``y``-only ``log(y!)``
    and ``log(dose)`` constants (don't affect gradients).

    ``dose`` is the KNOWN thinning fraction of the observation, i.e. ``y`` is
    assumed generated as :func:`sdate.tr_diffusion.noise.add_poisson_noise`
    does -- ``y = Poisson(lambda * dose) / dose`` -- which has the same mean as
    a native (``dose=1``) measurement but ``1/dose`` times the variance. At the
    default ``dose=1`` this reduces exactly to the plain Poisson(lambda)
    marginal. Getting this wrong (e.g. always assuming ``dose=1`` for data that
    was actually synthesised at ``dose=0.05``) means the network has no way to
    know the extra ``1/dose`` variance is a KNOWN artefact of the synthetic
    thinning rather than genuine uncertainty about ``lambda`` -- it then
    (correctly, given the wrong likelihood) inflates ``var`` by roughly
    ``1/dose`` to explain ``y``'s actual scatter, and the posterior combination
    below over-trusts that inflated ``var`` and leans on the noisy ``y`` far
    more than it should.
    """
    n = float(dose) * y
    alpha = mu.pow(2) / (var + eps)
    beta = mu / (var + eps)
    return (
        torch.lgamma(alpha)
        - torch.lgamma(n + alpha)
        - alpha * torch.log(beta + eps)
        + (alpha + n) * torch.log(beta + float(dose))
    )


def nb_nll_gaussian(y: torch.Tensor, mu: torch.Tensor, var: torch.Tensor, sigma_read2, dose: float = 1.0,
                    eps: float = 1e-6) -> torch.Tensor:
    """Gaussian-CLT approximation to Poisson(dose*mu)/dose convolved with the
    detector's own additive Gaussian read noise, per-pixel (no reduction).

    The real detector noise is Poisson (photon/dark-current shot noise) PLUS
    Gaussian (electronic read noise) -- :func:`nb_nll` only models the Poisson
    part (via a Gamma-mixed marginal), with no read-noise floor at all. That
    omission is harmless at the heavily-thinned dose=0.05 regime this project
    has used until now (counts are small, shot noise dominates, and the exact
    NB marginal's discreteness/skew genuinely matters there) but breaks down
    at NATIVE/full-dose counts (roughly 1/dose ~20x larger): immediately after
    the mu-only warmup phase, the ``var`` head is still near its untrained
    initialisation while ``mu`` is now large, so ``alpha = mu^2/var`` in
    :func:`nb_nll` can explode (e.g. mu=1000, var~1 gives alpha~1e6; var
    collapsing further makes it far worse) -- ``lgamma(alpha) - lgamma(n +
    alpha)`` and ``alpha*log(beta) - (alpha+n)*log(beta+dose)`` then subtract
    two float32 numbers of nearly equal huge magnitude, losing all precision
    and producing inf/nan. Confirmed: this is exactly what happened (NaN mid-
    epoch, first native-noise training run, see project memory
    project-tr-diffusion-jointfbpctx).

    Use this instead for native/high-count training. ``mu`` is large enough
    there for the Poisson component's own CLT Gaussian limit to be an
    excellent approximation, so the full noise model collapses to a single
    heteroscedastic Gaussian NLL:

        total_var = dose*mu   (Poisson shot-noise variance)
                   + var       (the head's own extra/epistemic overdispersion,
                                same role as in nb_nll -- Var_NB[y] = mu + var
                                is the exact NB identity this generalises)
                   + sigma_read2  (the detector's OWN additive Gaussian floor,
                                   estimated from real dark-frame variance --
                                   see scripts/tr_diffusion_estimate_read_noise.py --
                                   NOT learned, a fixed known per-pixel constant)

    ``sigma_read2`` is a hard floor on ``total_var`` (it cannot depend on the
    network), which is what structurally prevents the alpha-explosion failure
    mode above regardless of what the var head does. ``sigma_read2`` may be a
    python float (global) or a ``(H, W)``/broadcastable tensor (per-pixel,
    preferred -- see the calibration-map convention used for flat/dark
    elsewhere in this project).
    """
    total_var = (mu * float(dose) + var + sigma_read2).clamp_min(eps)
    return 0.5 * torch.log(total_var) + (y - mu).pow(2) / (2.0 * total_var)


def beta_nll_weight(mu: torch.Tensor, var: torch.Tensor, power: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """Detached rebalancing weight ``(alpha + beta) ** (-power)`` (beta-NLL style,
    Seitzer et al. 2022).

    The gradient on ``mu`` from :func:`nb_nll` scales inversely with ``var``, so
    the network can cheaply shrink the loss early on by inflating ``var``
    rather than improving ``mu``. Multiplying the per-pixel NB-NLL by this
    detached weight rebalances the gradient magnitude between low- and
    high-count pixels. Only worth enabling if collapse (``var`` growing while
    validation PSNR plateaus) is still observed after the warm start.
    """
    alpha = mu.pow(2) / (var + eps)
    beta = mu / (var + eps)
    return (alpha + beta).detach().clamp_min(eps).pow(-float(power))


def posterior_mean(y: torch.Tensor, mu: torch.Tensor, var: torch.Tensor, dose: float = 1.0,
                   eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact Gamma-Poisson posterior mean/variance of ``lambda`` given ``y`` (inference only).

    ``mu``/``var`` are the network's context-only prior belief; ``y`` is the
    REAL observed count at that pixel -- unlike training (which only scores
    ``(mu, var)`` at blind-spotted pixels), inference has no mask, so every
    pixel is combined with its own true observation. ``dose`` MUST match the
    thinning fraction ``y`` was actually generated at (see :func:`nb_nll`) --
    at the default ``dose=1`` (native, non-thinned measurement) this is
    ``(mu^2 + y*var) / (mu + var)``; for ``dose < 1`` (a synthetically
    thinned/noisier evaluation regime) ``y`` is known to be ``1/dose`` times
    noisier than a native measurement, so it is trusted proportionally less --
    as ``dose -> 0`` a single thinned observation carries almost no
    information and ``x_hat -> mu`` (the prior), rather than being pulled
    toward an untrustworthy ``y`` as the naive ``dose=1`` formula would do.

    Expected behaviour (log/sanity-check once, per the spec): at high counts /
    low local uncertainty (small ``var``, context confident and accurate),
    ``x_hat -> mu`` (same as the current denoiser); at low counts / high local
    uncertainty (large ``var``, e.g. a pixel straddling a moving edge where
    neighbouring context disagrees with itself), ``x_hat`` leans toward ``y``
    (defer to the real noisy count rather than smoothing it away), tempered by
    ``dose``. If ``var`` collapses to near-zero everywhere (``x_hat ~= mu``
    always, even at edges), that is the loss-attenuation failure mode --
    needs the warm start before drawing any conclusion about whether this
    resolves the blur.
    """
    dose = float(dose)
    denom = mu + dose * var + eps
    x_hat = (mu.pow(2) + dose * y * var) / denom
    var_post = x_hat * var / denom
    return x_hat, var_post


def gamma_sample(mu: torch.Tensor, var: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fresh draw from the moment-matched ``Gamma(alpha, beta)`` prior over the
    clean rate ``lambda`` (``alpha = mu^2/var``, ``beta = mu/var`` -- the same
    moment-matching :func:`nb_nll`/:func:`posterior_mean` use).

    An independently-noisy ALTERNATIVE to using ``mu`` (the prior's mean) as a
    deterministic point estimate -- see
    :class:`sdate.tr_diffusion.losses.BootstrapPoissonLoss` and
    :func:`sdate.tr_diffusion.pipeline.denoise_frames_bootstrap`, which condition
    a second-stage model on this sample instead of ``mu`` to test whether being
    forced to see through the prior's OWN local uncertainty (rather than always
    the same point value) yields a sharper final estimate.
    """
    alpha = mu.pow(2) / (var + eps)
    beta = mu / (var + eps)
    return torch.distributions.Gamma(alpha.clamp_min(eps), beta.clamp_min(eps)).sample()
