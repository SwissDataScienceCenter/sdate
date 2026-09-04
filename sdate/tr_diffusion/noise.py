"""Extra Poisson noise for the noise-sweep evaluation regime.

There is no clean ground truth for these projections, so absolute denoising
quality is hard to score.  The trick (requested design): take a measured frame —
already noisy at its native dose — and inject *additional* Poisson noise to
synthesise a **lower-dose, noisier** version.  The model then denoises the
noisier frame, and the *original* measured frame (which has strictly less noise)
serves as a pseudo-reference for PSNR/SSIM.  It is not a true GT, but it is
provably less noisy than the model input, so relative comparisons across noise
levels and between models are meaningful.

Dose reduction by Poisson thinning keeps the mean intensity fixed while raising
the relative variance::

    noisy = Poisson(counts * dose) / dose        (0 < dose <= 1)

``dose = 1`` layers one extra Poisson draw (slightly noisier); ``dose -> 0``
approaches a photon-starved regime.  Counts must be in true detector units
(non-negative), i.e. denormalised — see :mod:`sdate.tr_diffusion.frames`.
"""

from __future__ import annotations

from typing import Optional

import torch


def add_poisson_noise(
    counts: torch.Tensor,
    dose: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Return a noisier version of ``counts`` at effective dose fraction ``dose``.

    ``counts`` is any shape of non-negative detector counts (float).  The result
    has the same mean but higher variance for ``dose < 1``.
    """
    if not 0.0 < dose <= 1.0:
        raise ValueError("dose must be in (0, 1]")
    lam = (counts.clamp_min(0.0) * dose)
    noised = torch.poisson(lam, generator=generator)
    return noised / dose


def binomial_split(counts, dose, p, generator=None):
    """Split a ``dose``-fraction measurement of ``counts`` into two independent views.

    Noise2Noise via Poisson thinning + binomial splitting: draw the integer photon
    count of the fixed measurement ``N = Poisson(counts * dose)``, then split it
    ``N1 ~ Binomial(N, p)``, ``N2 = N - N1``. By the thinning property N1 and N2 are
    **independent** given the signal, so they form a valid N2N (input, target) pair
    from a single measurement. Both are rescaled to native count space::

        input  = N1 / (dose * p)          # ~ dose*p fraction of the photons
        target = N2 / (dose * (1 - p))    # ~ dose*(1-p) fraction

    each an unbiased estimate of ``counts`` (E = counts) with independent noise.
    e.g. dose=0.05, p=0.5 -> two independent 2.5%-dose views of the same frame.

    ``counts`` are non-negative native detector counts. ``generator`` seeds the
    Poisson draw (the binomial split is always fresh; N2N training wants fresh pairs).
    """
    if not 0.0 < dose <= 1.0:
        raise ValueError("dose must be in (0, 1]")
    if not 0.0 < p < 1.0:
        raise ValueError("split fraction p must be in (0, 1)")
    N = torch.poisson(counts.clamp_min(0.0) * dose, generator=generator)
    N1 = torch.binomial(N, torch.full_like(N, float(p)))
    N2 = N - N1
    return N1 / (dose * p), N2 / (dose * (1.0 - p))


def binomial_complementary_split(counts: torch.Tensor, p: float = 0.5,
                                 generator: Optional[torch.Generator] = None):
    """Split an ALREADY-REALISED measurement into two complementary halves (inference-time).

    Unlike :func:`binomial_split` (which draws a fresh Poisson realisation, for
    training), this operates on one real, already-measured count image ``counts``:
    ``N1 ~ Binomial(round(counts), p)``, ``N2 = round(counts) - N1``, both
    rescaled to native count space (``N1/p``, ``N2/(1-p)``). Unlike
    :func:`binomial_thin` (which discards the complement), this returns BOTH
    halves so the model can be conditioned on each in turn and the two resulting
    estimates averaged — the inference-time analogue of the swap-consistency
    training objective (:class:`sdate.tr_diffusion.losses.DiffusionN2NLoss` with
    ``consistency_weight > 0``).
    """
    if not 0.0 < p < 1.0:
        raise ValueError("split fraction p must be in (0, 1)")
    N = counts.clamp_min(0.0).round()
    N1 = torch.binomial(N, torch.full_like(N, float(p)), generator=generator)
    N2 = N - N1
    return N1 / p, N2 / (1.0 - p)


def binomial_thin(counts: torch.Tensor, q: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Further thin an ALREADY-REALISED measurement by fraction ``q`` (inference-time).

    Unlike :func:`binomial_split` (which draws a fresh Poisson realisation, for
    training), this operates on one real, already-measured count image ``counts``
    (e.g. the actual dose-0.05 acquired frame): ``N1 ~ Binomial(round(counts), q)``,
    rescaled ``N1 / q``. Used to synthesise "what a smaller input fraction of THIS
    measurement would have looked like" for the N2N inference-fraction ablation.
    ``q >= 1`` returns ``counts`` unchanged (no thinning -> the largest, least-noisy
    input, using the full available measurement).
    """
    if q >= 1.0:
        return counts
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1]")
    N = counts.clamp_min(0.0).round()
    N1 = torch.binomial(N, torch.full_like(N, float(q)), generator=generator)
    return N1 / q
