"""Raw-count measurement model + count-space data-fidelity losses.

The Poisson process lives in the **raw detector counts**, not in attenuation.
This module keeps that structure explicit so the *same* code path serves both a
simulation (fill counts by forward-projecting a phantom) and real GigaFrost data
(fill counts/flat/dark from the raw + calibration frames) — the model and losses
never change.

Measurement model (charge-integrating CMOS, g=1 photon units by default):

    lambda_j = flat_j * exp(-p_j)                 # expected photons (flat = I0)
    N_j      = gain * Poisson(lambda_j) + Normal(0, read_noise) + dark_j

so ``Var(N_j) ≈ gain^2 * lambda_j + read_noise^2`` — shot noise plus a Gaussian
read-noise floor.

Losses compare **predicted counts** ``lambda_hat_j = flat_j * exp(-p_hat_j)`` to
the measured ``N_j`` in count space (never ``p_hat`` vs ``-ln(N/flat)``):

* ``poisson_nll`` — ``sum_j (lambda_hat_j - N_j * ln lambda_hat_j)``; exact for a
  pure Poisson detector, a good approximation away from the read-noise floor.
* ``anscombe`` — generalized Anscombe variance-stabilizing transform, then MSE on
  the (≈unit-variance) residual; handles the Poisson-Gaussian floor at low counts.
"""

from __future__ import annotations

from typing import Optional

import torch


def simulate_counts(
    line_integral: torch.Tensor,        # (..., R, C) scaled line integrals p (>=0)
    flat: torch.Tensor,                 # (R, C) or scalar — I0 (expected photons, no sample)
    gain: float = 1.0,
    read_noise: float = 0.0,
    dark: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Draw raw counts ``N`` from the Poisson-Gaussian model. Returns float counts."""
    lam = flat * torch.exp(-line_integral)                      # expected photons
    shot = torch.poisson(lam.clamp_min(0.0), generator=generator)
    counts = gain * shot
    if read_noise > 0.0:
        counts = counts + torch.randn(counts.shape, device=counts.device,
                                      generator=generator) * read_noise
    counts = counts + dark
    return counts


def predict_lambda(p_hat: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
    """Predicted expected-counts ``lambda_hat = flat * exp(-p_hat)``."""
    return flat * torch.exp(-p_hat)


def poisson_nll(lambda_hat: torch.Tensor, counts: torch.Tensor,
                eps: float = 1e-6) -> torch.Tensor:
    """Mean Poisson negative log-likelihood (up to a data-only constant).

    ``mean_j (lambda_hat_j - N_j * ln lambda_hat_j)``.  ``counts`` clamped >= 0
    (read-noise can make raw counts slightly negative).
    """
    lam = lambda_hat.clamp_min(eps)
    n = counts.clamp_min(0.0)
    return (lam - n * torch.log(lam)).mean()


def _gat(x: torch.Tensor, read_noise: float, gain: float = 1.0) -> torch.Tensor:
    """Generalized Anscombe transform (Makitalo-Foi), unit-variance stabilization."""
    # For y = gain*Poisson(lam) + N(0, sigma): 2/gain * sqrt(gain*y + 3/8 gain^2 + sigma^2)
    inner = gain * x + (3.0 / 8.0) * gain * gain + read_noise * read_noise
    return (2.0 / gain) * torch.sqrt(inner.clamp_min(0.0))


def anscombe_loss(lambda_hat: torch.Tensor, counts: torch.Tensor,
                  read_noise: float = 0.0, gain: float = 1.0) -> torch.Tensor:
    """MSE between the Anscombe-stabilized measured counts and predicted mean.

    After the transform the residual is ~unit-variance Gaussian, so plain MSE is
    the (approximate) NLL and it correctly down-weights the darkest pixels.
    """
    t_meas = _gat(counts, read_noise, gain)
    t_pred = _gat(lambda_hat * gain, read_noise, gain)   # stabilized model mean
    return (t_meas - t_pred).pow(2).mean()


def counts_to_line_integral(counts: torch.Tensor, flat: torch.Tensor,
                            dark: float = 0.0, min_count: float = 1.0) -> torch.Tensor:
    """Classical log-correction ``p = -ln((N - dark) / flat)`` for the FBP baseline.

    Dark-subtracted counts are floored at ``min_count`` (>=1 photon) to avoid
    ``log(0)`` on starved rays.  Not used in the NAF loss (which stays in count
    space) — only to feed a conventional analytic reconstruction for comparison.
    """
    c = (counts - dark).clamp_min(min_count)
    trans = (c / flat).clamp(min=1e-8, max=1.0)
    return -torch.log(trans)
