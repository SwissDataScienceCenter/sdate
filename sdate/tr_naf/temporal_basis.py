"""Fixed smooth temporal bases for time-resolved NAF.

The time-resolved field emits ``K`` coefficients per voxel; the attenuation at
a given (normalised) scan time ``t`` is

    mu(x, y, z, t) = sigma( sum_k c_k(x, y, z) * phi_k(t) )

where ``phi_k`` is a **fixed** smooth basis over ``t in [0, 1]`` (the whole
scan spans the interval).  Because the acquisition times ``t_j`` are known
constants, the design matrix ``Phi[j, k] = phi_k(t_j)`` is computed *once* and
frozen in a buffer — autograd only ever needs gradients w.r.t. the
coefficients ``c`` (a plain multiply-and-sum), never w.r.t. ``t``.  Swap this
module for a torch-native differentiable evaluator only if the acquisition
times themselves become learnable (e.g. joint time-calibration refinement).

``K`` is the temporal degrees-of-freedom knob: ``K = 1`` is a static object
(full-angle problem, ~SW-FBP), and each extra coefficient buys temporal detail
at the cost of angular coverage.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import BSpline


class BSplineTemporalBasis(nn.Module):
    """Clamped cubic B-spline basis on the normalised scan interval ``[0, 1]``.

    Parameters
    ----------
    K : Number of basis functions = coefficients per voxel (``K >= degree + 1``).
    degree : Spline degree (3 = cubic).
    """

    def __init__(self, K: int = 6, degree: int = 3):
        super().__init__()
        if K < 1:
            raise ValueError(f"K={K} must be >= 1.")
        # Auto-lower the degree for small K so K=1 gives a static (degree-0,
        # piecewise-constant) field, K=2 linear, K=3 quadratic, K>=4 cubic.
        degree = min(int(degree), int(K) - 1)
        self.K = int(K)
        self.degree = int(degree)

        # Clamped knot vector: (degree+1) repeated knots at each end, interior
        # knots uniformly spaced.  #basis functions = len(knots) - degree - 1 = K.
        n_interior = K - degree - 1
        interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
        self.knots = np.concatenate(
            [np.zeros(degree + 1), interior, np.ones(degree + 1)]
        ).astype(np.float64)

        # Second-difference (P-spline) roughness operator: penalty = c^T R c per
        # voxel approximates the curvature energy of the temporal curve.  Needs
        # at least 3 coefficients to form a second difference; otherwise no
        # curvature penalty applies (K=1 static, K=2 linear).
        if self.K >= 3:
            D2 = np.diff(np.eye(self.K), n=2, axis=0)       # (K-2, K)
            R = D2.T @ D2
        else:
            R = np.zeros((self.K, self.K))
        self.register_buffer("R", torch.tensor(R, dtype=torch.float32))

    @torch.no_grad()
    def design(self, t_norm: torch.Tensor) -> torch.Tensor:
        """Return the ``(T, K)`` design matrix ``Phi[j, k] = phi_k(t_j)``.

        ``t_norm`` : ``(T,)`` normalised times in ``[0, 1]``.  Evaluated once at
        setup and stored by the caller; not part of the autograd graph.
        """
        t = t_norm.detach().cpu().numpy().astype(np.float64)
        # Guard the right endpoint: BSpline with extrapolate=False returns NaN
        # exactly at the final knot.  Nudge t=1 just inside the support.
        t = np.clip(t, 0.0, 1.0 - 1e-9)
        Phi = BSpline.design_matrix(t, self.knots, self.degree).toarray()
        return torch.tensor(Phi, dtype=torch.float32, device=t_norm.device)
