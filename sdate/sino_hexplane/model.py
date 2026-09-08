"""Sinogram-domain HexPlane INR: ``ProjectionField``.

Maps a 4-D projection-space coordinate ``(theta, s, v, t)`` (all normalised to
``[0, 1]``; ``theta`` periodic) to a **line integral** ``p_hat >= 0``.  The
measurement model then turns that into an expected photon count
``lambda_hat = flat * exp(-p_hat)`` (see :mod:`sdate.tr_naf.noise`), so the loss
lives in count space (Poisson NLL) while the field itself is the denoised
*projection* — exactly the quantity we score against the clean sinogram.

Architecture (K-Planes / HexPlane): factorise the 4-D input into all
``C(4,2) = 6`` coordinate pairs, one :class:`~sdate.sino_hexplane.planes.Plane2D`
each, combine the six feature vectors by an elementwise (Hadamard) **product**,
then decode with a tiny MLP.  Multiplication (not addition) is essential: it
selects a genuine joint coordinate instead of producing additive "ghost" support
along each plane's full extent.

The six planes and their roles (see the project spec / memory
``project-sino-hexplane``):

* ``theta_s`` classical sinogram (theta periodic)   * ``s_v`` static per-frame image
* ``theta_v`` axis-tilt diagnostic (near-flat)      * ``s_t`` transients localised in s
* ``theta_t`` rotation-time evolution (coarse)      * ``v_t`` transients localised in v

Only ``theta`` is periodic; ``s`` (detector column) and ``v`` (detector row /
rotation axis) are Euclidean.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .planes import Plane2D


# The six pairwise planes over (theta, s, v, t).  ``periodic`` marks which of the
# pair's (a, b) axes wraps — only theta ever does, and it is always axis ``a``.
_PLANES: Tuple[Tuple[str, str, str], ...] = (
    ("theta_s", "theta", "s"),
    ("theta_v", "theta", "v"),
    ("theta_t", "theta", "t"),
    ("s_v",     "s",     "v"),
    ("s_t",     "s",     "t"),
    ("v_t",     "v",     "t"),
)


def default_plane_config(n_theta: int = 180, n_s: int = 128, n_v: int = 128,
                         n_t: int = 25) -> Dict[str, dict]:
    """Sensible per-plane resolutions from the acquisition dimensions.

    ``theta_t`` is kept coarse in theta (rotation-time evolution is smooth);
    the ``*_t`` planes use ``auto`` backend so they stay dense at small frame
    counts and fall back to hashing at real detector/time scales.  These are the
    fixed v1 defaults — the per-plane count-splitting sweep (spec 5.5) that would
    tune them automatically is deferred.
    """
    theta_coarse = max(16, n_theta // 2)
    return {
        "theta_s": dict(max_res=(n_theta, n_s),        base_res=(8, 8)),
        "theta_v": dict(max_res=(n_theta, n_v),        base_res=(8, 8)),
        "theta_t": dict(max_res=(theta_coarse, n_t),   base_res=(8, min(4, n_t))),
        "s_v":     dict(max_res=(n_s, n_v),            base_res=(8, 8)),
        "s_t":     dict(max_res=(n_s, n_t),            base_res=(8, min(4, n_t))),
        "v_t":     dict(max_res=(n_v, n_t),            base_res=(8, min(4, n_t))),
    }


class HexPlaneEncoding(nn.Module):
    """Six 2-D planes over ``(theta, s, v, t)``, combined by Hadamard product."""

    def __init__(self, plane_config: Dict[str, dict], n_levels: int = 4,
                 n_features: int = 4, backend: str = "auto",
                 init_mean: float = 1.0, init_std: float = 0.1):
        super().__init__()
        self.planes = nn.ModuleDict()
        for name, axis_a, _axis_b in _PLANES:
            cfg = dict(plane_config[name])
            periodic_a = (axis_a == "theta")
            self.planes[name] = Plane2D(
                n_levels=n_levels, n_features=n_features, backend=backend,
                periodic_a=periodic_a, periodic_b=False,
                init_mean=init_mean, init_std=init_std, **cfg)
        self.output_dim = next(iter(self.planes.values())).output_dim
        for p in self.planes.values():
            if p.output_dim != self.output_dim:
                raise ValueError("All planes must share output_dim for Hadamard fusion.")

    def forward(self, theta: torch.Tensor, s: torch.Tensor,
                v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        coords = {"theta": theta, "s": s, "v": v, "t": t}
        f = None
        for name, axis_a, axis_b in _PLANES:
            g = self.planes[name](coords[axis_a], coords[axis_b])
            f = g if f is None else f * g
        return f

    @property
    def n_params(self) -> int:
        return sum(p.n_params for p in self.planes.values())

    def describe(self) -> str:
        return "\n".join(f"  {n:8s} {self.planes[n].describe()}" for n, _, _ in _PLANES)


class ProjectionField(nn.Module):
    """Coordinate encoding + tiny decoder MLP -> non-negative line integral ``p_hat``.

    Two swappable encodings (``encoding``):

    * ``"joint"`` (default, recommended) — a single fully-joint multi-resolution
      hash grid (:class:`~sdate.sino_hexplane.encodings.JointHashEncoding`).
      Empirically fits clean sinograms ~15 dB better than the HexPlane and trains
      faster; this is the spec's fallback and matches object-space NAF.
    * ``"hexplane"`` — the interpretable pairwise K-Planes factorisation
      (:class:`HexPlaneEncoding`), kept for the ANOVA / per-plane studies.

    Parameters
    ----------
    plane_config : per-plane resolutions for the HexPlane (see
        :func:`default_plane_config`); ignored for the joint encoding.
    joint_kwargs : kwargs for :class:`JointHashEncoding` (joint encoding only).
    n_levels, n_features, backend, init_mean, init_std : HexPlane hyper-parameters.
    hidden_dim, n_hidden_layers : decoder MLP.
    cold_start_value : initial uniform ``p_hat`` everywhere (decoder output layer
        zero-initialised, bias set so ``softplus(bias) == cold_start_value``).  A
        small value (~0) makes ``lambda_hat ~ flat = I0`` at init — the
        well-conditioned starting point for Poisson gradients (transparent object).
    """

    def __init__(self, plane_config: Optional[Dict[str, dict]] = None,
                 encoding: str = "joint", joint_kwargs: Optional[dict] = None,
                 n_levels: int = 4, n_features: int = 4, backend: str = "auto",
                 hidden_dim: int = 64, n_hidden_layers: int = 2,
                 cold_start_value: float = 0.01,
                 init_mean: float = 1.0, init_std: float = 0.1):
        super().__init__()
        self.encoding_type = encoding
        if encoding == "hexplane":
            if plane_config is None:
                raise ValueError("plane_config is required for encoding='hexplane'")
            self.encoding = HexPlaneEncoding(
                plane_config, n_levels=n_levels, n_features=n_features,
                backend=backend, init_mean=init_mean, init_std=init_std)
        elif encoding == "joint":
            from .encodings import JointHashEncoding
            self.encoding = JointHashEncoding(**(joint_kwargs or {}))
        else:
            raise ValueError(f"unknown encoding {encoding!r} (use 'joint' or 'hexplane')")

        layers = []
        in_dim = self.encoding.output_dim
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)

        # Cold start: zero output weights -> uniform p_hat = softplus(bias) everywhere,
        # independent of the encoding features. Gradients still flow to the encoding
        # via the head weight gradient (nonzero after step 1).
        nn.init.zeros_(self.head.weight)
        y = max(float(cold_start_value), 1e-6)
        bias = float(np.log(np.expm1(y))) if y < 20 else y   # softplus^{-1}
        nn.init.constant_(self.head.bias, bias)

    def forward(self, theta: torch.Tensor, s: torch.Tensor,
                v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return ``p_hat >= 0`` with the same batch shape as the inputs."""
        z = self.encoding(theta, s, v, t)
        out = self.head(self.trunk(z)).squeeze(-1)
        return F.softplus(out)

    def describe(self) -> str:
        return (f"ProjectionField [{self.encoding_type}]: "
                f"{self.encoding.n_params/1e6:.2f}M encoding params\n"
                + self.encoding.describe())
