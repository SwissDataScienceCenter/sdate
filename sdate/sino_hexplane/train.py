"""Training loop + inference helpers for the sinogram-domain HexPlane.

Poisson NLL in count space: the field predicts a line integral ``p_hat``, the
measurement model maps it to expected counts ``lambda_hat = flat*exp(-p_hat)``,
and we compare to the measured Poisson-Gaussian ``counts`` (never ``p_hat`` vs a
log-corrected measurement — that mismatch degrades low-dose fits, per the
project's prior work).  The optional Anscombe loss handles a read-noise floor.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import torch

from ..tr_naf.noise import anscombe_loss, poisson_nll, predict_lambda
from .data import HelixSinoDataset
from .model import ProjectionField, default_plane_config


def build_field_for(dataset: HelixSinoDataset, meta: dict, *, encoding: str = "joint",
                    n_levels: Optional[int] = None, n_features: Optional[int] = None,
                    hidden_dim: int = 64, n_hidden_layers: int = 2,
                    backend: str = "auto", n_theta: Optional[int] = None,
                    joint_kwargs: Optional[dict] = None,
                    cold_start_value: float = 0.01) -> ProjectionField:
    """Instantiate a :class:`ProjectionField` sized to the acquisition.

    ``encoding='joint'`` (default) builds a fully-joint hash grid whose finest
    resolution follows the detector size; ``'hexplane'`` builds the six planes.
    """
    if n_theta is None:
        # Angular sampling density over a full 360 turn.
        n_theta = int(round(meta["num_projs_per_frame"] * 360.0 / meta["wedge_deg"]))
    if encoding == "joint":
        jk = dict(max_resolution=max(dataset.C, dataset.R, n_theta),
                  n_levels=n_levels or 16, n_features=n_features or 2,
                  periodic_theta=True)
        jk.update(joint_kwargs or {})
        return ProjectionField(encoding="joint", joint_kwargs=jk,
                               hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers,
                               cold_start_value=cold_start_value)
    cfg = default_plane_config(n_theta=n_theta, n_s=dataset.C, n_v=dataset.R, n_t=dataset.F)
    return ProjectionField(
        cfg, encoding="hexplane", n_levels=n_levels or 4, n_features=n_features or 4,
        backend=backend, hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers,
        cold_start_value=cold_start_value)


def train_projection_field(
    dataset: HelixSinoDataset, meta: dict, *,
    n_iters: int = 3000, batch_size: int = 65536, lr: float = 1e-2,
    data_fidelity: str = "poisson_nll", encoding: str = "joint",
    n_levels: Optional[int] = None, n_features: Optional[int] = None,
    hidden_dim: int = 64, n_hidden_layers: int = 2, backend: str = "auto",
    n_theta: Optional[int] = None, joint_kwargs: Optional[dict] = None,
    cold_start_value: float = 0.01,
    log_every: int = 200, seed: int = 0, device: Optional[torch.device] = None,
    model: Optional[ProjectionField] = None,
) -> Dict:
    """Fit a projection field on count data. Returns model + loss history.

    ``encoding`` selects the joint hash grid (default) or the HexPlane."""
    device = device or dataset.device
    torch.manual_seed(seed)
    if model is None:
        model = build_field_for(
            dataset, meta, encoding=encoding, n_levels=n_levels, n_features=n_features,
            hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers, backend=backend,
            n_theta=n_theta, joint_kwargs=joint_kwargs, cold_start_value=cold_start_value)
    model = model.to(device)

    flat = float(meta["photons"])
    read_noise = float(meta.get("read_noise", 0.0))
    gain = float(meta.get("gain", 1.0))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses: List[float] = []
    t0 = time.time()
    for it in range(n_iters):
        batch = dataset.sample(batch_size)
        p_hat = model(batch["theta"], batch["s"], batch["v"], batch["t"])
        lam = predict_lambda(p_hat, flat)                        # flat*exp(-p_hat)
        if data_fidelity == "poisson_nll":
            loss = poisson_nll(lam, batch["counts"])
        elif data_fidelity == "anscombe":
            loss = anscombe_loss(lam, batch["counts"], read_noise=read_noise, gain=gain)
        else:
            raise ValueError(f"unknown data_fidelity {data_fidelity!r}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        losses.append(float(loss.detach()))
        if log_every and (it % log_every == 0 or it == n_iters - 1):
            print(f"[{it:5d}/{n_iters}] {data_fidelity}={losses[-1]:.4f}  "
                  f"({time.time()-t0:.1f}s)")

    return {"model": model, "losses": losses, "time": time.time() - t0,
            "config": dict(encoding=encoding, n_levels=n_levels, n_features=n_features,
                           hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers,
                           n_theta=n_theta, data_fidelity=data_fidelity)}


@torch.no_grad()
def predict(model: ProjectionField, coords: Dict[str, torch.Tensor],
            chunk: int = 1_000_000) -> torch.Tensor:
    """Chunked forward over flat coordinate tensors -> ``p_hat`` (same length)."""
    model.eval()
    n = coords["theta"].shape[0]
    out = torch.empty(n, device=coords["theta"].device)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        out[lo:hi] = model(coords["theta"][lo:hi], coords["s"][lo:hi],
                           coords["v"][lo:hi], coords["t"][lo:hi])
    return out


@torch.no_grad()
def predict_frame(model: ProjectionField, dataset: HelixSinoDataset, frame_idx: int,
                  chunk: int = 1_000_000) -> torch.Tensor:
    """Predicted clean line integrals for one frame -> ``(A, R, C)``."""
    coords = dataset.frame_coords(frame_idx)
    p = predict(model, coords, chunk=chunk)
    return p.reshape(dataset.A, dataset.R, dataset.C)


@torch.no_grad()
def predict_sino(model: ProjectionField, dataset: HelixSinoDataset, t_norm: float,
                 thetas_norm, chunk: int = 1_000_000) -> torch.Tensor:
    """Predicted line-integral sinogram ``(A', R, C)`` at time ``t_norm`` over an
    arbitrary set of normalised angles (measured or unseen)."""
    coords, (A, R, C) = dataset.sino_coords(t_norm, thetas_norm)
    p = predict(model, coords, chunk=chunk)
    return p.reshape(A, R, C)
