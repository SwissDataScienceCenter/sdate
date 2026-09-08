"""Per-scan self-supervised time-resolved NAF reconstruction.

Fits a :class:`~sdate.tr_naf.model.TimeResolvedNafField` to a set of
limited-angle frames (each with its own ``t`` and angular wedge).  Mirrors
``smartt.saxs_naf.reconstruct.saxs_naf_reconstruction`` in spirit:

* one cached ASTRA projector per frame (built once);
* AdamW + linear-warmup/cosine-decay LR;
* **coarse-to-fine spatial** annealing (reveal hash levels low->high);
* **temporal coarse-to-fine** via an *annealed roughness weight* that starts
  stiff (forcing a near-static field ~ the well-posed SW-FBP regime) and relaxes
  so motion emerges only as the data supports it;
* squared-TV spatial regulariser on the coefficient field.

The forward model is ``build_lamino_projector`` (a differentiable
``torch.autograd.Function``), so gradients flow field -> coefficients ->
volume(t) -> sinogram -> loss end to end.
"""

from __future__ import annotations

import math
import time as _time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .data import Frame
from .model import TimeResolvedNafField
from .temporal_basis import BSplineTemporalBasis
from .noise import (
    predict_lambda as noise_predict_lambda,
    poisson_nll as noise_poisson_nll,
    anscombe_loss as noise_anscombe,
)


def _cosine_warmup_lr(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))


def _linear_reveal(progress: float, n_units: int, frac: float) -> torch.Tensor:
    """FreeNeRF-style: fade in ``n_units`` low->high over the first ``frac`` of training."""
    w = torch.zeros(n_units)
    w[0] = 1.0
    if n_units == 1 or frac <= 0:
        return torch.ones(n_units) if frac <= 0 else w
    revealed = min(progress / frac, 1.0) * (n_units - 1)
    full = int(revealed)
    for i in range(1, min(full + 1, n_units)):
        w[i] = 1.0
    if full + 1 < n_units:
        w[full + 1] = revealed - full
    return w


def tr_naf_reconstruction(
    frames: List[Frame],
    meta: dict,
    K: int = 6,
    n_iterations: int = 2000,
    lr: float = 1e-2,
    warmup_steps: int = 50,
    reg_tv: float = 1e-7,
    reg_temporal_max: float = 1e-2,
    reg_temporal_tv: float = 0.0,
    temporal_anneal_frac: float = 0.6,
    data_fidelity: str = "mse",
    spatial_anneal: bool = True,
    spatial_anneal_frac: float = 0.5,
    checkpoint_frames: bool = True,
    field_kwargs: Optional[dict] = None,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """Reconstruct the 4-D field.  Returns dict with ``model``, ``basis``, ``coeffs``, ``losses``."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    vol_shape = tuple(meta["vol_shape"])
    det_shape = tuple(meta["det_shape"])
    field_kwargs = dict(field_kwargs or {})

    # Count-space regimes want the field to start at mu ~ 0 (p_hat ~ 0 -> lambda ~ I0)
    # for stable Poisson gradients; default the warm start unless the caller overrode it.
    count_mode = data_fidelity in ("poisson_nll", "anscombe")
    if count_mode and "cold_start_value" not in field_kwargs:
        field_kwargs["cold_start_value"] = 1e-3

    basis = BSplineTemporalBasis(K=K).to(device)
    model = TimeResolvedNafField(vol_shape, K=K, **field_kwargs).to(device)

    flat = meta["flat"].to(device) if count_mode else None
    gain = float(meta.get("gain", 1.0))
    read_noise = float(meta.get("read_noise", 0.0))

    # One cached projector + precomputed phi-row per frame.
    from astra_torch.lamino import build_lamino_projector
    times = torch.tensor([f.t_norm for f in frames], dtype=torch.float32, device=device)
    Phi = basis.design(times)                                   # (num_frames, K)
    groups = []
    for i, f in enumerate(frames):
        projector = build_lamino_projector(
            vol_shape=vol_shape, det_shape=det_shape, angles_deg=f.angles_deg,
            lamino_angle_deg=meta["lamino_angle_deg"], det_spacing_mm=meta["det_spacing_mm"],
            device=device,
        )
        g = {"projector": projector, "phi": Phi[i]}
        if count_mode:
            g["counts"] = f.counts.to(device)
        else:
            g["sino"] = f.sino.to(device)
        groups.append(g)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _cosine_warmup_lr(s, n_iterations, warmup_steps)
    )

    def frame_forward(coeffs, phi_row, projector):
        vol = TimeResolvedNafField.volume_at(coeffs, phi_row)   # (X,Y,Z)
        return projector(vol.unsqueeze(0).unsqueeze(0)).squeeze(0)  # (V,R,C)

    if verbose:
        print(f"TR-NAF: vol={vol_shape} K={K} frames={len(frames)} "
              f"sweeps={meta.get('num_sweeps', '?')} | {model.encoding.describe()}")

    losses: List[float] = []
    try:
        from tqdm import tqdm
        iterator = tqdm(range(n_iterations), disable=not verbose)
    except Exception:
        iterator = range(n_iterations)

    t0 = _time.time()
    for step in iterator:
        p = step / max(n_iterations - 1, 1)
        lam_t = reg_temporal_max * max(0.0, 1.0 - p / max(temporal_anneal_frac, 1e-9))
        level_w = None
        if spatial_anneal:
            level_w = _linear_reveal(p, model.encoding.n_levels, spatial_anneal_frac).to(device)

        optimizer.zero_grad(set_to_none=True)
        coeffs = model.coeffs(level_weights=level_w)            # (X,Y,Z,K)

        data_loss = coeffs.new_zeros(())
        for g in groups:
            if checkpoint_frames:
                pred = checkpoint(frame_forward, coeffs, g["phi"], g["projector"],
                                  use_reentrant=False)          # p_hat (V,R,C)
            else:
                pred = frame_forward(coeffs, g["phi"], g["projector"])
            if count_mode:
                lam_hat = noise_predict_lambda(pred, flat)      # flat*exp(-p_hat)
                if data_fidelity == "poisson_nll":
                    data_loss = data_loss + noise_poisson_nll(lam_hat, g["counts"])
                else:  # anscombe
                    data_loss = data_loss + noise_anscombe(lam_hat, g["counts"],
                                                           read_noise=read_noise, gain=gain)
            else:
                data_loss = data_loss + F.mse_loss(pred, g["sino"])
        data_loss = data_loss / len(groups)

        tv_loss = reg_tv * model.tv_regularization(coeffs) if reg_tv > 0 else coeffs.new_zeros(())
        temp_loss = lam_t * model.temporal_roughness(coeffs, basis.R) if lam_t > 0 else coeffs.new_zeros(())
        ttv_loss = reg_temporal_tv * model.temporal_tv(coeffs, Phi) if reg_temporal_tv > 0 else coeffs.new_zeros(())
        loss = data_loss + tv_loss + temp_loss + ttv_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(float(loss.detach()))
        if verbose and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{losses[-1]:.3e}", data=f"{float(data_loss):.3e}",
                                 lam_t=f"{lam_t:.2e}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    if device.type == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():
        final_coeffs = model.coeffs().detach()

    return {
        "model": model,
        "basis": basis,
        "coeffs": final_coeffs,      # (X, Y, Z, K)
        "losses": losses,
        "time": _time.time() - t0,
        "encoding": model.encoding.describe(),
    }


@torch.no_grad()
def reconstruct_volume_at(result: Dict, t_norm: float) -> torch.Tensor:
    """Evaluate the reconstructed attenuation volume ``(X, Y, Z)`` at time ``t_norm``."""
    basis, coeffs = result["basis"], result["coeffs"]
    phi = basis.design(torch.tensor([t_norm], dtype=torch.float32, device=coeffs.device))[0]
    return TimeResolvedNafField.volume_at(coeffs, phi)
