"""Gradient-descent volume reconstruction from (denoised) projections.

The final end-to-end test: feed the model's *denoised* projections into a
differentiable-forward-model GD reconstruction (``gd_reconstruction_masked`` —
the parallel-beam analog of the SAXS GD reconstruction) and compare the volume
to the ground-truth object.  Each frame is a complete 180deg parallel-beam scan
of a frozen object, so frames reconstruct independently.

Projection-space arms (all reconstructed with identical GD settings, no volume
regularisation, positivity via ``clamp_min=0``, then compared to ``true_volume``):

* ``noisy``     — classical log-corrected counts ``p = -ln((N-dark)/I0)`` [baseline]
* ``denoised``  — the model's ``p_hat`` at the measured angles [the test]
* ``clean``     — the noise-free ``p`` [reconstruction-operator ceiling]
* ``denoised360`` — measured half-turn + the model's *synthesised* complementary
  half-turn (unseen angles), a full 360deg set [interpolation-augmented, bonus]

All line integrals are divided by ``meta['scale']`` before reconstruction so the
recovered ``mu`` is in the same normalised units as ``true_volume``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..tr_naf.metrics import evaluate_frames
from ..tr_naf.noise import counts_to_line_integral
from .data import HelixSinoDataset, HelixFrame
from .model import ProjectionField
from .train import predict_frame, predict_sino


# Full-batch Adam (batch_size defaults to all views in ``gd_reconstruct``); these
# converge the clean arm to a ~34 dB ceiling at cube 48-64 (see calibration).
DEFAULT_GD = dict(max_epochs=800, lr=1e-1, clamp_min=0.0)

ARMS = ("noisy", "denoised", "clean", "denoised360")


def gd_reconstruct(projs_scaled: torch.Tensor, angles_deg: np.ndarray, meta: dict,
                   gd_kwargs: Optional[dict] = None,
                   device: Optional[torch.device] = None) -> torch.Tensor:
    """GD-reconstruct one volume from *scaled* line integrals -> normalised-mu ``(X,Y,Z)``.

    ``projs_scaled`` are line integrals in the acquisition's scaled units; they are
    divided by ``meta['scale']`` so the reconstruction is in normalised-mu units
    directly comparable to ``true_volume``.
    """
    from astra_torch.lamino import gd_reconstruction_masked

    device = device or projs_scaled.device
    gd = {**DEFAULT_GD, **(gd_kwargs or {})}
    projs = (projs_scaled / meta["scale"]).to(device)
    gd.setdefault("batch_size", int(projs.shape[0]))     # full-batch by default
    vol = gd_reconstruction_masked(
        projs_vrc=projs, angles_deg=np.asarray(angles_deg),
        lamino_angle_deg=meta["lamino_angle_deg"], vol_shape=meta["vol_shape"],
        det_spacing_mm=meta["det_spacing_mm"], device=device, verbose=False, **gd)
    if vol.dim() == 4:            # some wrappers return a leading batch dim
        vol = vol.squeeze(0)
    return vol.clamp_min(0.0)


def reconstruct_frame(model: ProjectionField, ds: HelixSinoDataset, frames: List[HelixFrame],
                      meta: dict, frame_idx: int, arms: Sequence[str] = ARMS,
                      gd_kwargs: Optional[dict] = None,
                      device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """Reconstruct the requested arms for one frame -> ``{arm: (X,Y,Z) volume}``."""
    device = device or ds.device
    f = frames[frame_idx]
    flat = meta["flat"].to(device)
    out: Dict[str, torch.Tensor] = {}

    if "noisy" in arms:
        p_meas = counts_to_line_integral(f.counts.to(device), flat, dark=meta["dark"])
        out["noisy"] = gd_reconstruct(p_meas, f.angles_deg, meta, gd_kwargs, device)
    if "clean" in arms:
        out["clean"] = gd_reconstruct(f.clean_p.to(device), f.angles_deg, meta, gd_kwargs, device)
    if "denoised" in arms:
        p_hat = predict_frame(model, ds, frame_idx)          # (A,R,C) scaled
        out["denoised"] = gd_reconstruct(p_hat, f.angles_deg, meta, gd_kwargs, device)
    if "denoised360" in arms:
        # measured half-turn (denoised) + synthesised complementary half-turn.
        p_meas_hat = predict_frame(model, ds, frame_idx)
        unseen_norm = (f.theta_norm + 0.5) % 1.0
        p_unseen = predict_sino(model, ds, f.t_norm, unseen_norm)
        projs = torch.cat([p_meas_hat, p_unseen], dim=0)
        angles = np.concatenate([f.angles_deg, f.angles_deg + 180.0])
        out["denoised360"] = gd_reconstruct(projs, angles, meta, gd_kwargs, device)
    return out


def reconstruct_and_evaluate(model: ProjectionField, ds: HelixSinoDataset,
                             frames: List[HelixFrame], meta: dict,
                             frame_indices: Optional[Sequence[int]] = None,
                             arms: Sequence[str] = ARMS,
                             gd_kwargs: Optional[dict] = None,
                             mask=None, slice_stride: int = 4,
                             device: Optional[torch.device] = None):
    """Reconstruct all arms over ``frame_indices`` and score each vs ``true_volume``.

    Returns ``(volumes, metrics)`` where ``volumes[arm]`` is a list of ``(X,Y,Z)``
    volumes and ``metrics[arm]`` is the :func:`evaluate_frames` dict (masked
    PSNR/SSIM/rel-err), all sharing one ``data_range`` for comparability.
    """
    device = device or ds.device
    if frame_indices is None:
        frame_indices = list(range(ds.F))
    true_vols = [frames[i].true_volume.to(device) for i in frame_indices]

    volumes: Dict[str, List[torch.Tensor]] = {a: [] for a in arms}
    for i in frame_indices:
        rec = reconstruct_frame(model, ds, frames, meta, i, arms=arms,
                                gd_kwargs=gd_kwargs, device=device)
        for a in arms:
            volumes[a].append(rec[a])

    # Shared data range over the GT volumes (masked), matching tr_naf convention.
    metrics: Dict[str, dict] = {}
    dr = None
    for a in arms:
        m = evaluate_frames(volumes[a], true_vols, mask=mask, data_range=dr,
                            slice_stride=slice_stride)
        dr = m["data_range"]      # reuse the first arm's range for all
        metrics[a] = m
    return volumes, metrics, [i for i in frame_indices]
