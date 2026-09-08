"""Metrics + persistence for time-resolved NAF reconstructions.

Circular-masked PSNR/SSIM per frame (matching the convention in
``notebooks/test_time_resolved_ddim.ipynb``), plus lightweight save/load of a
reconstruction result (coefficients + config) so a notebook can visualise a run
without retraining.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


def make_circular_mask(h: int, w: int, radius: Optional[float] = None,
                       device=None) -> torch.Tensor:
    """Boolean ``(h, w)`` circular mask centred on the image."""
    if radius is None:
        radius = w / 2.1
    cy, cx = h / 2.0, w / 2.0
    Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    m = ((X - cx) ** 2 + (Y - cy) ** 2) <= radius ** 2
    return m.to(device) if device is not None else m


def masked_psnr(gt: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor,
                data_range: float) -> float:
    """PSNR over the True region of ``mask`` (broadcast over leading slice dim)."""
    mse = torch.mean((gt[..., mask] - pred[..., mask]) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * torch.log10(torch.tensor(data_range ** 2) / mse))


def masked_ssim(gt: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor,
                data_range: float, slice_stride: int = 4) -> float:
    """Mean SSIM over slices (every ``slice_stride``-th), evaluated in the mask bbox."""
    from skimage.metrics import structural_similarity as _ssim

    ys, xs = torch.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    g = gt[:, y0:y1, x0:x1].cpu().numpy()
    p = pred[:, y0:y1, x0:x1].cpu().numpy()
    vals = [
        _ssim(g[i], p[i], data_range=data_range)
        for i in range(0, g.shape[0], slice_stride)
    ]
    return float(np.mean(vals))


def masked_gradient_energy(x: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean in-plane gradient magnitude within ``mask`` (broadcast over leading slice dim).

    A cheap sharpness proxy: higher = more high-frequency content. Only
    meaningful as a RATIO against a reference (e.g. GT) — on its own it can't
    distinguish genuine detail from noise, since both inflate the gradient.
    """
    dy = (x[..., 1:, :] - x[..., :-1, :]).abs()
    dx = (x[..., :, 1:] - x[..., :, :-1]).abs()
    mask_y = mask[1:, :] & mask[:-1, :]
    mask_x = mask[:, 1:] & mask[:, :-1]
    return float(torch.cat([dy[..., mask_y].reshape(-1), dx[..., mask_x].reshape(-1)]).mean())


def masked_sharpness_ratio(gt: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> float:
    """``pred``'s masked gradient energy relative to ``gt``'s: 1.0 = matched, <1 = blurred, >1 = over-sharp/noisy."""
    g_gt = masked_gradient_energy(gt, mask)
    g_pred = masked_gradient_energy(pred, mask)
    return g_pred / max(g_gt, 1e-8)


def evaluate_frames(recon_volumes: List[torch.Tensor],
                    true_volumes: List[torch.Tensor],
                    mask: Optional[torch.Tensor] = None,
                    data_range: Optional[float] = None,
                    slice_stride: int = 4) -> Dict[str, np.ndarray]:
    """Per-frame PSNR/SSIM/rel-error for a list of reconstructed vs true volumes."""
    dev = true_volumes[0].device
    h, w = true_volumes[0].shape[-2:]
    if mask is None:
        mask = make_circular_mask(h, w, device=dev)
    else:
        mask = mask.to(dev)
    if data_range is None:
        stacked = torch.stack([v[:, mask] for v in true_volumes])
        data_range = float(stacked.max() - stacked.min())

    psnr, ssim, rel = [], [], []
    for pred, gt in zip(recon_volumes, true_volumes):
        pred, gt = pred.to(dev), gt.to(dev)
        psnr.append(masked_psnr(gt, pred, mask, data_range))
        ssim.append(masked_ssim(gt, pred, mask, data_range, slice_stride))
        rel.append(float(((pred - gt)[:, mask]).norm() / gt[:, mask].norm()))
    return {"psnr": np.array(psnr), "ssim": np.array(ssim), "rel_err": np.array(rel),
            "data_range": data_range}


def save_result(result: Dict, meta: dict, path, extra: Optional[dict] = None) -> None:
    """Persist coefficients + config (not the nn.Module) for later visualisation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "coeffs": result["coeffs"].cpu(),
        "K": result["basis"].K,
        "degree": result["basis"].degree,
        "losses": result["losses"],
        "time": result["time"],
        "encoding": result["encoding"],
        "meta": meta,
    }
    if extra:
        bundle.update(extra)
    torch.save(bundle, path)


def load_result(path, device=None):
    """Load a saved bundle and rebuild ``basis`` so ``reconstruct_volume_at`` works.

    Returns ``(result_like, meta)`` where ``result_like`` has ``coeffs`` + ``basis``.
    """
    from .temporal_basis import BSplineTemporalBasis

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(path, map_location=device, weights_only=False)
    basis = BSplineTemporalBasis(K=bundle["K"], degree=bundle["degree"]).to(device)
    result = {
        "coeffs": bundle["coeffs"].to(device),
        "basis": basis,
        "losses": bundle["losses"],
        "time": bundle["time"],
        "encoding": bundle["encoding"],
    }
    return result, bundle["meta"]
