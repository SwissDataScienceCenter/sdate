"""Projection-space metrics: PSNR / SSIM of predicted vs clean line integrals.

The model is scored in **line-integral / attenuation space** (``p_hat`` vs the
noise-free scaled ``p``) — the denoised projection a physicist reads, and a
scale-stable target (unlike raw counts, dominated by the flat I0).  SSIM is
computed on the per-angle radiographs ``(R, C)``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


def psnr(gt: torch.Tensor, pred: torch.Tensor, data_range: float) -> float:
    mse = torch.mean((gt - pred) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * torch.log10(torch.tensor(data_range ** 2) / mse))


def _ssim_stack(gt: np.ndarray, pred: np.ndarray, data_range: float) -> float:
    """Mean SSIM over a stack of 2-D images ``(N, H, W)``."""
    from skimage.metrics import structural_similarity as _ssim
    return float(np.mean([_ssim(gt[i], pred[i], data_range=data_range)
                          for i in range(gt.shape[0])]))


def frame_metrics(pred_p: torch.Tensor, clean_p: torch.Tensor,
                  data_range: Optional[float] = None,
                  angle_stride: int = 8) -> Dict[str, float]:
    """PSNR over the whole ``(A, R, C)`` frame + mean SSIM over strided radiographs."""
    pred_p = pred_p.detach().cpu()
    clean_p = clean_p.detach().cpu()
    if data_range is None:
        data_range = float(clean_p.max() - clean_p.min())
    p = psnr(clean_p, pred_p, data_range)
    idx = range(0, clean_p.shape[0], angle_stride)
    g = clean_p[list(idx)].numpy()
    q = pred_p[list(idx)].numpy()
    s = _ssim_stack(g, q, data_range)
    return {"psnr": p, "ssim": s, "data_range": data_range}


def evaluate_all_frames(pred_frames: List[torch.Tensor], clean_frames: List[torch.Tensor],
                        data_range: Optional[float] = None,
                        angle_stride: int = 8) -> Dict[str, np.ndarray]:
    """Per-frame PSNR/SSIM. A shared ``data_range`` (over all clean frames) keeps
    frames comparable."""
    if data_range is None:
        mx = max(float(f.max()) for f in clean_frames)
        mn = min(float(f.min()) for f in clean_frames)
        data_range = mx - mn
    psnrs, ssims = [], []
    for pred, gt in zip(pred_frames, clean_frames):
        m = frame_metrics(pred, gt, data_range=data_range, angle_stride=angle_stride)
        psnrs.append(m["psnr"]); ssims.append(m["ssim"])
    return {"psnr": np.array(psnrs), "ssim": np.array(ssims),
            "data_range": float(data_range)}
