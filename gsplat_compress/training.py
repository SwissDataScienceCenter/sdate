"""
Training loops for Gaussian splatting 2D image fitting.

Provides:
- Full keyframe fitting with adaptive densification (prune + split)
- Fine-tuning from a warm start (for delta-frame encoding)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import optim
from torch.amp import GradScaler, autocast

from gsplat_compress.initialize import GaussianParams
from gsplat_compress.renderer import render


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    """Training hyperparameters for keyframe fitting."""

    num_iterations: int = 15_000
    use_fp16: bool = True
    log_interval: int = 100

    # Per-parameter learning rates
    lr_means: float = 1e-3
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_rgbs: float = 1e-2
    lr_opacities: float = 5e-2

    # Adaptive densification
    densify_from: int = 500
    densify_interval: int = 500
    densify_until: int = 12_000
    prune_opacity_thresh: float = 0.005
    split_scale_thresh: float = 10.0


@dataclass
class FinetuneConfig:
    """Training hyperparameters for delta-frame fine-tuning."""

    num_iterations: int = 10_000
    use_fp16: bool = True
    log_interval: int = 200
    lr_scale: float = 0.3  # multiplier on the base learning rates

    # Base learning rates (multiplied by lr_scale)
    lr_means: float = 1e-3
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_rgbs: float = 1e-2
    lr_opacities: float = 5e-2


@dataclass
class TrainResult:
    """Output of a training run."""

    params: GaussianParams
    loss_history: list[float] = field(default_factory=list)
    psnr_history: list[float] = field(default_factory=list)
    densify_iters: list[int] = field(default_factory=list)
    elapsed: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_optimizer(params: GaussianParams, cfg: TrainConfig, lr_mult: float = 1.0):
    """Build Adam optimiser over Gaussian parameters."""
    return optim.Adam([
        {"params": [params.means],      "lr": cfg.lr_means     * lr_mult},
        {"params": [params.log_scales], "lr": cfg.lr_scales    * lr_mult},
        {"params": [params.quats],      "lr": cfg.lr_quats     * lr_mult},
        {"params": [params.rgbs],       "lr": cfg.lr_rgbs      * lr_mult},
        {"params": [params.opacities],  "lr": cfg.lr_opacities * lr_mult},
    ])


def _enforce_grayscale(rgbs: torch.Tensor) -> None:
    """In-place: set all RGB channels to their mean (grayscale constraint)."""
    with torch.no_grad():
        rgb_mean = rgbs.data.mean(dim=-1, keepdim=True)
        rgbs.data.copy_(rgb_mean.expand_as(rgbs))


# ═══════════════════════════════════════════════════════════════════════════
# Keyframe training (with adaptive densification)
# ═══════════════════════════════════════════════════════════════════════════

def train_keyframe(
    params: GaussianParams,
    gt_image: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    W: int,
    H: int,
    cfg: TrainConfig | None = None,
    verbose: bool = True,
) -> TrainResult:
    """Train Gaussians from scratch to fit a target image.

    Includes adaptive pruning and densification.

    Parameters
    ----------
    params : GaussianParams
        Initialised Gaussians (modified in-place; tensors may be replaced
        during densification).
    gt_image : Tensor [H, W, 3]
        Target RGB image on device.
    viewmat, K : Tensor
        Camera matrices.
    W, H : int
        Image size.
    cfg : TrainConfig | None
        Training hyperparameters (default configuration if ``None``).
    verbose : bool
        Print progress.

    Returns
    -------
    TrainResult
        Contains the final ``GaussianParams`` and training history.
    """
    if cfg is None:
        cfg = TrainConfig()

    optimizer = _make_optimizer(params, cfg)
    scaler = GradScaler(enabled=cfg.use_fp16)
    mse_fn = torch.nn.MSELoss()

    loss_history: list[float] = []
    psnr_history: list[float] = []
    densify_iters: list[int] = []

    t_start = time.time()
    if verbose:
        print(f"Training keyframe: {cfg.num_iterations} iters, "
              f"N={params.n_gaussians:,}, fp16={cfg.use_fp16}")
        print("=" * 65)

    for iteration in range(cfg.num_iterations):
        # ── Forward ───────────────────────────────────────────────────
        with autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.use_fp16):
            rendered, alpha, info = render(
                params.means, params.log_scales, params.quats,
                params.rgbs, params.opacities,
                viewmat, K, W, H,
            )
            loss = mse_fn(rendered, gt_image)

        # ── Backward + update ─────────────────────────────────────────
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        _enforce_grayscale(params.rgbs)

        loss_val = loss.item()
        psnr = -10 * math.log10(loss_val + 1e-10)
        loss_history.append(loss_val)
        psnr_history.append(psnr)

        if verbose and (iteration % cfg.log_interval == 0 or iteration == cfg.num_iterations - 1):
            print(f"[{iteration:5d}/{cfg.num_iterations}]  "
                  f"loss={loss_val:.6f}  PSNR={psnr:.2f} dB  N={params.n_gaussians:,}")

        # ── Adaptive densification ────────────────────────────────────
        if (cfg.densify_from <= iteration < cfg.densify_until
                and iteration % cfg.densify_interval == 0
                and iteration > 0):
            params, n_pruned, n_split = _densify(params, cfg)
            lr_mult = 0.5 if iteration > cfg.num_iterations // 2 else 1.0
            optimizer = _make_optimizer(params, cfg, lr_mult)
            scaler = GradScaler(enabled=cfg.use_fp16)
            densify_iters.append(iteration)
            if verbose:
                print(f"  ↳ [densify @ {iteration}]  pruned={n_pruned}  "
                      f"split={n_split}  N={params.n_gaussians:,}")

    elapsed = time.time() - t_start
    if verbose:
        print(f"\n✅ Keyframe training done in {elapsed:.1f}s  "
              f"PSNR={psnr_history[-1]:.2f} dB  N={params.n_gaussians:,}")

    return TrainResult(
        params=params,
        loss_history=loss_history,
        psnr_history=psnr_history,
        densify_iters=densify_iters,
        elapsed=elapsed,
    )


def _densify(
    params: GaussianParams, cfg: TrainConfig,
) -> tuple[GaussianParams, int, int]:
    """Prune low-opacity and split over-large Gaussians.

    Returns a *new* ``GaussianParams`` with ``requires_grad=True``.
    """
    n_before = params.n_gaussians

    with torch.no_grad():
        # ── Prune ────────────────────────────────────────────────────
        opac_vals = torch.sigmoid(params.opacities)
        keep_mask = opac_vals > cfg.prune_opacity_thresh
        n_pruned = n_before - keep_mask.sum().item()

        means = params.means.data[keep_mask].clone()
        log_scales = params.log_scales.data[keep_mask].clone()
        quats = params.quats.data[keep_mask].clone()
        rgbs = params.rgbs.data[keep_mask].clone()
        opacities = params.opacities.data[keep_mask].clone()

        # ── Split ────────────────────────────────────────────────────
        scales_now = torch.exp(log_scales)
        split_mask = scales_now[:, :2].max(dim=1).values > cfg.split_scale_thresh
        n_split = split_mask.sum().item()

        if n_split > 0:
            p_means = means[split_mask]
            p_scales = log_scales[split_mask]
            p_quats = quats[split_mask]
            p_rgbs = rgbs[split_mask]
            p_opac = opacities[split_mask]

            offset = 0.5 * torch.exp(p_scales[:, :2])
            off3d = torch.zeros_like(p_means)
            off3d[:, :2] = offset

            c1_means = p_means + off3d
            c2_means = p_means - off3d
            c_scales = p_scales - math.log(2.0)

            keep = ~split_mask
            means = torch.cat([means[keep], c1_means, c2_means], 0)
            log_scales = torch.cat([log_scales[keep], c_scales, c_scales.clone()], 0)
            quats = torch.cat([quats[keep], p_quats, p_quats.clone()], 0)
            rgbs = torch.cat([rgbs[keep], p_rgbs, p_rgbs.clone()], 0)
            opacities = torch.cat([opacities[keep], p_opac, p_opac.clone()], 0)

    new_params = GaussianParams(means, log_scales, quats, rgbs, opacities)
    new_params.require_grad_()
    return new_params, n_pruned, n_split


# ═══════════════════════════════════════════════════════════════════════════
# Fine-tune from warm start (for next-frame encoding)
# ═══════════════════════════════════════════════════════════════════════════

def finetune_frame(
    base_params: GaussianParams,
    gt_image: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    W: int,
    H: int,
    cfg: FinetuneConfig | None = None,
    verbose: bool = True,
) -> TrainResult:
    """Fine-tune Gaussian parameters starting from *base_params* to a new frame.

    No densification is performed — the Gaussian count stays fixed.

    Parameters
    ----------
    base_params : GaussianParams
        Starting point (e.g. keyframe or reconstructed previous frame).
        A detached clone is made internally.
    gt_image : Tensor [H, W, 3]
        New frame target.
    cfg : FinetuneConfig | None
        Hyperparameters.
    verbose : bool
        Print progress.

    Returns
    -------
    TrainResult
        Fine-tuned parameters and training history.
    """
    if cfg is None:
        cfg = FinetuneConfig()

    # Work on a fresh copy so the caller's base_params are not modified
    params = base_params.detach_clone()
    params.require_grad_()

    lr_mult = cfg.lr_scale
    optimizer = optim.Adam([
        {"params": [params.means],      "lr": cfg.lr_means     * lr_mult},
        {"params": [params.log_scales], "lr": cfg.lr_scales    * lr_mult},
        {"params": [params.quats],      "lr": cfg.lr_quats     * lr_mult},
        {"params": [params.rgbs],       "lr": cfg.lr_rgbs      * lr_mult},
        {"params": [params.opacities],  "lr": cfg.lr_opacities * lr_mult},
    ])
    scaler = GradScaler(enabled=cfg.use_fp16)
    mse_fn = torch.nn.MSELoss()

    loss_history: list[float] = []
    psnr_history: list[float] = []

    t_start = time.time()
    if verbose:
        print(f"Fine-tuning frame: {cfg.num_iterations} iters, "
              f"N={params.n_gaussians:,}, lr_scale={cfg.lr_scale}")

    for it in range(cfg.num_iterations):
        with autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.use_fp16):
            rendered, _, _ = render(
                params.means, params.log_scales, params.quats,
                params.rgbs, params.opacities,
                viewmat, K, W, H,
            )
            loss = mse_fn(rendered, gt_image)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        _enforce_grayscale(params.rgbs)

        lv = loss.item()
        ps = -10 * math.log10(lv + 1e-10)
        loss_history.append(lv)
        psnr_history.append(ps)

        if verbose and (it % cfg.log_interval == 0 or it == cfg.num_iterations - 1):
            print(f"  [{it:5d}/{cfg.num_iterations}]  loss={lv:.6f}  PSNR={ps:.2f} dB")

    elapsed = time.time() - t_start
    if verbose:
        print(f"✅ Fine-tune done in {elapsed:.1f}s  PSNR={psnr_history[-1]:.2f} dB")

    return TrainResult(
        params=params,
        loss_history=loss_history,
        psnr_history=psnr_history,
        elapsed=elapsed,
    )
