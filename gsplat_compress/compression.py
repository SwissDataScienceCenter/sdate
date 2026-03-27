"""
Compression tools for inter-frame delta encoding with codebook quantisation.

Pipeline:
1. Compute per-Gaussian deltas between two frames
2. Standardise and run K-means to build a codebook
3. Fine-tune codebook entries via gradient descent
4. Reconstruct parameters from base + quantised deltas
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler, autocast

from gsplat_compress.initialize import GaussianParams
from gsplat_compress.renderer import render


# ═══════════════════════════════════════════════════════════════════════════
# Constants: parameter layout inside the flat delta vector
# ═══════════════════════════════════════════════════════════════════════════

# Full 14-dim: means(3) + log_scales(3) + quats(4) + rgbs(3) + opacities(1)
D_FULL = 14

# Column slices
S_M = slice(0, 3)    # Δmeans      [N, 3]
S_S = slice(3, 6)    # Δlog_scales [N, 3]
S_Q = slice(6, 10)   # Δquats      [N, 4]
S_R = slice(10, 13)  # Δrgbs       [N, 3]
S_O = 13             # Δopacities  [N]

# Effective delta dimensions after removing redundant/zero columns:
#   Δmean_z ≈ 0, Δscale_z ≈ 0, Δquat_x ≈ 0, Δquat_y ≈ 0  → 4 zeros
#   Δrgb_g = Δrgb_b = Δrgb_r                                 → 2 redundant
D_EFF = D_FULL - 6   # = 8

# Non-redundant parameters per Gaussian for the base frame
PARAMS_PER_GAUSSIAN_FULL = 14   # as stored (3+3+4+3+1)
PARAMS_PER_GAUSSIAN_EFF = 9    # non-redundant (3+2+2+1+1)


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CodebookConfig:
    """Configuration for codebook-based delta compression."""

    n_clusters: int = 2048
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 500
    kmeans_batch_size: int = 10_000
    finetune_iter: int = 2000
    finetune_lr: float = 5e-4
    use_fp16: bool = True
    seed: int = 42


@dataclass
class CompressedFrame:
    """Compressed representation of one delta frame.

    Storage footprint = codebook + labels.
    """

    codebook: torch.Tensor       # [K, 14] — delta codebook entries (FP32 during training)
    labels: torch.Tensor         # [N]     — integer cluster assignments (int16 or int32)
    scaler_mean: np.ndarray      # [14]    — StandardScaler mean
    scaler_scale: np.ndarray     # [14]    — StandardScaler std

    @property
    def n_clusters(self) -> int:
        return self.codebook.shape[0]

    @property
    def n_gaussians(self) -> int:
        return self.labels.shape[0]

    def storage_bytes(self, use_fp16: bool = True) -> dict[str, int]:
        """Compute storage sizes in bytes.

        Uses only the *effective* (non-redundant) delta dimensions for the
        codebook, and ceil(log2(K)) bits per label.
        """
        bpf = 2 if use_fp16 else 4  # bytes per float
        bits_per_idx = max(1, math.ceil(math.log2(self.n_clusters)))
        cb_bytes = self.n_clusters * D_EFF * bpf
        lbl_bytes = math.ceil(self.n_gaussians * bits_per_idx / 8)
        return {
            "codebook_bytes": cb_bytes,
            "label_bytes": lbl_bytes,
            "total_bytes": cb_bytes + lbl_bytes,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Delta computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_deltas(
    base: GaussianParams, target: GaussianParams,
) -> np.ndarray:
    """Compute per-Gaussian parameter deltas ``target − base``.

    Returns
    -------
    deltas : ndarray [N, 14]
    """
    d_means = (target.means - base.means).detach().cpu()
    d_scales = (target.log_scales - base.log_scales).detach().cpu()
    d_quats = (target.quats - base.quats).detach().cpu()
    d_rgbs = (target.rgbs - base.rgbs).detach().cpu()
    d_opac = (target.opacities - base.opacities).detach().cpu().unsqueeze(-1)
    return torch.cat([d_means, d_scales, d_quats, d_rgbs, d_opac], dim=-1).numpy()


def split_delta(delta_np: np.ndarray, device: torch.device):
    """Split a flat [N, 14] delta array into parameter groups on *device*.

    Returns (d_means, d_scales, d_quats, d_rgbs, d_opacities).
    """
    d = torch.from_numpy(delta_np).float().to(device)
    return d[:, S_M], d[:, S_S], d[:, S_Q], d[:, S_R], d[:, S_O]


def reconstruct_from_delta(
    base: GaussianParams,
    d_means: torch.Tensor,
    d_scales: torch.Tensor,
    d_quats: torch.Tensor,
    d_rgbs: torch.Tensor,
    d_opac: torch.Tensor,
) -> GaussianParams:
    """Apply deltas to base params and return a new GaussianParams (detached)."""
    return GaussianParams(
        means=(base.means + d_means).detach().clone(),
        log_scales=(base.log_scales + d_scales).detach().clone(),
        quats=(base.quats + d_quats).detach().clone(),
        rgbs=(base.rgbs + d_rgbs).detach().clone(),
        opacities=(base.opacities + d_opac).detach().clone(),
    )


# ═══════════════════════════════════════════════════════════════════════════
# K-means codebook
# ═══════════════════════════════════════════════════════════════════════════

def build_codebook(
    deltas: np.ndarray,
    cfg: CodebookConfig | None = None,
    init_centroids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Run K-means on standardised deltas.

    Parameters
    ----------
    deltas : ndarray [N, 14]
        Per-Gaussian delta vectors.
    cfg : CodebookConfig | None
        K-means hyperparameters.
    init_centroids : ndarray [K, 14] | None
        Optional warm-start centroids in *original* (unscaled) delta space,
        e.g. the centroids returned by the previous frame's ``build_codebook``
        call.  When provided the new frame's scaler re-scales them before
        passing to K-means, and ``n_init`` is forced to 1 (sklearn requirement
        when ``init`` is an array).  This encourages cluster assignments to
        stay ordered across frames, which is key for entropy-coding the
        sorted label sequence.

    Returns
    -------
    labels : ndarray [N]  — cluster assignments
    centroids : ndarray [K, 14]  — in *original* scale
    scaler : StandardScaler  — for inverse_transform
    """
    if cfg is None:
        cfg = CodebookConfig()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(deltas)

    if init_centroids is not None:
        # Re-scale previous-frame centroids into the current frame's
        # standardised space and use them as the K-means starting point.
        scaled_init = scaler.transform(init_centroids)
        km = MiniBatchKMeans(
            n_clusters=cfg.n_clusters,
            init=scaled_init,
            n_init=1,            # required when init is an array
            max_iter=cfg.kmeans_max_iter,
            batch_size=min(cfg.kmeans_batch_size, len(deltas)),
            random_state=cfg.seed,
            verbose=0,
        )
    else:
        km = MiniBatchKMeans(
            n_clusters=cfg.n_clusters,
            init="k-means++",
            n_init=cfg.kmeans_n_init,
            max_iter=cfg.kmeans_max_iter,
            batch_size=min(cfg.kmeans_batch_size, len(deltas)),
            random_state=cfg.seed,
            verbose=0,
        )
    km.fit(scaled)

    centroids = scaler.inverse_transform(km.cluster_centers_)
    return km.labels_, centroids, scaler


# ═══════════════════════════════════════════════════════════════════════════
# Codebook fine-tuning via gradient descent
# ═══════════════════════════════════════════════════════════════════════════

def finetune_codebook(
    base: GaussianParams,
    labels: np.ndarray,
    centroids: np.ndarray,
    gt_image: torch.Tensor,
    viewmat: torch.Tensor,
    K_cam: torch.Tensor,
    W: int,
    H: int,
    cfg: CodebookConfig | None = None,
    verbose: bool = True,
) -> tuple[torch.Tensor, list[float], list[float]]:
    """Fine-tune codebook centroids via gradient descent through the renderer.

    The base Gaussian parameters are *frozen*.  Only the codebook is learnable.

    Parameters
    ----------
    base : GaussianParams
        Frozen base-frame Gaussians.
    labels : ndarray [N]
        Per-Gaussian cluster assignments (fixed).
    centroids : ndarray [K, 14]
        Initial codebook centroids (original scale).
    gt_image : Tensor [H, W, 3]
        Target image.
    viewmat, K_cam : Tensor
        Camera matrices.
    W, H : int
        Image size.
    cfg : CodebookConfig
        Training config.
    verbose : bool
        Print progress.

    Returns
    -------
    codebook : Tensor [K, 14]
        Fine-tuned codebook (on device).
    loss_history : list[float]
    psnr_history : list[float]
    """
    if cfg is None:
        cfg = CodebookConfig()

    device = base.device

    # Frozen base
    b_means = base.means.detach()
    b_scales = base.log_scales.detach()
    b_quats = base.quats.detach()
    b_rgbs = base.rgbs.detach()
    b_opac = base.opacities.detach()

    # Learnable codebook
    codebook = torch.tensor(centroids, dtype=torch.float32, device=device,
                            requires_grad=True)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)

    optimizer = torch.optim.Adam([codebook], lr=cfg.finetune_lr)
    scaler = GradScaler(enabled=cfg.use_fp16)
    mse_fn = torch.nn.MSELoss()

    loss_history: list[float] = []
    psnr_history: list[float] = []

    if verbose:
        print(f"Codebook fine-tune: K={cfg.n_clusters}, "
              f"{cfg.finetune_iter} iters, lr={cfg.finetune_lr}")

    t0 = time.time()
    for it in range(cfg.finetune_iter):
        with autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.use_fp16):
            delta = codebook[labels_t]  # [N, 14]
            rendered, _, _ = render(
                b_means + delta[:, S_M],
                b_scales + delta[:, S_S],
                b_quats + delta[:, S_Q],
                b_rgbs + delta[:, S_R],
                b_opac + delta[:, S_O],
                viewmat, K_cam, W, H,
            )
            loss = mse_fn(rendered, gt_image)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        lv = loss.item()
        ps = -10 * math.log10(lv + 1e-10)
        loss_history.append(lv)
        psnr_history.append(ps)

        if verbose and (it % 500 == 0 or it == cfg.finetune_iter - 1):
            print(f"  [{it:5d}/{cfg.finetune_iter}]  loss={lv:.6f}  PSNR={ps:.2f} dB")

    elapsed = time.time() - t0
    if verbose:
        print(f"✅ Codebook fine-tune done in {elapsed:.1f}s  "
              f"PSNR={psnr_history[-1]:.2f} dB")

    return codebook.detach(), loss_history, psnr_history


# ═══════════════════════════════════════════════════════════════════════════
# Full delta-frame encoding pipeline
# ═══════════════════════════════════════════════════════════════════════════

def encode_delta_frame(
    base_params: GaussianParams,
    finetuned_params: GaussianParams,
    gt_image: torch.Tensor,
    viewmat: torch.Tensor,
    K_cam: torch.Tensor,
    W: int,
    H: int,
    cfg: CodebookConfig | None = None,
    verbose: bool = True,
) -> tuple[CompressedFrame, GaussianParams]:
    """Full delta-frame compression: deltas → K-means → codebook fine-tune.

    Parameters
    ----------
    base_params : GaussianParams
        Reference frame Gaussians (frozen).
    finetuned_params : GaussianParams
        Gaussians fine-tuned to the new frame (the "teacher" signal).
    gt_image : Tensor [H, W, 3]
        Target image for codebook fine-tuning.
    viewmat, K_cam : Tensor
        Camera.
    W, H : int
        Image size.
    cfg : CodebookConfig
        Compression configuration.
    verbose : bool

    Returns
    -------
    compressed : CompressedFrame
    reconstructed : GaussianParams
        Reconstructed Gaussians (base + quantised delta, to be used as
        starting point for the next frame).
    """
    if cfg is None:
        cfg = CodebookConfig()

    # 1. Compute deltas
    deltas = compute_deltas(base_params, finetuned_params)
    if verbose:
        print(f"Delta matrix: {deltas.shape[0]:,} × {deltas.shape[1]}")

    # 2. K-means codebook
    labels, centroids, scaler = build_codebook(deltas, cfg)
    if verbose:
        print(f"K-means done: K={cfg.n_clusters}, "
              f"cluster sizes min={np.bincount(labels).min()} "
              f"max={np.bincount(labels).max()}")

    # 3. Fine-tune codebook
    codebook, cb_loss, cb_psnr = finetune_codebook(
        base_params, labels, centroids, gt_image,
        viewmat, K_cam, W, H, cfg, verbose,
    )

    # 4. Reconstruct
    labels_t = torch.tensor(labels, dtype=torch.long, device=base_params.device)
    with torch.no_grad():
        delta_ft = codebook[labels_t]
        rec = reconstruct_from_delta(
            base_params,
            delta_ft[:, S_M], delta_ft[:, S_S], delta_ft[:, S_Q],
            delta_ft[:, S_R], delta_ft[:, S_O],
        )

    compressed = CompressedFrame(
        codebook=codebook.cpu(),
        labels=labels_t.cpu(),
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
    )

    return compressed, rec


def reconstruct_compressed_frame(
    base: GaussianParams,
    compressed: CompressedFrame,
) -> GaussianParams:
    """Reconstruct Gaussians from a base + compressed delta.

    This is the *decoder* side.
    """
    device = base.device
    codebook = compressed.codebook.to(device)
    labels_t = compressed.labels.to(device)

    with torch.no_grad():
        delta = codebook[labels_t]
        return reconstruct_from_delta(
            base,
            delta[:, S_M], delta[:, S_S], delta[:, S_Q],
            delta[:, S_R], delta[:, S_O],
        )
