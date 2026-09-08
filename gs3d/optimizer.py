"""
Gaussian Splatting optimiser with adaptive density control.

Implements the three-phase loop from 3D Gaussian Splatting (Kerbl et al.):

    1. **Gradient-descent step** (AdamW) — update means, rotations, scales,
       intensities to minimise reconstruction loss.
    2. **Pruning** — remove Gaussians whose intensity (opacity) is below a
       threshold, or whose scale is negligibly small.
    3. **Densification** — duplicate / split Gaussians in regions where the
       position-gradient magnitude is large.

All operations run entirely on GPU with batched tensor ops — no Python loops
over individual Gaussians.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gs3d.model import GaussianModel3D, GaussianModelConfig, quaternion_to_rotation_matrix
from gs3d.renderer import VolumeRenderer
from gs3d.spatial import SpatialGrid, world_frame_margin


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OptimConfig:
    """Hyper-parameters for the Gaussian Splatting optimiser."""

    # ── Optimiser ──
    lr_means: float = 1.6e-4
    lr_quats: float = 1e-3
    lr_scales: float = 5e-3
    lr_intensity: float = 5e-2
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)

    # ── Training loop ──
    num_epochs: int = 3000
    samples_per_step: int = 262_144
    """Number of random voxel coordinates sampled per gradient step
    (only used when window_size is None — legacy scattered sampling)."""
    log_interval: int = 50
    eval_interval: int = 200

    # ── Windowed SGD ──
    window_size: Tuple[int, int, int] = (64, 64, 64)
    """Size of the random 3-D sub-volume (D, H, W) sampled each step.
    Defines the spatial region for Gaussian culling.  Only Gaussians whose
    support overlaps the window are evaluated.
    Set to (0, 0, 0) to fall back to scattered random-voxel sampling."""
    voxels_per_step: int = 16_384
    """Number of voxels actually evaluated per gradient step.  If smaller
    than the total window volume, a random subset is drawn from the window
    coordinates.  This avoids building huge (B, N_local, 3) tensors.
    Typical values: 8_192 – 32_768."""
    cutoff_sigma: float = 3.0
    """Number of standard deviations beyond which a Gaussian is considered
    to have zero contribution.  Used both for window-based culling and
    the bounding-box cull in evaluate_gaussians."""
    eval_samples: int = 100_000
    """Number of random voxels used for periodic PSNR evaluation.
    Lower = faster but noisier estimate."""
    grid_threshold: int = 50_000
    """Gaussian count above which the spatial grid is used for culling.
    Below this threshold, a brute-force rotation-aware bbox check is used
    (O(N) with tiny constant, faster than grid overhead for small N)."""

    # ── Learning rate schedule ──
    lr_decay_factor: float = 0.01
    """Final LR = initial * decay_factor  (exponential decay over training)."""

    # ── Densification (standard 3DGS rules) ──
    densify_from_epoch: int = 200
    densify_until_epoch: int = 2000
    densify_interval: int = 100
    grad_threshold: float = 0.0002
    """Positional-gradient norm above which a Gaussian is densified."""
    max_gaussians: int = 5_000_000
    split_scale_threshold: float = 0.01
    """Gaussians with scale > this (in volume-extent units) are *split*
    rather than *cloned* when their gradient is above ``grad_threshold``."""
    densify_scale_factor: float = 1.6
    """After cloning, new Gaussian positions are offset by this × scale."""

    # ── Pruning ──
    prune_from_epoch: int = 200
    prune_interval: int = 100
    prune_intensity_threshold: float = 0.005
    """Gaussians with sigmoid(intensity_logit) < this are pruned."""
    prune_scale_threshold: float = 1e-5
    """Gaussians with max-scale < this are pruned."""
    prune_large_scale_threshold: float = 0.5
    """Gaussians with max-scale > this × max_extent also pruned (too big)."""

    # ── Spatial index ──
    grid_resolution: Tuple[int, int, int] = (16, 16, 16)
    """Resolution of the uniform 3-D grid used for spatial culling.
    Higher → finer queries but more cells.  16³ is good for extent ≈1."""
    spatial_rebuild_interval: int = 50
    """Rebuild the spatial grid every N training steps.  Between rebuilds,
    cell assignment can become slightly stale as means drift.  Set to 1
    for exact behaviour (recommended for N < 100 K);  increase for
    multi-million-Gaussian models to amortise the O(N) rebuild cost."""

    # ── Opacity reset (periodically refresh opacities) ──
    opacity_reset_interval: int = 500
    opacity_reset_value: float = 0.01
    """Logit corresponding to ~sigmoid(v) = opacity_reset_value."""


# ---------------------------------------------------------------------------
# Gradient accumulator  (running mean of ||∂L/∂μ||)
# ---------------------------------------------------------------------------

class _GradAccumulator:
    """Track running mean of positional-gradient norms."""

    def __init__(self, n: int, device: torch.device):
        self.grad_accum = torch.zeros(n, device=device)
        self.count = torch.zeros(n, device=device, dtype=torch.long)

    def update(self, grad_norms: torch.Tensor) -> None:
        self.grad_accum += grad_norms
        self.count += 1

    def mean(self) -> torch.Tensor:
        return self.grad_accum / self.count.clamp(min=1).float()

    def reset(self, n: int, device: torch.device) -> None:
        self.grad_accum = torch.zeros(n, device=device)
        self.count = torch.zeros(n, device=device, dtype=torch.long)


# ---------------------------------------------------------------------------
# Main optimiser
# ---------------------------------------------------------------------------

class GaussianOptimizer:
    """Full training loop for Gaussian Splatting of a 3-D volume.

    Parameters
    ----------
    model : GaussianModel3D
        The Gaussian model (will be mutated in place).
    target_volume : torch.Tensor
        (D, H, W) reference volume on **CPU** (will be copied to GPU).
    config : OptimConfig
        Hyper-parameters.
    device : torch.device
        GPU device.
    """

    def __init__(
        self,
        model: GaussianModel3D,
        target_volume: torch.Tensor,
        config: OptimConfig | None = None,
        device: torch.device | str = "cuda",
    ):
        self.device = torch.device(device)
        self.config = config or OptimConfig()
        self.model = model.to(self.device)

        # Store target volume flat on GPU
        D, H, W = target_volume.shape
        self.volume_shape = (D, H, W)
        self._target_flat = target_volume.reshape(-1).to(self.device)

        # Renderer (pre-builds coordinate grid)
        self.renderer = VolumeRenderer(
            volume_shape=(D, H, W),
            volume_extent=model.config.volume_extent,
            device=self.device,
        )

        # Optimiser — per-parameter-group LR
        self.optimizer = self._build_optimizer()

        # Spatial grid index for fast Gaussian-to-window culling
        self._grid = SpatialGrid(
            volume_extent=model.config.volume_extent,
            grid_resolution=self.config.grid_resolution,
            device=self.device,
        )
        self._build_grid()
        self._step_count = 0

        # Gradient accumulator for densification
        self._grad_acc = _GradAccumulator(model.n_gaussians, self.device)

    # ──────────────────────────────────────────────── optimiser construction
    def _build_optimizer(self) -> torch.optim.AdamW:
        c = self.config
        params = [
            {"params": [self.model._means],           "lr": c.lr_means,     "name": "means"},
            {"params": [self.model._quats],            "lr": c.lr_quats,     "name": "quats"},
            {"params": [self.model._log_scales],       "lr": c.lr_scales,    "name": "scales"},
            {"params": [self.model._intensity_logit],  "lr": c.lr_intensity, "name": "intensity"},
        ]
        return torch.optim.AdamW(
            params,
            lr=c.lr_means,  # fallback
            betas=c.betas,
            weight_decay=c.weight_decay,
        )

    def _rebuild_optimizer(self) -> None:
        """Rebuild the optimiser after parameter replacement (prune/densify)."""
        self.optimizer = self._build_optimizer()
        self._grad_acc.reset(self.model.n_gaussians, self.device)
        self._build_grid()  # structural change → must rebuild
        torch.cuda.empty_cache()  # reclaim fragmented memory from old params

    def _build_grid(self) -> None:
        """(Re)build the spatial grid from current Gaussian parameters."""
        with torch.no_grad():
            self._grid.build(
                self.model._means.detach(),
                self.model._quats.detach(),
                self.model._log_scales.detach(),
                cutoff_sigma=self.config.cutoff_sigma,
            )

    # ──────────────────────────────────────────────── LR schedule
    def _get_lr_factor(self, epoch: int) -> float:
        """Exponential decay from 1 to ``config.lr_decay_factor``."""
        t = epoch / max(self.config.num_epochs, 1)
        return math.exp(t * math.log(max(self.config.lr_decay_factor, 1e-10)))

    def _set_lr(self, factor: float) -> None:
        c = self.config
        base = [c.lr_means, c.lr_quats, c.lr_scales, c.lr_intensity]
        for pg, blr in zip(self.optimizer.param_groups, base):
            pg["lr"] = blr * factor

    # ──────────────────────────────────────────────── single training step
    def train_step(self) -> float:
        """One gradient step using windowed SGD.

        1. Sample a random 3-D window inside the volume.
        2. Cull Gaussians — keep only those whose support overlaps the window.
        3. Evaluate **only** those local Gaussians on the window voxels.
        4. MSE loss → backprop → AdamW update.

        Memory cost is O(W_d × W_h × W_w × N_local) instead of
        O(samples × N_total), which is drastically lower when N is large.

        Falls back to scattered random sampling when ``window_size == (0,0,0)``.

        Returns the MSE loss value.
        """
        c = self.config
        self.model.train()
        ws = c.window_size

        # ── Legacy scattered sampling when window is disabled ──
        if ws[0] <= 0 or ws[1] <= 0 or ws[2] <= 0:
            coords, idx = self.renderer.sample_coords(c.samples_per_step)
            target = self._target_flat[idx]
            pred = VolumeRenderer.evaluate_gaussians(
                self.model, coords, cutoff_sigma=c.cutoff_sigma,
            )
            loss = F.mse_loss(pred, target)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.model._means.grad is not None:
                self._grad_acc.update(
                    self.model._means.grad.detach().norm(dim=-1))
            self.optimizer.step()
            return loss.item()

        # ── Windowed SGD ──────────────────────────────────────────────
        # Periodic spatial-grid rebuild (positions drift between rebuilds)
        self._step_count += 1
        if self._step_count % c.spatial_rebuild_interval == 0:
            self._build_grid()

        coords, idx, win_min, win_max = self.renderer.sample_window(
            ws, self.volume_shape)

        # Sub-sample voxels from the window to keep B small
        B_win = coords.shape[0]
        B_use = min(c.voxels_per_step, B_win)
        if B_use < B_win:
            sel = torch.randperm(B_win, device=self.device)[:B_use]
            coords = coords[sel]
            idx = idx[sel]

        target = self._target_flat[idx]                          # (B,)

        # Spatial cull: choose method based on Gaussian count
        N = self.model.n_gaussians
        if N >= c.grid_threshold:
            local_mask = self._grid.query_window(
                win_min, win_max,
                self.model._means.detach(),
                self.model._quats.detach(),
                self.model._log_scales.detach(),
                cutoff_sigma=c.cutoff_sigma,
            )
        else:
            local_mask = self._gaussians_in_box(
                win_min, win_max, c.cutoff_sigma)
        n_local = int(local_mask.sum().item())

        if n_local == 0:
            # Window is empty of Gaussians — nothing to learn here
            return 0.0

        # ── Dynamic B reduction ──────────────────────────────────
        # Cap voxels so that the diff tensor (B, L, 3) fits in memory.
        # Peak memory ≈ B × L × 20 bytes (diff + mahal + gauss, fwd+bwd).
        _STEP_MEM_BUDGET = 512 * 1024 * 1024  # 512 MB
        B_cap = max(256, _STEP_MEM_BUDGET // max(n_local * 20, 1))
        if B_cap < coords.shape[0]:
            sel2 = torch.randperm(coords.shape[0], device=self.device)[:B_cap]
            coords = coords[sel2]
            idx    = idx[sel2]
            target = self._target_flat[idx]

        pred = self._evaluate_local(
            coords, local_mask, cutoff_sigma=c.cutoff_sigma)

        loss = F.mse_loss(pred, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Accumulate positional gradient norms (full N — zero where not in mask)
        if self.model._means.grad is not None:
            gnorms = self.model._means.grad.detach().norm(dim=-1)  # (N,)
            self._grad_acc.update(gnorms)

        self.optimizer.step()
        return loss.item()

    # ──────────────────── helpers for windowed SGD ──────────────────────
    def _gaussians_in_box(
        self,
        box_min: torch.Tensor,
        box_max: torch.Tensor,
        cutoff_sigma: float,
    ) -> torch.Tensor:
        """Return a boolean (N,) mask of Gaussians overlapping the box.

        Uses **rotation-aware** axis-aligned bounding boxes:
        margin_j = cutoff × √Σ_{jj} where Σ = R diag(s²) Rᵀ.
        This correctly handles anisotropic Gaussians under rotation.
        """
        means = self.model._means.detach()                  # (N, 3)
        margin = world_frame_margin(
            self.model._quats.detach(),
            self.model._log_scales.detach(),
            cutoff_sigma,
        )                                                    # (N, 3)
        upper = means + margin
        lower = means - margin
        # Overlap when  lower_i < box_max_i  AND  upper_i > box_min_i  for all axes
        overlap = (
            (lower < box_max.unsqueeze(0)) &
            (upper > box_min.unsqueeze(0))
        ).all(dim=-1)                                        # (N,)
        return overlap

    def _evaluate_local(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        cutoff_sigma: float = 3.0,
    ) -> torch.Tensor:
        """Differentiable evaluation using only the Gaussians in *mask*.

        Directly indexes into the model's parameter tensors so that
        gradients flow back to the *full* parameter arrays (important for
        AdamW state correspondence).  Gaussians outside the mask simply
        contribute nothing and receive zero gradient.

        Args:
            coords: (B, 3) query points (the window voxels).
            mask:   (N,) boolean — True for local Gaussians.

        Returns:
            (B,) predicted values.
        """
        B = coords.shape[0]
        # Index into *parameters* (not .data) to preserve the graph
        m  = self.model._means[mask]              # (L, 3)
        q  = self.model._quats[mask]              # (L, 4)
        ls = self.model._log_scales[mask]         # (L, 3)
        il = self.model._intensity_logit[mask]    # (L,)
        L  = m.shape[0]

        intensities = torch.sigmoid(il)           # (L,)

        R    = quaternion_to_rotation_matrix(q)   # (L, 3, 3)
        invs = torch.exp(-ls)                     # (L, 3)
        RiS  = R * invs.unsqueeze(-2)             # (L, 3, 3)
        prec = RiS @ RiS.transpose(-1, -2)       # (L, 3, 3)

        # Tile over Gaussians to keep peak memory bounded.
        # Budget: diff(B,T,3) + mahal(B,T) + gauss(B,T) ≈ B*T*5 floats (20 bytes).
        # Target peak ≤ 512 MB.
        _MEM_BUDGET = 512 * 1024 * 1024   # bytes
        TILE = max(1, min(L, _MEM_BUDGET // max(B * 20, 1)))
        result = torch.zeros(B, device=coords.device)
        for g0 in range(0, L, TILE):
            g1 = min(g0 + TILE, L)
            diff  = coords.unsqueeze(1) - m[g0:g1].unsqueeze(0)        # (B, T, 3)
            mahal = torch.einsum("bti,tij,btj->bt",
                                 diff, prec[g0:g1], diff)             # (B, T)
            gauss = torch.exp(-0.5 * mahal)                            # (B, T)
            result = result + (gauss * intensities[g0:g1].unsqueeze(0)).sum(-1)
        return result

    # ──────────────────────────────────────────────── pruning
    def prune(self) -> int:
        """Remove weak / tiny / huge Gaussians.  Returns number pruned."""
        c = self.config
        with torch.no_grad():
            intensities = self.model.intensities          # (N,) in (0,1)
            scales = self.model.scales                    # (N, 3)
            max_scale = scales.max(dim=-1).values         # (N,)
            max_extent = max(self.model.config.volume_extent)

            keep = (
                (intensities >= c.prune_intensity_threshold)
                & (max_scale >= c.prune_scale_threshold)
                & (max_scale <= c.prune_large_scale_threshold * max_extent)
            )
            n_before = self.model.n_gaussians
            if keep.all():
                return 0

            self.model._replace_parameters(
                self.model._means.data[keep],
                self.model._quats.data[keep],
                self.model._log_scales.data[keep],
                self.model._intensity_logit.data[keep],
            )
            self._rebuild_optimizer()
            return n_before - self.model.n_gaussians

    # ──────────────────────────────────────────────── densification
    def densify(self) -> int:
        """Clone small Gaussians and split large ones where gradients are high.

        Returns the number of **new** Gaussians added.
        """
        c = self.config
        if self.model.n_gaussians >= c.max_gaussians:
            return 0

        with torch.no_grad():
            grad_mean = self._grad_acc.mean()                   # (N,)
            scales = self.model.scales                          # (N, 3)
            max_scale = scales.max(dim=-1).values               # (N,)

            high_grad = grad_mean > c.grad_threshold

            # ── clone-candidates: small Gaussians with high grad ──
            clone_mask = high_grad & (max_scale <= c.split_scale_threshold)
            # ── split-candidates: large Gaussians with high grad ──
            split_mask = high_grad & (max_scale > c.split_scale_threshold)

            new_means = []
            new_quats = []
            new_ls = []
            new_il = []

            # Clone: duplicate and offset
            if clone_mask.any():
                m = self.model._means.data[clone_mask]
                q = self.model._quats.data[clone_mask]
                ls = self.model._log_scales.data[clone_mask]
                il = self.model._intensity_logit.data[clone_mask]
                # offset in a random direction proportional to scale
                offset = torch.randn_like(m) * scales[clone_mask] * c.densify_scale_factor
                new_means.append(m + offset)
                new_quats.append(q.clone())
                new_ls.append(ls.clone())
                new_il.append(il.clone())

            # Split: replace one Gaussian by two smaller ones
            if split_mask.any():
                m = self.model._means.data[split_mask]
                q = self.model._quats.data[split_mask]
                ls = self.model._log_scales.data[split_mask]
                il = self.model._intensity_logit.data[split_mask]
                sc = scales[split_mask]

                # Shrink scale by factor of 1.6
                new_log_s = ls - math.log(1.6)

                # Two children: parent ± offset
                offset = torch.randn_like(m) * sc * 0.5
                new_means.extend([m + offset, m - offset])
                new_quats.extend([q.clone(), q.clone()])
                new_ls.extend([new_log_s.clone(), new_log_s.clone()])
                new_il.extend([il.clone(), il.clone()])

            if not new_means:
                self._grad_acc.reset(self.model.n_gaussians, self.device)
                return 0

            all_means = torch.cat([self.model._means.data] + new_means, dim=0)
            all_quats = torch.cat([self.model._quats.data] + new_quats, dim=0)
            all_ls    = torch.cat([self.model._log_scales.data] + new_ls, dim=0)
            all_il    = torch.cat([self.model._intensity_logit.data] + new_il, dim=0)

            # Cap at max_gaussians
            n_new = all_means.shape[0]
            if n_new > c.max_gaussians:
                all_means = all_means[:c.max_gaussians]
                all_quats = all_quats[:c.max_gaussians]
                all_ls    = all_ls[:c.max_gaussians]
                all_il    = all_il[:c.max_gaussians]

            n_added = all_means.shape[0] - self.model.n_gaussians
            self.model._replace_parameters(all_means, all_quats, all_ls, all_il)
            self._rebuild_optimizer()
            return n_added

    # ──────────────────────────────────────────────── opacity reset
    def reset_opacity(self) -> None:
        """Set all opacities to a low value to prune dead Gaussians next cycle."""
        with torch.no_grad():
            logit = math.log(self.config.opacity_reset_value /
                             (1.0 - self.config.opacity_reset_value))
            self.model._intensity_logit.data.clamp_(max=logit)

    # ──────────────────────────────────────────────── evaluation
    @torch.no_grad()
    def evaluate(
        self,
        n_samples: int | None = None,
    ) -> Dict[str, float]:
        """Estimate reconstruction quality via windowed random sampling.

        Samples random windows, culls Gaussians spatially (just like
        training), and evaluates locally.  This keeps evaluation
        O(n_samples × N_local_avg) instead of O(n_samples × N_total).

        Returns dict with 'mse', 'psnr', 'n_gaussians', 'model_mb'.
        """
        self.model.eval()
        c = self.config
        cs = c.cutoff_sigma
        n_samples = n_samples or c.eval_samples
        ws = c.window_size

        # Choose evaluation method
        use_windowed = ws[0] > 0 and ws[1] > 0 and ws[2] > 0

        sse = 0.0
        n_evaluated = 0
        CHUNK = min(32_768, n_samples)

        while n_evaluated < n_samples:
            n_this = min(CHUNK, n_samples - n_evaluated)
            if use_windowed:
                # Sample a window, subsample voxels, cull Gaussians
                coords, idx, wmin, wmax = self.renderer.sample_window(
                    ws, self.volume_shape)
                if coords.shape[0] > n_this:
                    sel = torch.randperm(coords.shape[0], device=self.device)[:n_this]
                    coords = coords[sel]
                    idx = idx[sel]
                target = self._target_flat[idx]

                N = self.model.n_gaussians
                if N >= c.grid_threshold:
                    local_mask = self._grid.query_window(
                        wmin, wmax,
                        self.model._means.detach(),
                        self.model._quats.detach(),
                        self.model._log_scales.detach(),
                        cutoff_sigma=cs,
                    )
                else:
                    local_mask = self._gaussians_in_box(wmin, wmax, cs)
                if local_mask.any():
                    pred = self._evaluate_local_nograd(
                        coords, local_mask, cutoff_sigma=cs)
                else:
                    pred = torch.zeros(coords.shape[0], device=self.device)
            else:
                coords, idx = self.renderer.sample_coords(n_this)
                target = self._target_flat[idx]
                pred = VolumeRenderer.evaluate_gaussians(
                    self.model, coords, cutoff_sigma=cs)

            sse += ((pred - target) ** 2).sum().item()
            n_evaluated += coords.shape[0]

        mse_val = sse / max(n_evaluated, 1)
        max_val = float(self._target_flat.max() - self._target_flat.min())
        psnr_val = (10 * math.log10(max_val ** 2 / mse_val)
                    if mse_val > 0 else float("inf"))

        return {
            "mse": mse_val,
            "psnr": psnr_val,
            "n_gaussians": self.model.n_gaussians,
            "model_mb": self.model.state_size_bytes() / 1e6,
        }

    def _evaluate_local_nograd(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        cutoff_sigma: float = 3.0,
    ) -> torch.Tensor:
        """Non-differentiable version of _evaluate_local for fast evaluation."""
        m  = self.model._means.data[mask]
        q  = self.model._quats.data[mask]
        ls = self.model._log_scales.data[mask]
        il = self.model._intensity_logit.data[mask]
        L  = m.shape[0]
        B  = coords.shape[0]

        intensities = torch.sigmoid(il)
        R    = quaternion_to_rotation_matrix(q)
        invs = torch.exp(-ls)
        RiS  = R * invs.unsqueeze(-2)
        prec = RiS @ RiS.transpose(-1, -2)

        _MEM = 512 * 1024 * 1024
        TILE = max(1, min(L, _MEM // max(B * 20, 1)))
        result = torch.zeros(B, device=coords.device)
        for g0 in range(0, L, TILE):
            g1 = min(g0 + TILE, L)
            diff  = coords.unsqueeze(1) - m[g0:g1].unsqueeze(0)
            mahal = torch.einsum("bti,tij,btj->bt", diff, prec[g0:g1], diff)
            gauss = torch.exp(-0.5 * mahal)
            result += (gauss * intensities[g0:g1].unsqueeze(0)).sum(-1)
        return result

    # ──────────────────── fast full render with precomputation ─────────
    @staticmethod
    def _eval_block_precomputed(
        coords: torch.Tensor,       # (B, 3)
        local_ids: torch.LongTensor, # (L,)
        means: torch.Tensor,         # (N, 3)   precomputed
        prec: torch.Tensor,          # (N, 3, 3) precomputed
        intensities: torch.Tensor,   # (N,)      precomputed
        mem_budget: int = 512 * 1024 * 1024,
    ) -> torch.Tensor:
        """Evaluate Gaussian mixture at coords using pre-computed arrays.

        No rotation-matrix or precision recomputation — just indexing + einsum.
        """
        m = means[local_ids]          # (L, 3)
        p = prec[local_ids]           # (L, 3, 3)
        a = intensities[local_ids]    # (L,)
        B = coords.shape[0]
        L = m.shape[0]
        TILE = max(1, min(L, mem_budget // max(B * 20, 1)))
        result = torch.zeros(B, device=coords.device)
        for g0 in range(0, L, TILE):
            g1 = min(g0 + TILE, L)
            diff = coords.unsqueeze(1) - m[g0:g1].unsqueeze(0)     # (B, T, 3)
            mahal = torch.einsum("bti,tij,btj->bt",
                                 diff, p[g0:g1], diff)             # (B, T)
            gauss = torch.exp(-0.5 * mahal)                        # (B, T)
            result += (gauss * a[g0:g1].unsqueeze(0)).sum(-1)
        return result

    # ──────────────────── full render with spatial culling ──────────────
    @torch.no_grad()
    def render_full_spatial(
        self,
        block_size: Tuple[int, int, int] = (16, 64, 64),
        verbose: bool = True,
    ) -> torch.Tensor:
        """Render entire volume block-by-block with spatial Gaussian culling.

        **Key optimisations** (why this is ~50-100× faster than the naive version):

        1. **Precompute once**: rotation matrices, precision matrices, margins,
           and bounding boxes are computed ONCE for all N Gaussians, not
           re-derived per block.
        2. **Small blocks → tight culling**: smaller spatial blocks mean far
           fewer Gaussians overlap each block (N_local ∝ block_volume).
           With 16×64×64 blocks, N_local is typically 2-5K instead of 30-50K
           for 64×128×128 blocks.
        3. **Vectorised bbox culling on pre-computed bounds**: per-block culling
           is just 6 comparisons per Gaussian on pre-computed ``lower``/``upper``
           arrays — essentially free compared to recomputing rotation-aware
           margins every block.

        Args:
            block_size: (bd, bh, bw) voxels per block.  Smaller = tighter
                culling but more Python-loop overhead.  (16, 64, 64) is a
                good default for ~1M Gaussians.
            verbose: print progress + ETA.

        Returns:
            (D, H, W) CPU float32 tensor.
        """
        D, H, W = self.volume_shape
        cs = self.config.cutoff_sigma
        N = self.model.n_gaussians
        values = torch.zeros(D * H * W, device=self.device)

        # ── 1.  Precompute ALL Gaussian data ONCE ────────────────────
        means = self.model._means.data                              # (N, 3)
        intensities = torch.sigmoid(self.model._intensity_logit.data)  # (N,)
        R = quaternion_to_rotation_matrix(self.model._quats.data)   # (N, 3, 3)
        inv_s = torch.exp(-self.model._log_scales.data)             # (N, 3)
        RiS = R * inv_s.unsqueeze(-2)                               # (N, 3, 3)
        prec = RiS @ RiS.transpose(-1, -2)                         # (N, 3, 3)

        margin = world_frame_margin(
            self.model._quats.data, self.model._log_scales.data, cs,
        )                                                           # (N, 3)
        lower = means - margin                                      # (N, 3)
        upper = means + margin                                      # (N, 3)

        if verbose:
            print(f"  Precomputed prec/margins for {N:,} Gaussians  "
                  f"(prec: {prec.numel()*4/1e6:.0f} MB)")

        # ── 2.  Block iteration ──────────────────────────────────────
        bd, bh, bw = block_size
        n_d = (D + bd - 1) // bd
        n_h = (H + bh - 1) // bh
        n_w = (W + bw - 1) // bw
        n_total = n_d * n_h * n_w

        t0 = time.time()
        done = 0
        total_evals = 0

        for id_ in range(n_d):
            d0, d1 = id_ * bd, min((id_ + 1) * bd, D)
            for ih in range(n_h):
                h0, h1 = ih * bh, min((ih + 1) * bh, H)
                for iw in range(n_w):
                    w0, w1 = iw * bw, min((iw + 1) * bw, W)

                    # Flat voxel indices
                    dd = torch.arange(d0, d1, device=self.device)
                    hh = torch.arange(h0, h1, device=self.device)
                    ww = torch.arange(w0, w1, device=self.device)
                    gd, gh, gw = torch.meshgrid(dd, hh, ww, indexing="ij")
                    flat_idx = (gd * H * W + gh * W + gw).reshape(-1)

                    coords = self.renderer._coords[flat_idx]  # (B, 3)
                    win_min = coords.min(dim=0).values
                    win_max = coords.max(dim=0).values

                    # Cull with precomputed bbox (no R recomputation!)
                    overlap = (
                        (lower < win_max.unsqueeze(0))
                        & (upper > win_min.unsqueeze(0))
                    ).all(dim=-1)  # (N,)
                    local_ids = torch.where(overlap)[0]

                    n_local = local_ids.numel()
                    if n_local == 0:
                        done += 1
                        continue

                    pred = self._eval_block_precomputed(
                        coords, local_ids, means, prec, intensities,
                    )
                    values[flat_idx] = pred
                    total_evals += coords.shape[0] * n_local

                    done += 1
                    if verbose and done % max(1, n_total // 20) == 0:
                        elapsed = time.time() - t0
                        rate = done / max(elapsed, 1e-6)
                        eta = (n_total - done) / max(rate, 1e-6)
                        print(f"  Block {done}/{n_total}  "
                              f"N_local={n_local:,}  "
                              f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

        elapsed = time.time() - t0
        if verbose:
            print(f"  Done: {n_total} blocks in {elapsed:.1f}s  "
                  f"({total_evals/1e9:.1f}B evaluations)")

        return values.reshape(D, H, W).cpu()

    # ──────────────────────────────────────────────── full training loop
    def train(
        self,
        verbose: bool = True,
    ) -> Dict:
        """Run the complete training loop.

        Returns a dict with 'loss_history', 'metrics_history', 'final_loss',
        'final_metrics', 'training_time'.
        """
        c = self.config
        loss_history: List[float] = []
        metrics_history: List[Dict] = []

        t0 = time.time()
        for epoch in range(c.num_epochs):
            # LR schedule
            factor = self._get_lr_factor(epoch)
            self._set_lr(factor)

            loss = self.train_step()
            loss_history.append(loss)

            # ── adaptive density control ──
            do_densify = (
                c.densify_from_epoch <= epoch < c.densify_until_epoch
                and epoch % c.densify_interval == 0
                and epoch > 0
            )
            do_prune = (
                epoch >= c.prune_from_epoch
                and epoch % c.prune_interval == 0
                and epoch > 0
            )
            do_reset = (
                epoch > 0
                and epoch % c.opacity_reset_interval == 0
                and epoch < c.densify_until_epoch
            )

            n_pruned = n_added = 0
            if do_prune:
                n_pruned = self.prune()
            if do_densify:
                n_added = self.densify()
            if do_reset:
                self.reset_opacity()

            # ── logging ──
            if verbose and (epoch % c.log_interval == 0 or epoch == c.num_epochs - 1):
                msg = (f"[Epoch {epoch:5d}/{c.num_epochs}]  "
                       f"loss={loss:.6f}  N={self.model.n_gaussians:,}")
                if n_pruned:
                    msg += f"  pruned={n_pruned}"
                if n_added:
                    msg += f"  added={n_added}"
                print(msg)

            if epoch % c.eval_interval == 0 or epoch == c.num_epochs - 1:
                metrics = self.evaluate()
                metrics["epoch"] = epoch
                metrics_history.append(metrics)
                if verbose and epoch % c.log_interval == 0:
                    print(f"         PSNR={metrics['psnr']:.2f} dB  "
                          f"MSE={metrics['mse']:.6f}  "
                          f"model={metrics['model_mb']:.2f} MB")

        training_time = time.time() - t0
        final_metrics = self.evaluate()

        return {
            "loss_history": loss_history,
            "metrics_history": metrics_history,
            "final_loss": loss_history[-1] if loss_history else 0.0,
            "final_metrics": final_metrics,
            "training_time": training_time,
        }
