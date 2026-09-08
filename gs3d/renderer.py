"""
Differentiable volume renderer for 3-D Gaussian mixtures.

Instead of projecting Gaussians to a 2-D image (classical splatting), we
evaluate the mixture on a regular 3-D voxel grid and compare to a target
volume.  The renderer is designed for the *optimisation* inner-loop and
needs to be **memory-efficient**.

Strategy
--------
1.  Random-coordinate sampling  — draw B random voxel coordinates per step,
    evaluate the mixture, compare to target values at those coordinates.
2.  Full-volume rendering  — for evaluation / visualisation only.

Both paths are tiled over the Gaussian index dimension when N is large
to keep peak GPU memory bounded.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn.functional as F

from gs3d.model import GaussianModel3D, quaternion_to_rotation_matrix


class VolumeRenderer:
    """Evaluate a ``GaussianModel3D`` on a 3-D voxel grid.

    The renderer holds cached coordinate tensors and precomputes per-voxel
    indexing so that random-coordinate sampling is zero-copy.

    Parameters
    ----------
    volume_shape : (D, H, W)
        Target voxel grid dimensions.
    volume_extent : (ez, ey, ex)
        Physical extent matching the model's ``volume_extent``.
    device : torch.device
        GPU device.
    """

    def __init__(
        self,
        volume_shape: Tuple[int, int, int],
        volume_extent: Tuple[float, float, float],
        device: torch.device,
    ):
        self.shape = volume_shape
        self.extent = volume_extent
        self.device = device
        D, H, W = volume_shape
        self.total_voxels = D * H * W

        # Pre-build the full coordinate grid (float32) — stored on GPU.
        ez, ey, ex = volume_extent
        zz = torch.linspace(0, ez, D, device=device)
        yy = torch.linspace(0, ey, H, device=device)
        xx = torch.linspace(0, ex, W, device=device)
        grid = torch.stack(torch.meshgrid(zz, yy, xx, indexing="ij"), dim=-1)
        self._coords = grid.reshape(-1, 3)     # (D*H*W, 3)

    # ───────────────────────────────── random-sample evaluation (training) ──
    def sample_coords(self, n_samples: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Return n_samples random voxel coordinates **and their flat indices**.

        Returns
        -------
        coords : (B, 3)
        indices : (B,)  — flat indices into a (D*H*W,) array.
        """
        idx = torch.randint(0, self.total_voxels, (n_samples,), device=self.device)
        return self._coords[idx], idx

    # ─────────────────────── random 3-D window sampling (windowed SGD) ──
    def sample_window(
        self,
        window_size: Tuple[int, int, int],
        volume_shape: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor]:
        """Sample a contiguous 3-D sub-block of voxels at a random position.

        Parameters
        ----------
        window_size : (wd, wh, ww)
            Size of the window in voxels.  Clamped to volume dimensions.
        volume_shape : (D, H, W)
            Full volume shape (used for flat-index computation).

        Returns
        -------
        coords    : (wd*wh*ww, 3)  — physical coordinates of the window voxels.
        flat_idx  : (wd*wh*ww,)    — flat indices into the (D*H*W,) target array.
        win_min   : (3,)           — physical lower-corner of the window.
        win_max   : (3,)           — physical upper-corner of the window.
        """
        D, H, W = volume_shape
        wd = min(window_size[0], D)
        wh = min(window_size[1], H)
        ww = min(window_size[2], W)

        # Random origin (in voxel indices)
        d0 = torch.randint(0, max(D - wd, 0) + 1, (1,), device=self.device).item()
        h0 = torch.randint(0, max(H - wh, 0) + 1, (1,), device=self.device).item()
        w0 = torch.randint(0, max(W - ww, 0) + 1, (1,), device=self.device).item()

        # Build flat indices for the sub-block
        dd = torch.arange(d0, d0 + wd, device=self.device)
        hh = torch.arange(h0, h0 + wh, device=self.device)
        ww_idx = torch.arange(w0, w0 + ww, device=self.device)
        gd, gh, gw = torch.meshgrid(dd, hh, ww_idx, indexing="ij")
        flat_idx = (gd * H * W + gh * W + gw).reshape(-1)

        coords = self._coords[flat_idx]  # (B, 3)

        # Physical extent of the window
        win_min = coords.min(dim=0).values  # (3,)
        win_max = coords.max(dim=0).values  # (3,)

        return coords, flat_idx, win_min, win_max

    # ─────────────────────────── differentiable forward on sampled coords ──
    @staticmethod
    def evaluate_gaussians(
        model: GaussianModel3D,
        coords: torch.Tensor,
        tile_gaussians: int = 0,
        cutoff_sigma: float = 4.0,
    ) -> torch.Tensor:
        """Evaluate the Gaussian mixture at *coords* differentiably.

        When *tile_gaussians* > 0 the evaluation is tiled across the
        Gaussian-index dimension to keep peak memory bounded.

        Gaussians whose centres are farther than ``cutoff_sigma`` × max_scale
        from **all** query points in a tile are skipped entirely (the
        contribution is < exp(-8) ≈ 3e-4 and negligible).

        Args:
            model: GaussianModel3D instance.
            coords: (B, 3) query points.
            tile_gaussians: 0 = auto, else explicit tile size.
            cutoff_sigma: distance cutoff in units of max per-axis scale.

        Returns:
            (B,) predicted values.
        """
        N = model.n_gaussians
        B = coords.shape[0]

        if tile_gaussians <= 0:
            # heuristic: keep B*TILE*3 ≤ ~500 M floats
            tile_gaussians = max(1, min(N, 500_000_000 // max(3 * B, 1)))

        result = torch.zeros(B, device=coords.device)
        intensities = torch.sigmoid(model._intensity_logit)

        # Pre-compute max scale per Gaussian for distance culling
        max_scales = torch.exp(model._log_scales).max(dim=-1).values  # (N,)
        cutoff_dists = max_scales * cutoff_sigma                       # (N,)

        # Bounding box of query points for coarse culling
        coords_min = coords.min(dim=0).values  # (3,)
        coords_max = coords.max(dim=0).values  # (3,)

        for g0 in range(0, N, tile_gaussians):
            g1 = min(g0 + tile_gaussians, N)
            m = model._means[g0:g1]                              # (T, 3)

            # Coarse bounding-box culling: skip Gaussians too far from
            # the entire query batch
            cd = cutoff_dists[g0:g1]                             # (T,)
            box_dist = torch.clamp(coords_min.unsqueeze(0) - m, min=0).pow(2) + \
                       torch.clamp(m - coords_max.unsqueeze(0), min=0).pow(2)
            box_dist = box_dist.sum(-1).sqrt()                   # (T,)
            active = box_dist < cd
            if not active.any():
                continue

            m = m[active]
            q = model._quats[g0:g1][active]                     # (T', 4)
            ls = model._log_scales[g0:g1][active]               # (T', 3)
            a  = intensities[g0:g1][active]                      # (T',)

            R = quaternion_to_rotation_matrix(q)                 # (T', 3, 3)
            inv_s = torch.exp(-ls)                               # (T', 3)
            RiS = R * inv_s.unsqueeze(-2)                        # (T', 3, 3)
            prec = RiS @ RiS.transpose(-1, -2)                  # (T', 3, 3)

            diff = coords.unsqueeze(1) - m.unsqueeze(0)          # (B, T', 3)
            mahal = torch.einsum("bti,tij,btj->bt", diff, prec, diff)  # (B, T')
            gauss = torch.exp(-0.5 * mahal)                     # (B, T')
            result = result + (gauss * a.unsqueeze(0)).sum(-1)

        return result

    # ────────────────────────────────── full-volume (non-differentiable) ──
    @torch.no_grad()
    def render_full(
        self,
        model: GaussianModel3D,
        batch_size: int = 262144,
        tile_gaussians: int = 0,
    ) -> torch.Tensor:
        """Evaluate Gaussians on the entire grid — returns (D, H, W) CPU tensor.

        Both the voxel dimension and the Gaussian dimension are tiled so that
        peak memory ≈ batch_size × tile_gaussians × 3 floats, keeping well
        within GPU budget regardless of N or grid size.

        Args:
            batch_size: number of voxels evaluated per chunk.
            tile_gaussians: Gaussians evaluated per chunk (0 = auto).
        """
        D, H, W = self.shape
        N = model.n_gaussians
        values = torch.zeros(self.total_voxels, device=self.device)

        if tile_gaussians <= 0:
            # keep peak_mem ≈ batch_size * TILE * 3 * 4 B ≤ 2 GB
            tile_gaussians = max(1, min(N, 2_000_000_000 // max(batch_size * 3 * 4, 1)))

        intensities = torch.sigmoid(model._intensity_logit)  # (N,)

        for s in range(0, self.total_voxels, batch_size):
            e = min(s + batch_size, self.total_voxels)
            pts = self._coords[s:e]  # (B, 3)
            acc = torch.zeros(e - s, device=self.device)

            for g0 in range(0, N, tile_gaussians):
                g1 = min(g0 + tile_gaussians, N)
                m  = model._means[g0:g1]           # (T, 3)
                q  = model._quats[g0:g1]           # (T, 4)
                ls = model._log_scales[g0:g1]      # (T, 3)
                a  = intensities[g0:g1]            # (T,)

                R    = quaternion_to_rotation_matrix(q)   # (T, 3, 3)
                invs = torch.exp(-ls)                     # (T, 3)
                RiS  = R * invs.unsqueeze(-2)             # (T, 3, 3)
                prec = RiS @ RiS.transpose(-1, -2)       # (T, 3, 3)

                diff  = pts.unsqueeze(1) - m.unsqueeze(0)              # (B, T, 3)
                mahal = torch.einsum("bti,tij,btj->bt", diff, prec, diff)
                gauss = torch.exp(-0.5 * mahal)                        # (B, T)
                acc  += (gauss * a.unsqueeze(0)).sum(-1)

            values[s:e] = acc

        return values.reshape(D, H, W).cpu()
