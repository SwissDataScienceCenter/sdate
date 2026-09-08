"""
elm_inr.py — Dimension-agnostic ELM-INR (2D + 3D)
===================================================

Supports:
  - 2D grayscale images  (H × W)
  - 3D scalar volumes    (H × W × T  or  T × H × W)

Method (identical for both):
  1. Tile the normalised coordinate domain [-1,1]^d with overlapping
     rectangular (2D) or cuboidal (3D) subdomains.
  2. Each subdomain gets a local ELM with frozen random hidden weights
     and a single output weight vector α solved in closed form.
  3. Reconstruction blends all local predictions via a normalised
     partition-of-unity (PoU):
         f_hat(x) = Σ_i φ_i(x) f_hat_i(x)  /  Σ_i φ_i(x)

No backpropagation, no BEAM, no adaptive partitions.

Usage::

    from inct.elm_inr import ELMINR, ELMINRConfig

    # ── 2D ───────────────────────────────────
    cfg = ELMINRConfig(n_parts=(8, 8), hidden_dim=512)
    model = ELMINR(cfg, device)
    model.fit(image_hw)
    recon = model.reconstruct_image()          # → [H, W]

    # ── 3D ───────────────────────────────────
    cfg = ELMINRConfig(n_parts=(4, 4, 5), hidden_dim=1024)
    model = ELMINR(cfg, device)
    model.fit(volume_hwt)
    recon = model.reconstruct_volume()         # → [H, W, T]

Based on *"Escaping Spectral Bias without Backpropagation: Fast Implicit
Neural Representations with Extreme Learning Machines"*.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from itertools import product as _product
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ELMINRConfig:
    """Configuration for the dimension-agnostic ELM-INR model.

    Parameters
    ----------
    n_parts : tuple of int
        Number of subdomains along each axis.
        ``(ny, nx)`` for 2D, ``(ny, nx, nt)`` for 3D.
        The *first* entries are spatial, the *last* is temporal when ``dim=3``.
    overlap : float or tuple of float
        Fractional overlap per axis (0 = no overlap).  A single float is
        broadcast to every axis.
    hidden_dim : int
        Number of hidden units per local ELM.
    expand_dim : int
        If ``> 0``, use a two-layer frozen hidden network:
        ``input → expand_dim → hidden_dim``.  Both layers are regenerable
        from the seed (zero stored cost).
    activation : str
        Hidden-layer activation: ``'relu'``, ``'tanh'``, or ``'sigmoid'``.
    use_rff : bool
        Use Random Fourier Features for input encoding.
    rff_dim : int
        Number of RFF frequencies.  Output feature dimension = ``2 * rff_dim``.
    rff_sigma : float
        Standard deviation of the Gaussian frequency matrix **B**.
    ridge_lambda : float
        Tikhonov regularisation parameter for the augmented least-squares
        solver.
    window : str
        Partition-of-unity window type: ``'hann'``, ``'triangular'``, or
        ``'gaussian'``.
    seed : int
        Random seed for reproducible frozen weights.
    data_layout : str
        For 3D data only.  ``'HWT'`` means the input tensor has shape
        ``(H, W, T)``; ``'THW'`` means ``(T, H, W)``.  Ignored for 2D.
    """

    # Subdomain grid
    n_parts: Tuple[int, ...] = (8, 8)

    # Overlap
    overlap: Union[float, Tuple[float, ...]] = 0.25

    # Local ELM architecture
    hidden_dim: int = 512
    expand_dim: int = 0
    activation: Literal["relu", "tanh", "sigmoid"] = "relu"

    # Input encoding
    use_rff: bool = True
    rff_dim: int = 128
    rff_sigma: float = 10.0

    # Solver
    ridge_lambda: float = 1e-4

    # Partition-of-unity window
    window: Literal["hann", "triangular", "gaussian"] = "hann"

    # Reproducibility
    seed: int = 42

    # 3D data layout (ignored for 2D)
    data_layout: Literal["HWT", "THW"] = "HWT"

    # ── derived helpers ────────────────────────────────────────────────

    @property
    def dim(self) -> int:
        """Spatial + temporal dimensionality (2 or 3)."""
        return len(self.n_parts)

    def overlap_per_axis(self) -> Tuple[float, ...]:
        """Return overlap ratio for every axis as a tuple."""
        if isinstance(self.overlap, (int, float)):
            return (float(self.overlap),) * self.dim
        assert len(self.overlap) == self.dim
        return tuple(float(o) for o in self.overlap)


# ═══════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════

class ELMINR:
    """Dimension-agnostic ELM-INR for 2D images and 3D scalar volumes.

    The same code path handles both dimensionalities — most helper methods
    operate on *d*-dimensional coordinates / tuples so that almost nothing
    is duplicated between the 2D and 3D cases.

    Architecture per subdomain (single layer, ``expand_dim=0``)::

        H_i(x) = σ(z(x) @ W_h + b_h)        [N_i, hidden_dim]
        f_hat_i(x) = H_i(x) @ α_i            [N_i, 1]

    Two-layer variant (``expand_dim > 0``)::

        H1 = σ(z @ W_h1 + b_h1)              [N_i, expand_dim]
        H  = σ(H1 @ W_h2 + b_h2)             [N_i, hidden_dim]

    All hidden weights are random, frozen, and regenerable from the seed.
    Only α_i (and the shared RFF matrix B) must be stored.
    """

    def __init__(
        self,
        cfg: ELMINRConfig,
        device: torch.device = torch.device("cpu"),
    ):
        self.cfg = cfg
        self.dim = cfg.dim
        self.device = device
        self.fitted = False

        assert self.dim in (2, 3), f"Only dim=2 and dim=3 supported, got {self.dim}"

        # Reproducible RNG (same stream for all frozen weights)
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(cfg.seed)

        # RFF frequency matrix B: [d, rff_dim]
        if cfg.use_rff:
            self.B = (
                torch.randn(self.dim, cfg.rff_dim, generator=self.rng, device=device)
                * cfg.rff_sigma
            )
            self.input_dim = 2 * cfg.rff_dim  # cos + sin
        else:
            self.B = None
            self.input_dim = self.dim

        self.subdomains: List[dict] = []
        self.alphas: List[torch.Tensor] = []
        self.data_shape: Optional[Tuple[int, ...]] = None  # canonical (H,W) or (H,W,T)

    # ─── Coordinate generation ─────────────────────────────────────────

    @staticmethod
    def make_coords(
        shape: Tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        """Normalised coordinates in [-1, 1]^d.

        Args:
            shape: canonical data shape — ``(H, W)`` or ``(H, W, T)``.
        Returns:
            coords: ``[N, d]``, where ``N = prod(shape)``.
        """
        linspaces = [torch.linspace(-1, 1, s, device=device) for s in shape]
        grids = torch.meshgrid(*linspaces, indexing="ij")
        return torch.stack([g.flatten() for g in grids], dim=-1)

    # ─── Feature encoding ──────────────────────────────────────────────

    def encode(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode *d*-dimensional coordinates into input features.

        RFF:  ``z(x) = [cos(2π B^T x), sin(2π B^T x)]``  (2 × rff_dim)
        Raw:  ``z(x) = x``
        """
        if self.cfg.use_rff:
            proj = 2.0 * math.pi * (coords @ self.B)  # [N, rff_dim]
            return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return coords

    # ─── Subdomain generation ──────────────────────────────────────────

    def _init_subdomains(self) -> None:
        """Create a regular d-dim grid of overlapping subdomains.

        2D → rectangles in (y, x).
        3D → cuboids   in (y, x, t).
        """
        cfg = self.cfg
        d = self.dim
        n_parts = cfg.n_parts
        overlap = cfg.overlap_per_axis()

        # Per-axis centres and half-sizes
        centers_per_axis: List[List[float]] = []
        half_per_axis: List[float] = []
        for axis in range(d):
            n = n_parts[axis]
            base_half = 1.0 / n
            half = base_half * (1.0 + overlap[axis])
            centers = [
                -1.0 + base_half + 2.0 * base_half * i for i in range(n)
            ]
            centers_per_axis.append(centers)
            half_per_axis.append(half)

        self.subdomains = []

        # Cartesian product over all axes → one subdomain per grid cell
        for idx in _product(*(range(n) for n in n_parts)):
            center = tuple(
                centers_per_axis[axis][idx[axis]] for axis in range(d)
            )
            half_size = tuple(half_per_axis)

            # Frozen random hidden weights (regenerable from seed)
            if cfg.expand_dim > 0:
                std1 = 1.0 / math.sqrt(self.input_dim)
                W_h1 = (
                    torch.randn(
                        self.input_dim,
                        cfg.expand_dim,
                        generator=self.rng,
                        device=self.device,
                    )
                    * std1
                )
                b_h1 = torch.rand(
                    cfg.expand_dim, generator=self.rng, device=self.device
                )
                std2 = 1.0 / math.sqrt(cfg.expand_dim)
                W_h2 = (
                    torch.randn(
                        cfg.expand_dim,
                        cfg.hidden_dim,
                        generator=self.rng,
                        device=self.device,
                    )
                    * std2
                )
                b_h2 = torch.rand(
                    cfg.hidden_dim, generator=self.rng, device=self.device
                )
                self.subdomains.append(
                    {
                        "center": center,
                        "half_size": half_size,
                        "W_h1": W_h1,
                        "b_h1": b_h1,
                        "W_h2": W_h2,
                        "b_h2": b_h2,
                    }
                )
            else:
                # U[0,1] bias keeps ReLU units alive (avoids dead-neuron NaN)
                std = 1.0 / math.sqrt(self.input_dim)
                W_h = (
                    torch.randn(
                        self.input_dim,
                        cfg.hidden_dim,
                        generator=self.rng,
                        device=self.device,
                    )
                    * std
                )
                b_h = torch.rand(
                    cfg.hidden_dim, generator=self.rng, device=self.device
                )
                self.subdomains.append(
                    {
                        "center": center,
                        "half_size": half_size,
                        "W_h": W_h,
                        "b_h": b_h,
                    }
                )

    # ─── Partition-of-unity windows ────────────────────────────────────

    def _window_1d(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate a 1-D window on normalised distance ``t ∈ [-1, 1]``."""
        t = t.clamp(-1, 1)
        if self.cfg.window == "hann":
            return 0.5 * (1.0 + torch.cos(math.pi * t))
        elif self.cfg.window == "triangular":
            return 1.0 - t.abs()
        elif self.cfg.window == "gaussian":
            return torch.exp(-0.5 * (t / 0.4) ** 2)
        raise ValueError(f"Unknown window: {self.cfg.window}")

    def _eval_window(
        self, coords: torch.Tensor, sub: dict
    ) -> torch.Tensor:
        """Separable d-dim window  φ(x) = ∏_k w₁d((x_k − c_k) / h_k).

        Returns a non-negative weight vector ``[N]``.  Zero outside the
        subdomain support.
        """
        center = sub["center"]
        half_size = sub["half_size"]
        d = self.dim

        # Determine which points fall inside the cuboid support
        inside = torch.ones(
            coords.shape[0], dtype=torch.bool, device=coords.device
        )
        for axis in range(d):
            t = (coords[:, axis] - center[axis]) / half_size[axis]
            inside &= t.abs() <= 1.0

        phi = torch.zeros(coords.shape[0], device=coords.device)
        if inside.any():
            val = torch.ones(inside.sum(), device=coords.device)
            for axis in range(d):
                t = (coords[inside, axis] - center[axis]) / half_size[axis]
                val = val * self._window_1d(t)
            phi[inside] = val
        return phi

    # ─── Hidden-layer forward pass ─────────────────────────────────────

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.activation == "relu":
            return F.relu(x)
        elif self.cfg.activation == "tanh":
            return torch.tanh(x)
        elif self.cfg.activation == "sigmoid":
            return torch.sigmoid(x)
        raise ValueError(f"Unknown activation: {self.cfg.activation}")

    def _hidden_matrix(
        self, features: torch.Tensor, sub: dict
    ) -> torch.Tensor:
        """Hidden activations H for one subdomain's ELM.

        Single layer:  ``H = σ(z @ W_h + b_h)``
        Two layer:     ``H = σ( σ(z @ W_h1 + b_h1) @ W_h2 + b_h2 )``
        """
        if "W_h1" in sub:
            h1 = self._activation(features @ sub["W_h1"] + sub["b_h1"])
            return self._activation(h1 @ sub["W_h2"] + sub["b_h2"])
        return self._activation(features @ sub["W_h"] + sub["b_h"])

    # ─── Ridge-regression solver ───────────────────────────────────────

    def _solve_alpha(
        self, H: torch.Tensor, Y: torch.Tensor
    ) -> torch.Tensor:
        """Augmented least-squares ridge regression.

        Solves ``[H; √λ I] α = [Y; 0]`` via ``torch.linalg.lstsq``.
        This avoids forming ``H^T H`` (which squares the condition number)
        and is robust to rank-deficient H.
        """
        m = H.shape[1]
        sqrt_lam = math.sqrt(self.cfg.ridge_lambda)
        H_aug = torch.cat(
            [H, sqrt_lam * torch.eye(m, device=H.device, dtype=H.dtype)],
            dim=0,
        )
        Y_aug = torch.cat(
            [Y, torch.zeros(m, 1, device=Y.device, dtype=Y.dtype)], dim=0
        )
        alpha = torch.linalg.lstsq(H_aug, Y_aug).solution
        return torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)

    # ─── Data layout handling ──────────────────────────────────────────

    def _to_canonical(self, data: torch.Tensor) -> torch.Tensor:
        """Convert input to canonical layout: ``(H, W)`` or ``(H, W, T)``."""
        if self.dim == 2:
            assert data.ndim == 2, f"Expected 2D tensor, got {data.shape}"
            return data
        assert data.ndim == 3, f"Expected 3D tensor, got {data.shape}"
        if self.cfg.data_layout == "THW":
            return data.permute(1, 2, 0)  # T,H,W → H,W,T
        return data  # already H,W,T

    # ─── Fit ───────────────────────────────────────────────────────────

    def fit(
        self, data: torch.Tensor, verbose: bool = True
    ) -> "ELMINR":
        """Fit the ELM-INR to scalar data (2D image or 3D volume).

        Args:
            data: ``[H, W]`` for 2D **or** ``[H, W, T]`` / ``[T, H, W]``
                  for 3D (``data_layout`` controls interpretation).
            verbose: print fitting progress.
        Returns:
            self
        """
        data = self._to_canonical(data.to(self.device))
        self.data_shape = data.shape

        t0 = time.time()

        # Coordinates & targets
        coords = self.make_coords(self.data_shape, self.device)  # [N, d]
        Y = data.flatten().unsqueeze(-1)  # [N, 1]
        features = self.encode(coords)  # [N, input_dim]

        # Subdomains
        self._init_subdomains()
        n_sub = len(self.subdomains)

        if verbose:
            shape_str = " × ".join(str(s) for s in self.data_shape)
            parts_str = " × ".join(str(n) for n in self.cfg.n_parts)
            N = int(np.prod(self.data_shape))
            if self.cfg.expand_dim > 0:
                arch = f"{self.input_dim} → {self.cfg.expand_dim} → {self.cfg.hidden_dim}"
            else:
                arch = f"{self.input_dim} → {self.cfg.hidden_dim}"
            print(
                f"ELM-INR {self.dim}D: {parts_str} = {n_sub} subdomains, "
                f"arch=[{arch}], overlap={self.cfg.overlap}"
            )
            print(f"  Data shape: {shape_str} = {N:,} {'voxels' if self.dim == 3 else 'pixels'}")
            print(
                f"  Features: {self.input_dim}  "
                f"({'RFF σ=' + str(self.cfg.rff_sigma) if self.cfg.use_rff else 'raw coords'})"
            )

        # Fit each subdomain independently
        self.alphas = []
        total_points = 0

        for sub in self.subdomains:
            center = sub["center"]
            half_size = sub["half_size"]

            # Mask: points inside this subdomain's cuboid support
            mask = torch.ones(
                coords.shape[0], dtype=torch.bool, device=self.device
            )
            for axis in range(self.dim):
                mask &= (coords[:, axis] - center[axis]).abs() <= half_size[axis]

            n_pts = mask.sum().item()
            total_points += n_pts

            # Read actual hidden_dim from the subdomain's own weights
            actual_hdim = (
                sub["W_h2"] if "W_h2" in sub else sub["W_h"]
            ).shape[1]

            if n_pts == 0:
                self.alphas.append(
                    torch.zeros(actual_hdim, 1, device=self.device)
                )
                continue

            H_sub = self._hidden_matrix(features[mask], sub)
            alpha = self._solve_alpha(H_sub, Y[mask])
            self.alphas.append(alpha)

        self.fitted = True
        elapsed = time.time() - t0

        if verbose:
            print(f"  Avg pts/subdomain: {total_points / n_sub:,.0f}")
            print(f"  Fit completed in {elapsed:.2f}s (no backpropagation)")

        return self

    # ─── Predict ───────────────────────────────────────────────────────

    def predict(
        self,
        coords: Optional[torch.Tensor] = None,
        chunk_size: int = 0,
    ) -> torch.Tensor:
        """Evaluate the ELM-INR at arbitrary d-dim coordinates.

        Blend:  ``f_hat(x) = Σ_i φ_i(x) f_hat_i(x) / Σ_i φ_i(x)``

        Args:
            coords: ``[N, d]`` query points; ``None`` → training grid.
            chunk_size: if ``> 0``, process in chunks to limit GPU memory.
        Returns:
            ``[N]`` predicted scalar values.
        """
        assert self.fitted, "Call fit() first"

        if coords is None:
            coords = self.make_coords(self.data_shape, self.device)

        if chunk_size > 0:
            return self._predict_chunked(coords, chunk_size)

        N = coords.shape[0]
        features = self.encode(coords)

        weighted_sum = torch.zeros(N, device=self.device)
        weight_sum = torch.zeros(N, device=self.device)

        for sub, alpha in zip(self.subdomains, self.alphas):
            phi = self._eval_window(coords, sub)
            active = phi > 0
            if not active.any():
                continue
            H_k = self._hidden_matrix(features[active], sub)
            pred_k = (H_k @ alpha).squeeze(-1)
            weighted_sum[active] += phi[active] * pred_k
            weight_sum[active] += phi[active]

        return weighted_sum / weight_sum.clamp(min=1e-10)

    def _predict_chunked(
        self, coords: torch.Tensor, chunk_size: int
    ) -> torch.Tensor:
        """Split ``predict()`` into memory-friendly chunks."""
        parts = []
        for i in range(0, coords.shape[0], chunk_size):
            parts.append(self.predict(coords[i : i + chunk_size]))
        return torch.cat(parts)

    # ─── Reconstruction helpers ────────────────────────────────────────

    def reconstruct_image(self, **predict_kw) -> torch.Tensor:
        """Reconstruct a 2D image → ``[H, W]``."""
        assert self.dim == 2, "reconstruct_image() is for 2D data"
        return self.predict(**predict_kw).reshape(self.data_shape)

    def reconstruct_volume(
        self, original_layout: bool = False, **predict_kw
    ) -> torch.Tensor:
        """Reconstruct a 3D volume → ``[H, W, T]`` (canonical).

        Args:
            original_layout: if ``True`` and ``data_layout='THW'``,
                return ``[T, H, W]`` matching the original input layout.
        """
        assert self.dim == 3, "reconstruct_volume() is for 3D data"
        vol = self.predict(**predict_kw).reshape(self.data_shape)  # H,W,T
        if original_layout and self.cfg.data_layout == "THW":
            vol = vol.permute(2, 0, 1)  # → T,H,W
        return vol

    def reconstruct(self, **predict_kw) -> torch.Tensor:
        """Generic reconstruct in canonical layout."""
        return self.predict(**predict_kw).reshape(self.data_shape)

    # ─── Storage analysis ──────────────────────────────────────────────

    def storage_bytes(self, dtype_bytes: int = 4) -> dict:
        """Estimate stored vs regenerable byte counts.

        **Stored**: α weights + RFF matrix B + metadata.
        **Regenerable**: all hidden weights (from seed).
        """
        n_sub = len(self.subdomains)

        # α: sum actual sizes (handles mixed hidden_dim)
        if self.alphas:
            alpha_bytes = sum(a.numel() for a in self.alphas) * dtype_bytes
        else:
            alpha_bytes = n_sub * self.cfg.hidden_dim * dtype_bytes

        # RFF matrix B: [d, rff_dim]
        b_bytes = (self.dim * self.cfg.rff_dim * dtype_bytes) if self.cfg.use_rff else 0

        # Hidden weights (regenerable from seed — informational only)
        if self.subdomains:
            if self.cfg.expand_dim > 0:
                wh_bytes = sum(
                    sub["W_h1"].numel()
                    + sub["b_h1"].numel()
                    + sub["W_h2"].numel()
                    + sub["b_h2"].numel()
                    for sub in self.subdomains
                ) * dtype_bytes
            else:
                wh_bytes = sum(
                    sub["W_h"].numel() + sub["b_h"].numel()
                    for sub in self.subdomains
                ) * dtype_bytes
        else:
            wh_bytes = 0

        meta_bytes = 64

        total_stored = alpha_bytes + b_bytes + meta_bytes
        return {
            "alpha_bytes": alpha_bytes,
            "rff_B_bytes": b_bytes,
            "hidden_bytes (regenerable)": wh_bytes,
            "meta_bytes": meta_bytes,
            "total_stored": total_stored,
            "total_if_all_stored": total_stored + wh_bytes,
        }

    def compression_ratio(self, dtype_bytes: int = 4, raw_dtype_bytes: int = 2) -> float:
        """Compression ratio: raw data size / stored size.

        ``raw_dtype_bytes=2`` ↔ int16 baseline (the typical raw format).
        """
        N = int(np.prod(self.data_shape))
        raw = N * raw_dtype_bytes
        return raw / self.storage_bytes(dtype_bytes)["total_stored"]


# ═══════════════════════════════════════════════════════════════════════════
#  Quality metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_psnr(pred: np.ndarray, tgt: np.ndarray) -> float:
    dr = float(tgt.max() - tgt.min())
    mse = float(np.mean((np.clip(pred, tgt.min(), tgt.max()) - tgt) ** 2))
    return 10.0 * math.log10(dr ** 2 / (mse + 1e-10))


def compute_ssim(pred: np.ndarray, tgt: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return float("nan")
    dr = float(tgt.max() - tgt.min())
    return float(
        structural_similarity(
            tgt, np.clip(pred, tgt.min(), tgt.max()), data_range=dr
        )
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Synthetic data generators (for demos)
# ═══════════════════════════════════════════════════════════════════════════

def _make_synthetic_2d(H: int = 256, W: int = 256) -> np.ndarray:
    """Checkerboard + Gaussian bumps → smooth 2D test image in [0, 1]."""
    y = np.linspace(-1, 1, H)
    x = np.linspace(-1, 1, W)
    Y, X = np.meshgrid(y, x, indexing="ij")

    # Checkerboard
    img = 0.5 + 0.3 * np.sin(5 * np.pi * X) * np.sin(5 * np.pi * Y)
    # Gaussian bump
    img += 0.4 * np.exp(-((X - 0.3) ** 2 + (Y + 0.2) ** 2) / 0.08)
    # Normalise to [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img.astype(np.float32)


def _make_synthetic_3d(H: int = 64, W: int = 64, T: int = 20) -> np.ndarray:
    """Moving Gaussian blob + drifting sinusoid → smooth 3D scalar field.

    Layout: ``(H, W, T)`` (canonical).
    """
    y = np.linspace(-1, 1, H)
    x = np.linspace(-1, 1, W)
    t = np.linspace(-1, 1, T)
    Y, X, TT = np.meshgrid(y, x, t, indexing="ij")

    # Moving Gaussian: centre traces a circle in (x, y), constant in t
    cx = 0.4 * np.sin(np.pi * TT)
    cy = 0.4 * np.cos(np.pi * TT)
    blob = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / 0.05)

    # Drifting sinusoidal background
    wave = 0.3 * np.sin(3 * np.pi * X + 2 * np.pi * TT) * np.cos(3 * np.pi * Y)

    vol = blob + wave
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    return vol.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  Demos
# ═══════════════════════════════════════════════════════════════════════════

def demo_2d(device: torch.device | None = None) -> None:
    """Demo A — 2D grayscale image."""
    import matplotlib.pyplot as plt

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  Demo A: 2D grayscale image")
    print("=" * 60)

    # Synthetic image
    img_np = _make_synthetic_2d(256, 256)
    image = torch.from_numpy(img_np).to(device)

    # Config — similar to the 2D notebook defaults
    cfg = ELMINRConfig(
        n_parts=(8, 8),
        overlap=0.25,
        hidden_dim=512,
        use_rff=True,
        rff_dim=128,
        rff_sigma=10.0,
        ridge_lambda=1e-4,
        window="hann",
        seed=42,
    )

    model = ELMINR(cfg, device)
    model.fit(image)

    recon = model.reconstruct_image()
    recon_np = recon.clamp(0, 1).cpu().numpy()

    psnr = compute_psnr(recon_np, img_np)
    ssim = compute_ssim(recon_np, img_np)
    cr = model.compression_ratio()
    print(f"\n  PSNR = {psnr:.2f} dB  |  SSIM = {ssim:.4f}  |  CR = {cr:.2f}×")

    # Visualise
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(recon_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"ELM-INR  PSNR={psnr:.2f} dB")
    axes[1].axis("off")

    err = np.abs(img_np - recon_np)
    im = axes[2].imshow(err, cmap="hot", vmin=0, vmax=err.max())
    axes[2].set_title(f"|Error|  max={err.max():.4f}")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.suptitle("Demo A: 2D ELM-INR", fontsize=14)
    plt.tight_layout()
    plt.show()


def demo_3d(device: torch.device | None = None) -> None:
    """Demo B — 3D spatiotemporal volume."""
    import matplotlib.pyplot as plt

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("  Demo B: 3D spatiotemporal volume")
    print("=" * 60)

    H, W, T = 64, 64, 20
    vol_np = _make_synthetic_3d(H, W, T)
    volume = torch.from_numpy(vol_np).to(device)

    # Config — 3D with separate spatial / temporal partitioning
    cfg = ELMINRConfig(
        n_parts=(4, 4, 5),         # 4×4 spatial, 5 temporal chunks
        overlap=(0.25, 0.25, 0.2), # slightly less temporal overlap
        hidden_dim=512,
        use_rff=True,
        rff_dim=128,
        rff_sigma=10.0,
        ridge_lambda=1e-4,
        window="hann",
        seed=42,
    )

    model = ELMINR(cfg, device)
    model.fit(volume)

    recon_vol = model.reconstruct_volume()
    recon_np = recon_vol.clamp(0, 1).cpu().numpy()

    psnr_all = compute_psnr(recon_np, vol_np)
    cr = model.compression_ratio()
    print(f"\n  Volume PSNR = {psnr_all:.2f} dB  |  CR = {cr:.2f}×")

    # Per-frame PSNR
    print("  Per-frame PSNR:", end="")
    for t_idx in range(T):
        p = compute_psnr(recon_np[:, :, t_idx], vol_np[:, :, t_idx])
        if t_idx < 10 or t_idx == T - 1:
            print(f"  t={t_idx}: {p:.1f}", end="")
    print()

    # Visualise a few temporal slices
    n_show = min(6, T)
    show_idxs = np.linspace(0, T - 1, n_show, dtype=int)

    fig, axes = plt.subplots(2, n_show, figsize=(3.5 * n_show, 7))
    for col, t_idx in enumerate(show_idxs):
        axes[0, col].imshow(vol_np[:, :, t_idx], cmap="viridis", vmin=0, vmax=1)
        axes[0, col].set_title(f"GT  t={t_idx}")
        axes[0, col].axis("off")

        axes[1, col].imshow(recon_np[:, :, t_idx], cmap="viridis", vmin=0, vmax=1)
        p = compute_psnr(recon_np[:, :, t_idx], vol_np[:, :, t_idx])
        axes[1, col].set_title(f"Recon  {p:.1f} dB")
        axes[1, col].axis("off")

    plt.suptitle(
        f"Demo B: 3D ELM-INR  ({H}×{W}×{T})  —  PSNR={psnr_all:.2f} dB,  CR={cr:.2f}×",
        fontsize=13,
    )
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    demo_2d(device)
    demo_3d(device)


if __name__ == "__main__":
    main()
