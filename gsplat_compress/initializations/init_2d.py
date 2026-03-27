"""
2-D Gaussian initialisation strategies for orthographic image fitting.

Available strategies
--------------------
uniform_2d
    Regular grid with sub-pixel jitter — even coverage.

intensity_2d
    Positions sampled proportional to image intensity.

multiresolution_residual_2d
    Coarse-to-fine greedy matching pursuit.

    Idea
    ~~~~
    1. Place a small number of *large* Gaussians on a coarse grid.
    2. Quickly optimise their amplitudes/opacities only (~100–200 iterations).
    3. Compute the residual  ``r = target − render``.
    4. Build a new sampling weight  ``w = |r| + η |∇r|``
       to concentrate the next batch of Gaussians where the error is large
       *and* where fine texture is present (high gradient).
    5. Place a batch of *smaller* Gaussians sampled from ``w``.
    6. Repeat, halving the scale each stage.

    This is the cleanest answer to
    "place more Gaussians where there are more high-frequency details".
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F

from gsplat_compress.initialize import GaussianParams


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_params(
    means_xy: torch.Tensor,   # [N, 2]  pixel coords
    scale_px: float,
    init_opacity: float,
    image: torch.Tensor,      # [H, W]
    device: torch.device,
) -> GaussianParams:
    """Assemble a GaussianParams from pixel-space 2-D positions."""
    N = means_xy.shape[0]
    H, W = image.shape

    means_z = 1.0 + 0.1 * torch.randn(N, 1, device=device)
    means = torch.cat([means_xy, means_z], dim=-1)  # [N, 3]

    log_scales = torch.full((N, 3), math.log(scale_px), device=device)
    log_scales[:, 2] = math.log(0.5)

    quats = torch.zeros(N, 4, device=device)
    quats[:, 0] = 1.0

    px_idx = means_xy[:, 0].long().clamp(0, W - 1)
    py_idx = means_xy[:, 1].long().clamp(0, H - 1)
    sampled = image.to(device)[py_idx, px_idx]
    rgbs = torch.logit(sampled.clamp(0.01, 0.99)).unsqueeze(-1).repeat(1, 3)

    opacities = torch.logit(torch.full((N,), init_opacity, device=device))
    return GaussianParams(means, log_scales, quats, rgbs, opacities)


def _uniform_positions(n_points: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    """Return a n_side² × 2 tensor of uniform grid positions (pixel space)."""
    n_side = int(math.ceil(math.sqrt(n_points)))
    xs = torch.linspace(0, W, n_side, device=device)
    ys = torch.linspace(0, H, n_side, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # [n_side², 2]


def _sample_positions(
    weight: torch.Tensor,  # [H, W] non-negative — used as probability map
    n_points: int,
    H: int,
    W: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample n_points positions from the pixel-weight distribution, with sub-pixel jitter."""
    w = weight.to(device).reshape(-1).float()
    w = (w + 1e-6) / (w + 1e-6).sum()
    indices = torch.multinomial(w, n_points, replacement=True)
    px = (indices % W).float() + torch.rand(n_points, device=device) - 0.5
    py = (indices // W).float() + torch.rand(n_points, device=device) - 0.5
    px.clamp_(0.0, W - 1e-3)
    py.clamp_(0.0, H - 1e-3)
    return torch.stack([px, py], dim=-1)  # [N, 2]


def _image_gradient_mag(image: torch.Tensor) -> torch.Tensor:
    """Return spatial gradient magnitude map  ‖∇I‖  in [H, W] on the same device."""
    # Sobel kernels
    img = image.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=image.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=image.device
    ).view(1, 1, 3, 3)
    gx = F.conv2d(img, sobel_x, padding=1)
    gy = F.conv2d(img, sobel_y, padding=1)
    return (gx.pow(2) + gy.pow(2)).sqrt().squeeze()  # [H, W]


def _quick_amplitude_opt(
    params: GaussianParams,
    target: torch.Tensor,       # [H, W, 3]
    viewmat: torch.Tensor,
    K: torch.Tensor,
    W: int,
    H: int,
    n_iters: int,
    lr: float,
) -> None:
    """Optimise only rgbs + opacities in-place (no-grad on means/scales/quats)."""
    from gsplat_compress.renderer import render as _render

    params.means.requires_grad_(False)
    params.log_scales.requires_grad_(False)
    params.quats.requires_grad_(False)
    params.rgbs.requires_grad_(True)
    params.opacities.requires_grad_(True)

    opt = torch.optim.Adam(
        [params.rgbs, params.opacities], lr=lr
    )
    for _ in range(n_iters):
        opt.zero_grad(set_to_none=True)
        rendered, _, _ = _render(
            params.means, params.log_scales, params.quats,
            params.rgbs, params.opacities, viewmat, K, W, H,
        )
        loss = F.mse_loss(rendered, target.float())
        loss.backward()
        opt.step()

    # re-enable gradients on all params for subsequent full training
    params.means.requires_grad_(True)
    params.log_scales.requires_grad_(True)
    params.quats.requires_grad_(True)


# ── public API ────────────────────────────────────────────────────────────────

def uniform_2d(
    image: torch.Tensor,
    n_points: int,
    device: torch.device,
    init_scale_px: float = 3.0,
    init_opacity: float = 0.3,
    seed: int = 42,
) -> GaussianParams:
    """Regular grid initialisation with sub-pixel jitter.

    Parameters
    ----------
    image : Tensor [H, W]
        Grayscale target in [0, 1].
    n_points : int
        Desired number of Gaussians (rounded up to a perfect square).
    device : torch.device
    init_scale_px : float
        Initial Gaussian σ in pixels.
    init_opacity : float
        Initial sigmoid-space opacity.
    seed : int

    Returns
    -------
    GaussianParams with ``requires_grad = True``.
    """
    H, W = image.shape
    torch.manual_seed(seed)

    n_side = int(math.ceil(math.sqrt(n_points)))
    means_xy = _uniform_positions(n_side * n_side, H, W, device)
    n_actual = means_xy.shape[0]

    params = _build_params(means_xy, init_scale_px, init_opacity, image, device)
    params.require_grad_()
    return params


def intensity_2d(
    image: torch.Tensor,
    n_points: int,
    device: torch.device,
    init_scale_px: float = 1.5,
    init_opacity: float = 0.3,
    seed: int = 42,
) -> GaussianParams:
    """Positions sampled proportional to image intensity.

    Concentrates the Gaussian budget where the signal is brightest.

    Parameters
    ----------
    image : Tensor [H, W]
        Grayscale target in [0, 1].
    n_points : int
    device : torch.device
    init_scale_px : float
        Initial Gaussian σ.  Typically smaller than ``uniform_2d`` because
        Gaussians land in high-detail regions.
    init_opacity : float
    seed : int

    Returns
    -------
    GaussianParams with ``requires_grad = True``.
    """
    H, W = image.shape
    torch.manual_seed(seed)

    means_xy = _sample_positions(image.to(device), n_points, H, W, device)

    params = _build_params(means_xy, init_scale_px, init_opacity, image, device)
    params.require_grad_()
    return params


def multiresolution_residual_2d(
    image: torch.Tensor,
    n_points: int,
    device: torch.device,
    n_stages: int = 3,
    coarse_fraction: float = 0.35,
    eta: float = 0.30,
    init_scale_px: float = 2.0,
    init_opacity: float = 0.5,
    amplitude_iters: int = 150,
    amplitude_lr: float = 0.05,
    seed: int = 42,
    return_stages: bool = False,
) -> "GaussianParams | tuple[GaussianParams, list]":
    """Coarse-to-fine Gaussian initialisation via residual sampling.

    Algorithm
    ---------
    Stage 1
        Place ``coarse_fraction × n_points`` Gaussians on a coarse uniform
        grid with scale ``init_scale_px × 2^(n_stages - 1)``.
        Quick amplitude-only optimisation (~``amplitude_iters`` steps).
    Stages 2 … n_stages
        Compute residual  ``r = |target − render|``.
        Build sampling weight ``w = r + η ‖∇r‖``.
        Sample the next batch of Gaussians from ``w``.
        Scale is halved relative to the previous stage.
    Final render
        All stage Gaussians are merged.  Full gradients enabled on every
        parameter tensor, ready for subsequent full training.

    Parameters
    ----------
    image : Tensor [H, W]
        Grayscale target in [0, 1].
    n_points : int
        Total Gaussians across all stages.
    device : torch.device
    n_stages : int
        Number of coarse-to-fine stages (default 3).
    coarse_fraction : float
        Fraction of *n_points* allocated to stage 1 (default 0.35).
    eta : float
        Weight of the gradient term in the sampling map (default 0.30).
        Higher → more Gaussians on edges/texture; lower → pure residual.
    init_scale_px : float
        σ of the *finest* stage.  Coarser stages use
        ``init_scale_px × 2^(n_stages − stage − 1)``.
    init_opacity : float
        Initial opacity for all Gaussians.
    amplitude_iters : int
        Gradient-descent steps for the quick amplitude pass (per stage).
    amplitude_lr : float
        Adam learning-rate for the amplitude pass.
    seed : int
    return_stages : bool
        If ``True`` return ``(params, stage_info)`` where ``stage_info`` is a
        list of dicts with keys ``weight_map``, ``n_gaussians``, ``scale_px``
        for each stage after stage 1.

    Returns
    -------
    GaussianParams
        All Gaussians concatenated, ``requires_grad = True``.
    (GaussianParams, list)
        Only when ``return_stages=True``.
    """
    from gsplat_compress.renderer import render as _render
    from gsplat_compress.camera import ortho_camera

    H, W = image.shape
    torch.manual_seed(seed)
    img_dev = image.to(device)
    gt_rgb = img_dev.unsqueeze(-1).repeat(1, 1, 3)  # [H, W, 3]

    K, viewmat = ortho_camera(device)

    # ── budget allocation per stage ───────────────────────────────────────
    n_stage1 = max(4, int(round(coarse_fraction * n_points)))
    remaining = n_points - n_stage1
    # distribute remaining gaussians evenly across stages 2..n_stages
    extra_stages = n_stages - 1
    if extra_stages > 0:
        base = remaining // extra_stages
        extras = remaining - base * extra_stages
        stage_counts = [n_stage1] + [
            base + (1 if i < extras else 0) for i in range(extra_stages)
        ]
    else:
        stage_counts = [n_stage1]

    # ── scale schedule (coarsest first) ──────────────────────────────────
    # stage 1 → init_scale_px * 2^(n_stages-1)
    # stage k → init_scale_px * 2^(n_stages-k)
    scales = [init_scale_px * (2 ** max(0, n_stages - 1 - s)) for s in range(n_stages)]

    all_params: List[GaussianParams] = []
    stage_info: List[dict] = []

    for stage_idx in range(n_stages):
        n_stage = stage_counts[stage_idx]
        scale = scales[stage_idx]

        if stage_idx == 0:
            # Coarse uniform grid ─────────────────────────────────────────
            n_side = int(math.ceil(math.sqrt(n_stage)))
            means_xy = _uniform_positions(n_side * n_side, H, W, device)
            # take only n_stage (uniform grid is n_side² which may be ≥ n_stage)
            means_xy = means_xy[:n_stage]
        else:
            # Sample from residual weight map ─────────────────────────────
            # Render so far (merge all accumulated params)
            with torch.no_grad():
                merged = _merge(all_params, device)
                rendered, _, _ = _render(
                    merged.means, merged.log_scales, merged.quats,
                    merged.rgbs, merged.opacities, viewmat, K, W, H,
                )
                # residual: absolute error, channel-averaged
                residual = (gt_rgb - rendered).abs().mean(dim=-1).clamp(0.0, 1.0)  # [H, W]

            # sampling weight = |r| + eta * |∇r|
            grad_mag = _image_gradient_mag(residual.detach())
            weight = residual + eta * grad_mag
            weight = weight / (weight.max() + 1e-8)

            means_xy = _sample_positions(weight, n_stage, H, W, device)

            stage_info.append({
                "weight_map": weight.cpu(),
                "residual_map": residual.detach().cpu(),
                "n_gaussians": n_stage,
                "scale_px": scale,
                "stage": stage_idx + 1,
            })

        # Build params for this stage ─────────────────────────────────────
        p = _build_params(means_xy, scale, init_opacity, img_dev, device)
        all_params.append(p)

        # Quick amplitude optimisation ────────────────────────────────────
        if amplitude_iters > 0:
            # Merge existing + current stage for rendering context
            merged_so_far = _merge(all_params, device)
            _quick_amplitude_opt(
                merged_so_far, gt_rgb, viewmat, K, W, H,
                amplitude_iters, amplitude_lr,
            )
            # Write back only the current stage's updated rgbs / opacities
            n_prev = sum(q.n_gaussians for q in all_params[:-1])
            with torch.no_grad():
                p.rgbs.copy_(merged_so_far.rgbs[n_prev:])
                p.opacities.copy_(merged_so_far.opacities[n_prev:])

    # ── merge all stages ──────────────────────────────────────────────────
    final = _merge(all_params, device)
    final.require_grad_()

    if return_stages:
        return final, stage_info
    return final


# ── internal ──────────────────────────────────────────────────────────────────

def _merge(params_list: List[GaussianParams], device: torch.device) -> GaussianParams:
    """Concatenate a list of GaussianParams along the Gaussian axis."""
    return GaussianParams(
        means=torch.cat([p.means.detach() for p in params_list], dim=0).to(device),
        log_scales=torch.cat([p.log_scales.detach() for p in params_list], dim=0).to(device),
        quats=torch.cat([p.quats.detach() for p in params_list], dim=0).to(device),
        rgbs=torch.cat([p.rgbs.detach() for p in params_list], dim=0).to(device),
        opacities=torch.cat([p.opacities.detach() for p in params_list], dim=0).to(device),
    )
