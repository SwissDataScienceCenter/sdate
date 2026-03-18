"""
Differentiable 2D image renderer using gsplat's rasterization pipeline.

Wraps ``gsplat.rasterization()`` with activation functions and provides a
clean interface for orthographic rendering.
"""

from __future__ import annotations

import torch
from gsplat import rasterization


def render(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    quats: torch.Tensor,
    rgbs: torch.Tensor,
    opacities: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    W: int,
    H: int,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Render image via gsplat's full rasterization pipeline (orthographic).

    Parameters
    ----------
    means : Tensor [N, 3]
        Gaussian centres (x, y, z) in pixel coordinates.
    log_scales : Tensor [N, 3]
        Log-space scale factors.
    quats : Tensor [N, 4]
        Quaternion rotations (w, x, y, z).
    rgbs : Tensor [N, 3]
        Sigmoid-space colour intensities.
    opacities : Tensor [N]
        Sigmoid-space opacities.
    viewmat : Tensor [4, 4]
        View matrix.
    K : Tensor [3, 3]
        Intrinsic matrix.
    W, H : int
        Output image width and height.
    near_plane, far_plane : float
        Clipping planes.

    Returns
    -------
    rendered : Tensor [H, W, 3]
        Rendered RGB image.
    alphas : Tensor [H, W, 1]
        Alpha channel.
    info : dict
        Additional rasterization info from gsplat.
    """
    scales = torch.exp(log_scales)                        # [N, 3]
    colors = torch.sigmoid(rgbs)                          # [N, 3]
    opac = torch.sigmoid(opacities)                       # [N]
    q = quats / quats.norm(dim=-1, keepdim=True)          # [N, 4]

    renders, alphas, info = rasterization(
        means=means,
        quats=q,
        scales=scales,
        opacities=opac,
        colors=colors,
        viewmats=viewmat[None],
        Ks=K[None],
        width=W,
        height=H,
        near_plane=near_plane,
        far_plane=far_plane,
        packed=False,
        camera_model="ortho",
    )

    return renders[0], alphas[0], info
