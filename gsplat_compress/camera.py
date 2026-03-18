"""
Orthographic camera setup for 2D image fitting with gsplat.

For orthographic projection, rays are parallel to the z-axis.
The intrinsic matrix K maps camera-space coordinates directly to pixel
coordinates:  u = f_x * x_cam + c_x,   v = f_y * y_cam + c_y

With f_x = f_y = 1 and c_x = c_y = 0, Gaussian means in world space
are directly in pixel coordinates.  The view matrix is identity, placing
the camera at the origin looking down +z.
"""

from __future__ import annotations

import torch


def ortho_camera(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return orthographic intrinsics K (3x3) and identity view matrix (4x4).

    Parameters
    ----------
    device : torch.device
        Target device for the tensors.

    Returns
    -------
    K : Tensor [3, 3]
        Intrinsic matrix (identity for orthographic).
    viewmat : Tensor [4, 4]
        View matrix (identity — camera at origin looking +z).
    """
    K = torch.eye(3, device=device)
    viewmat = torch.eye(4, device=device)
    return K, viewmat
