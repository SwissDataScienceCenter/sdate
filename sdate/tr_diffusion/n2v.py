"""Noise2Void blind-spot corruption of the central frame.

The central frame is the diffusion target ``x_0`` but must never be fed to the
model uncorrupted, or the task is trivial (the model just copies it).  Following
Noise2Void (Krull et al., 2019) we replace a small random fraction of pixels
with a value sampled from a nearby neighbour ("uniform pixel selection"), and
train so that the loss is evaluated **only at those replaced (blind-spot)
pixels**.  Because the network cannot see the true value there, it can only
predict the underlying signal — i.e. it denoises — while the correlated but
independently-noisy rotation / temporal neighbours supply the signal
(Noise2Noise).

At inference the corruption is resampled on every diffusion step so no single
blind-spot pattern biases the result.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def blind_spot_corrupt(
    x: torch.Tensor,
    ratio: float = 0.02,
    window: int = 5,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(corrupted, mask)`` for a batch of frames.

    Parameters
    ----------
    x:
        ``(B, 1, H, W)`` frames.
    ratio:
        Fraction of pixels to turn into blind spots (~1-5 % is typical).
    window:
        Side length of the odd square neighbourhood a replacement value is drawn
        from (the centre pixel itself is excluded).
    generator:
        Optional RNG for reproducible masks (used at inference for per-step
        resampling).

    ``corrupted`` equals ``x`` everywhere except at masked pixels, whose values
    are copied from a random neighbour within ``window``.  ``mask`` is a bool
    tensor (same shape) that is ``True`` at the blind-spot pixels.
    """
    if x.dim() != 4 or x.shape[1] != 1:
        raise ValueError(f"expected (B, 1, H, W), got {tuple(x.shape)}")
    if window % 2 == 0 or window < 3:
        raise ValueError("window must be an odd integer >= 3")
    if not 0.0 < ratio < 1.0:
        raise ValueError("ratio must be in (0, 1)")

    b, _, h, w = x.shape
    device = x.device
    r = window // 2

    mask = (torch.rand((b, 1, h, w), device=device, generator=generator) < ratio)

    # For every pixel draw an integer neighbour offset in [-r, r]^2, excluding
    # (0, 0); the replacement is gathered from the shifted (clamped) coordinate.
    dy = torch.randint(-r, r + 1, (b, 1, h, w), device=device, generator=generator)
    dx = torch.randint(-r, r + 1, (b, 1, h, w), device=device, generator=generator)
    zero = (dy == 0) & (dx == 0)
    dx = torch.where(zero, torch.ones_like(dx), dx)  # nudge the degenerate case

    yy = torch.arange(h, device=device).view(1, 1, h, 1)
    xx = torch.arange(w, device=device).view(1, 1, 1, w)
    src_y = (yy + dy).clamp_(0, h - 1)
    src_x = (xx + dx).clamp_(0, w - 1)
    flat = (src_y * w + src_x).reshape(b, -1)
    gathered = torch.gather(x.reshape(b, -1), 1, flat).reshape(b, 1, h, w)

    corrupted = torch.where(mask, gathered, x)
    return corrupted, mask
