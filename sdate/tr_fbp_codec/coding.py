"""Arithmetic coding of the model's predicted per-pixel distribution via torchac.

Requires ``torchac`` + ``ninja`` (JIT-compiles a small C++ extension on first
import -- if that fails with "Ninja is required", make sure ``ninja`` is on
PATH, not just importable: ``PATH=$(python -c 'import sysconfig;
print(sysconfig.get_path("scripts"))'):$PATH``).

Because context (raw preceding frames + FBP-prior tap) comes entirely from
OTHER, already-known frames -- never from other pixels within the same
target frame -- the whole frame's per-pixel distributions can be computed
in a single forward pass and then coded/decoded in one arithmetic-coder
pass over all pixels at once. This is NOT a pixel-autoregressive model
(PixelCNN-style): there is no sequential per-pixel decode dependency within
a frame, which is what makes this fast to decode.
"""

from __future__ import annotations

import torch


def logits_to_cdf(logits: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) logits -> (B, H, W, C+1) float32 CDF for torchac.

    torchac wants the class dimension last, with C+1 cumulative values per
    pixel (CDF at each class boundary, 0 at the low end, 1 at the high end).
    """
    probs = torch.softmax(logits.float(), dim=1)  # (B, C, H, W)
    cdf = torch.cumsum(probs, dim=1)  # (B, C, H, W), cdf[...,-1] ~= 1
    cdf = torch.nn.functional.pad(cdf, (0, 0, 0, 0, 1, 0))  # prepend 0 -> (B, C+1, H, W)
    cdf = cdf.clamp(0.0, 1.0)
    return cdf.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C+1)


def encode(logits: torch.Tensor, target_q: torch.Tensor) -> bytes:
    """logits: (1, C, H, W). target_q: (1, H, W) int64 in [0, C). Batch size must be 1
    -- torchac's byte-string output is per-call, not separable per batch element."""
    import torchac

    if logits.shape[0] != 1:
        raise ValueError("encode() takes one frame at a time (batch size 1)")
    cdf = logits_to_cdf(logits).cpu()
    sym = target_q.to(torch.int16).cpu()
    return torchac.encode_float_cdf(cdf, sym, needs_normalization=True, check_input_bounds=False)


def decode(logits: torch.Tensor, byte_stream: bytes) -> torch.Tensor:
    """Inverse of :func:`encode`. logits must be EXACTLY what the encoder used
    (same model, same inputs) -- the decoder has no access to the true
    target, only to the context/prior it's conditioned on, and recomputes
    the identical per-pixel distribution from that before decoding symbols."""
    import torchac

    cdf = logits_to_cdf(logits).cpu()
    return torchac.decode_float_cdf(cdf, byte_stream, needs_normalization=True)


def bits_per_pixel(byte_stream: bytes, n_pixels: int) -> float:
    return len(byte_stream) * 8 / n_pixels
