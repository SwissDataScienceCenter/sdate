"""Cross-entropy loss over 12-bit quantized pixel classes.

Cross-entropy in nats is the training loss; converted to bits it is exactly
the bits-per-pixel an ideal arithmetic coder would achieve against the
model's predicted distribution -- the metric the whole project cares about
(see README.md "Metric"). Track ``nats_to_bits(loss)`` alongside raw loss
during training, not just the loss itself.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """logits: (B, n_classes, H, W). target: (B, H, W) int64 in [0, n_classes)."""
    return F.cross_entropy(logits, target, reduction="mean")


def nats_to_bits(loss_nats: torch.Tensor) -> torch.Tensor:
    return loss_nats / math.log(2)
