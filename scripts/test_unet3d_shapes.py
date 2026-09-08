#!/usr/bin/env python3
"""Probe UNet3DConditionModel with various (D, H, W) shapes on CPU.

Tests which volume sizes survive a full forward pass with the same
architecture config used in train_isonet_3d.py.
"""

import traceback
import torch
from diffusers.models import UNet3DConditionModel

CHANNELS = (32, 64, 128)
CROSS_ATTENTION_DIM = 128
NORM_NUM_GROUPS = 16
LAYERS_PER_BLOCK = 1

DOWN_BLOCKS = ("DownBlock3D", "DownBlock3D", "CrossAttnDownBlock3D")
UP_BLOCKS = ("CrossAttnUpBlock3D", "UpBlock3D", "UpBlock3D")


def make_model(sample_size, attention_head_dim):
    return UNet3DConditionModel(
        sample_size=sample_size,
        in_channels=1,
        out_channels=1,
        down_block_types=DOWN_BLOCKS,
        up_block_types=UP_BLOCKS,
        block_out_channels=CHANNELS,
        layers_per_block=LAYERS_PER_BLOCK,
        cross_attention_dim=CROSS_ATTENTION_DIM,
        attention_head_dim=attention_head_dim,
        norm_num_groups=NORM_NUM_GROUPS,
    )


def try_forward(sample_size, attention_head_dim=8, batch_size=1):
    if isinstance(sample_size, int):
        volume_size = [sample_size, sample_size, sample_size]
        label = f"cubic {sample_size}"
    else:
        d, h, w = sample_size
        volume_size = list(sample_size)
        label = f"slab {d}x{h}x{w}"

    try:
        model = make_model(volume_size, attention_head_dim)
        model.eval()
        x = torch.randn(batch_size, 1, *volume_size)
        timesteps = torch.zeros(batch_size, dtype=torch.long)
        encoder_hidden_states = torch.zeros(batch_size, 1, CROSS_ATTENTION_DIM)
        with torch.no_grad():
            out = model(x, timestep=timesteps, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        print(f"  OK   [{label}]  head_dim={attention_head_dim}  out={tuple(out.shape)}")
        return True
    except Exception as e:
        short = str(e).split("\n")[0][:120]
        print(f"  FAIL [{label}]  head_dim={attention_head_dim}  -> {type(e).__name__}: {short}")
        # Uncomment for full traceback:
        # traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("Cubic volumes")
    print("=" * 70)
    for size in [16, 32, 48, 64, 80, 96, 112, 128]:
        try_forward(size, attention_head_dim=8)

    print()
    print("=" * 70)
    print("Non-cubic (D, 96, 96) — varying D")
    print("=" * 70)
    for d in [8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 80, 96]:
        try_forward((d, 96, 96), attention_head_dim=8)

    print()
    print("=" * 70)
    print("Non-cubic (D, 128, 128) — varying D")
    print("=" * 70)
    for d in [8, 16, 24, 32, 40, 48, 64, 96, 128]:
        try_forward((d, 128, 128), attention_head_dim=8)

    print()
    print("=" * 70)
    print("Non-cubic (D, 64, 64) — varying D")
    print("=" * 70)
    for d in [8, 16, 24, 32, 40, 48, 64]:
        try_forward((d, 64, 64), attention_head_dim=8)

    print()
    print("=" * 70)
    print("Varying attention_head_dim on a known-good shape (64,96,96)")
    print("=" * 70)
    for hdim in [2, 4, 8, 16, 32]:
        try_forward((64, 96, 96), attention_head_dim=hdim)
