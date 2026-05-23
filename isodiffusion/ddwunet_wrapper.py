"""DeepDeWedge 3-D U-Net adapted for the isodiffusion / isonet pipeline.

Architecture source:
  https://github.com/MLI-lab/DeepDeWedge/blob/master/ddw/utils/unet.py

Only the core network classes are retained (LitUnet3D and training utilities
from the original file are omitted).  A thin DDWUNetIsoNet wrapper adds the
UNet3DConditionModel-compatible forward signature so it drops in without
modifying loss functions or inference pipelines.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Core architecture (from DeepDeWedge, MIT licence)
# ---------------------------------------------------------------------------

class _DownConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_chans),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv3d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_chans),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv3d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_chans),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _UpConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_chans, in_chans // 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_chans // 2),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv3d(in_chans // 2, in_chans // 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_chans // 2),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv3d(in_chans // 2, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_chans),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _SpatialDownSampling(nn.Module):
    def __init__(self, chans: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(chans, chans, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _SpatialUpSampling(nn.Module):
    def __init__(self, in_chans: int, out_chans: int) -> None:
        super().__init__()
        self.tconv = nn.ConvTranspose3d(
            in_chans, out_chans,
            kernel_size=3, stride=2, padding=1, output_padding=1,
        )
        self.activation = nn.LeakyReLU(negative_slope=0.05, inplace=True)

    def forward(self, x: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        out = self.tconv(x)
        out = torch.cat([out, cat], dim=1)
        return self.activation(out)


class DDWUnet3D(nn.Module):
    """3-D U-Net from DeepDeWedge with built-in z-score normalization.

    Parameters
    ----------
    chans:
        Base feature width at the first encoder level; doubles at each level.
    num_downsample_layers:
        Encoder depth (default 3 gives 4 resolution levels total).
    drop_prob:
        Spatial dropout probability applied in every conv block.
    residual:
        Add a skip connection from the (normalized) input to the output before
        denormalization.
    normalization_loc, normalization_scale:
        Z-score statistics applied inside the forward pass.  Set these from
        training-data statistics before the first epoch; the values are stored
        as non-trainable parameters and therefore persist in the checkpoint
        state_dict so inference reconstruction is exact.
    """

    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 1,
        chans: int = 32,
        num_downsample_layers: int = 3,
        drop_prob: float = 0.0,
        residual: bool = True,
        normalization_loc: float = 0.0,
        normalization_scale: float = 1.0,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_downsample_layers = num_downsample_layers
        self.drop_prob = drop_prob
        self.residual = residual
        # Stored as non-trainable parameters so they end up in state_dict.
        self._norm_loc = nn.Parameter(
            torch.tensor(float(normalization_loc)), requires_grad=False
        )
        self._norm_scale = nn.Parameter(
            torch.tensor(float(normalization_scale)), requires_grad=False
        )
        self._build_layers()

    @property
    def normalization_loc(self) -> float:
        return self._norm_loc.item()

    @normalization_loc.setter
    def normalization_loc(self, value: float) -> None:
        self._norm_loc.data.fill_(float(value))

    @property
    def normalization_scale(self) -> float:
        return self._norm_scale.item()

    @normalization_scale.setter
    def normalization_scale(self, value: float) -> None:
        self._norm_scale.data.fill_(float(value))

    def _build_layers(self) -> None:
        self.down_blocks = nn.ModuleList(
            [_DownConvBlock(self.in_chans, self.chans, self.drop_prob)]
        )
        self.down_samplers = nn.ModuleList([_SpatialDownSampling(self.chans)])

        ch = self.chans
        for _ in range(self.num_downsample_layers - 1):
            self.down_blocks.append(_DownConvBlock(ch, ch * 2, self.drop_prob))
            self.down_samplers.append(_SpatialDownSampling(ch * 2))
            ch *= 2

        self.bottleneck = nn.Sequential(
            nn.Conv3d(ch, ch * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv3d(ch * 2, ch, kernel_size=3, padding=1),
        )

        # First upsampler: ch→ch (output concatenated with skip → 2*ch for up_block[0])
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList([_SpatialUpSampling(ch, ch)])
        for _ in range(self.num_downsample_layers - 1):
            self.up_blocks.append(_UpConvBlock(2 * ch, ch, self.drop_prob))
            self.upsamplers.append(_SpatialUpSampling(ch, ch // 2))
            ch //= 2
        self.up_blocks.append(_UpConvBlock(2 * ch, ch, self.drop_prob))

        self.final_conv = nn.Conv3d(ch, self.out_chans, kernel_size=1)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._norm_loc) / (self._norm_scale + 1e-6)

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self._norm_scale + 1e-6) + self._norm_loc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize(x)

        stack: list[torch.Tensor] = []
        out = x
        for block, downsampler in zip(self.down_blocks, self.down_samplers):
            out = block(out)
            stack.append(out)
            out = downsampler(out)

        out = self.bottleneck(out)

        for upsampler, block in zip(self.upsamplers, self.up_blocks):
            out = upsampler(out, stack.pop())
            out = block(out)

        out = self.final_conv(out)
        if self.residual:
            out = out + x

        return self._denormalize(out)


# ---------------------------------------------------------------------------
# Pipeline-compatible wrapper
# ---------------------------------------------------------------------------

class DDWUNetIsoNet(nn.Module):
    """Isonet wrapper for DDWUnet3D with UNet3DConditionModel-compatible API.

    ``timestep`` and ``encoder_hidden_states`` are accepted for drop-in
    compatibility with IsoNetLoss and inference pipelines but are not used.
    """

    def __init__(
        self,
        chans: int = 32,
        normalization_loc: float = 0.0,
        normalization_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.unet = DDWUnet3D(
            in_chans=1,
            out_chans=1,
            chans=chans,
            normalization_loc=normalization_loc,
            normalization_scale=normalization_scale,
        )
        # Minimal config so IsoNetLoss can read cross_attention_dim without branching.
        self.config = SimpleNamespace(cross_attention_dim=1)

    def forward(
        self,
        x: torch.Tensor,
        timestep=None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        out = self.unet(x)
        return {"sample": out} if return_dict else (out,)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def compute_ddwunet_normalization_stats(
    dataset,
    num_samples: int = 200,
    seed: int = 0,
) -> tuple[float, float]:
    """Compute voxel mean and std over a random subset of *dataset* samples.

    Returns (mean, std) as Python floats suitable for setting
    ``DDWUnet3D.normalization_loc`` and ``DDWUnet3D.normalization_scale``.
    """
    n = min(num_samples, len(dataset))
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=rng)[:n].tolist()

    vsum = 0.0
    vsumsq = 0.0
    count = 0
    for i in indices:
        v = dataset[i].float()
        vsum += v.sum().item()
        vsumsq += (v * v).sum().item()
        count += v.numel()

    mean = vsum / count
    variance = max(vsumsq / count - mean * mean, 1e-10)
    std = math.sqrt(variance)
    return mean, std
