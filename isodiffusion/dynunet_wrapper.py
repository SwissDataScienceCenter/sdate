"""DynUNet wrappers providing a UNet3DConditionModel-compatible forward API.

Two wrappers are provided:
- DynUNetIsoNet: direct regression (timestep and encoder_hidden_states ignored).
- DynUNetDiffusion: noise prediction with timestep injected via sinusoidal
  embedding + MLP, added as a spatial bias before the first DynUNet conv.

Both implement the same call signature as diffusers' UNet3DConditionModel so
they drop in without changing loss functions or inference pipelines.

Architecture is derived entirely from ``filters``:
  - n = len(filters) levels total (encoder + bottleneck)
  - kernel_size: [[3,3,3]] * n
  - strides:     [[1,1,1]] + [[2,2,2]] * (n - 1)  -- first level no downsample
  - upsample_kernel_size: [[2,2,2]] * (n - 1)

For 96^3 volumes, filters=[32,64,128,256,320] (5 levels) gives a 6^3 bottleneck.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import List, Optional, Union

import torch
import torch.nn as nn


def parse_dynunet_filters(value) -> List[int]:
    """Accept a comma-separated string or an iterable of ints."""
    if isinstance(value, str):
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    return [int(v) for v in value]


def build_dynunet(in_channels: int, out_channels: int, filters: List[int]) -> nn.Module:
    """Instantiate a MONAI DynUNet for 3-D volumes."""
    from monai.networks.nets import DynUNet

    n = len(filters)
    if n < 2:
        raise ValueError("DynUNet requires at least 2 levels (len(filters) >= 2)")

    kernel_sizes = [[3, 3, 3]] * n
    strides = [[1, 1, 1]] + [[2, 2, 2]] * (n - 1)
    upsample_kernel_sizes = [[2, 2, 2]] * (n - 1)

    return DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_sizes,
        strides=strides,
        upsample_kernel_size=upsample_kernel_sizes,
        filters=filters,
        deep_supervision=False,
    )


class _SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)


class DynUNetIsoNet(nn.Module):
    """DynUNet wrapper for direct regression.

    ``timestep`` and ``encoder_hidden_states`` are accepted for API compatibility
    but ignored — no noise schedule is used.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filters: List[int],
        cross_attention_dim: int = 32,
    ) -> None:
        super().__init__()
        self.dynunet = build_dynunet(in_channels, out_channels, filters)
        # Dummy config so inference code can read model.config.cross_attention_dim
        self.config = SimpleNamespace(cross_attention_dim=cross_attention_dim)

    def forward(
        self,
        x: torch.Tensor,
        timestep=None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        out = self.dynunet(x)
        return {"sample": out} if return_dict else (out,)


class DynUNetDiffusion(nn.Module):
    """DynUNet wrapper for noise prediction.

    Timestep conditioning is injected by projecting a sinusoidal embedding to
    ``in_channels`` and adding it as a spatial bias to the input before the
    first DynUNet convolution.  ``encoder_hidden_states`` is accepted for API
    compatibility but ignored — conditioning is carried via input channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filters: List[int],
        time_embed_dim: int = 256,
        cross_attention_dim: int = 32,
    ) -> None:
        super().__init__()
        self.dynunet = build_dynunet(in_channels, out_channels, filters)
        self.time_embedding = nn.Sequential(
            _SinusoidalEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, in_channels),
        )
        self.config = SimpleNamespace(cross_attention_dim=cross_attention_dim)

    def _coerce_timestep(
        self, timestep: Union[torch.Tensor, int], batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if isinstance(timestep, torch.Tensor):
            t = timestep.to(device)
            return t.expand(batch_size).long() if t.dim() == 0 else t.long()
        return torch.full((batch_size,), int(timestep), device=device, dtype=torch.long)

    def forward(
        self,
        x: torch.Tensor,
        timestep: Union[torch.Tensor, int] = 0,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        t = self._coerce_timestep(timestep, x.shape[0], x.device)
        t_bias = self.time_embedding(t).view(x.shape[0], -1, 1, 1, 1)
        out = self.dynunet(x + t_bias)
        return {"sample": out} if return_dict else (out,)
