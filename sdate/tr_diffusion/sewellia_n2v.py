"""N2V+context training pieces specific to the Sewellia lineolata (theta, phi)
dataset -- a plain (N, H, W) array with no `.mov`/rotation-window structure,
unlike every other dataset in this project.

Reuses the SAME core machinery as the rest of `tr_diffusion` --
:class:`sdate.tr_diffusion.losses.BaselineN2VLoss`,
:func:`sdate.tr_diffusion.model.create_baseline_unet`,
:func:`sdate.tr_diffusion.pipeline.denoise_frames_baseline` -- only the data
loading (:class:`SewelliaN2VDataset`, no `.mov`/`MemmapFrameSource`) and the
model's input-shape handling (:class:`PaddedUNet`, the detector is only 6
pixels tall, but the stock 6-stage UNet2DModel downsamples by 2**5=32) are new.

Context = the T phi-gated joint-FBP taps (see
:mod:`sdate.tr_diffusion.phi_context`), fed as `aux_channel` -- there is no
rotation/temporal context (`k=0`), since theta here plays the role phi plays
in phi_context, not the free/repeating axis the rest of the project assumes.
No known raw-count/darks-flats for this h5 preview file, so the loss is plain
Gaussian/L2 (`loss_type="mse"`, `poisson_head=False`) rather than the
Poisson-based losses used elsewhere -- see project memory
project-tr-diffusion-sewellia-phi-context.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SewelliaN2VDataset(torch.utils.data.Dataset):
    """Plain in-memory N2V+context dataset: no `.mov`, no rotation taps (k=0).

    ``tap_paths``: the phi_context tap memmaps (float16, shape (N,H,W) each,
    with a `<path>.meta.npz` sidecar giving `first_index`/`num_frames`) --
    must cover the FULL dataset (first_index=0, num_frames=N), matching how
    ``scripts/sewellia_phi_context_prototype.py`` writes them.
    """

    def __init__(self, h5_path: str, tap_paths: Sequence[str], norm_percentiles=(0.5, 99.5)):
        with h5py.File(h5_path, "r") as f:
            self.sino = f["sinogram"][:].astype(np.float32)  # (N, H, W)
        self.n, self.h, self.w = self.sino.shape

        aux = []
        for p in tap_paths:
            meta = np.load(str(p) + ".meta.npz")
            first, n_frames = int(meta["first_index"]), int(meta["num_frames"])
            if first != 0 or n_frames != self.n:
                raise ValueError(f"tap {p!r} covers [{first},{first+n_frames}), expected [0,{self.n})")
            arr = np.memmap(p, dtype=np.float16, mode="r", shape=(n_frames, self.h, self.w))
            aux.append(np.array(arr))  # force real copy into RAM (~130MB/tap here)
        self.aux = np.stack(aux, axis=1)  # (N, T, H, W)
        self.T = len(tap_paths)

        lo, hi = np.percentile(self.sino, norm_percentiles)
        if hi - lo < 1e-6:
            hi = lo + 1.0
        self.norm_min, self.norm_max = float(lo), float(hi)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * (x - self.norm_min) / (self.norm_max - self.norm_min) - 1.0

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 1.0) * 0.5 * (self.norm_max - self.norm_min) + self.norm_min

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        central = self.normalize(torch.from_numpy(self.sino[idx][None]).float())  # (1,H,W)
        aux = self.normalize(torch.from_numpy(self.aux[idx].astype(np.float32)))  # (T,H,W)
        context = torch.zeros((0, self.h, self.w), dtype=torch.float32)  # k=0: no rotation context
        return {"central": central, "context": context, "aux_channel": aux}


class PaddedUNet(nn.Module):
    """Wrap a UNet2DModel to accept an (H, W) not divisible by its downsample
    factor (here 2**5=32): replicate-pads up to the next multiple of ``pad_to``
    before the forward pass, crops back to (H, W) after. Transparent to
    callers using the standard ``model(x, timestep, class_labels, return_dict=False)[0]``
    contract (:class:`sdate.tr_diffusion.losses.BaselineN2VLoss`,
    :func:`sdate.tr_diffusion.pipeline.denoise_frames_baseline` both use it as-is).

    Replicate (not reflect) padding: reflect padding requires pad size <= dim
    size - 1, which H=6 violates (needs 13 per side); replicate has no such
    restriction and is a more physically reasonable border assumption than
    zero-padding for a detector strip with no real content beyond its edges.
    """

    def __init__(self, unet: nn.Module, orig_hw: Tuple[int, int], pad_to: int = 32):
        super().__init__()
        self.unet = unet
        h, w = orig_hw
        self.h, self.w = h, w
        ph = ((h + pad_to - 1) // pad_to) * pad_to
        pw = ((w + pad_to - 1) // pad_to) * pad_to
        self.top, self.bottom = (ph - h) // 2, ph - h - (ph - h) // 2
        self.left, self.right = (pw - w) // 2, pw - w - (pw - w) // 2
        self.padded_hw = (ph, pw)

    def forward(self, sample, timestep, class_labels=None, return_dict=True):
        x = F.pad(sample, (self.left, self.right, self.top, self.bottom), mode="replicate")
        out = self.unet(x, timestep, class_labels=class_labels, return_dict=False)[0]
        cropped = out[..., self.top:self.top + self.h, self.left:self.left + self.w]
        return (cropped,)
