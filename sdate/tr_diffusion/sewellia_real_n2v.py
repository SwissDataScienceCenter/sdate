"""N2V+context training pieces for the REAL full Sewellia lineolata dataset
(genuine raw counts + real darks/flats), superseding
:mod:`sdate.tr_diffusion.sewellia_n2v` (built for the small precorrected
preview file, which had no real counts and used a plain MSE loss).

Context = the T=5 phi-gated joint-FBP-groups taps (see
:mod:`sdate.tr_diffusion.phi_context`'s ``reconstruct_phi_gated_groups``,
built by ``scripts/sewellia_real_phi_context_v2.py``), fed as ``aux_channel``.
These are FULL native resolution (580x576), matching the central frame
exactly -- no upsampling needed (unlike the abandoned v1 det_bin=4 cache).

Real per-pixel darks/flats are available, so training uses the project's
standard Poisson-based loss (``BaselineN2VLoss(poisson_head=True,
gaussian_floor=True, sigma_read2=<dark variance>)``) rather than the preview
file's plain MSE -- see project memory project-tr-diffusion-t5native-gaussianfloor
for why nb_nll_gaussian (not the exact NB-NLL) is used at this native-count
regime.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import h5py
import numpy as np
import torch


class SewelliaRealN2VDataset(torch.utils.data.Dataset):
    """Central = real raw counts (preloaded fully into RAM, ~12.4GB, uint16 ->
    read on the fly as float32 per-sample to keep RAM low). Aux = T=5
    full-resolution phi-context-groups taps, left as memmaps (NOT force-copied
    into RAM like the tiny preview dataset did -- at ~13.4GB/tap x 5 this would
    pin ~67GB of anonymous memory; leaving them as memmaps lets the OS page
    cache serve repeated epochs from file-backed pages, which it can evict
    under pressure unlike a plain numpy copy).

    No h5py file handle is kept open past ``__init__`` (central is fully
    preloaded), and memmaps are safe to share across forked DataLoader worker
    processes, so ``num_workers > 0`` is safe here (unlike a live h5py handle).
    """

    def __init__(self, h5_path: str, tap_paths: Sequence[str], calib_path: str,
                norm_percentiles=(0.1, 99.9)):
        with h5py.File(h5_path, "r") as f:
            self.sino = f["exchange/data"][:]  # (N,H,W) uint16, ~12.4GB
        self.n, self.h, self.w = self.sino.shape

        self.aux_mms = []
        for p in tap_paths:
            meta = np.load(str(p) + ".meta.npz")
            n_proj = int(meta["n_proj"])
            if n_proj != self.n:
                raise ValueError(f"tap {p!r} covers n_proj={n_proj}, expected {self.n}")
            self.aux_mms.append(np.memmap(p, dtype=np.float16, mode="r", shape=(self.n, self.h, self.w)))
        self.T = len(tap_paths)

        z = np.load(calib_path)
        self.dark_mean = z["dark_mean"].astype(np.float32)   # (H,W)
        self.white_mean = z["white_mean"].astype(np.float32)  # (H,W)
        self.dark_var = z["dark_var"].astype(np.float32)      # (H,W) -- read-noise floor for gaussian_floor loss

        sample_idx = np.linspace(0, self.n - 1, 2000, dtype=int)
        lo, hi = np.percentile(self.sino[sample_idx].astype(np.float32), norm_percentiles)
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
        central = self.normalize(torch.from_numpy(self.sino[idx][None].astype(np.float32)))  # (1,H,W)
        aux_raw = np.stack([mm[idx] for mm in self.aux_mms]).astype(np.float32)  # (T,H,W), counts-domain
        aux = self.normalize(torch.from_numpy(aux_raw))
        context = torch.zeros((0, self.h, self.w), dtype=torch.float32)  # k=0: no rotation/time context
        return {"central": central, "context": context, "aux_channel": aux}
