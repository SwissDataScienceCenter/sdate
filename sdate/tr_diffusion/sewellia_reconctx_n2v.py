"""Dataset for the reconstruction-domain N2V+context track (see project memory
project-sewellia-recon-space-n2v.md) -- denoises reconstructed SLICES instead
of projections.

v2 design (Noise2Noise anchor pair, supersedes the v1 blind-spot-N2V design --
see the postmortem in scripts/sewellia_recon_context_build.py's docstring):
``a1``/``a2`` are two INDEPENDENT reconstructions of the same phi bin from
disjoint ~200-view halves of a ~400-view gate -- genuinely independent noise,
same underlying structure. ``a1`` is the model's input, ``a2`` is the
regression target (plain full-frame MSE, no blind-spot masking needed).
Context = T=3 phi-gated joint-FBP levels, built with the anchor pair's ~400
views EXCLUDED from every context gate (see ``exclude_idx`` in
``phi_context.reconstruct_phi_gated_groups``) so context and anchor share no
photon draws.

Both are cached to disk by ``scripts/sewellia_recon_context_build.py``,
indexed by (phi_bin, row); training samples are simply every (bin, z) pair.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class SewelliaReconContextDataset(torch.utils.data.Dataset):
    def __init__(self, out_dir: str, tag: str = "sewellia_reconctx", norm_sample_size: int = 4000):
        out_dir = Path(out_dir)
        meta = np.load(out_dir / f"{tag}_ctx0.f16.meta.npz", allow_pickle=True)
        self.n_bins = int(meta["n_bins"])
        self.crop_rows = int(meta["crop_rows"])
        self.n_pix = int(meta["n_pix"])
        self.T = int(meta["T"])
        shape = (self.n_bins, self.crop_rows, self.n_pix, self.n_pix)

        done_bins_path = out_dir / f"{tag}_done_bins.json"
        with open(done_bins_path) as f:
            self.usable_bins = np.array(sorted(json.load(f)), dtype=np.int64)
        if len(self.usable_bins) == 0:
            raise RuntimeError(f"no completed bins in {done_bins_path} -- has the build finished any bins?")

        self.a1_mm = np.memmap(out_dir / f"{tag}_a1.f16", dtype=np.float16, mode="r", shape=shape)
        self.a2_mm = np.memmap(out_dir / f"{tag}_a2.f16", dtype=np.float16, mode="r", shape=shape)
        self.ctx_mms = [np.memmap(out_dir / f"{tag}_ctx{c}.f16", dtype=np.float16, mode="r", shape=shape)
                        for c in range(self.T)]

        rng = np.random.default_rng(0)
        sample_bins = rng.choice(self.usable_bins, size=min(norm_sample_size // self.crop_rows + 1,
                                                            len(self.usable_bins)), replace=False)
        sample_rows = rng.integers(0, self.crop_rows, size=len(sample_bins))
        samples = np.concatenate([
            self.a1_mm[sample_bins, sample_rows].astype(np.float32).ravel(),
            self.a2_mm[sample_bins, sample_rows].astype(np.float32).ravel(),
        ] + [mm[sample_bins, sample_rows].astype(np.float32).ravel() for mm in self.ctx_mms])
        lo, hi = np.percentile(samples, [0.1, 99.9])
        if hi - lo < 1e-8:
            hi = lo + 1.0
        self.norm_min, self.norm_max = float(lo), float(hi)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * (x - self.norm_min) / (self.norm_max - self.norm_min) - 1.0

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 1.0) * 0.5 * (self.norm_max - self.norm_min) + self.norm_min

    def __len__(self) -> int:
        return len(self.usable_bins) * self.crop_rows

    def __getitem__(self, idx: int):
        bin_pos, z = divmod(idx, self.crop_rows)
        b = int(self.usable_bins[bin_pos])
        a1 = self.normalize(torch.from_numpy(self.a1_mm[b, z][None].astype(np.float32)))  # (1,H,W)
        a2 = self.normalize(torch.from_numpy(self.a2_mm[b, z][None].astype(np.float32)))  # (1,H,W)
        ctx = np.stack([mm[b, z] for mm in self.ctx_mms]).astype(np.float32)  # (T,H,W)
        aux = self.normalize(torch.from_numpy(ctx))
        return {"input": a1, "target": a2, "aux_channel": aux}


class SewelliaReconContextKSplitDataset(torch.utils.data.Dataset):
    """v3: Noise2Inverse-style K-way anchor split (see
    scripts/sewellia_recon_context_build_ksplit.py's docstring for the full
    rationale). Instead of a FIXED (a1, a2) pair reused identically every
    epoch (v2's flaw -- the network could partially memorize a bin-specific
    correction that reproduces a1/a2's own shared artifact fingerprint),
    this loads K disjoint sub-anchor reconstructions per bin and, on EVERY
    `__getitem__` call, randomly redraws which one is this step's regression
    TARGET and averages the other K-1 as the (lower-noise) INPUT ("X:1"
    strategy). Since the same (bin, row) sample gets a different random
    combination each time it's drawn, no single input/target pairing is
    ever memorized -- only the shared underlying structure is a consistent
    prediction target across draws.

    Context (`ctx0..ctx{T-1}`) is untouched from v2 -- the K splits share the
    exact same anchor pool/gate as v2's a1/a2, so `exclude_idx` (and hence
    the context caches) is unchanged; ctx files are read directly here.
    """

    def __init__(self, out_dir: str, tag: str = "sewellia_reconctx", k_splits: int = 4,
                norm_sample_size: int = 4000):
        out_dir = Path(out_dir)
        meta = np.load(out_dir / f"{tag}_ctx0.f16.meta.npz", allow_pickle=True)
        self.n_bins = int(meta["n_bins"])
        self.crop_rows = int(meta["crop_rows"])
        self.n_pix = int(meta["n_pix"])
        self.T = int(meta["T"])
        self.K = k_splits
        shape = (self.n_bins, self.crop_rows, self.n_pix, self.n_pix)

        ksplit_done_path = out_dir / f"{tag}_ksplit_done_bins.json"
        with open(ksplit_done_path) as f:
            self.usable_bins = np.array(sorted(json.load(f)), dtype=np.int64)
        if len(self.usable_bins) == 0:
            raise RuntimeError(f"no completed bins in {ksplit_done_path} -- has the ksplit build finished any bins?")

        self.asub_mms = [np.memmap(out_dir / f"{tag}_asub{k}.f16", dtype=np.float16, mode="r", shape=shape)
                         for k in range(self.K)]
        self.ctx_mms = [np.memmap(out_dir / f"{tag}_ctx{c}.f16", dtype=np.float16, mode="r", shape=shape)
                        for c in range(self.T)]

        rng = np.random.default_rng(0)
        sample_bins = rng.choice(self.usable_bins, size=min(norm_sample_size // self.crop_rows + 1,
                                                            len(self.usable_bins)), replace=False)
        sample_rows = rng.integers(0, self.crop_rows, size=len(sample_bins))
        samples = np.concatenate(
            [mm[sample_bins, sample_rows].astype(np.float32).ravel() for mm in self.asub_mms] +
            [mm[sample_bins, sample_rows].astype(np.float32).ravel() for mm in self.ctx_mms])
        lo, hi = np.percentile(samples, [0.1, 99.9])
        if hi - lo < 1e-8:
            hi = lo + 1.0
        self.norm_min, self.norm_max = float(lo), float(hi)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * (x - self.norm_min) / (self.norm_max - self.norm_min) - 1.0

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 1.0) * 0.5 * (self.norm_max - self.norm_min) + self.norm_min

    def __len__(self) -> int:
        return len(self.usable_bins) * self.crop_rows

    def __getitem__(self, idx: int):
        bin_pos, z = divmod(idx, self.crop_rows)
        b = int(self.usable_bins[bin_pos])
        # re-drawn every call (torch's per-worker-seeded RNG, safe under DataLoader workers)
        j = int(torch.randint(0, self.K, (1,)).item())
        target_raw = self.asub_mms[j][b, z].astype(np.float32)
        input_raw = np.mean([self.asub_mms[k][b, z] for k in range(self.K) if k != j], axis=0).astype(np.float32)

        input_t = self.normalize(torch.from_numpy(input_raw)[None])
        target_t = self.normalize(torch.from_numpy(target_raw)[None])
        ctx = np.stack([mm[b, z] for mm in self.ctx_mms]).astype(np.float32)
        aux = self.normalize(torch.from_numpy(ctx))
        return {"input": input_t, "target": target_t, "aux_channel": aux}
