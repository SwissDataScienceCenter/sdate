"""Annealed-N2N: iterative renoising/annealing Noise2Noise denoiser.

See ``sdate/tr_diffusion/CONTEXT.md`` for the glossary (round, iterate,
anneal weight, round denoiser, ...) and ``docs/adr/0001``/``0002`` for the two
deliberate deviations from the obvious path (binomial-split y1/y2 instead of
independent draws; fixed per-round epoch schedule instead of loss-curve-based
early stopping).

One round: (1) train the round denoiser ``D`` (plain MSE Noise2Noise, in
Anscombe/Gaussian z-space) on the current iterate `x_hat_k`, targeting `y1`;
(2) run ``D`` over the full dataset to get `D(x_hat_k)`; (3) renoise:
``x_hat_{k+1} = alpha_k * y2 + (1 - alpha_k) * D(x_hat_k)``. `y1`/`y2` are the
two halves of a fixed (same seed every round) binomial split of one Poisson
draw at combined dose -- see ADR-0001 -- so they never change; only the
`x_hat` iterate evolves, in a mutable float16 memmap indexed 1:1 with the
underlying ``TimeResolvedFrameDataset``'s ``.indices``.

The renoising update is done directly in NORMALISED z-space (the
``[-1, 1]``-affine-mapped Anscombe value the model actually sees), not raw
z-units: affine normalisation commutes with a convex combination
(``normalize(a*z1+(1-a)*z2) == a*normalize(z1)+(1-a)*normalize(z2)``), so
this is exact, and it avoids an extra denormalize/renormalize round trip
inside the hot loop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import TimeResolvedFrameDataset
from .noise import anscombe_transform_thinned, inverse_anscombe


# --------------------------------------------------------------------------- #
# anneal-weight schedules
# --------------------------------------------------------------------------- #

def alpha_schedule(num_rounds: int, kind: str = "cosine") -> np.ndarray:
    """``alpha_k`` for ``k=0..num_rounds-1``: ``alpha_0=1`` (x_hat_0 = y2 exactly),
    ``alpha_{num_rounds-1}=0`` (the last round's renoise step, if it ran, would
    return ``D(x_hat_k)`` unchanged -- which is exactly why the pipeline stops
    there and reports that as the final estimate instead of taking one more
    no-op renoise step)."""
    if num_rounds < 2:
        raise ValueError("num_rounds must be >= 2")
    k = np.arange(num_rounds)
    if kind == "cosine":
        return 0.5 * (1.0 + np.cos(np.pi * k / (num_rounds - 1)))
    if kind == "linear":
        return 1.0 - k / (num_rounds - 1)
    raise ValueError(f"kind must be 'cosine' or 'linear', got {kind!r}")


# --------------------------------------------------------------------------- #
# Anscombe z-space normalisation (this pipeline's own fit -- distinct from
# TimeResolvedFrameDataset's `anscombe=True` mode, which is wired only for
# --mode ambient_tweedie and doesn't know about the binomial-split y1/y2 here)
# --------------------------------------------------------------------------- #

def fit_z_norm(base_ds: TimeResolvedFrameDataset, split_dose: float,
               n_sample: int = 64, percentiles: Tuple[float, float] = (0.5, 99.5),
               seed: int = 0) -> Tuple[float, float]:
    """Percentile range of ``Anscombe(y1)`` over a sample of frames -- the
    model's input/output range. Fit on the target ``y1`` (not the noisier,
    round-0 ``y2`` anchor) since that's the distribution ``D`` is trained to
    land on."""
    rng = np.random.default_rng(seed + 5)
    n = len(base_ds)
    picks = rng.choice(n, size=min(n_sample, n), replace=False)
    vals = []
    for i in picks:
        item = base_ds[int(i)]
        target_raw = base_ds.denormalize(item["central_target"])
        z = anscombe_transform_thinned(target_raw, split_dose)
        vals.append(z.numpy().ravel())
    lo, hi = np.percentile(np.concatenate(vals), percentiles)
    if hi - lo < 1e-6:
        hi = lo + 1.0
    return float(lo), float(hi)


def normalize_z(z: torch.Tensor, z_min: float, z_max: float) -> torch.Tensor:
    return 2.0 * (z - z_min) / (z_max - z_min) - 1.0


def denormalize_z(z_norm: torch.Tensor, z_min: float, z_max: float) -> torch.Tensor:
    return (z_norm + 1.0) * 0.5 * (z_max - z_min) + z_min


def z_norm_to_counts(z_norm: torch.Tensor, z_min: float, z_max: float, dose: float) -> torch.Tensor:
    """Invert normalised z-space back to native-scale raw counts (same scale as
    ``reference``/``TimeResolvedFrameDataset.denormalize``): undo the affine
    z-normalisation, undo the Anscombe transform (recovering the actual
    photon draw at this thinned dose), then rescale by ``dose`` -- the same
    rescaling :func:`sdate.tr_diffusion.noise.add_poisson_noise`/
    ``binomial_split`` apply to keep the mean at native-count scale."""
    raw_z = denormalize_z(z_norm, z_min, z_max)
    thinned_draw = inverse_anscombe(raw_z)
    return thinned_draw / dose


# --------------------------------------------------------------------------- #
# dataset wrapper: fixed y1/y2/context (from a seeded binomial-split N2N
# TimeResolvedFrameDataset) + the mutable x_hat iterate (a float16 memmap,
# NORMALISED z-space, indexed 1:1 with base_ds.indices)
# --------------------------------------------------------------------------- #

class AnnealDataset(Dataset):
    def __init__(self, base_ds: TimeResolvedFrameDataset, x_hat_path,
                z_min: float, z_max: float, split_dose: float, combined_dose: float):
        if not base_ds.n2n:
            raise ValueError("AnnealDataset requires a TimeResolvedFrameDataset built with n2n=True")
        self.base_ds = base_ds
        self.x_hat_path = str(x_hat_path)
        self.z_min, self.z_max = float(z_min), float(z_max)
        self.split_dose = float(split_dose)
        self.combined_dose = float(combined_dose)
        self.crop = base_ds.crop
        self._mm: Optional[np.memmap] = None  # opened lazily, once per worker

    def _memmap(self) -> np.memmap:
        if self._mm is None:
            self._mm = np.memmap(self.x_hat_path, dtype=np.float16, mode="r+",
                                 shape=(len(self.base_ds), *self.crop))
        return self._mm

    def _z(self, raw_counts: torch.Tensor, dose: float) -> torch.Tensor:
        return normalize_z(anscombe_transform_thinned(raw_counts, dose), self.z_min, self.z_max)

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.base_ds[idx]
        target_z = self._z(self.base_ds.denormalize(item["central_target"]), self.split_dose)   # y1
        y2_z = self._z(self.base_ds.denormalize(item["central_input"]), self.split_dose)         # y2 (renoise anchor)
        context_z = self._z(self.base_ds.denormalize(item["context"]), self.combined_dose)
        x_hat = torch.from_numpy(np.asarray(self._memmap()[idx]).astype(np.float32)).unsqueeze(0)
        return {
            "input": x_hat, "target": target_z, "context": context_z, "y2": y2_z,
            "reference": item["reference"], "index": torch.tensor(idx, dtype=torch.long),
        }

    def init_x_hat_to_y2(self, batch_size: int = 64, num_workers: int = 4) -> None:
        """Round-0 setup: x_hat_0 = y2 for every index. Only call on a fresh run
        (never on resume -- that would overwrite an already-evolved iterate)."""
        mm = self._memmap()
        loader = DataLoader(self.base_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        pos = 0
        for batch in loader:
            y2_raw = self.base_ds.denormalize(batch["central_input"])
            y2_z = self._z(y2_raw, self.split_dose).numpy().astype(np.float16)[:, 0]
            n = y2_z.shape[0]
            mm[pos:pos + n] = y2_z
            pos += n
        mm.flush()


def create_x_hat_memmap(path, num_indices: int, crop: Tuple[int, int]) -> np.memmap:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(str(path), dtype=np.float16, mode="w+", shape=(num_indices, *crop))


def create_counts_memmap(path, num_indices: int, crop: Tuple[int, int]) -> np.memmap:
    """A raw-count output cache in the exact same layout/sidecar convention as
    :func:`sdate.tr_diffusion.reconstruct.denoise_sequence`'s output, so
    :func:`sdate.tr_diffusion.reconstruct.run_windows` can consume it directly."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(str(path), dtype=np.float16, mode="w+", shape=(num_indices, *crop))


def write_counts_meta(path, first_index: int, num_frames: int, crop: Tuple[int, int],
                      dose: float, noise_seed: int, **extra) -> None:
    np.savez(str(path) + ".meta.npz", first_index=first_index, num_frames=num_frames,
             crop=list(crop), dose=dose, noise_seed=noise_seed, **extra)


# --------------------------------------------------------------------------- #
# one round: train, then infer + (optionally) renoise, streamed batch-by-batch
# --------------------------------------------------------------------------- #

def _model_forward(model, inp: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
    """Round-blind single-pass regression: ``timestep=0`` (no diffusion), a
    constant ``class_labels`` (the architecture's ``class_embed_type="timestep"``
    slot needs SOME value even though this model has no p_bin/round conditioning
    to put there -- same convention as :class:`sdate.tr_diffusion.losses.NoiseToCleanLoss`)."""
    model_input = torch.cat([inp, ctx], dim=1)
    bsz = model_input.shape[0]
    t = torch.zeros(bsz, device=model_input.device, dtype=torch.long)
    cls = torch.ones(bsz, device=model_input.device, dtype=torch.long)
    return model(model_input, timestep=t, class_labels=cls, return_dict=False)[0]


def train_round(model, dataset: AnnealDataset, epochs: int, device, batch_size: int = 16,
                lr: float = 1e-4, weight_decay: float = 1e-3, num_workers: int = 4,
                log_every: int = 100) -> List[float]:
    """Plain MSE Noise2Noise fine-tune for ``epochs`` epochs. A fresh
    ``AdamW`` is created every call (fresh optimizer state each round, even
    though the model weights warm-start from the previous round) -- Adam's
    momentum/variance estimates were accumulated against the PREVIOUS round's
    (differently-distributed) input iterate, so carrying them over would be
    optimizing against a stale landscape rather than genuinely warm-starting."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                        pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    history = []
    for epoch in range(epochs):
        running, seen = 0.0, 0
        for bi, batch in enumerate(loader):
            inp = batch["input"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
            ctx = batch["context"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = _model_forward(model, inp, ctx)
            loss = torch.nn.functional.mse_loss(pred, tgt)
            loss.backward()
            optimizer.step()
            bsz = inp.shape[0]
            running += loss.item() * bsz
            seen += bsz
            if log_every and bi % log_every == 0:
                print(f"    epoch {epoch + 1}/{epochs}  step {bi}/{len(loader)}  loss {loss.item():.5f}", flush=True)
        mean_loss = running / max(seen, 1)
        history.append(mean_loss)
        print(f"  epoch {epoch + 1}/{epochs}  mean_loss={mean_loss:.5f}", flush=True)
    return history


@torch.no_grad()
def infer_round(model, dataset: AnnealDataset, device, alpha_k: Optional[float],
                data_range: float, batch_size: int = 32, num_workers: int = 4,
                trajectory_indices: Optional[np.ndarray] = None,
                out_counts_mm: Optional[np.memmap] = None) -> Dict:
    """One full sequential pass over the dataset: compute ``D(x_hat_k)`` for every
    index, score it against the native ``reference`` (projection-domain PSNR/SSIM,
    the trajectory diagnostic ADR-0002 relies on), and -- if ``alpha_k`` is not
    None -- overwrite the ``x_hat`` memmap in place with the renoised
    ``alpha_k*y2 + (1-alpha_k)*D(x_hat_k)`` for the NEXT round. Pass
    ``alpha_k=None`` on the final round: no renoise write, ``D(x_hat_k)`` is
    the pipeline's final estimate.

    Returns ``{"psnr": float, "ssim": float, "trajectory": {idx: {"x_hat":.., "D":..}}}``
    with the trajectory arrays in native raw-count units (for the handful of
    ``trajectory_indices``, if given). If ``out_counts_mm`` is given (a
    pre-allocated ``(N, H, W)`` float16 memmap, positionally indexed exactly
    like the ``x_hat`` memmap), ``D(x_hat_k)`` in native raw-count units is
    written there for EVERY index -- used to cache the round-0 MMSE anchor and
    the final estimate for the downstream reconstruction eval.
    """
    from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=True)
    model.eval()
    mm = dataset._memmap()
    traj_set = set(int(i) for i in trajectory_indices) if trajectory_indices is not None else set()
    trajectory: Dict[int, Dict[str, np.ndarray]] = {}
    psnrs, ssims = [], []

    for batch in loader:
        inp = batch["input"].to(device, non_blocking=True)
        ctx = batch["context"].to(device, non_blocking=True)
        y2 = batch["y2"].to(device, non_blocking=True)
        ref_raw = dataset.base_ds.denormalize(batch["reference"]).numpy()[:, 0]
        idx = batch["index"].numpy()

        pred = _model_forward(model, inp, ctx)  # D(x_hat_k), normalised z-space
        pred_counts = z_norm_to_counts(pred, dataset.z_min, dataset.z_max, dataset.split_dose)
        pred_counts_np = pred_counts.detach().float().cpu().numpy()[:, 0]

        for j in range(pred_counts_np.shape[0]):
            psnrs.append(psnr(ref_raw[j], pred_counts_np[j], data_range=data_range))
            ssims.append(ssim(ref_raw[j], pred_counts_np[j], data_range=data_range))

        if out_counts_mm is not None:
            out_counts_mm[idx] = pred_counts_np.astype(np.float16)

        for j, i in enumerate(idx):
            if int(i) in traj_set:
                x_hat_counts = z_norm_to_counts(inp[j:j + 1], dataset.z_min, dataset.z_max,
                                                dataset.split_dose).detach().float().cpu().numpy()[0, 0]
                trajectory[int(i)] = {"x_hat": x_hat_counts, "D": pred_counts_np[j]}

        if alpha_k is not None:
            x_hat_next = alpha_k * y2 + (1.0 - alpha_k) * pred
            mm[idx] = x_hat_next.detach().float().cpu().numpy().astype(np.float16)[:, 0]

    if alpha_k is not None:
        mm.flush()
    if out_counts_mm is not None:
        out_counts_mm.flush()

    return {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)), "trajectory": trajectory}


def pick_trajectory_indices(base_ds: TimeResolvedFrameDataset, frame_lo: int, frame_hi: int,
                            n: int = 6, seed: int = 0) -> np.ndarray:
    """``n`` positional indices into ``base_ds``, one drawn at random from each of
    ``n`` equal-width bins spanning the frames in ``[frame_lo, frame_hi)`` that
    ``base_ds`` actually covers -- stratified so the trajectory visuals span the
    eval window's rotation angles without hand-picking specific content."""
    frame_ids = base_ds.indices  # frame index for each positional index
    in_window = np.where((frame_ids >= frame_lo) & (frame_ids < frame_hi))[0]
    if len(in_window) < n:
        raise ValueError(f"only {len(in_window)} usable frames in [{frame_lo}, {frame_hi}); need >= {n}")
    rng = np.random.default_rng(seed)
    bins = np.array_split(in_window, n)
    return np.array([int(rng.choice(b)) for b in bins])


# --------------------------------------------------------------------------- #
# checkpoint / resume state
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# N2V-based Annealed pipeline: reuses an EXISTING baseline N2V+Huber+context
# checkpoint (already trained, already known to produce a blurred-but-noiseless
# E[x|context] estimate) as the round denoiser, via the project's own
# BaselineN2VLoss/denoise_frames_baseline -- no Anscombe transform, no
# binomial-split y1/y2. Superseded the plain-MSE N2N design above: that design
# let the network route noise straight through via skip connections since
# nothing forced it to ignore the directly-visible noisy input pixel: N2V's
# blind-spot masking removes that shortcut structurally (the network never
# sees the pixel it's predicting), which is the whole reason for the switch.
#
# Round 0 is free (the existing checkpoint IS the round-0 denoiser, no
# training) -- the driver script runs one inference pass on the fixed noisy
# measurement to get D(x_hat_0)=D(measurement), then renoises straight into
# x_hat_1. Every round after that warm-starts the SAME architecture and
# fine-tunes via the SAME BaselineN2VLoss (same ratio/window/conditioning_
# probability/loss_type the checkpoint was originally trained with), but
# self-supervising against the CURRENT iterate x_hat_k instead of a fresh
# video read. The renoise blend always mixes back in the ORIGINAL fixed
# measurement (never the evolving iterate), in the model's native normalised
# [-1, 1] raw-count space -- the same space the reused checkpoint already
# operates in, so no transform is needed going in or out.
# --------------------------------------------------------------------------- #

class _IndexedWrapper(Dataset):
    """Adds an ``"index"`` field (the wrapped dataset's own positional index,
    0..len-1) to every item -- lets round 0's plain ``base_ds`` loader (no
    ``x_hat`` involved yet) emit the same ``"index"`` key ``AnnealN2VDataset``
    already provides, so :func:`infer_round_n2v` can treat both loaders
    uniformly regardless of round."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = dict(self.base[idx])
        item["index"] = torch.tensor(idx, dtype=torch.long)
        return item


class AnnealN2VDataset(Dataset):
    """Wraps a fixed-seed ``TimeResolvedFrameDataset`` (plain ``extra_noise_dose``
    branch, NOT n2n) -- ``context``/``reference``/the fixed noisy ``measurement``
    are read live each access (deterministic given the fixed ``noise_seed``) --
    with the mutable ``x_hat`` iterate (a float16 memmap, normalised [-1, 1]
    raw-count space, indexed 1:1 with ``base_ds.indices``) standing in for
    ``central`` -- the image ``BaselineN2VLoss``'s blind-spot masking
    self-supervises against.
    """

    def __init__(self, base_ds: TimeResolvedFrameDataset, x_hat_path):
        if base_ds.n2n or base_ds.extra_noise_dose is None:
            raise ValueError("AnnealN2VDataset requires a plain TimeResolvedFrameDataset "
                             "(n2n=False, extra_noise_dose set) -- the fixed dose-thinned "
                             "measurement branch, not the binomial-split N2N branch")
        self.base_ds = base_ds
        self.x_hat_path = str(x_hat_path)
        self.crop = base_ds.crop
        self._mm: Optional[np.memmap] = None

    def _memmap(self) -> np.memmap:
        if self._mm is None:
            self._mm = np.memmap(self.x_hat_path, dtype=np.float16, mode="r+",
                                 shape=(len(self.base_ds), *self.crop))
        return self._mm

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.base_ds[idx]  # "central" here IS the fixed noisy measurement
        x_hat = torch.from_numpy(np.asarray(self._memmap()[idx]).astype(np.float32)).unsqueeze(0)
        return {
            "central": x_hat, "measurement": item["central"], "context": item["context"],
            "reference": item["reference"], "index": torch.tensor(idx, dtype=torch.long),
        }


def train_round_n2v(model, dataset, loss_fn, epochs: int, device, batch_size: int = 16,
                    lr: float = 1e-4, weight_decay: float = 1e-3, num_workers: int = 4,
                    log_every: int = 100) -> List[float]:
    """Fine-tune ``model`` for ``epochs`` epochs with an existing ``BaseLoss``
    instance (``BaselineN2VLoss``, already constructed with the checkpoint's own
    ratio/window/conditioning_probability/loss_type) -- ``compute_loss`` handles
    its own device placement, so batches are passed through unmodified. Fresh
    optimizer state every round (see ``train_round``'s docstring for why), even
    though the model weights warm-start from the previous round."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                        pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    history = []
    for epoch in range(epochs):
        running, seen = 0.0, 0
        for bi, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = loss_fn.compute_loss(batch, model)
            loss.backward()
            optimizer.step()
            bsz = batch["central"].shape[0]
            running += loss.item() * bsz
            seen += bsz
            if log_every and bi % log_every == 0:
                print(f"    epoch {epoch + 1}/{epochs}  step {bi}/{len(loader)}  loss {loss.item():.5f}", flush=True)
        mean_loss = running / max(seen, 1)
        history.append(mean_loss)
        print(f"  epoch {epoch + 1}/{epochs}  mean_loss={mean_loss:.5f}", flush=True)
    return history


@torch.no_grad()
def infer_round_n2v(model, loader, device, alpha_k: Optional[float], data_range: float,
                    norm_min: float, norm_max: float, x_hat_mm: Optional[np.memmap] = None,
                    trajectory_indices: Optional[np.ndarray] = None,
                    out_counts_mm: Optional[np.memmap] = None) -> Dict:
    """One full sequential pass over ``loader`` (either the plain fixed-measurement
    ``base_ds`` for round 0, or an ``AnnealN2VDataset`` for every round after):
    run the reused ``denoise_frames_baseline`` on ``central`` (the fixed
    measurement at round 0, the current ``x_hat_k`` iterate otherwise), score
    projection-domain PSNR/SSIM vs the native ``reference``, and -- if
    ``alpha_k`` is not None -- renoise into ``x_hat_mm`` for the NEXT round:
    ``x_hat_{k+1} = alpha_k*measurement + (1-alpha_k)*D(central)``, ALWAYS
    blending back the fixed original measurement, never the iterate itself.
    Pass ``alpha_k=None`` on the final round: no renoise write, ``D(central)``
    is the pipeline's final estimate. ``out_counts_mm``, if given, gets
    ``D(central)`` in native raw-count units for every index (used for the
    round-0 MMSE anchor and the final estimate's reconstruction-eval cache).
    """
    from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

    from .pipeline import denoise_frames_baseline

    model.eval()
    span = float(norm_max - norm_min)

    def to_counts(x_norm: torch.Tensor) -> torch.Tensor:
        return (x_norm.clamp(-1, 1) + 1.0) * 0.5 * span + norm_min

    traj_set = set(int(i) for i in trajectory_indices) if trajectory_indices is not None else set()
    trajectory: Dict[int, Dict[str, np.ndarray]] = {}
    psnrs, ssims = [], []

    for batch in loader:
        central = batch["central"].to(device, non_blocking=True)
        context = batch["context"].to(device, non_blocking=True)
        measurement = batch.get("measurement", batch["central"]).to(device, non_blocking=True)
        ref_counts = to_counts(batch["reference"].to(device)).cpu().numpy()[:, 0]
        idx = batch["index"].numpy()

        pred = denoise_frames_baseline(model, central, context, present=True, poisson_head=False,
                                       norm_min=norm_min, norm_max=norm_max)  # normalised [-1, 1]
        pred_counts_np = to_counts(pred).detach().float().cpu().numpy()[:, 0]

        for j in range(pred_counts_np.shape[0]):
            psnrs.append(psnr(ref_counts[j], pred_counts_np[j], data_range=data_range))
            ssims.append(ssim(ref_counts[j], pred_counts_np[j], data_range=data_range))

        if out_counts_mm is not None:
            out_counts_mm[idx] = pred_counts_np.astype(np.float16)

        for j, i in enumerate(idx):
            if int(i) in traj_set:
                central_counts = to_counts(central[j:j + 1]).detach().float().cpu().numpy()[0, 0]
                trajectory[int(i)] = {"x_hat": central_counts, "D": pred_counts_np[j]}

        if alpha_k is not None and x_hat_mm is not None:
            x_hat_next = alpha_k * measurement + (1.0 - alpha_k) * pred
            x_hat_mm[idx] = x_hat_next.detach().float().cpu().numpy().astype(np.float16)[:, 0]

    if alpha_k is not None and x_hat_mm is not None:
        x_hat_mm.flush()
    if out_counts_mm is not None:
        out_counts_mm.flush()

    return {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)), "trajectory": trajectory}


def save_state(state_path, model, round_idx: int, config: Dict) -> None:
    Path(state_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, str(state_path) + ".pt")
    Path(str(state_path) + ".json").write_text(json.dumps({
        "last_completed_round": round_idx, **config,
    }, indent=2))


def load_state(state_path) -> Optional[Dict]:
    p = Path(str(state_path) + ".json")
    if not p.exists():
        return None
    return json.loads(p.read_text())
