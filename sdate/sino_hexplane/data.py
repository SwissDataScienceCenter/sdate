"""Synthetic helix-acquisition data for the sinogram-domain HexPlane.

Reuses the object phantoms + forward projector of :mod:`sdate.tr_naf.data`, but
with two differences that make the synthetic stream resemble a real continuous
GigaFrost acquisition (see project memory ``project-sino-hexplane``):

* **Advancing wedge.**  Consecutive frames continue the rotation
  (frame 0 -> ``[0, 180)``, frame 1 -> ``[180, 360)``, frame 2 -> ``[360, 540) ==
  [0, 180)`` ...) instead of repeating the same ``[0, 180]`` scan.  The *physical*
  angle fed to the model is ``theta = angle mod 360`` (genuinely 2*pi-periodic);
  the monotonic advance is carried entirely by ``t``.  Each frame still measures
  only its own angular window, so asking the model for the complementary
  half-turn is a genuine unseen-projection synthesis test.
* **Count space.**  Each frame is turned into raw Poisson-Gaussian detector
  counts (``lambda = flat*exp(-p)``), the interface a real loader will fill.

Axis convention (confirmed empirically): the projection tensor is ``(A, R, C)``
= ``(angles, v, s)`` — detector **rows R = v** (rotation axis), **cols C = s**
(sinogram axis).

The measurement (noisy) target is ``counts``; the clean, noise-free scaled line
integral ``p`` is kept per frame as the ground truth for PSNR/SSIM (the model is
scored in this line-integral space).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..tr_naf.data import resample_to_cube
from ..tr_naf.noise import simulate_counts


@dataclass
class HelixFrame:
    """One acquired frame: clean scaled line integrals + noisy counts, ``(A, R, C)``."""
    t_norm: float
    angles_deg: np.ndarray        # (A,) absolute projection angles (may exceed 360)
    theta_norm: np.ndarray        # (A,) physical angle mod 360, normalised to [0, 1)
    clean_p: torch.Tensor         # (A, R, C) noise-free scaled line integrals (GT)
    counts: torch.Tensor          # (A, R, C) raw detector counts (training target)
    true_volume: torch.Tensor = field(default=None, repr=False)


def build_helix_noise_acquisition(
    data_path: Union[str, Path],
    num_frames: int = 25,
    wedge_deg: float = 180.0,             # angular window measured per frame (full scan)
    angle_advance_deg: Optional[float] = None,  # start-angle step between frames (default = wedge)
    angle_start_deg: float = 0.0,
    num_projs_per_frame: int = 180,
    full_angle_span_deg: float = 180.0,
    timestep_skip: int = 0,
    cube_size: int = 128,
    det_spacing_mm: float = 1.0,
    lamino_angle_deg: float = 0.0,
    photons: float = 1e3,                 # I0 (dose knob): 1e3 / 5e2 are the interesting regimes
    p_max: float = 2.5,
    gain: float = 1.0,
    read_noise: float = 0.0,
    dark: float = 0.0,
    normalize_range: Optional[Tuple[float, float]] = None,
    noise_seed: Optional[int] = 0,
    device: Optional[torch.device] = None,
) -> Tuple[List[HelixFrame], dict]:
    """Build a helix (advancing-wedge) count-space acquisition.

    Returns ``(frames, meta)``.  ``meta`` carries ``flat`` (I0), ``scale``
    (line-integral scaling), ``det_shape`` and the noise parameters — the same
    fields a real GigaFrost loader would populate.
    """
    from ladiff.datasets import TifVolumeSliceDataset
    from astra_torch.lamino import build_lamino_projector

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if angle_advance_deg is None:
        angle_advance_deg = wedge_deg
    gen = torch.Generator(device=device)
    if noise_seed is not None:
        gen.manual_seed(int(noise_seed))

    dataset = TifVolumeSliceDataset(
        data_path=data_path, file_range=num_frames, resize=cube_size,
        normalize_range=normalize_range, augment=False, skip=timestep_skip,
    )
    if dataset.num_files < num_frames:
        raise ValueError(f"Requested {num_frames} frames but only {dataset.num_files} tif files resolved.")

    num_projs = max(1, int(num_projs_per_frame))
    vol_shape = (cube_size, cube_size, cube_size)
    det_shape = (cube_size, cube_size)  # (R, C) = (v, s)

    # Pass 1: project every frame over its advancing wedge; keep clean line integrals.
    vols, clean_sinos, angles_all = [], [], []
    for i in range(num_frames):
        vol = resample_to_cube(dataset.get_volume(i).to(device), cube_size)
        a_start = angle_start_deg + i * angle_advance_deg
        angles = np.linspace(a_start, a_start + wedge_deg, num_projs, endpoint=False)
        projector = build_lamino_projector(
            vol_shape=vol_shape, det_shape=det_shape, angles_deg=angles,
            lamino_angle_deg=lamino_angle_deg, det_spacing_mm=det_spacing_mm, device=device,
        )
        with torch.no_grad():
            sino = projector(vol.unsqueeze(0).unsqueeze(0)).squeeze(0)  # (A, R, C)
        vols.append(vol); clean_sinos.append(sino); angles_all.append(angles)

    global_max = max(float(s.max()) for s in clean_sinos)
    scale = p_max / max(global_max, 1e-8)
    flat = torch.full(det_shape, float(photons), device=device)

    # Pass 2: scale to [0, p_max] and draw counts.
    frames: List[HelixFrame] = []
    for i in range(num_frames):
        p = clean_sinos[i] * scale                                  # (A, R, C) clean GT
        counts = simulate_counts(p, flat, gain=gain, read_noise=read_noise, dark=dark, generator=gen)
        theta_norm = (angles_all[i] % 360.0) / 360.0
        t_norm = i / max(num_frames - 1, 1)
        frames.append(HelixFrame(
            t_norm=t_norm, angles_deg=angles_all[i], theta_norm=theta_norm,
            clean_p=p, counts=counts, true_volume=vols[i]))

    meta = dict(
        regime="helix_noise", vol_shape=vol_shape, det_shape=det_shape, cube_size=cube_size,
        det_spacing_mm=det_spacing_mm, lamino_angle_deg=lamino_angle_deg,
        num_frames=num_frames, wedge_deg=wedge_deg, angle_advance_deg=angle_advance_deg,
        num_projs_per_frame=num_projs, full_angle_span_deg=full_angle_span_deg,
        photons=photons, p_max=p_max, scale=scale, gain=gain, read_noise=read_noise, dark=dark,
        flat=flat.cpu(), norm_min=dataset.norm_min, norm_max=dataset.norm_max,
    )
    return frames, meta


class HelixSinoDataset:
    """Random per-pixel ``(theta, s, v, t, counts)`` batch sampler over all frames.

    Stacks the per-frame ``(A, R, C)`` tensors (constant shape) into ``(F, A, R, C)``
    and samples uniform random ``(frame, angle, row, col)`` indices.  Counts (the
    training target) stay on ``device``; the clean GT line integrals are held on
    CPU (only needed for evaluation).
    """

    def __init__(self, frames: List[HelixFrame], device: Optional[torch.device] = None):
        self.device = device or (frames[0].counts.device)
        self.F = len(frames)
        self.A, self.R, self.C = frames[0].counts.shape

        self.counts = torch.stack([f.counts for f in frames]).to(self.device)      # (F,A,R,C)
        self.clean_p = torch.stack([f.clean_p.cpu() for f in frames])              # (F,A,R,C) cpu
        self.theta = torch.tensor(np.stack([f.theta_norm for f in frames]),
                                  dtype=torch.float32, device=self.device)         # (F, A)
        self.t = torch.tensor([f.t_norm for f in frames],
                              dtype=torch.float32, device=self.device)             # (F,)
        # Detector-axis coordinate lookups in [0, 1].
        self.s_coord = torch.linspace(0, 1, self.C, device=self.device)            # cols = s
        self.v_coord = torch.linspace(0, 1, self.R, device=self.device)            # rows = v
        self.angles_deg = [f.angles_deg for f in frames]

    def sample(self, batch_size: int, generator: Optional[torch.Generator] = None):
        """Return dict of ``theta, s, v, t`` (each ``(B,)`` in [0,1]) and ``counts (B,)``."""
        g = generator
        fi = torch.randint(0, self.F, (batch_size,), device=self.device, generator=g)
        ai = torch.randint(0, self.A, (batch_size,), device=self.device, generator=g)
        ri = torch.randint(0, self.R, (batch_size,), device=self.device, generator=g)
        ci = torch.randint(0, self.C, (batch_size,), device=self.device, generator=g)
        return {
            "theta": self.theta[fi, ai],
            "s": self.s_coord[ci],
            "v": self.v_coord[ri],
            "t": self.t[fi],
            "counts": self.counts[fi, ai, ri, ci],
        }

    def frame_coords(self, frame_idx: int, device: Optional[torch.device] = None):
        """Full ``(theta, s, v, t)`` grid for one frame, flattened to ``(A*R*C,)`` each.

        Order matches ``clean_p[frame_idx].reshape(-1)`` so predictions can be
        reshaped straight back to ``(A, R, C)``.
        """
        device = device or self.device
        A, R, C = self.A, self.R, self.C
        theta = self.theta[frame_idx].to(device)                    # (A,)
        th = theta.view(A, 1, 1).expand(A, R, C).reshape(-1)
        v = self.v_coord.to(device).view(1, R, 1).expand(A, R, C).reshape(-1)
        s = self.s_coord.to(device).view(1, 1, C).expand(A, R, C).reshape(-1)
        t = self.t[frame_idx].to(device).expand(A * R * C)
        return {"theta": th, "s": s, "v": v, "t": t}

    def sino_coords(self, t_norm: float, thetas_norm: np.ndarray,
                    device: Optional[torch.device] = None):
        """Coords for a full sinogram ``(A', R, C)`` at fixed time ``t_norm`` over an
        arbitrary set of normalised angles ``thetas_norm`` (may be unmeasured).

        Order matches ``reshape(A', R, C)``.  Used to predict a denoised sinogram at
        angles the model never saw (for the interpolated 360deg reconstruction arm).
        """
        device = device or self.device
        R, C = self.R, self.C
        th = torch.as_tensor(thetas_norm, dtype=torch.float32, device=device)
        A = th.shape[0]
        theta = th.view(A, 1, 1).expand(A, R, C).reshape(-1)
        v = self.v_coord.to(device).view(1, R, 1).expand(A, R, C).reshape(-1)
        s = self.s_coord.to(device).view(1, 1, C).expand(A, R, C).reshape(-1)
        t = torch.full((A * R * C,), float(t_norm), device=device)
        return {"theta": theta, "s": s, "v": v, "t": t}, (A, R, C)

    def interp_plane_coords(self, t_norm: float, v_index: int, thetas_norm: np.ndarray,
                            device: Optional[torch.device] = None):
        """Coords for a ``(theta, s)`` image at fixed time ``t_norm`` and detector row ``v``.

        ``thetas_norm`` is an arbitrary set of normalised angles in [0,1) (may lie
        between measured ones) — used for the theta-interpolation visualisation.
        Returns coords dict shaped ``(len(thetas) * C,)`` and the grid shape.
        """
        device = device or self.device
        C = self.C
        th = torch.tensor(thetas_norm, dtype=torch.float32, device=device)
        nth = th.shape[0]
        theta = th.view(nth, 1).expand(nth, C).reshape(-1)
        s = self.s_coord.to(device).view(1, C).expand(nth, C).reshape(-1)
        v = torch.full((nth * C,), float(self.v_coord[v_index]), device=device)
        t = torch.full((nth * C,), float(t_norm), device=device)
        return {"theta": theta, "s": s, "v": v, "t": t}, (nth, C)
