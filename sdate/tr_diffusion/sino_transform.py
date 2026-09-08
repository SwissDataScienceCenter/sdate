"""Per-frame 2D Radon transform ("virtual sinogram") of a single projection.

Each measured projection frame is treated as its own standalone 2D image (an
"object" in the Radon-transform sense) — this has NO relation to the real 3D
acquisition geometry or rotation axis. Its "sinogram" is the plain 2D
parallel-beam Radon transform of that one image at ``num_angles`` synthetic
angles spanning ``[0, angle_max_deg)`` (180 deg by default: a parallel-beam
transform at angle theta+180 is the mirror of theta, so 180 already fully
determines the image).

Rationale (see :mod:`sdate.tr_diffusion.losses` ``SinogramN2VLoss``): a
sinogram value is a line-integral -- a weighted SUM over many raw pixels --
so it has much better SNR than any single raw pixel (the noise averages down
the way any sum of many samples does). That same summing also means the
result is no longer cleanly Poisson-distributed, so a plain Huber/MAE/MSE
loss (not this project's usual NB-NLL) is the right choice for this domain.

Built on the existing ``astra_torch.parallel2D`` module (2D parallel-beam
projector/FBP already used by :mod:`sdate.sino_hexplane` and the
``astra-torch`` test suite) -- no new ASTRA geometry code needed here.
"""

from __future__ import annotations

import math
import sys
from typing import Optional, Tuple

import numpy as np
import torch

if "/myhome/astra-torch" not in sys.path:
    sys.path.insert(0, "/myhome/astra-torch")

from astra_torch.parallel2D import (  # noqa: E402
    _apply_ramp_filter, _AstraParallel2DOp, _create_parallel2d_geometry, build_parallel2d_projector,
)


def round_up_to_multiple(x: float, multiple: int = 32) -> int:
    """Smallest multiple of ``multiple`` that is >= ``x``.

    The UNet has 5 downsampling stages (6 ``DownBlock2D``s, the last without
    downsampling -- see ``model.py``'s ``_DOWN``), so both spatial dims of
    whatever it's fed must be divisible by ``2**5 = 32``.
    """
    m = int(multiple)
    return int(math.ceil(float(x) / m)) * m


def choose_sino_shape(vol_shape: Tuple[int, int], unet_divisor: int = 32) -> Tuple[int, int]:
    """``(num_angles, det_cols)`` for a square, corner-clipping-free sinogram of
    an image with shape ``vol_shape=(ny, nx)``.

    ``det_cols`` is padded to the image's diagonal (the widest extent any
    oblique-angle projection can span) rounded up to a multiple of
    ``unet_divisor``; ``num_angles`` is set equal to it, keeping the sinogram
    square (matching the crop's width was the original ask -- padding for the
    diagonal and keeping it square both push in the same direction here).
    """
    ny, nx = vol_shape
    diag = math.sqrt(float(ny) ** 2 + float(nx) ** 2)
    n = round_up_to_multiple(diag, unet_divisor)
    return n, n


class SinoTransform:
    """Forward (Radon) / inverse (FBP) transform between a projection frame and
    its per-frame virtual sinogram, batched and multi-channel.

    ``vol_shape=(ny, nx)`` is the projection's own crop shape (the "volume" the
    forward projector treats each frame as). ``num_angles``/``det_cols``
    default to :func:`choose_sino_shape`.
    """

    def __init__(self, vol_shape: Tuple[int, int], num_angles: Optional[int] = None,
                 det_cols: Optional[int] = None, angle_max_deg: float = 180.0,
                 device: Optional[torch.device] = None):
        self.vol_shape = (int(vol_shape[0]), int(vol_shape[1]))
        if num_angles is None or det_cols is None:
            auto_n, auto_c = choose_sino_shape(self.vol_shape)
            num_angles = num_angles if num_angles is not None else auto_n
            det_cols = det_cols if det_cols is not None else auto_c
        self.num_angles = int(num_angles)
        self.det_cols = int(det_cols)
        self.angle_max_deg = float(angle_max_deg)
        self.angles_deg = np.linspace(0.0, self.angle_max_deg, self.num_angles, endpoint=False)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._forward_layer = build_parallel2d_projector(
            self.vol_shape, self.det_cols, self.angles_deg, device=self.device,
        )
        # ``fbp_reconstruction_masked`` (the library's own FBP helper) has a CuPy
        # zero-copy bug against the ASTRA version installed in this environment
        # (``astra.data2d.link`` rejects a CuPy array with "Data should be
        # numpy.ndarray" -- confirmed via a real-GPU smoke test). Its FORWARD
        # projector (used above) and ``_AstraParallel2DOp.adjoint`` avoid this
        # entirely by going through ``astra.data2d.create`` + ``cp.asnumpy``
        # instead of ``.link`` -- confirmed working. So ``inverse`` below
        # implements FBP itself (ramp-filter, numpy-side, then ``.adjoint``)
        # rather than calling the library's broken helper.
        vol_geom, proj_geom = _create_parallel2d_geometry(
            vol_shape=self.vol_shape, det_cols=self.det_cols, angles_deg=self.angles_deg,
        )
        self._op = _AstraParallel2DOp(vol_geom, proj_geom, self.vol_shape, self.det_cols, self.num_angles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, ny, nx)`` raw-count images -> ``(B, C, num_angles, det_cols)`` sinograms.

        Every channel (central AND every context tap) is transformed
        independently -- each is its own standalone 2D image. ``C=0`` (a k=0
        context) returns an empty ``(B, 0, num_angles, det_cols)`` tensor.
        """
        if x.dim() != 4:
            raise ValueError(f"expected (B, C, ny, nx), got {tuple(x.shape)}")
        b, c, h, w = x.shape
        if (h, w) != self.vol_shape:
            raise ValueError(f"expected spatial shape {self.vol_shape}, got {(h, w)}")
        if c == 0:
            return x.new_zeros(b, 0, self.num_angles, self.det_cols)
        flat = x.reshape(b * c, 1, h, w).to(device=self.device, dtype=torch.float32)
        sino = self._forward_layer(flat)  # (B*C, num_angles, det_cols)
        return sino.reshape(b, c, self.num_angles, self.det_cols)

    @torch.no_grad()
    def inverse(self, sino: torch.Tensor, filter_type: str = "hann") -> torch.Tensor:
        """``(B, 1, num_angles, det_cols)`` sinograms -> ``(B, 1, ny, nx)`` images (FBP).

        Inference-only (no gradient) -- loops over the batch (ramp-filter on
        CPU/numpy, then ``_AstraParallel2DOp.adjoint`` for the backprojection
        half of FBP; see the CuPy-bug note in ``__init__``).
        """
        if sino.dim() != 4 or sino.shape[1] != 1:
            raise ValueError(f"expected (B, 1, num_angles, det_cols), got {tuple(sino.shape)}")
        b = sino.shape[0]
        # Matches astra_torch.parallel2D.fbp_reconstruction_masked's own analytical-FBP
        # scaling -- plain summed backprojection over `num_angles` discrete angles isn't
        # itself a Riemann-sum approximation of the continuous backprojection integral
        # without this factor (confirmed empirically: omitting it gave a ~1e5x-too-large,
        # wrong-scale reconstruction in a round-trip smoke test).
        norm_factor = math.pi / (2.0 * self.num_angles) if self.num_angles > 0 else 1.0
        out = torch.empty(b, 1, *self.vol_shape, device=self.device, dtype=torch.float32)
        for i in range(b):
            sino_np = sino[i, 0].detach().cpu().numpy().astype(np.float32)
            filtered_np = _apply_ramp_filter(sino_np, use_gpu=False, filter_type=filter_type, det_spacing_mm=1.0)
            filtered_t = torch.from_numpy(np.ascontiguousarray(filtered_np)).to(device=self.device, dtype=torch.float32)
            out[i, 0] = self._op.adjoint(filtered_t) * norm_factor
        return out


@torch.no_grad()
def fit_sino_norm_range(dataset, sino_transform: SinoTransform, n_sample: int = 32,
                        percentiles: Tuple[float, float] = (0.5, 99.5), seed: int = 0) -> Tuple[float, float]:
    """Fit ``(sino_norm_min, sino_norm_max)`` over a sample of the dataset's own
    frames, mirroring :class:`sdate.tr_diffusion.data.TimeResolvedFrameDataset`'s
    own ``_fit_norm`` convention (same default percentiles).

    A sinogram's line-integral values live on a very different scale than raw
    projection counts, so this is a SEPARATE normalisation range from
    ``dataset.norm_min``/``norm_max`` -- fit it once, up front, over real data
    rather than guessing a constant.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(n_sample, len(dataset)), replace=False)
    device = sino_transform.device
    span = None
    vals = []
    for i in idx:
        item = dataset[int(i)]
        central = item["central"].unsqueeze(0).float().to(device)
        if span is None:
            span = (float(dataset.norm_max) - float(dataset.norm_min))
        counts = (central + 1.0) * 0.5 * span + float(dataset.norm_min)
        sino = sino_transform.forward(counts)
        vals.append(sino.reshape(-1).cpu().numpy())
    all_vals = np.concatenate(vals)
    lo, hi = np.percentile(all_vals, percentiles)
    return float(lo), float(hi)
