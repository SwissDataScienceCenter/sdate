"""Time-resolved neural implicit tomography (NAF-style) for limited-angle 4-D CT.

A coordinate network emits ``K`` temporal-basis coefficients per spatial voxel;
contracted against a fixed smooth basis (cubic B-splines) it represents a smooth
per-voxel attenuation curve over the scan.  The forward model is the
differentiable ASTRA laminography projector, so the field is fit
self-supervisedly to rotating limited-angle frames — the analog of SAXS-NAF
(``smartt.saxs_naf``) with a temporal basis in place of spherical harmonics.
"""

from .temporal_basis import BSplineTemporalBasis
from .model import TimeResolvedNafField
from .data import (
    Frame, build_acquisition, sliding_window_fbp, resample_to_cube,
    build_noise_acquisition, per_frame_fbp,
)
from .reconstruct import tr_naf_reconstruction, reconstruct_volume_at
from .metrics import (
    make_circular_mask,
    masked_psnr,
    masked_ssim,
    evaluate_frames,
    save_result,
    load_result,
)
from .movies import generate_slice_movies, movies_from_result, generate_comparison_movie

__all__ = [
    "BSplineTemporalBasis",
    "TimeResolvedNafField",
    "Frame",
    "build_acquisition",
    "sliding_window_fbp",
    "resample_to_cube",
    "build_noise_acquisition",
    "per_frame_fbp",
    "tr_naf_reconstruction",
    "reconstruct_volume_at",
    "make_circular_mask",
    "masked_psnr",
    "masked_ssim",
    "evaluate_frames",
    "save_result",
    "load_result",
    "generate_slice_movies",
    "movies_from_result",
    "generate_comparison_movie",
]
