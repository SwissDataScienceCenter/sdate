"""ladiff – Limited-angle diffusion model project.

This package provides datasets and utilities for training diffusion models
on FBP reconstructions from limited-angle tomography data.

Public API
----------
BaseLimitedAngleReconstructions : torch.utils.data.Dataset
    Dataset of FBP reconstructions obtained by sliding a window of size
    `total_projections` over the temporally-assembled sinogram.
"""

from ladiff.datasets import TifVolumeSliceDataset
from ladiff.fourier_wedge import (
    apply_missing_wedge,
    build_missing_wedge_mask,
    extract_central_slice_at_angle,
    inpaint_fourier_wedge,
)

__all__ = [
    "BaseLimitedAngleReconstructions",
    "DDIMPipeline",
    "GuidedDDIMScheduler",
    "TifVolumeSliceDataset",
    "apply_missing_wedge",
    "build_missing_wedge_mask",
    "extract_central_slice_at_angle",
    "inpaint_fourier_wedge",
]

# Heavy dependencies (astra_torch, chip, diffusers) are optional;
# import them only when available so that ladiff.datasets can be used standalone.
try:
    from sdate.limited_angle_tomo import BaseLimitedAngleReconstructions
    from ladiff.schedulers.pipeline_ddim import DDIMPipeline
    from ladiff.schedulers.scheduling_ddim import GuidedDDIMScheduler
except ImportError:
    pass
