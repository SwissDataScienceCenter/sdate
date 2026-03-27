"""
gsplat_compress.initializations — Gaussian parameter initialisation strategies.

Sub-modules
-----------
init_2d
    Strategies for 2-D orthographic image fitting:
    ``uniform_2d``, ``intensity_2d``, ``multiresolution_residual_2d``

init_3d
    Strategies for 3-D volumetric / perspective scenes:
    ``uniform_3d``  (placeholder — extend as needed)
"""

from gsplat_compress.initializations.init_2d import (
    uniform_2d,
    intensity_2d,
    multiresolution_residual_2d,
)

from gsplat_compress.initializations.init_3d import (
    uniform_3d,
)

__all__ = [
    # 2-D
    "uniform_2d",
    "intensity_2d",
    "multiresolution_residual_2d",
    # 3-D
    "uniform_3d",
]
