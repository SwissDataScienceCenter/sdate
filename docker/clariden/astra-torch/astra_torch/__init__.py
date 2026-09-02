"""ASTRA-Torch: GPU-accelerated tomographic reconstruction library.

This library provides PyTorch-compatible implementations of tomographic reconstruction
algorithms using the ASTRA toolbox, including support for CBCT, laminography, and 2D parallel beam.

Modules:
--------
cbct : Cone-beam CT reconstruction functions
lamino : Laminography reconstruction functions
parallel2D : 2D parallel beam reconstruction functions

Examples:
---------
>>> import astra_torch
>>> from astra_torch import fdk_reconstruction_masked
>>> volume = fdk_reconstruction_masked(projs, vecs, voxel_per_mm=10)
"""

# Import CBCT functions
from .cbct import (
    project_conebeam_masked, 
    build_conebeam_projector, 
    CBCTAcquisition,
    fdk_reconstruction_masked as cbct_fdk_reconstruction_masked,
    gd_reconstruction_masked as cbct_gd_reconstruction_masked
)

# Import laminography functions  
from .lamino import (
    fbp_reconstruction_masked as lamino_fbp_reconstruction_masked,
    sirt_reconstruction_masked as lamino_sirt_reconstruction_masked,
    gd_reconstruction_masked as lamino_gd_reconstruction_masked,
    build_lamino_projector, 
    LaminoAcquisition
)

# Import 2D parallel beam functions
from .parallel2D import (
    fbp_reconstruction_masked as parallel2d_fbp_reconstruction_masked,
    sirt_reconstruction_masked as parallel2d_sirt_reconstruction_masked,
    gd_reconstruction_masked as parallel2d_gd_reconstruction_masked,
    build_parallel2d_projector,
    Parallel2DAcquisition
)

# For backward compatibility, expose some functions without prefixes
from .cbct import fdk_reconstruction_masked, gd_reconstruction_masked

__version__ = "0.1.0"
__author__ = "CHIP Project"

__all__ = [
    # CBCT functions
    'cbct_fdk_reconstruction_masked',
    'cbct_gd_reconstruction_masked', 
    'fdk_reconstruction_masked',  # Backward compatibility
    'gd_reconstruction_masked',   # Backward compatibility
    'project_conebeam_masked',
    'build_conebeam_projector',
    'CBCTAcquisition',
    
    # Laminography functions
    'lamino_fbp_reconstruction_masked',
    'lamino_sirt_reconstruction_masked',
    'lamino_gd_reconstruction_masked',
    'build_lamino_projector',
    'LaminoAcquisition',
    
    # 2D Parallel beam functions
    'parallel2d_fbp_reconstruction_masked',
    'parallel2d_sirt_reconstruction_masked',
    'parallel2d_gd_reconstruction_masked',
    'build_parallel2d_projector',
    'Parallel2DAcquisition',
    
    # Version info
    '__version__',
    '__author__'
]
