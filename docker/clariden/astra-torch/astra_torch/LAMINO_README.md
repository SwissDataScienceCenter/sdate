# Laminography Reconstruction Module

This module (`astra_torch.lamino`) provides GPU-accelerated laminography reconstruction functions using the ASTRA toolbox. It implements true parallel beam geometry for laminography.

## Key Features

- **FBP reconstruction** (Filtered Back Projection) with optional view masking for laminography data
- **Gradient descent reconstruction** using differentiable projectors  
- **PyTorch integration** with autograd support for optimization-based methods
- **CUDA acceleration** via ASTRA toolbox (requires ASTRA compiled with CUDA)
- **True parallel beam geometry** - source/detector distances are irrelevant

## Laminography Geometry

Laminography uses parallel beam geometry where:

1. **Lamino angle**: All X-ray beams are parallel and tilted by a lamino angle (typically 45-75°) relative to the sample
2. **Rotation**: The sample rotates 0-360° around its axis while beam direction remains fixed
3. **Limited angle problem**: Due to the tilted geometry, some spatial frequencies are missing, leading to anisotropic resolution

## Main Functions

### `fbp_reconstruction_masked`

Performs FBP (Filtered Back Projection) reconstruction for laminography data using parallel beam geometry.

```python
volume = fbp_reconstruction_masked(
    projs_vrc,           # (V,R,C) tensor: projections 
    angles_deg,          # (V,) array: rotation angles in degrees
    lamino_angle_deg,    # float: laminography angle (e.g., 61°)
    mask=None,           # optional: view indices to use
    tilt_angle_deg=0.0,  # additional tilt parameter
    voxel_per_mm=10,     # resolution parameter
    vol_shape=None,      # optional: explicit (nx,ny,nz)
    det_spacing_mm=1.0,  # detector pixel spacing
)
```
```

### `gd_reconstruction_masked`  

Performs iterative gradient descent reconstruction using differentiable forward projector.

```python
volume = gd_reconstruction_masked(
    projs_vrc,           # (V,R,C) tensor: projections
    angles_deg,          # (V,) array: rotation angles  
    lamino_angle_deg,    # float: laminography angle
    mask=None,           # optional: view indices to use
    vol_init=None,       # optional: initialization (e.g., from FDK)
    max_epochs=30,       # number of optimization epochs
    batch_size=10,       # views per mini-batch
    lr=1e-3,             # learning rate
    optimizer_type="adam", # "adam" or "sgd"
    verbose=True,        # show progress bars
)
```

### `build_lamino_projector`

Creates a differentiable laminography projector for custom applications.

```python
projector = build_lamino_projector(
    vol_shape,           # (nx,ny,nz): volume dimensions
    det_shape,           # (rows,cols): detector dimensions  
    angles_deg,          # rotation angles
    lamino_angle_deg,    # laminography angle
    voxel_size_mm=1.0,   # voxel size
    det_spacing_mm=1.0,  # detector pixel spacing
)

# Use as: projections = projector(volume_tensor)
```

## Usage Examples

### Basic FBP Reconstruction

```python
import numpy as np
import torch
from astra_torch.lamino import fbp_reconstruction_masked

# Load your laminography projection data
projections = torch.load('lamino_projections.pt')  # (V,R,C)
angles = np.linspace(0, 360, projections.shape[0])
lamino_angle = 61.0

# Reconstruct
volume = fbp_reconstruction_masked(
    projections, angles, lamino_angle,
    voxel_per_mm=10
)
```

### Gradient Descent with FBP Initialization

```python
# First get FBP reconstruction
vol_fbp = fbp_reconstruction_masked(projections, angles, lamino_angle)

# Refine with gradient descent
vol_refined = gd_reconstruction_masked(
    projections, angles, lamino_angle,
    vol_init=vol_fbp,    # Initialize with FBP
    max_epochs=20,
    lr=5e-4,
    verbose=True
)
```

### Undersampled Reconstruction

```python
# Use only every 4th projection for faster/undersampled reconstruction
mask = np.arange(0, len(angles), 4)

vol_sparse = gd_reconstruction_masked(
    projections, angles, lamino_angle,
    mask=mask,           # Sparse view reconstruction
    max_epochs=50,       # More epochs for sparse data
    lr=1e-3
)
```

### Implementation Notes

### Geometry Parameterization

The laminography geometry is parameterized using ASTRA's `parallel3d_vec` format with 12 parameters per projection view:

- `[0-2]`: Ray direction (rx, ry, rz) - parallel for all detector pixels
- `[3-5]`: Detector center (dx, dy, dz) 
- `[6-8]`: Detector u-direction vector  
- `[9-11]`: Detector v-direction vector

**Key difference from cone-beam CT**: All rays are parallel (no point source), and ray direction is determined solely by the laminography angle.

### Distance Parameters

**Important**: In this parallel beam implementation, `source_detector_distance_mm` and `source_origin_distance_mm` parameters are **completely ignored**. They are kept in function signatures for API compatibility but have no effect on the reconstruction.

This is correct behavior for parallel beam geometry where all rays are parallel and distance parameters are meaningless.

### Memory and Performance

- All computations use `float32` precision for GPU efficiency
- The differentiable projector supports batching for memory efficiency
- Large reconstructions may require adjusting batch sizes to fit in GPU memory
- Progress bars help monitor long-running optimizations

### Comparison to CBCT Module

| Feature | CBCT | Laminography |
|---------|------|--------------|  
| Geometry | Circular cone-beam | Parallel beam (tilted) |
| Projector | `_AstraConeOp` | `_AstraLaminoOp` |
| Reconstruction | FDK algorithm | FBP algorithm |
| Key parameter | - | `lamino_angle_deg` |
| Distance params | Used (cone beam) | Ignored (parallel beam) |

## Requirements

- ASTRA toolbox with CUDA support
- PyTorch (preferably with CUDA)
- NumPy
- tqdm (for progress bars)

The module will automatically use CUDA if available, falling back to CPU computation otherwise.
