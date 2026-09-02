# ASTRA-Torch

GPU-accelerated tomographic reconstruction library with PyTorch integration.

ASTRA-Torch provides PyTorch-compatible implementations of tomographic reconstruction algorithms using the ASTRA toolbox, with support for both cone-beam CT (CBCT) and laminography.

## Features

- **CBCT Reconstruction**: FDK and gradient descent algorithms for cone-beam CT
- **Laminography Reconstruction**: FBP, SIRT, and gradient descent algorithms for parallel-beam laminography
- **2D Parallel Beam**: FBP, SIRT, and gradient descent algorithms for 2D parallel beam tomography
- **PyTorch Integration**: Differentiable operators with autograd support
- **GPU Acceleration**: CUDA-accelerated reconstruction via ASTRA toolbox
- **Flexible Data Loading**: Built-in support for Walnut dataset format
- **Optimization-Based Methods**: Gradient descent reconstruction with customizable loss functions

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- ASTRA Toolbox compiled with CUDA support

### Install from PyPI (when available)

```bash
pip install astra-torch
```

### Install from source

```bash
git clone https://github.com/chip-project/astra-torch.git
cd astra-torch
pip install -e .
```

### Development installation

```bash
git clone https://github.com/chip-project/astra-torch.git
cd astra-torch
pip install -e ".[dev,notebooks]"
```

## Quick Start

### CBCT Reconstruction

```python
import astra_torch
import torch

# Load Walnut dataset
projs, vecs, meta = astra_torch.load_walnut_projections(
    '/path/to/walnut/data', 
    walnut_id=1,
    orbit_id=2
)

# FDK reconstruction
volume_fdk = astra_torch.fdk_reconstruction_masked(
    projs, vecs, 
    voxel_per_mm=10
)

# Gradient descent reconstruction (sparse views)
mask = torch.arange(0, len(vecs), 4)  # Every 4th view
volume_gd = astra_torch.gd_reconstruction_masked(
    projs, vecs,
    mask=mask,
    max_epochs=30,
    lr=1e-3
)
```

### Laminography Reconstruction

```python
from astra_torch import build_lamino_projector, lamino_fbp_reconstruction_masked

# Build laminography projector  
projector = build_lamino_projector(
    vol_shape=(256, 512, 512),
    det_shape=(501, 501), 
    angles_deg=np.linspace(0, 360, 360),
    lamino_angle_deg=61.0
)

# Forward projection
projections = projector(volume_tensor)

# FBP reconstruction
reconstructed = lamino_fbp_reconstruction_masked(
    projections,
    angles_deg=np.linspace(0, 360, 360),
    lamino_angle_deg=61.0
)
```

### 2D Parallel Beam Reconstruction

```python
from astra_torch import parallel2d_fbp_reconstruction_masked, build_parallel2d_projector
import numpy as np

# Generate projections
angles_deg = np.linspace(0, 180, 180, endpoint=False)
projector = build_parallel2d_projector(
    vol_shape=(256, 256),
    det_cols=384,
    angles_deg=angles_deg
)

# Forward projection
projections = projector(volume_2d)  # volume_2d shape: (1, 1, H, W)

# FBP reconstruction
reconstructed = parallel2d_fbp_reconstruction_masked(
    projs_vc=projections[0],  # Shape: (V, C)
    angles_deg=angles_deg,
    vol_shape=(256, 256),
    filter_type='hann'
)

# SIRT reconstruction (more robust to noise)
from astra_torch import parallel2d_sirt_reconstruction_masked
reconstructed_sirt = parallel2d_sirt_reconstruction_masked(
    projs_vc=projections[0],
    angles_deg=angles_deg,
    vol_shape=(256, 256),
    num_iterations=100,
    min_constraint=0.0  # Non-negativity
)
```

## Documentation

### CBCT Functions

- `load_walnut_projections()`: Load Walnut dataset projections
- `fdk_reconstruction_masked()`: FDK reconstruction with view masking
- `gd_reconstruction_masked()`: Gradient descent reconstruction  
- `build_conebeam_projector()`: Create differentiable cone-beam projector

### Laminography Functions

- `lamino_fbp_reconstruction_masked()`: FBP reconstruction for laminography
- `lamino_sirt_reconstruction_masked()`: SIRT reconstruction for laminography
- `lamino_gd_reconstruction_masked()`: Gradient descent laminography reconstruction
- `build_lamino_projector()`: Create differentiable laminography projector

### 2D Parallel Beam Functions

- `parallel2d_fbp_reconstruction_masked()`: FBP reconstruction for 2D parallel beam
- `parallel2d_sirt_reconstruction_masked()`: SIRT reconstruction for 2D parallel beam
- `parallel2d_gd_reconstruction_masked()`: Gradient descent reconstruction for 2D parallel beam
- `build_parallel2d_projector()`: Create differentiable 2D parallel beam projector

## Requirements

- `torch>=1.9.0` (with `__cuda_array_interface__` support for CUDA tensors)
- `numpy>=1.21.0`
- `astra-toolbox>=2.0.0` (compiled with CUDA support)
- `tqdm>=4.60.0`
- `h5py>=3.0.0`
- `scipy>=1.7.0`

**Note:** PyTorch 1.7+ CUDA tensors work directly with ASTRA via `__cuda_array_interface__`, 
enabling GPU-only operations without CPU memory overhead.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `pytest`
5. Format code: `black astra_torch/`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use ASTRA-Torch in your research, please cite:

```bibtex
@software{astra_torch,
  title={ASTRA-Torch: GPU-accelerated tomographic reconstruction with PyTorch},
  author={CHIP Project},
  year={2024},
  url={https://github.com/chip-project/astra-torch}
}
```

## Acknowledgments

- Built on top of the excellent [ASTRA Toolbox](https://www.astra-toolbox.com/)
- Inspired by the PyTorch ecosystem for differentiable programming
