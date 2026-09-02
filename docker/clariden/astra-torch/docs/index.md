# ASTRA-Torch Documentation

Welcome to the ASTRA-Torch documentation! This library provides GPU-accelerated tomographic reconstruction algorithms with PyTorch integration.

```{toctree}
:maxdepth: 2
:caption: Contents:

installation
quickstart
api/index
examples/index
contributing
```

## Features

- **CBCT Reconstruction**: FDK and gradient descent algorithms for cone-beam CT
- **Laminography Reconstruction**: FBP, SIRT, and gradient descent algorithms for parallel-beam laminography
- **2D Parallel Beam**: FBP, SIRT, and gradient descent algorithms for 2D parallel beam tomography
- **PyTorch Integration**: Differentiable operators with autograd support
- **GPU Acceleration**: CUDA-accelerated reconstruction via ASTRA toolbox
- **Flexible Data Loading**: Built-in support for Walnut dataset format
- **Optimization-Based Methods**: Gradient descent reconstruction with customizable loss functions

## Quick Links

- [Installation Guide](installation.md)
- [Quick Start Tutorial](quickstart.md)
- [API Reference](api/index.md)
- [Examples](examples/index.md)

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
