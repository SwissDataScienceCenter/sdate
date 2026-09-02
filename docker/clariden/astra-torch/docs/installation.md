# Installation

## Prerequisites

Before installing ASTRA-Torch, ensure you have:

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for best performance)
- ASTRA Toolbox compiled with CUDA support

## ASTRA Toolbox Installation

The ASTRA Toolbox is a required dependency. Install it using conda:

```bash
conda install -c astra-toolbox astra-toolbox
```

Or follow the official [ASTRA installation guide](https://github.com/astra-toolbox/astra-toolbox) for other installation methods.

## Installing ASTRA-Torch

### From PyPI (Recommended)

```bash
pip install astra-torch
```

### From Source

```bash
git clone https://github.com/chip-project/astra-torch.git
cd astra-torch
pip install -e .
```

### Development Installation

For development work, install with additional dependencies:

```bash
git clone https://github.com/chip-project/astra-torch.git
cd astra-torch
pip install -e ".[dev,notebooks]"
```

This installs:
- Testing dependencies (pytest, coverage)
- Code quality tools (black, flake8, mypy)
- Jupyter notebook support

## Verifying Installation

Test your installation:

```python
import astra_torch
import torch

# Check CUDA availability
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

# Check ASTRA import
try:
    import astra
    print("ASTRA Toolbox successfully imported")
except ImportError:
    print("ASTRA Toolbox not found - install with conda")

# Test basic functionality
print(f"ASTRA-Torch version: {astra_torch.__version__}")
```

## Docker Installation

A Docker image with all dependencies is also available:

```bash
docker pull lfbarba/sdsc_image:1.0.0
docker run --gpus all -it lfbarba/sdsc_image:1.0.0
```

## Troubleshooting

### CUDA Issues

If you encounter CUDA-related errors:

1. Verify CUDA installation: `nvidia-smi`
2. Check PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
3. Ensure ASTRA is compiled with CUDA support

### ASTRA Import Errors

If ASTRA fails to import:

1. Install via conda: `conda install -c astra-toolbox astra-toolbox`
2. Check your PATH and LD_LIBRARY_PATH environment variables
3. Try installing in a fresh conda environment
