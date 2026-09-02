# API Reference

```{toctree}
:maxdepth: 2

cbct
lamino
```

## Overview

The ASTRA-Torch API is organized into two main modules:

- **cbct**: Cone-beam CT reconstruction functions and classes
- **lamino**: Laminography reconstruction functions and classes

Both modules provide:
- Forward projection operators
- Reconstruction algorithms (analytical and iterative)
- Acquisition geometry handling
- PyTorch integration with automatic differentiation

## Quick Function Index

### CBCT Functions

```{autosummary}
astra_torch.cbct.fdk_reconstruction_masked
astra_torch.cbct.gd_reconstruction_masked
astra_torch.cbct.project_conebeam_masked
astra_torch.cbct.build_conebeam_projector
astra_torch.cbct.CBCTAcquisition
```

### Laminography Functions

```{autosummary}
astra_torch.lamino.fbp_reconstruction_masked
astra_torch.lamino.gd_reconstruction_masked
astra_torch.lamino.build_lamino_projector
astra_torch.lamino.LaminoAcquisition
```
