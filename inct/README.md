# INCT: Instant Neural Compression for Tomography

A PyTorch library implementing Instant Neural Graphics Primitives (NGP) for compressing tomographic projection data.

Based on: ["Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"](https://arxiv.org/abs/2201.05989)

## Overview

INCT learns a continuous function that maps coordinates to intensity values using:
- **Multi-resolution hash encoding**: Captures features at multiple scales using hash tables
- **Small MLP decoder**: Maps encoded features to output values
- **Coordinate-based learning**: Learns f(x, y, z) → intensity for 3D projection volumes

## Installation

```bash
# From the sdate project root
pip install -e .

# Or add to your Python path
import sys
sys.path.insert(0, '/path/to/sdate')
```

## Quick Start

```python
import torch
from inct import InstantNGPModel, BatchVoxelDataset, Trainer, TrainingConfig

# Create or load your volume (H, W, N_projections)
volume = torch.rand(256, 256, 10)  # Example: 10 projections of 256x256

# Create dataset
dataset = BatchVoxelDataset(
    volume,
    batch_size=65536,  # Voxels per batch
    n_batches=500,     # Batches per epoch
    normalize_values=True,
)

# Create model
model = InstantNGPModel(
    n_dims=3,                  # 3D coordinates (height, width, projection)
    n_levels=16,               # Resolution levels
    n_features_per_level=2,    # Features per level
    base_resolution=16,        # Coarsest resolution
    max_resolution=512,        # Finest resolution
    table_size=2**19,          # Hash table size
    hidden_dims=[64, 64],      # MLP architecture
)

# Train
config = TrainingConfig(
    learning_rate=1e-2,
    num_epochs=100,
    checkpoint_dir='checkpoints_inct',
)

trainer = Trainer(model, config)
trainer.train(train_loader, val_loader)

# Reconstruct volume
reconstructed = model.predict_volume(volume.shape)
```

## Components

### Hash Encoding (`hash_encoding.py`)

- `HashEncoding`: Single-level spatial hash encoding
- `MultiResolutionHashEncoding`: Multi-level encoding with geometric resolution progression

### Model (`model.py`)

- `TinyMLP`: Small feedforward network
- `InstantNGPModel`: Complete model combining hash encoding and MLP

### Dataset (`dataset.py`)

- `VoxelDataset`: Basic dataset returning (coord, value) pairs
- `BatchVoxelDataset`: Efficient batch sampling
- `ProjectionVolumeDataset`: Load from TomographyFolderProcessor

### Training (`trainer.py`)

- `TrainingConfig`: Configuration dataclass
- `Trainer`: Training loop with AdamW, LR scheduling, checkpointing

## Training Script

```bash
# Quick test
python scripts/train_inct.py \
    --data_path /path/to/data \
    --num_projections 5 \
    --target_size 256 \
    --num_epochs 50

# Full training
python scripts/train_inct.py \
    --data_path /path/to/data \
    --num_projections 100 \
    --target_size 512 \
    --n_levels 16 \
    --table_size 524288 \
    --num_epochs 200
```

## Demo Notebook

See `notebooks/INCT_Demo.ipynb` for a complete walkthrough including:
- Hash encoding visualization
- Loading tomographic data
- Training the model
- Evaluating reconstruction quality
- Saving and loading models

## Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `n_levels` | Number of resolution levels | 8-16 |
| `n_features_per_level` | Features per hash entry | 2 |
| `base_resolution` | Coarsest resolution | 16 |
| `max_resolution` | Finest resolution | 256-1024 |
| `table_size` | Hash table entries per level | 2^14 - 2^20 |
| `hidden_dims` | MLP hidden layers | [64, 64] or [64, 64, 64] |

## Compression vs Quality Trade-off

- **Higher table_size**: Better quality, larger model
- **More n_levels**: Better multi-scale features
- **Higher max_resolution**: Better fine detail
- **Deeper MLP**: More capacity (but hash tables dominate)

## References

- [Instant Neural Graphics Primitives](https://nvlabs.github.io/instant-ngp/)
- [Original Paper](https://arxiv.org/abs/2201.05989)
