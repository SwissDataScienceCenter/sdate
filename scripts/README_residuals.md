# Computing and Using Residuals with TiffVolumeDataset

This guide explains how to compute residuals from a trained HEIC-to-TIFF model and use them as an additional channel in the TiffVolumeDataset.

## Overview

Residuals are the difference between model predictions and ground truth targets:
```
residual = predicted_tiff - target_tiff
```

These residuals can be useful for:
- Understanding model errors and biases
- Training error-correction models
- Analyzing spatial patterns in prediction quality
- Creating residual-aware training pipelines

## Step 1: Train Your Model

First, train a HEIC-to-TIFF model using the standard training script:

```bash
python scripts/run_heic_to_tiff_training.py \
    --data_path=/path/to/tiff/data \
    --volume_size=64 \
    --stride=64 \
    --num_frames=100 \
    --output_dir=outputs/heic_to_tiff
```

## Step 2: Compute Residuals

Once training is complete, compute residuals using the trained model:

```bash
python scripts/compute_residuals.py \
    --data_path=/path/to/tiff/data \
    --checkpoint_path=outputs/heic_to_tiff/checkpoint-1000 \
    --output_path=outputs/residuals/data_residuals.npy \
    --volume_size=64 \
    --stride=64 \
    --num_frames=100 \
    --heic_quality=85 \
    --batch_size=4
```

This will create three files:
- `data_residuals.npy`: Memory-mapped residuals array (num_subvolumes, 64, 64, 64)
- `data_positions.npy`: Position indices for each sub-volume (num_subvolumes, 3)
- `data_metadata.npz`: Metadata including volume_size, stride, num_frames, etc.

### Arguments

- `--data_path`: Path to the TIFF data directory (same as training)
- `--checkpoint_path`: Path to trained model checkpoint
- `--output_path`: Where to save residuals (.npy file)
- `--volume_size`: Sub-volume size (must match training)
- `--stride`: Stride for sub-volume extraction (must match training)
- `--num_frames`: Number of frames to process
- `--heic_quality`: HEIC compression quality (must match training)
- `--batch_size`: Batch size for inference (default: 4)
- `--device`: Device to use (default: auto-detect cuda/cpu)

## Step 3: Use Residuals in Dataset

Load the dataset with residuals as a third channel:

```python
from sdate.datasets import TiffVolumeDataset

# Create dataset with residuals
dataset = TiffVolumeDataset(
    data_path='/path/to/tiff/data',
    volume_size=64,
    stride=64,
    num_frames=100,
    use_heic_compression=True,
    heic_quality=85,
    dual_channel=True,
    use_residuals=True,  # Enable residuals
    residuals_path='outputs/residuals/data_residuals.npy',
    normalize=True,
    global_normalize=True,
)

# Get a sample with residuals
sub_volume, position = dataset[0]
print(f"Sub-volume shape: {sub_volume.shape}")  # (3, 64, 64, 64)
print(f"Channels: TIFF, HEIC, RESIDUAL")

# Access individual channels
tiff_channel = sub_volume[0]    # (64, 64, 64)
heic_channel = sub_volume[1]    # (64, 64, 64)
residual_channel = sub_volume[2]  # (64, 64, 64)
```

### Channel Order

When `use_residuals=True`, the sub-volumes have 3 channels:
1. **Channel 0**: Target TIFF (ground truth)
2. **Channel 1**: HEIC compressed input
3. **Channel 2**: Residuals (predicted - target)

## Example: Training with Residuals

You can use residuals to train error-correction models:

```python
from torch.utils.data import DataLoader

# Load dataset with residuals
dataset = TiffVolumeDataset(
    data_path='/path/to/data',
    volume_size=64,
    use_heic_compression=True,
    dual_channel=True,
    use_residuals=True,
    residuals_path='residuals.npy',
)

# Create dataloader
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Training loop
for batch_volumes, batch_positions in loader:
    # batch_volumes shape: (batch_size, 3, 64, 64, 64)
    
    tiff = batch_volumes[:, 0:1]       # Ground truth
    heic = batch_volumes[:, 1:2]       # Compressed input
    residuals = batch_volumes[:, 2:3]  # Model residuals
    
    # Train a residual correction model
    # corrected = heic + model(heic, residuals)
    # loss = criterion(corrected, tiff)
```

## Requirements

The residuals functionality requires:
- `use_residuals=True`: Enable residual loading
- `dual_channel=True`: Must have both TIFF and HEIC channels
- `residuals_path`: Path to the residuals .npy file

The dataset will automatically validate:
- Residuals file exists
- Volume size matches
- Number of frames matches
- Number of sub-volumes matches

## Performance

Residuals are loaded using memory-mapped arrays (`mmap_mode='r'`), which:
- Doesn't load entire array into RAM
- Loads sub-volumes on-demand as accessed
- Efficient for large datasets
- Safe for multi-process data loading

## File Format

The residuals are saved in NumPy format with three files:

### `*_residuals.npy`
- Memory-mapped float32 array
- Shape: (num_subvolumes, volume_size, volume_size, volume_size)
- Contains: predicted_tiff - target_tiff for each sub-volume

### `*_positions.npy`
- int32 array
- Shape: (num_subvolumes, 3)
- Contains: (d_start, h_start, w_start) for each sub-volume

### `*_metadata.npz`
- Compressed NumPy archive
- Contains: volume_size, stride, num_frames, heic_quality, data_path, checkpoint_path, etc.

## Troubleshooting

### Error: "residuals_path must be provided"
You set `use_residuals=True` but didn't provide `residuals_path`. Add the path:
```python
residuals_path='path/to/residuals.npy'
```

### Error: "use_residuals=True requires dual_channel=True"
Residuals require both TIFF and HEIC channels. Set:
```python
use_heic_compression=True
dual_channel=True
```

### Error: "Residuals volume_size does not match"
The residuals were computed with different `volume_size`. Recompute residuals with matching parameters.

### Error: "Number of residual sub-volumes does not match"
The stride or data changed. Recompute residuals with current dataset configuration.

## Complete Example

```bash
# 1. Train model
python scripts/run_heic_to_tiff_training.py \
    --data_path=/myhome/data/file_3_extracted \
    --volume_size=64 \
    --stride=64 \
    --num_frames=100 \
    --output_dir=outputs/heic_to_tiff

# 2. Compute residuals
python scripts/compute_residuals.py \
    --data_path=/myhome/data/file_3_extracted \
    --checkpoint_path=outputs/heic_to_tiff/checkpoint-final \
    --output_path=outputs/residuals/file_3_residuals.npy \
    --volume_size=64 \
    --stride=64 \
    --num_frames=100

# 3. Use in Python
python -c "
from sdate.datasets import TiffVolumeDataset

dataset = TiffVolumeDataset(
    data_path='/myhome/data/file_3_extracted',
    volume_size=64,
    stride=64,
    num_frames=100,
    use_heic_compression=True,
    dual_channel=True,
    use_residuals=True,
    residuals_path='outputs/residuals/file_3_residuals.npy',
)

print(f'Dataset: {len(dataset)} sub-volumes')
sub_volume, pos = dataset[0]
print(f'Shape: {sub_volume.shape}')  # (3, 64, 64, 64)
print(f'Channels: TIFF, HEIC, RESIDUAL')
"
```
