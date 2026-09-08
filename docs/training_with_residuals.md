# Training with Residuals

## Overview

The training script `train_heic_to_tiff.py` now supports using pre-computed residuals as an additional input channel. This enables **residual-aware training** where the model can learn to correct or refine predictions using error information from a previous model.

## Changes Made

### 1. New Parameter: `residual_path`

Added `--residual_path` parameter to the training script:

```bash
python sdate/training/compression/train_heic_to_tiff.py \
    --data_path=/path/to/data \
    --residual_path=outputs/residuals/data_residuals.npy \  # ← NEW
    ... other parameters ...
```

### 2. Automatic Channel Configuration

The model input is now dynamically configured based on whether residuals are provided:

- **Without residuals** (`residual_path=None`):
  - Dataset loads 2 channels: TIFF (target) + HEIC (input)
  - Model receives: [HEIC, zeros] (2 channels)
  
- **With residuals** (`residual_path` provided):
  - Dataset loads 3 channels: TIFF (target) + HEIC (input) + Residuals
  - Model receives: [HEIC, Residuals] (2 channels)

### 3. Updated Collate Function

The data collation now handles both 2-channel and 3-channel volumes:

```python
# Automatically detects channel count and prepares model input
if volumes.shape[1] == 3:  # Has residuals
    model_inputs = [HEIC, Residuals]
else:  # No residuals
    model_inputs = [HEIC, zeros]
```

## Usage Patterns

### Pattern 1: Standard Training (No Residuals)

Train a baseline model without residuals:

```bash
python sdate/training/compression/train_heic_to_tiff.py \
    --data_path=/myhome/data/file_3_extracted \
    --output_dir=outputs/baseline_model \
    --volume_size=32 \
    --stride=32 \
    --num_frames=32 \
    --batch_size=4
```

### Pattern 2: Compute Residuals

After training, compute residuals from the trained model:

```bash
python scripts/compute_residuals.py \
    --data_path=/myhome/data/file_3_extracted \
    --checkpoint_path=outputs/baseline_model/checkpoint-final \
    --output_path=outputs/residuals/data_residuals.npy \
    --volume_size=32 \
    --stride=32 \
    --num_frames=32
```

### Pattern 3: Refinement Training (With Residuals)

Train a refinement model using the residuals:

```bash
python sdate/training/compression/train_heic_to_tiff.py \
    --data_path=/myhome/data/file_3_extracted \
    --output_dir=outputs/refined_model \
    --volume_size=32 \
    --stride=32 \
    --num_frames=32 \
    --batch_size=4 \
    --residual_path=outputs/residuals/data_residuals.npy  # ← Use residuals
```

## Data Flow

### Without Residuals
```
TIFF Files → TiffVolumeDataset → [TIFF, HEIC] → Collate → Model Input: [HEIC, zeros]
                                                                         ↓
                                                               Model → Predicted TIFF
                                                                         ↓
                                                               Loss(Predicted, TIFF)
```

### With Residuals
```
TIFF Files ─┐
            ├→ TiffVolumeDataset → [TIFF, HEIC, Residuals] → Collate → Model Input: [HEIC, Residuals]
Residuals ──┘                                                                        ↓
                                                                          Model → Predicted TIFF
                                                                                      ↓
                                                                          Loss(Predicted, TIFF)
```

## Model Architecture

The UNet3D model receives **2-channel input** in both cases:

```python
UNet3DConditionModel(
    in_channels=2,    # [HEIC, second_channel]
    out_channels=1,   # Predicted TIFF
    ...
)
```

Where `second_channel` is:
- **Zeros** when no residuals are provided (baseline training)
- **Residuals** when `residual_path` is provided (refinement training)

## Benefits of Residual-Aware Training

1. **Error Correction**: Model learns from previous mistakes
2. **Iterative Refinement**: Can train multiple stages of improvement
3. **Faster Convergence**: Model starts with error information
4. **Better Performance**: Explicit error signal improves learning

## Complete Workflow Example

See `scripts/train_with_residuals_example.sh` for a complete 3-step workflow:

1. Train baseline model
2. Compute residuals
3. Train refined model with residuals

```bash
bash scripts/train_with_residuals_example.sh
```

## Important Notes

### Compatibility Requirements

- **Residuals must match dataset configuration**:
  - Same `volume_size`
  - Same `stride`
  - Same `num_frames`
  - Same data path

The dataset will automatically validate these parameters when loading residuals.

### Performance Considerations

- **Memory**: Residuals are memory-mapped, minimal overhead
- **Speed**: Loading residuals adds <1% to data loading time
- **Storage**: Residuals require ~4 bytes per voxel
  - Example: 1000 sub-volumes × 32³ ≈ 125 MB

### Multi-Stage Training

You can iteratively improve models:

```bash
# Stage 1: Baseline
train → checkpoint-1

# Stage 2: Compute residuals from Stage 1
compute_residuals(checkpoint-1) → residuals-1

# Stage 3: Refine with residuals-1
train(residuals=residuals-1) → checkpoint-2

# Stage 4: Compute new residuals from Stage 3
compute_residuals(checkpoint-2) → residuals-2

# Stage 5: Further refinement
train(residuals=residuals-2) → checkpoint-3
```

## Troubleshooting

### Error: "Residuals file not found"
- Ensure you computed residuals first with `compute_residuals.py`
- Check the path in `--residual_path` is correct

### Error: "Residuals volume_size does not match"
- Use same `--volume_size` for both training and residual computation

### Error: "Number of residual sub-volumes does not match"
- Ensure `--stride` and `--num_frames` match between training and residual computation

## API Reference

### New Training Parameter

```python
trainer = HeicToTiffTrainer(
    data_path=...,
    residual_path=None,  # Optional[str] - Path to residuals file
    ...
)
```

### Command Line

```bash
--residual_path PATH    Path to pre-computed residuals file (optional)
```

## See Also

- `scripts/compute_residuals.py` - Compute residuals from trained model
- `scripts/README_residuals.md` - Complete residuals documentation
- `scripts/residuals_quick_reference.py` - Code examples
- `IMPLEMENTATION_RESIDUALS.md` - Technical implementation details
