# HEIC to TIFF Translation Training

This module provides training scripts for learning HEIC-to-TIFF translation using a 3D UNet model from the diffusers library.

## Overview

The training pipeline:
1. Loads TIFF sequences and creates HEIC compressed versions
2. Uses dual-channel TiffVolumeDataset (TIFF + HEIC)
3. Trains UNet3D to translate HEIC → TIFF
4. Uses positional encodings as conditioning information
5. Supports mixed precision training and gradient accumulation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_heic_training.txt
```

### 2. Simple Training Run

```bash
cd /myhome/sdate
python scripts/run_heic_to_tiff_training.py
```

### 3. Command Line Training

```bash
python -m sdate.training.compression.train_heic_to_tiff \
    --data_path /path/to/tiff/data \
    --output_dir ./outputs \
    --volume_size 64 \
    --batch_size 2 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --use_wandb
```

## Configuration

### Key Parameters

- **`volume_size`**: Size of cubic sub-volumes (e.g., 64 → 64×64×64)
- **`stride`**: Overlap between volumes (stride=32 with size=64 → 50% overlap)
- **`batch_size`**: Keep small (1-4) for 3D volumes due to memory
- **`heic_quality`**: HEIC compression quality (0-100)
- **`mixed_precision`**: Use "fp16" or "bf16" to save memory
- **`gradient_accumulation_steps`**: Increase to simulate larger batches

### Model Architecture

The training uses `UNet3DConditionModel` with:
- **Input**: 1-channel HEIC volumes (compressed)
- **Output**: 1-channel TIFF volumes (target quality)
- **Conditioning**: Positional encodings from volume coordinates
- **Attention**: Cross-attention layers for spatial awareness

### Data Flow

```
TIFF Files → HEIC Compression → Dual-Channel Dataset
                                      ↓
                  HEIC Input + Position → UNet3D → TIFF Output
                                      ↓
                              Loss(Predicted, Target)
```

## Training Features

### Multi-GPU Support
Uses Accelerate for easy multi-GPU training:
```bash
accelerate config  # Setup once
accelerate launch python -m sdate.training.compression.train_heic_to_tiff --args...
```

### Memory Optimization
- Mixed precision training (fp16/bf16)
- Gradient accumulation for larger effective batch sizes
- Gradient clipping for stability
- Efficient data loading with multiple workers

### Monitoring
- Progress bars with loss tracking
- Configurable logging intervals
- Weights & Biases integration
- Regular validation evaluation
- Automatic checkpointing

### Loss Function
Combined loss with:
- **L1 Loss**: Preserves overall structure
- **L2 Loss**: Smooth pixel-wise reconstruction
- Configurable weights for each component

## File Structure

```
sdate/training/compression/
├── __init__.py                    # Module exports
├── train_heic_to_tiff.py         # Main training script
└── README.md                     # This file

scripts/
├── run_heic_to_tiff_training.py  # Simple runner script

configs/
├── heic_to_tiff_training.yaml    # Configuration template

requirements_heic_training.txt     # Dependencies
```

## Usage Examples

### Basic Training
```python
from sdate.training.compression import HeicToTiffTrainer

trainer = HeicToTiffTrainer(
    data_path="/path/to/tiff/data",
    output_dir="./outputs",
    volume_size=64,
    batch_size=2,
    num_epochs=50
)
trainer.train()
```

### Custom Configuration
```python
trainer = HeicToTiffTrainer(
    data_path="/path/to/tiff/data",
    output_dir="./outputs",
    
    # Volume parameters
    volume_size=32,        # Smaller volumes for faster training
    stride=16,             # More overlap
    num_frames=100,        # Subset of data
    
    # Training parameters
    batch_size=4,          # Larger batch with smaller volumes
    learning_rate=2e-4,    # Higher learning rate
    num_epochs=100,        # More epochs
    
    # HEIC parameters
    heic_quality=75,       # Lower quality for more compression
    
    # System parameters
    mixed_precision="bf16", # Use bfloat16 if available
    gradient_accumulation_steps=4,
    
    # Logging
    use_wandb=True,
    wandb_project="my-heic-project"
)
```

### Loading Checkpoints
```python
# TODO: Add checkpoint loading functionality
# The trainer saves checkpoints that can be resumed
```

## Performance Tips

1. **Memory Management**:
   - Start with `volume_size=32` and `batch_size=1`
   - Increase gradually based on available GPU memory
   - Use `mixed_precision="fp16"` or `"bf16"`

2. **Training Speed**:
   - Use multiple workers for data loading (`max_workers=4-8`)
   - Enable gradient accumulation for larger effective batches
   - Consider using smaller sub-volumes for faster iteration

3. **Data Efficiency**:
   - Use appropriate `stride` for overlap (32 with size=64 is good)
   - Start with subset of frames (`num_frames=100-200`)
   - Balance HEIC quality vs. compression ratio

4. **Monitoring**:
   - Watch validation loss to avoid overfitting
   - Monitor gradient norms for training stability
   - Use W&B for experiment tracking

## Expected Results

The model learns to:
- Reduce HEIC compression artifacts
- Restore fine details lost in compression
- Maintain spatial consistency across the volume
- Preserve quantitative accuracy for scientific data

Training typically converges within 20-50 epochs depending on:
- Data complexity
- Volume size
- HEIC compression quality
- Model capacity

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce `batch_size` and `volume_size`
   - Enable mixed precision
   - Increase gradient accumulation steps

2. **Slow Training**:
   - Increase `max_workers`
   - Use smaller volumes initially
   - Check data loading bottlenecks

3. **Poor Convergence**:
   - Adjust learning rate
   - Check data normalization
   - Verify HEIC quality settings

4. **Import Errors**:
   - Install all requirements
   - Check Python path setup
   - Verify diffusers version compatibility

### Getting Help

Check the logs for detailed error messages and training progress. The training script includes comprehensive logging and error handling.