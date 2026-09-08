# Checkpoint Resume Training

## Overview

The training script now supports resuming from checkpoints, allowing you to:
- **Continue interrupted training** (power loss, timeout, etc.)
- **Train for additional epochs** without starting from scratch
- **Fine-tune** a pretrained model with different hyperparameters

## Usage

### Basic Resume

```bash
python sdate/training/compression/train_heic_to_tiff.py \
    --data_path=/path/to/data \
    --output_dir=outputs/training \
    --resume_from_checkpoint=outputs/training/checkpoint-2400 \
    ... other parameters ...
```

### Command-Line Argument

```bash
--resume_from_checkpoint PATH    Path to checkpoint directory to resume from
```

## What Gets Restored

When resuming from a checkpoint, the following are loaded:

1. **Model Weights** (`unet/`)
   - Full UNet3D architecture and parameters
   - All trained weights from the checkpoint

2. **Positional Encoder** (`positional_encoder.pth`)
   - Positional encoding model state

3. **Optimizer State** (`training_state.pth`)
   - Adam optimizer momentum and state
   - Ensures smooth continuation of training

4. **Learning Rate Scheduler** (`training_state.pth`)
   - Current position in LR schedule
   - Warmup/cosine annealing state

5. **Training Progress** (`training_state.pth`)
   - Global step counter
   - Current epoch number

## Checkpoint Structure

Each checkpoint saved by the script has this structure:

```
checkpoint-{step}/
├── unet/                          # UNet3D model (Diffusers format)
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── positional_encoder.pth         # Positional encoder weights
└── training_state.pth             # Optimizer, scheduler, step info
```

## Examples

### Example 1: Resume After Interruption

Your training was interrupted at step 2400:

```bash
# Original training command
python train_heic_to_tiff.py \
    --data_path=/data/file_3 \
    --output_dir=outputs/run1 \
    --num_epochs=20

# Resume from where it stopped
python train_heic_to_tiff.py \
    --data_path=/data/file_3 \
    --output_dir=outputs/run1 \
    --resume_from_checkpoint=outputs/run1/checkpoint-2400 \
    --num_epochs=20  # Will continue to complete 20 epochs
```

### Example 2: Train for More Epochs

You trained for 10 epochs, now want 10 more:

```bash
# Initial training (10 epochs)
python train_heic_to_tiff.py \
    --num_epochs=10 \
    --output_dir=outputs/initial

# Continue for 10 more epochs (total 20)
python train_heic_to_tiff.py \
    --num_epochs=20 \
    --output_dir=outputs/extended \
    --resume_from_checkpoint=outputs/initial/checkpoint-final
```

### Example 3: Fine-tune with Different Settings

Start from a checkpoint but change hyperparameters:

```bash
# Resume but with lower learning rate
python train_heic_to_tiff.py \
    --resume_from_checkpoint=outputs/run1/checkpoint-2400 \
    --learning_rate=1e-5  # Lower LR for fine-tuning
    --num_epochs=25
```

**Note**: Changing `batch_size`, `volume_size`, or data-related parameters when resuming is not recommended as it may cause issues.

### Example 4: Resume with Residuals

If your checkpoint was trained without residuals, you can resume with them:

```bash
python train_heic_to_tiff.py \
    --resume_from_checkpoint=outputs/baseline/checkpoint-2400 \
    --residual_path=outputs/residuals/data_residuals.npy \
    --num_epochs=20
```

## How It Works

### Step Calculation

The script automatically calculates which epoch and step to resume from:

```python
global_step = loaded_from_checkpoint  # e.g., 2400
steps_per_epoch = len(train_dataloader)  # e.g., 480
start_epoch = global_step // steps_per_epoch  # e.g., 5

# Training continues from epoch 5, step 2400
```

### Epoch Range

The training loop runs from `start_epoch` to `num_epochs`:

```python
for epoch in range(start_epoch, num_epochs):
    # Continue training
```

This means:
- If you resumed at epoch 5 with `--num_epochs=20`
- Training will run epochs 5, 6, 7, ..., 19 (15 more epochs)

## Important Notes

### Checkpoint Path

The checkpoint path should point to a **directory** containing the checkpoint files:

```bash
# ✓ Correct
--resume_from_checkpoint=outputs/training/checkpoint-2400

# ✗ Incorrect
--resume_from_checkpoint=outputs/training/checkpoint-2400/unet
--resume_from_checkpoint=outputs/training/checkpoint-2400/training_state.pth
```

### Checkpoint Compatibility

**Must Match**:
- Model architecture (automatically handled)
- Volume size (should be same)
- Data preprocessing (normalization, etc.)

**Can Change**:
- Learning rate
- Number of epochs
- Batch size (may affect LR schedule)
- Logging/saving frequency

### Missing Components

If some checkpoint files are missing:

- **No `unet/`**: Training will fail with error
- **No `positional_encoder.pth`**: Warning logged, new encoder initialized
- **No `training_state.pth`**: Warning logged, optimizer starts fresh (not recommended)

### Output Directory

You can use:
- **Same output dir**: New checkpoints will be saved alongside old ones
- **Different output dir**: Creates a new directory for continued training

```bash
# Same directory
--output_dir=outputs/run1 \
--resume_from_checkpoint=outputs/run1/checkpoint-2400

# Different directory  
--output_dir=outputs/run1_continued \
--resume_from_checkpoint=outputs/run1/checkpoint-2400
```

## Troubleshooting

### Error: "Checkpoint path provided but not found"

**Cause**: The path doesn't exist

**Solution**: Check the path and ensure checkpoint was saved
```bash
ls -la outputs/training/checkpoint-2400
```

### Error: "UNet checkpoint not found"

**Cause**: The checkpoint directory doesn't contain `unet/` subfolder

**Solution**: Ensure you're pointing to the correct checkpoint directory, not a subdirectory

### Warning: "Training state not found, starting optimizer from scratch"

**Cause**: `training_state.pth` is missing

**Impact**: Optimizer state and step counter are lost, but model weights are loaded

**Solution**: 
- If acceptable, training will continue (slightly suboptimal)
- Otherwise, use a different checkpoint or retrain

### Training Restarts from Epoch 0

**Cause**: Either:
- `training_state.pth` is missing
- Global step is 0 in the checkpoint

**Solution**: Check that you're using a checkpoint saved during training, not just model weights

## Best Practices

1. **Regular Checkpoints**: Save checkpoints frequently (e.g., every 500 steps)
   ```bash
   --save_steps=500
   ```

2. **Keep Multiple Checkpoints**: Don't delete old checkpoints immediately
   - Useful if latest checkpoint is corrupted
   - Can resume from earlier point if needed

3. **Verify Checkpoint**: Before long training runs, verify checkpoint loads:
   ```bash
   # Quick test: train for 1 step from checkpoint
   python train.py --resume_from_checkpoint=... --num_epochs=1 --max_steps=1
   ```

4. **Match Configurations**: When resuming, use the same data and model configuration

5. **Log Resume**: The script logs when resuming:
   ```
   Loading model from checkpoint: outputs/training/checkpoint-2400
   Resuming from step 2400, epoch 5
   ```

## See Also

- `scripts/resume_training_example.sh` - Complete example script
- `save_checkpoint()` method in `train_heic_to_tiff.py` - How checkpoints are saved
- `setup_model()` and `setup_optimizer()` - Where checkpoint loading happens
