# Reset Optimizer Flag

## Overview

The `--reset_optimizer` flag allows you to load model weights from a checkpoint while starting with a fresh optimizer. This is useful for fine-tuning scenarios where you want to change the training dynamics.

## Usage

```bash
python sdate/training/compression/train_heic_to_tiff.py \
    --resume_from_checkpoint=outputs/checkpoint-2400 \
    --reset_optimizer \
    --learning_rate=1e-5 \
    ... other parameters ...
```

## Behavior Comparison

### Without `--reset_optimizer` (Normal Resume)

```bash
--resume_from_checkpoint=outputs/checkpoint-2400
```

**What gets loaded:**
- ✅ Model weights (UNet3D)
- ✅ Positional encoder
- ✅ Optimizer state (momentum, etc.)
- ✅ LR scheduler state
- ✅ Global step counter
- ✅ Epoch number

**Result:** Training continues exactly where it left off

---

### With `--reset_optimizer` (Fine-tuning Mode)

```bash
--resume_from_checkpoint=outputs/checkpoint-2400 \
--reset_optimizer
```

**What gets loaded:**
- ✅ Model weights (UNet3D)
- ✅ Positional encoder
- ❌ Optimizer state (fresh initialization)
- ❌ LR scheduler state (fresh initialization)
- ❌ Global step counter (starts at 0)
- ❌ Epoch number (starts at 0)

**Result:** Model weights loaded, but training starts fresh from epoch 0

## Use Cases

### 1. Fine-tuning with Lower Learning Rate

Train a model, then fine-tune with a lower LR:

```bash
# Initial training
python train.py \
    --learning_rate=1e-4 \
    --num_epochs=10

# Fine-tune with lower LR
python train.py \
    --resume_from_checkpoint=outputs/checkpoint-final \
    --reset_optimizer \
    --learning_rate=1e-5 \
    --num_epochs=5
```

### 2. Changing Optimizer Hyperparameters

Want to change warmup steps, weight decay, etc.:

```bash
python train.py \
    --resume_from_checkpoint=outputs/checkpoint-2400 \
    --reset_optimizer \
    --warmup_steps=500 \
    --num_epochs=10
```

### 3. Avoiding Stale Momentum

If previous training had different data distribution or batch size:

```bash
python train.py \
    --resume_from_checkpoint=outputs/checkpoint-2400 \
    --reset_optimizer \
    --batch_size=8  # Different from original
```

### 4. Transfer Learning Setup

Load pretrained weights but train from scratch:

```bash
python train.py \
    --resume_from_checkpoint=pretrained/checkpoint \
    --reset_optimizer \
    --data_path=/new/dataset \
    --learning_rate=1e-4
```

## Examples

### Example 1: Standard Fine-tuning

```bash
# Step 1: Train base model
python train.py \
    --data_path=/data/train \
    --output_dir=outputs/base \
    --learning_rate=1e-4 \
    --num_epochs=20

# Step 2: Fine-tune with lower LR
python train.py \
    --data_path=/data/train \
    --output_dir=outputs/finetuned \
    --resume_from_checkpoint=outputs/base/checkpoint-final \
    --reset_optimizer \
    --learning_rate=1e-5 \
    --num_epochs=10
```

### Example 2: Multi-stage Training

```bash
# Stage 1: Fast convergence with high LR
python train.py --learning_rate=1e-4 --num_epochs=10

# Stage 2: Refinement with medium LR
python train.py \
    --resume_from_checkpoint=outputs/checkpoint-4800 \
    --reset_optimizer \
    --learning_rate=5e-5 \
    --num_epochs=10

# Stage 3: Final polish with low LR
python train.py \
    --resume_from_checkpoint=outputs/checkpoint-9600 \
    --reset_optimizer \
    --learning_rate=1e-5 \
    --num_epochs=5
```

### Example 3: Experiment with Different Schedules

```bash
# Try different warmup schedules
python train.py \
    --resume_from_checkpoint=outputs/checkpoint-2400 \
    --reset_optimizer \
    --warmup_steps=1000 \
    --num_epochs=10
```

## Command-Line Flags

```bash
--resume_from_checkpoint PATH    Load model from checkpoint
--reset_optimizer                Reset optimizer (don't load optimizer state)
```

**Note**: `--reset_optimizer` only has effect when used with `--resume_from_checkpoint`

## Technical Details

### What Gets Reset

When `--reset_optimizer` is True:

1. **Optimizer**: New AdamW instance with fresh momentum buffers
2. **LR Scheduler**: New warmup + cosine schedule starting from beginning
3. **Global Step**: Reset to 0
4. **Start Epoch**: Reset to 0

### What Gets Kept

1. **Model Weights**: All UNet3D parameters
2. **Positional Encoder**: All embedding weights
3. **Model Architecture**: Exactly as saved in checkpoint

### Code Flow

```python
# In setup_optimizer()
if resume_from_checkpoint and not reset_optimizer:
    # Load optimizer state
    optimizer.load_state_dict(saved_state)
elif resume_from_checkpoint and reset_optimizer:
    # Skip loading, use fresh optimizer
    logger.info("Reset optimizer flag set: using fresh optimizer state")
```

## Performance Considerations

### When to Use `--reset_optimizer`

✅ **Use when:**
- Fine-tuning with significantly different learning rate
- Previous training was unstable or diverged
- Changing batch size significantly
- Applying to different data distribution
- Want to avoid momentum from previous training

❌ **Don't use when:**
- Simply continuing interrupted training
- Want to preserve learning dynamics
- Training for more epochs with same settings

### Impact on Training

**With reset optimizer:**
- Training may take longer to stabilize initially
- Learning rate starts from scratch (warmup phase)
- No momentum carried over from previous training
- More suitable for exploration/experimentation

**Without reset optimizer (normal resume):**
- Seamless continuation
- Preserves momentum and learning dynamics
- Faster initial progress
- Better for interrupted training scenarios

## Best Practices

1. **Document Your Experiments**: Track which checkpoints used reset optimizer

2. **Lower Learning Rate**: When resetting optimizer, often use lower LR than original training
   ```bash
   # Original: --learning_rate=1e-4
   # Reset:    --learning_rate=1e-5
   ```

3. **Shorter Training**: Fine-tuning with reset optimizer usually needs fewer epochs
   ```bash
   --num_epochs=5  # vs 20 for original training
   ```

4. **Test First**: Try both with and without reset to see which works better for your use case

5. **Monitor Metrics**: Watch validation loss closely at start of fine-tuning

## Troubleshooting

### Training Starts from Epoch 0

**Expected behavior** when using `--reset_optimizer`. The model weights are loaded but training counters are reset.

### Learning Rate Too High

If training is unstable after reset:
```bash
--learning_rate=1e-5  # Lower than original
```

### Validation Loss Increases

May happen initially due to fresh optimizer state. Give it a few epochs to stabilize.

### Checkpoint Not Found

Ensure you're using `--resume_from_checkpoint` with `--reset_optimizer`:
```bash
# ✓ Correct
--resume_from_checkpoint=outputs/checkpoint-2400 --reset_optimizer

# ✗ Incorrect (reset_optimizer has no effect without checkpoint)
--reset_optimizer
```

## See Also

- `scripts/finetune_with_reset_optimizer.sh` - Complete example
- `docs/checkpoint_resume.md` - Full checkpoint documentation
- `setup_optimizer()` in `train_heic_to_tiff.py` - Implementation details
