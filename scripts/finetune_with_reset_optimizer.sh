#!/bin/bash
# Example: Fine-tuning with reset optimizer
#
# This demonstrates how to load model weights from a checkpoint
# but start with a fresh optimizer. Useful for fine-tuning with
# different learning rates or optimizer parameters.

DATA_PATH="/myhome/data/sdate/shared/compression_paper/file_3_extracted"
CHECKPOINT="/myhome/data/sdate/shared/compression_paper/checkpoint-2400"
OUTPUT_DIR="outputs/finetuned_model"

echo "===================================================================="
echo "Fine-tuning: Load model weights but reset optimizer"
echo "===================================================================="
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo ""
echo "What happens:"
echo "  ✓ Model weights loaded from checkpoint"
echo "  ✓ Positional encoder loaded from checkpoint"
echo "  ✗ Optimizer state RESET (fresh start)"
echo "  ✗ LR scheduler RESET (fresh start)"
echo "  ✗ Training starts from epoch 0, step 0"
echo ""

# Fine-tune with reset optimizer and lower learning rate
python /myhome/sdate/sdate/training/compression/train_heic_to_tiff.py \
    --data_path=$DATA_PATH \
    --output_dir=$OUTPUT_DIR \
    --resume_from_checkpoint=$CHECKPOINT \
    --reset_optimizer \
    --volume_size=32 \
    --stride=32 \
    --num_frames=32 \
    --batch_size=4 \
    --learning_rate=1e-5 \
    --num_epochs=10 \
    --heic_quality=85 \
    --validation_split=0.1

echo ""
echo "===================================================================="
echo "Fine-tuning completed!"
echo "===================================================================="
echo ""
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "Use Cases for --reset_optimizer:"
echo "  1. Fine-tuning with lower learning rate"
echo "  2. Changing optimizer parameters"
echo "  3. Starting fresh training schedule"
echo "  4. Avoiding momentum from previous training"
