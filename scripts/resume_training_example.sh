#!/bin/bash
# Example: Resume training from a checkpoint
#
# This demonstrates how to resume training if it was interrupted
# or if you want to continue training for more epochs

DATA_PATH="/myhome/data/sdate/shared/compression_paper/file_3_extracted"
OUTPUT_DIR="outputs/heic_to_tiff_training"
CHECKPOINT="outputs/heic_to_tiff_training/checkpoint-2400"  # Adjust to your checkpoint

echo "===================================================================="
echo "Resuming training from checkpoint"
echo "===================================================================="
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo ""

# Resume training from the checkpoint
python /myhome/sdate/sdate/training/compression/train_heic_to_tiff.py \
    --data_path=$DATA_PATH \
    --output_dir=$OUTPUT_DIR \
    --resume_from_checkpoint=$CHECKPOINT \
    --volume_size=32 \
    --stride=32 \
    --num_frames=32 \
    --batch_size=4 \
    --learning_rate=1e-4 \
    --num_epochs=20 \
    --heic_quality=85 \
    --validation_split=0.1

echo ""
echo "===================================================================="
echo "Training resumed and completed!"
echo "===================================================================="
echo ""
echo "The model will continue from where it left off:"
echo "  - Model weights loaded from checkpoint"
echo "  - Optimizer state restored"
echo "  - Learning rate scheduler restored"
echo "  - Training continues from the same global step and epoch"
