#!/bin/bash
# Example: Training HEIC-to-TIFF model with pre-computed residuals
#
# This script demonstrates how to train the model using residuals as an additional
# input channel. The residuals should be pre-computed using compute_residuals.py

# Step 1: Train initial model (without residuals)
echo "===================================================================="
echo "STEP 1: Training initial model without residuals"
echo "===================================================================="

python /myhome/sdate/sdate/training/compression/train_heic_to_tiff.py \
    --data_path=/myhome/data/sdate/shared/compression_paper/file_3_extracted \
    --output_dir=outputs/heic_to_tiff_initial \
    --volume_size=32 \
    --stride=32 \
    --num_frames=32 \
    --batch_size=4 \
    --learning_rate=1e-4 \
    --num_epochs=10 \
    --heic_quality=85 \
    --validation_split=0.1

# Step 2: Compute residuals from trained model
echo ""
echo "===================================================================="
echo "STEP 2: Computing residuals from trained model"
echo "===================================================================="

python scripts/compute_residuals.py \
    --data_path=/myhome/data/sdate/shared/compression_paper/file_3_extracted \
    --checkpoint_path=outputs/heic_to_tiff_initial/checkpoint-final \
    --output_path=outputs/residuals/file_3_residuals.npy \
    --volume_size=32 \
    --stride=32 \
    --num_frames=32 \
    --heic_quality=85 \
    --batch_size=4

# Step 3: Train refinement model with residuals
echo ""
echo "===================================================================="
echo "STEP 3: Training refinement model WITH residuals"
echo "===================================================================="

python /myhome/sdate/sdate/training/compression/train_heic_to_tiff.py \
    --data_path=/myhome/data/sdate/shared/compression_paper/file_3_extracted \
    --output_dir=outputs/heic_to_tiff_with_residuals \
    --volume_size=32 \
    --stride=32 \
    --num_frames=32 \
    --batch_size=4 \
    --learning_rate=1e-4 \
    --num_epochs=10 \
    --heic_quality=85 \
    --validation_split=0.1 \
    --residual_path=outputs/residuals/file_3_residuals.npy

echo ""
echo "===================================================================="
echo "Training complete!"
echo "===================================================================="
echo ""
echo "Models saved:"
echo "  - Initial model:    outputs/heic_to_tiff_initial/"
echo "  - Refined model:    outputs/heic_to_tiff_with_residuals/"
echo "  - Residuals:        outputs/residuals/file_3_residuals.npy"
