#!/bin/bash
# Example training script for INCT

# Quick test with few projections (fast)
echo "Running quick INCT training test..."
python scripts/train_inct.py \
    --data_path /myhome/data/sdate/shared/compression_paper/file_1_extracted \
    --num_projections 5 \
    --target_size 256 \
    --num_epochs 50 \
    --n_levels 12 \
    --table_size 131072 \
    --batch_size 65536 \
    --n_batches 200 \
    --checkpoint_dir checkpoints_inct_test

echo "Training complete!"
echo "Check checkpoints_inct_test/ for results"
