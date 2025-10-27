#!/bin/bash

# Simple TIFF Compression Analysis Sweep Script
# This is a simplified version for quick testing

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_BASE_DIR="$PROJECT_ROOT/data/ct_files"

# Quality levels to test
CQ_VALUES=(100 98 96 94 92 90 85 80)

echo "🚀 Starting TIFF Compression Sweep"
echo "Data directory: $DATA_BASE_DIR"
echo "Quality levels: ${CQ_VALUES[*]}"
echo ""

# Find first dataset directory with TIFF files for testing
TEST_DATASET=""
for folder in "$DATA_BASE_DIR"/*; do
    if [ -d "$folder" ] && find "$folder" -name "*.tif" -o -name "*.tiff" | head -1 | grep -q .; then
        TEST_DATASET="$folder"
        break
    fi
done

if [ -z "$TEST_DATASET" ]; then
    echo "❌ No datasets with TIFF files found"
    exit 1
fi

echo "📁 Using test dataset: $(basename "$TEST_DATASET")"
echo ""

# Run analysis for each quality level
for cq_value in "${CQ_VALUES[@]}"; do
    echo "🎬 Running analysis with CQ=$cq_value..."
    
    python "$SCRIPT_DIR/tiff_compression_analysis.py" \
        --data_path "$TEST_DATASET" \
        --cq_hw "$cq_value" \
        --experiment_name "sweep_test_cq${cq_value}" \
        --wandb_project "tiff-compression-sweep-test"
    
    echo "✅ Completed CQ=$cq_value"
    echo ""
done

echo "🎉 Sweep completed successfully!"
