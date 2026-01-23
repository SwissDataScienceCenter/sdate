#!/bin/bash
#SBATCH --job-name=tomo_compress
#SBATCH --output=/das/home/barbaf_l/sdate/scripts/logs/tomo_compress_%j.out
#SBATCH --error=/das/home/barbaf_l/sdate/scripts/logs/tomo_compress_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=128G
#SBATCH --partition=day

# Tomographic Data Compression Pipeline - SLURM Launcher
# ========================================================
# This script runs the batch compression pipeline on SLURM
# 
# Usage:
#   sbatch run_tomography_compression.sh                  # Process all folders
#   sbatch run_tomography_compression.sh <folder_id>       # Process single folder
#
# Examples:
#   sbatch run_tomography_compression.sh                  # All folders
#   sbatch run_tomography_compression.sh 12345            # Only file_12345_extracted
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/tomo_compress_JOBID.out

# Parse optional folder_id argument
FOLDER_ID=${1:-}  # Empty string if not provided

echo "========================================================================"
echo "Tomographic Data Compression Pipeline"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "Working directory: $(pwd)"
if [ -n "$FOLDER_ID" ]; then
    echo "Folder ID: $FOLDER_ID (processing file_${FOLDER_ID}_extracted only)"
else
    echo "Folder ID: Not specified (processing all folders)"
fi
echo ""

# Create logs directory if it doesn't exist
mkdir -p /das/home/barbaf_l/sdate/scripts/logs

# Load conda module
echo "Loading anaconda module..."
module load anaconda/2019.07

# Activate conda environment
echo "Activating conda environment: /das/home/barbaf_l/Xiangyu/conda/mled"
source activate /das/home/barbaf_l/Xiangyu/conda/mled

# Verify environment
echo ""
echo "Python version:"
python --version
echo ""
echo "Python path:"
which python
echo ""

# Change to script directory
cd /das/home/barbaf_l/sdate/scripts

# Run the compression pipeline
echo "========================================================================"
echo "Starting compression pipeline..."
echo "========================================================================"
echo ""

# Build command with optional folder_id
if [ -n "$FOLDER_ID" ]; then
    python run_tomography_compression.py --folder-id "$FOLDER_ID"
else
    python run_tomography_compression.py
fi

# Capture exit code
EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Pipeline completed with exit code: $EXIT_CODE"
echo "Finished at: $(date)"
echo "========================================================================"

exit $EXIT_CODE
