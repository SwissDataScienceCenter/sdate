# TIFF Compression Analysis Script

This script automates the TIFF video compression analysis from your notebook, with configurable parameters and wandb logging capabilities.

## Features

- ✅ Loads TIFF image sequences and normalizes to [0,1] range
- ✅ Compresses using HEVC 10-bit encoding with configurable quality
- ✅ Computes PSNR and SSIM quality metrics with GPU acceleration
- ✅ Generates comparison visualizations (Original vs Compressed vs Difference)
- ✅ Logs all results, metrics, and plots to wandb
- ✅ Saves summary data to CSV files
- ✅ Configurable parameters via command line

## Installation

Make sure you have all required dependencies installed:

```bash
# If using conda/mamba
conda install pytorch torchvision matplotlib pillow pandas tqdm wandb

# If using pip
pip install torch torchvision matplotlib pillow pandas tqdm wandb
```

## Basic Usage

```bash
# Basic analysis with high quality compression
python tiff_compression_analysis.py --data_path /path/to/tiff/folder --cq_hw 100

# Medium quality with custom parameters
python tiff_compression_analysis.py \
    --data_path /path/to/tiff/folder \
    --cq_hw 23 \
    --skip_frames 5 \
    --max_frames 500 \
    --experiment_name my_experiment

# Run without wandb logging
python tiff_compression_analysis.py \
    --data_path /path/to/tiff/folder \
    --cq_hw 15 \
    --disable_wandb
```

## Parameters

### Required Parameters
- `--data_path`: Path to folder containing TIFF sequence files

### Compression Parameters
- `--cq_hw`: HEVC compression quality (0-51, lower is better quality, default: 100)
- `--fps`: Frames per second for video encoding (default: 30)

### Data Loading Parameters
- `--max_frames`: Maximum number of frames to load (default: 400)
- `--start_offset`: Starting frame offset (default: 150)

### Quality Metric Parameters
- `--skip_frames`: Skip frames for metric computation to speed up analysis (default: 10)
- `--crop_pixels`: Pixels to crop from edges for metric computation (default: 5)

### Output Parameters
- `--output_dir`: Output directory for results (default: ./compression_analysis_output)
- `--experiment_name`: wandb experiment name (default: tiff_hevc_compression)

### wandb Parameters
- `--wandb_project`: wandb project name (default: tiff-compression-analysis)
- `--disable_wandb`: Disable wandb logging (flag)

## Output

The script generates:

1. **Compressed video file**: `compressed_cq{quality}.mov`
2. **Comparison visualization**: `compression_comparison.png` showing original vs compressed vs difference
3. **Summary CSV**: `compression_summary.csv` with all metrics
4. **Console output** with formatted summary:

```
📋 Compression Analysis Summary:
============================================================
Original Size (MB)....... 4423.68
Compressed Size (MB)..... 254.87
Compression Ratio........ 17.4:1
Space Savings............ 94.2%
PSNR (dB)................ 53.13
SSIM..................... 0.995672
Bits/Pixel Original...... 16.00
Bits/Pixel Compressed.... 0.92
```

## wandb Integration

When wandb logging is enabled (default), the script logs:
- All metrics and parameters
- Comparison visualization plot
- Summary table
- CSV and image files as artifacts

## Quality Parameters Guide

The `cq_hw` parameter controls compression quality:
- `100`: Visually lossless (largest file size)
- `51`: Still very high quality
- `23`: Good balance of quality vs size (recommended starting point)
- `10`: Higher compression, some quality loss
- `0`: Maximum compression, significant quality loss

## Example Configurations

### High Quality Analysis (Visually Lossless)
```bash
python tiff_compression_analysis.py \
    --data_path ../data/ct_files/your_data \
    --cq_hw 100 \
    --skip_frames 5 \
    --experiment_name high_quality_analysis
```

### Balanced Quality/Size Analysis
```bash
python tiff_compression_analysis.py \
    --data_path ../data/ct_files/your_data \
    --cq_hw 23 \
    --skip_frames 10 \
    --experiment_name balanced_analysis
```

### High Compression Analysis
```bash
python tiff_compression_analysis.py \
    --data_path ../data/ct_files/your_data \
    --cq_hw 10 \
    --skip_frames 15 \
    --experiment_name high_compression_analysis
```

## Performance Notes

- GPU acceleration (MPS on Apple Silicon, CUDA on NVIDIA) is automatically used when available
- Use `--skip_frames` to balance analysis detail vs computation time
- Larger `max_frames` values will require more memory and time
- The script automatically handles memory management for large datasets

## Troubleshooting

1. **Import errors**: Make sure all dependencies are installed in your Python environment
2. **TIFF files not found**: Verify the `--data_path` points to a folder with .tif or .tiff files
3. **Memory issues**: Reduce `--max_frames` or increase `--skip_frames`
4. **wandb issues**: Use `--disable_wandb` flag if you don't want to use wandb logging

## File Structure

After running the script, your output directory will contain:
```
compression_analysis_output/
├── compressed_cq100.mov           # Compressed video file
├── compression_comparison.png     # Visualization plot
└── compression_summary.csv        # Summary metrics
```
