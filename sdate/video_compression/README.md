# Video Compression Module

This module provides utilities for video compression optimized for scientific data, particularly grayscale video tensors.

## Installation

The module is part of the `sdate` package. Make sure the package is installed in your environment.

## Usage

### Basic Import

```python
from sdate.video_compression import encode_hevc_grayscale_10bit, decode_hevc_grayscale_10bit
```

### Encoding

```python
import torch

# Create or load your video tensor (T, H, W) with float32 values in range [0, 1]
video_tensor = torch.rand(100, 512, 512, dtype=torch.float32)

# Encode to HEVC 10-bit
encode_hevc_grayscale_10bit(
    v=video_tensor,
    outfile="my_video.mov",
    fps=24,
    cq_hw=90,      # VideoToolbox quality (0-100, lower = better)
    crf_sw=14,     # libx265 CRF (0 = lossless)
    preset_sw="veryslow"  # libx265 preset
)
```

### Decoding

```python
# Decode HEVC video back to tensor
decoded_tensor = decode_hevc_grayscale_10bit("my_video.mov", device="cuda")
print(f"Decoded shape: {decoded_tensor.shape}")  # (T, H, W)
print(f"Value range: [{decoded_tensor.min():.3f}, {decoded_tensor.max():.3f}]")
```

## Features

- **Hardware Acceleration**: Automatically tries Apple VideoToolbox for hardware encoding, falls back to libx265 software encoding if needed
- **10-bit Precision**: Maintains high dynamic range for scientific data
- **Robust Error Handling**: Comprehensive input validation and error reporting  
- **Device Support**: Automatic device placement for decoded tensors
- **Optimized for Grayscale**: Specifically designed for single-channel scientific video data

## Requirements

- PyTorch
- NumPy
- FFmpeg with HEVC support
- For hardware acceleration on macOS: VideoToolbox support

## Parameters

### encode_hevc_grayscale_10bit

- `v`: Input tensor (T, H, W) with float32 values in [0, 1]
- `outfile`: Output video file path (default: "out_grayscale_10bit.mov")
- `fps`: Frames per second (default: 24)
- `cq_hw`: VideoToolbox quality 0-100, lower = better (default: 90)
- `crf_sw`: libx265 CRF, 0 = lossless (default: 14)
- `preset_sw`: libx265 preset for speed/quality tradeoff (default: "veryslow")

### decode_hevc_grayscale_10bit

- `infile`: Input video file path
- `device`: PyTorch device for output tensor (optional)

Returns: torch.Tensor of shape (T, H, W) with float32 values in [0, 1]

## Examples in Notebooks

See `notebooks/RawProjectionOverview.ipynb` for complete usage examples and quality analysis.
