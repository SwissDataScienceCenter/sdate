#!/usr/bin/env python3
"""
TIFF Video Compression Analysis Script

This script performs HEVC compression analysis on TIFF image sequences,
computing quality metrics and logging results to wandb.

Usage:
    python tiff_compression_analysis.py --data_path /path/to/tiff/folder --cq_hw 100 --skip_frames 10

Features:
    - Loads TIFF sequence and normalizes to [0,1]
    - Compresses using HEVC 10-bit encoding
    - Computes PSNR and SSIM quality metrics
    - Generates comparison visualizations
    - Logs all results and plots to wandb
"""

import argparse
import os
import sys
import time
from pathlib import Path
import re
import glob

# Add the project root directory to Python path to find sdate module
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import wandb

# Import video compression utilities
from sdate.video_compression import encode_hevc_grayscale_10bit, decode_hevc_grayscale_10bit


def setup_device():
    """Set up optimal compute device"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ MPS (Apple Silicon GPU) acceleration available and will be used")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ CUDA GPU acceleration available and will be used")
    else:
        device = torch.device("cpu")
        print(f"ℹ️  Using CPU for computations (no GPU acceleration available)")
    
    return device


def ssim2d_per_slice(
    vol_pred: torch.Tensor,
    vol_gt: torch.Tensor,
    data_range: float = 1.0,
    win_size: int = 11,
    sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
    device: torch.device = None,
    slice_batch: int = 64
):
    """
    Computes SSIM on each 2D slice (WxH), averages over pixels per slice,
    then returns the mean over all D slices. Also returns per-slice SSIMs.
    """
    import torch.nn.functional as F
    
    assert vol_pred.shape == vol_gt.shape and vol_pred.ndim == 3, "Expected (D,W,H)"
    D, W, H = vol_pred.shape

    if win_size % 2 == 0:
        win_size += 1  # ensure odd
    if win_size > min(W, H):
        raise ValueError(f"win_size {win_size} larger than slice size {(W,H)}")

    # Pick device automatically if not given
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    x = vol_pred.to(device=device, dtype=torch.float32)
    y = vol_gt.to(device=device, dtype=torch.float32)

    # Build 2D Gaussian kernel (1x1)
    coords = torch.arange(win_size, device=device, dtype=torch.float32) - (win_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).reshape(-1, 1) @ (g / g.sum()).reshape(1, -1)
    kernel = (g / g.sum()).unsqueeze(0).unsqueeze(0).contiguous()  # [1,1,ks,ks]

    # Constants
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # Helpers
    pad = win_size // 2
    def conv2_same(z):  # z: [N,1,W,H]
        z = F.pad(z, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(z, kernel, groups=1)

    per_slice_vals = []
    for s0 in range(0, D, slice_batch):
        s1 = min(D, s0 + slice_batch)
        xs = x[s0:s1].unsqueeze(1)  # [n,1,W,H]
        ys = y[s0:s1].unsqueeze(1)

        mu_x = conv2_same(xs)
        mu_y = conv2_same(ys)
        mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y

        sigma_x2 = conv2_same(xs * xs) - mu_x2
        sigma_y2 = conv2_same(ys * ys) - mu_y2
        sigma_xy = conv2_same(xs * ys) - mu_xy

        num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
        ssim_map = num / (den + 1e-12)            # [n,1,W,H]
        ssim_slice = ssim_map.mean(dim=(1,2,3))   # [n]
        per_slice_vals.append(ssim_slice)

    per_slice_vals = torch.cat(per_slice_vals, dim=0)  # [D]
    mean_ssim = per_slice_vals.mean().item()
    return mean_ssim, per_slice_vals.detach().cpu()


def calculate_quality_metrics(original, reconstructed, crop_pixels=5, use_gpu=True, device=None):
    """
    Calculate PSNR and SSIM between two tensors using PyTorch
    """
    if device is None:
        device = setup_device()
    
    # Choose device based on user preference and availability
    compute_device = device if use_gpu else torch.device("cpu")
    
    # Ensure tensors are float and move to compute device
    orig = original.float().to(compute_device)
    recon = reconstructed.float().to(compute_device)
    
    # Handle different tensor dimensions
    if orig.dim() == 2:  # Single 2D image
        orig = orig.unsqueeze(0)  # Add batch dimension
        recon = recon.unsqueeze(0)
    elif orig.dim() == 4:  # Video tensor (N, C, H, W)
        orig = orig.squeeze(1) if orig.shape[1] == 1 else orig.mean(dim=1)
        recon = recon.squeeze(1) if recon.shape[1] == 1 else recon.mean(dim=1)
    
    # Calculate PSNR using PyTorch
    mse = torch.mean((orig - recon) ** 2)
    if mse > 0:
        # Assume normalized input [0, 1] range
        max_val = torch.max(orig)
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    else:
        psnr = torch.tensor(float('inf'), device=compute_device)

    # Calculate SSIM
    ssim, _ = ssim2d_per_slice(recon, orig, data_range=1.0, win_size=11, sigma=1.5, slice_batch=16)

    # Move results back to CPU for final output
    return {
        'psnr': psnr.cpu().item() if not torch.isinf(psnr) else float('inf'),
        'ssim': ssim,
        'mse': mse.cpu().item(),
        'cropped_shape': orig.shape,
        'crop_pixels': crop_pixels,
        'compute_device': str(compute_device)
    }


def load_tiff_sequence(tiff_files, start_offset=0, max_frames=None, check_bit_depth=True):
    """
    Load a sequence of TIFF files into a video tensor.
    """
    if max_frames is not None:
        tiff_files = tiff_files[start_offset:start_offset+max_frames]
    
    print(f"Loading {len(tiff_files)} TIFF files...")
    
    # Load first image to get dimensions and check data type
    first_img = Image.open(tiff_files[0])
    first_array = np.array(first_img)
    height, width = first_array.shape[:2]
    
    print(f"Image dimensions: {width} x {height}")
    print(f"First image dtype: {first_array.dtype}")
    print(f"First image shape: {first_array.shape}")
    print(f"First image range: [{first_array.min()}, {first_array.max()}]")
    
    # Determine if we can use 16-bit or need 32-bit
    if check_bit_depth:
        print("\n🔍 Analyzing bit depth requirements...")
        global_min = float('inf')
        global_max = float('-inf')
        
        # Sample some files to check range
        sample_indices = np.linspace(0, len(tiff_files)-1, min(20, len(tiff_files)), dtype=int)
        
        for idx in tqdm(sample_indices, desc="Sampling files for range analysis"):
            img = Image.open(tiff_files[idx])
            arr = np.array(img)
            global_min = min(global_min, arr.min())
            global_max = max(global_max, arr.max())
        
        print(f"Sampled range: [{global_min}, {global_max}]")
        
        # Determine optimal data type
        if global_max <= 65535 and global_min >= 0:
            optimal_dtype = torch.uint16
            print("✅ Data fits in 16-bit unsigned integers")
        elif global_max <= 2147483647 and global_min >= -2147483648:
            optimal_dtype = torch.int32  
            print("⚠️  Data requires 32-bit integers")
        else:
            optimal_dtype = torch.float32
            print("⚠️  Data requires 32-bit float")
    else:
        optimal_dtype = torch.float32
        global_min, global_max = None, None
    
    # Initialize tensor
    if first_array.ndim == 2:  # Grayscale
        video_tensor = torch.zeros(len(tiff_files), height, width, dtype=optimal_dtype)
    else:  # Color - take first channel or convert to grayscale
        video_tensor = torch.zeros(len(tiff_files), height, width, dtype=optimal_dtype)
        print("⚠️  Color images detected - will convert to grayscale using first channel")
    
    # Load all images
    min_val = float('inf')
    max_val = float('-inf')
    
    for i, tiff_file in enumerate(tqdm(tiff_files, desc="Loading TIFF files")):
        img = Image.open(tiff_file)
        arr = np.array(img)
        
        # Handle different image formats
        if arr.ndim == 3:  # Color image
            arr = arr[:, :, 0]  # Take first channel
        
        video_tensor[i] = torch.from_numpy(arr).to(optimal_dtype)
        
        # Track actual min/max
        min_val = min(min_val, arr.min())
        max_val = max(max_val, arr.max())
    
    dtype_info = {
        'original_dtype': first_array.dtype,
        'torch_dtype': optimal_dtype,
        'fits_16bit': max_val <= 65535 and min_val >= 0
    }
    
    return video_tensor, min_val, max_val, dtype_info


def create_comparison_plot(video_normalized, reconstructed_video, min_frames, output_path="comparison_plot.png"):
    """
    Create and save a comparison plot showing original vs compressed vs difference
    """
    # Select frames for visualization
    comparison_frames = [0, min_frames // 4, min_frames // 2, min_frames - 1]
    
    fig, axes = plt.subplots(3, len(comparison_frames), figsize=(5*len(comparison_frames), 15))
    fig.suptitle('TIFF HEVC Compression Analysis: Original vs Compressed vs Difference', 
                 fontsize=16, fontweight='bold')
    
    for i, frame_idx in enumerate(comparison_frames):
        if frame_idx < min_frames:
            original_frame = video_normalized[frame_idx]
            compressed_frame = reconstructed_video[frame_idx]
            difference = torch.abs(original_frame - compressed_frame)
            
            # Original
            axes[0, i].imshow(original_frame.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'Original Frame {frame_idx}')
            axes[0, i].axis('off')
            
            # Compressed
            axes[1, i].imshow(compressed_frame.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title(f'Compressed Frame {frame_idx}')
            axes[1, i].axis('off')
            
            # Difference
            im = axes[2, i].imshow(difference.cpu().numpy(), cmap='hot')
            axes[2, i].set_title(f'Difference\nMax: {difference.max():.2e}')
            axes[2, i].axis('off')
            plt.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='TIFF Video Compression Analysis with wandb logging')
    
    # Required parameters
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to TIFF sequence folder')
    
    # Compression parameters
    parser.add_argument('--cq_hw', type=int, default=100,
                        help='HEVC compression quality (0-51, lower is better quality) (default: 100)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for video encoding (default: 30)')
    
    # Data loading parameters
    parser.add_argument('--max_frames', type=int, default=200,
                        help='Maximum number of frames to load (default: 400)')
    parser.add_argument('--start_offset', type=int, default=150,
                        help='Starting frame offset (default: 150)')
    
    # Quality metric parameters
    parser.add_argument('--skip_frames', type=int, default=10,
                        help='Skip frames for metric computation (default: 10)')
    parser.add_argument('--crop_pixels', type=int, default=5,
                        help='Pixels to crop from edges for metric computation (default: 5)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./compression_analysis_output',
                        help='Output directory for results (default: ./compression_analysis_output)')
    parser.add_argument('--experiment_name', type=str, default='tiff_hevc_compression',
                        help='wandb experiment name (default: tiff_hevc_compression)')
    
    # wandb parameters
    parser.add_argument('--wandb_project', type=str, default='tiff-compression-analysis',
                        help='wandb project name (default: tiff-compression-analysis)')
    parser.add_argument('--disable_wandb', action='store_true',
                        help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set up device
    device = setup_device()
    
    # Initialize wandb if not disabled
    if not args.disable_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            config=vars(args)
        )
    
    print(f"\n🎬 TIFF Video Compression Analysis")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Compression quality (cq_hw): {args.cq_hw}")
    print(f"Max frames: {args.max_frames}")
    print(f"Skip frames for metrics: {args.skip_frames}")
    
    try:
        # Load TIFF sequence
        data_path = Path(args.data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        
        # Find TIFF files
        tiff_files = sorted(list(data_path.glob('*.tif')) + list(data_path.glob('*.tiff')))
        if len(tiff_files) == 0:
            raise FileNotFoundError(f"No TIFF files found in {data_path}")
        
        print(f"\n📁 Found {len(tiff_files)} TIFF files")
        
        # Load and normalize video tensor
        video_tensor_original, min_val, max_val, dtype_info = load_tiff_sequence(
            tiff_files, 
            start_offset=args.start_offset, 
            max_frames=args.max_frames, 
            check_bit_depth=True
        )
        
        # Normalize to [0, 1]
        print(f"\n🔄 Normalizing video tensor to [0, 1] range...")
        video_tensor_original = video_tensor_original.float()
        video_tensor_original -= min_val
        video_tensor_original /= (max_val - min_val)
        video_normalized = video_tensor_original
        
        print(f"✅ Normalization completed:")
        print(f"  Shape: {video_normalized.shape}")
        print(f"  Range: [{video_normalized.min():.6f}, {video_normalized.max():.6f}]")
        
        # Compress video
        compressed_video_path = output_dir / f"compressed_cq{args.cq_hw}.mov"
        print(f"\n🎬 Compressing video to: {compressed_video_path}")
        
        encode_hevc_grayscale_10bit(
            video_normalized, 
            str(compressed_video_path), 
            fps=args.fps, 
            cq_hw=args.cq_hw
        )
        
        # Decode compressed video
        reconstructed_video = decode_hevc_grayscale_10bit(str(compressed_video_path), device="cpu")
        print(f"✅ Video compression completed")
        print(f"Reconstructed video shape: {reconstructed_video.shape}")
        
        # Calculate file sizes and compression metrics
        compressed_file_size = os.path.getsize(compressed_video_path)
        original_tensor_size = video_normalized.numel() * video_normalized.element_size() / 2  # bytes (assuming original was uint16)
        
        compression_ratio = original_tensor_size / compressed_file_size
        space_savings = (1 - compressed_file_size / original_tensor_size) * 100
        
        # Calculate quality metrics
        print(f"\n📊 Computing quality metrics...")
        test_original = video_normalized[::args.skip_frames]
        test_reconstructed = reconstructed_video[::args.skip_frames]
        
        start_time = time.time()
        metrics = calculate_quality_metrics(
            test_original, 
            test_reconstructed, 
            crop_pixels=args.crop_pixels,
            use_gpu=True,
            device=device
        )
        compute_time = time.time() - start_time
        
        # Extract metrics
        psnr_value = metrics['psnr']
        ssim_value = metrics['ssim']
        mse = metrics['mse']
        
        # Calculate bits per pixel
        min_frames = min(video_normalized.shape[0], reconstructed_video.shape[0])
        min_height = min(video_normalized.shape[1], reconstructed_video.shape[1])
        min_width = min(video_normalized.shape[2], reconstructed_video.shape[2])
        
        bits_per_pixel_original = (original_tensor_size * 8) / (min_frames * min_height * min_width)
        bits_per_pixel_compressed = (compressed_file_size * 8) / (min_frames * min_height * min_width)
        
        # Create summary data
        summary_data = {
            'Source': 'TIFF Sequence',
            'Original Size (MB)': original_tensor_size / 1e6,
            'Compressed Size (MB)': compressed_file_size / 1e6,
            'Compression Ratio': compression_ratio,
            'Space Savings (%)': space_savings,
            'PSNR (dB)': psnr_value,
            'SSIM': ssim_value,
            'MSE': mse,
            'Bits/Pixel Original': bits_per_pixel_original,
            'Bits/Pixel Compressed': bits_per_pixel_compressed,
            'Compute Time (s)': compute_time,
            'CQ_HW': args.cq_hw,
            'FPS': args.fps,
            'Frames Analyzed': len(test_original),
            'Skip Frames': args.skip_frames
        }
        
        # Create comparison plot
        print(f"\n📊 Creating comparison visualization...")
        plot_path = output_dir / "compression_comparison.png"
        create_comparison_plot(video_normalized, reconstructed_video, min_frames, str(plot_path))
        
        # Print summary
        print(f"\n📋 Compression Analysis Summary:")
        print("=" * 60)
        print(f"Original Size (MB)....... {summary_data['Original Size (MB)']:.2f}")
        print(f"Compressed Size (MB)..... {summary_data['Compressed Size (MB)']:.2f}")
        print(f"Compression Ratio........ {summary_data['Compression Ratio']:.1f}:1")
        print(f"Space Savings............ {summary_data['Space Savings (%)']:.1f}%")
        print(f"PSNR (dB)................ {summary_data['PSNR (dB)']:.2f}")
        print(f"SSIM..................... {summary_data['SSIM']:.6f}")
        print(f"Bits/Pixel Original...... {summary_data['Bits/Pixel Original']:.2f}")
        print(f"Bits/Pixel Compressed.... {summary_data['Bits/Pixel Compressed']:.2f}")
        
        # Save summary to CSV
        summary_csv_path = output_dir / "compression_summary.csv"
        df_summary = pd.DataFrame([summary_data])
        df_summary.to_csv(summary_csv_path, index=False)
        print(f"\n💾 Summary saved to: {summary_csv_path}")
        
        # Log to wandb
        if not args.disable_wandb:
            print(f"\n🔄 Logging results to wandb...")
            
            # Log metrics
            wandb.log(summary_data)
            
            # Log comparison plot
            wandb.log({"compression_comparison": wandb.Image(str(plot_path))})
            
            # Log summary table
            wandb.log({"summary_table": wandb.Table(dataframe=df_summary)})
            
            # Log artifacts
            wandb.save(str(summary_csv_path))
            wandb.save(str(plot_path))
            
            print(f"✅ Results logged to wandb project: {args.wandb_project}")
        
        print(f"\n🎉 Analysis completed successfully!")
        print(f"   Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        if not args.disable_wandb:
            wandb.log({"error": str(e)})
        raise
    
    finally:
        if not args.disable_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
