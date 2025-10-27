"""
H.264 compression utilities for video tensor analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import platform
import math
import json
from PIL import Image


def transform_video_for_compression(video_float32, max_value=255.0):
    """
    Transforms a float32 video tensor with a 0-170 range to a
    uint8 tensor with a 0-255 range for video compression.

    Args:
        video_float32 (torch.Tensor): A tensor with dtype float32 and values in [0, 170].

    Returns:
        torch.Tensor: A tensor with dtype uint8 and values in [0, 255].
    """
    # Step 1: Clip the data to the expected range [0, 170]
    maximum = video_float32.max()
    minimum = video_float32.min()
    
    # Step 2: Normalize to [0, 1] by dividing by the max value of the range
    video_normalized = (video_float32 - minimum) / maximum

    # Step 3: Scale to [0, 255]
    video_scaled = video_normalized * max_value

    # Step 4: Convert to uint8. This will truncate any decimal part.
    video_uint8 = video_scaled.to(torch.uint8)

    return video_uint8


def tensor_to_raw_video(video_tensor, filename):
    """Saves a torch tensor to a raw RGB video file."""
    if video_tensor.device.type != 'cpu':
        video_tensor = video_tensor.cpu()
    video_np = video_tensor.numpy()
    with open(filename, 'wb') as f:
        f.write(video_np.tobytes())


def read_raw_video(filename, width, height, channels=3, pix_fmt='rgb24'):
    """Reads a raw video file and returns a numpy array."""
    if pix_fmt == 'rgb24':
        frame_size = width * height * channels
        dtype = np.uint8
    else: # yuv420p
        frame_size = int(width * height * 1.5)
        dtype = np.uint8
    
    frames = []
    with open(filename, 'rb') as f:
        while True:
            frame_data = f.read(frame_size)
            if len(frame_data) < frame_size:
                break
            
            if pix_fmt == 'yuv420p':
                 y_size = width * height
                 y_data = np.frombuffer(frame_data[:y_size], dtype=dtype)
                 frame = y_data.reshape((height, width))
            else: # rgb24
                 frame = np.frombuffer(frame_data, dtype=dtype).reshape((height, width, channels))

            frames.append(frame)
    return np.stack(frames) if frames else np.array([])


def calculate_psnr(original_frame, compressed_frame):
    """Calculates PSNR between two frames."""
    mse = np.mean((original_frame.astype(np.float64) - compressed_frame.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel_value = 255.0
    psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse))
    return psnr


def analyze_h264_compression(video_tensor, preset='medium', crf=23, save_compressed_video=True):
    """
    Performs a full analysis of H.264 compression on a video tensor.
    - Calculates and prints the compression ratio (overall and per-frame).
    - Calculates and prints the overall PSNR.
    - Determines the type of each frame (I, P, B).
    - Plots the motion compensation residuals by frame type.
    - Plots the PSNR per frame by frame type.
    - Plots the compression ratio per frame by frame type.
    - Optionally saves the final compressed video.

    Args:
        video_tensor (torch.Tensor): The input video tensor (uint8).
        preset (str): The x264 encoding preset. Controls the trade-off between
                      compression speed and efficiency.
                      Possible values: 'ultrafast', 'superfast', 'veryfast',
                      'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'.
        crf (int): The Constant Rate Factor (0-51). Lower values mean higher
                   quality and larger file size. 23 is a good default. 0 is
                   lossless.
        save_compressed_video (bool): If True, the final .mp4 file is kept.
    """
    if not isinstance(video_tensor, torch.Tensor) or video_tensor.dtype != torch.uint8:
        raise TypeError("Input must be a torch.uint8 tensor.")
        
    num_frames, height, width, channels = video_tensor.shape
    
    original_raw_file = 'original_video.rgb'
    predicted_yuv_file = 'predicted_frames.yuv'
    compressed_output_file = 'compressed_video.mp4'
    reconstructed_raw_file = 'reconstructed_video.rgb'
    
    temp_files = [original_raw_file, predicted_yuv_file, reconstructed_raw_file]
    if not save_compressed_video:
        temp_files.append(compressed_output_file)

    try:
        # --- 1. Calculate Uncompressed Size and Save Raw Video ---
        uncompressed_frame_size = height * width * channels
        uncompressed_total_size = num_frames * uncompressed_frame_size
        print(f"Uncompressed Size: {uncompressed_total_size / 1e6:.2f} MB ({uncompressed_frame_size / 1e3:.2f} KB per frame)")
        tensor_to_raw_video(video_tensor, original_raw_file)
        
        # --- 2. Encode the Video ---
        x264_params = f'dump-yuv={predicted_yuv_file}'
        ffmpeg_command = [
            'ffmpeg', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
            '-r', '24', '-i', original_raw_file, '-vcodec', 'libx264',
            '-preset', preset, '-crf', str(crf),
            '-x264-params', x264_params, '-f', 'mp4', '-y', compressed_output_file
        ]
        print(f"\nRunning FFmpeg to encode the video (preset={preset}, crf={crf})...")
        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        print(f"Successfully created '{compressed_output_file}'")
        
        # --- 3. Calculate Overall Compression Ratio ---
        compressed_total_size = os.path.getsize(compressed_output_file)
        print(f"Compressed Size: {compressed_total_size / 1e3:.2f} KB")
        if compressed_total_size > 0:
            compression_ratio = uncompressed_total_size / compressed_total_size
            print(f"-> Overall Compression Ratio: {compression_ratio:.1f} : 1")

        # --- 4. Decode Video and Calculate Overall PSNR ---
        decode_command = [
            'ffmpeg', '-i', compressed_output_file, '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-y', reconstructed_raw_file
        ]
        subprocess.run(decode_command, check=True, capture_output=True, text=True)
        reconstructed_video_np = read_raw_video(reconstructed_raw_file, width, height)
        original_video_np = video_tensor.numpy()
        overall_psnr = calculate_psnr(original_video_np, reconstructed_video_np)
        print(f"-> Overall Video PSNR: {overall_psnr:.2f} dB")
        
        # --- 5. Get Per-Frame Data (Type, Size) using ffprobe (JSON output) ---
        ffprobe_command = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_frames',
            '-select_streams', 'v:0', '-show_entries', 'frame=pict_type,pkt_size',
            compressed_output_file
        ]
        print("\nRunning ffprobe to extract per-frame data...")
        result = subprocess.run(ffprobe_command, check=True, capture_output=True, text=True)
        ffprobe_data = json.loads(result.stdout)
        
        frame_types = []
        frame_sizes_bytes = []
        if 'frames' in ffprobe_data:
            for frame in ffprobe_data['frames']:
                frame_types.append(frame.get('pict_type', '?'))
                frame_sizes_bytes.append(int(frame.get('pkt_size', '0')))
        
        print(f"Cleaned and detected frame types: {''.join(frame_types)}")
        
        # --- 6. Calculate Per-Frame Metrics ---
        psnr_values = [calculate_psnr(original_video_np[i], reconstructed_video_np[i]) for i in range(len(reconstructed_video_np))]
        
        compression_ratios = [uncompressed_frame_size / size if size > 0 else 1 for size in frame_sizes_bytes]

        predicted_y_np = read_raw_video(predicted_yuv_file, width, height, pix_fmt='yuv420p')
        rgb_weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 1, 1, 3)
        original_y_tensor = torch.sum(video_tensor.float() * rgb_weights, dim=3)
        original_y_np = original_y_tensor.numpy().astype(np.uint8)
        
        residuals = [0.0]
        num_comparisons = min(len(predicted_y_np), len(original_y_np) - 1)
        for i in range(num_comparisons):
            mae = np.mean(np.abs(original_y_np[i + 1].astype(np.float32) - predicted_y_np[i].astype(np.float32)))
            residuals.append(mae)

        # --- 7. Plotting ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(3, 1, figsize=(16, 18), sharex=True)
        fig.suptitle('H.264 Compression Analysis', fontsize=20, fontweight='bold')
        
        num_frames_to_plot = min(len(frame_types), len(residuals), len(psnr_values), len(compression_ratios))
        frame_indices = range(num_frames_to_plot)

        plot_data = {
            'I': {'x': [], 'res': [], 'psnr': [], 'ratio': []},
            'P': {'x': [], 'res': [], 'psnr': [], 'ratio': []},
            'B': {'x': [], 'res': [], 'psnr': [], 'ratio': []}
        }
        
        for i in frame_indices:
            f_type = frame_types[i]
            if f_type in plot_data: # Only process valid frame types
                plot_data[f_type]['x'].append(i)
                plot_data[f_type]['res'].append(residuals[i])
                plot_data[f_type]['psnr'].append(psnr_values[i])
                plot_data[f_type]['ratio'].append(compression_ratios[i])

        # Plot 1: Residuals
        ax1 = axes[0]
        ax1.plot(frame_indices, residuals[:num_frames_to_plot], color='gray', linestyle='--', lw=0.8, zorder=0)
        ax1.scatter(plot_data['I']['x'], plot_data['I']['res'], color='gold', s=150, zorder=3, label='I-Frame', edgecolors='black', marker='*')
        ax1.scatter(plot_data['P']['x'], plot_data['P']['res'], color='salmon', s=70, zorder=2, label='P-Frame', edgecolors='darkred')
        ax1.scatter(plot_data['B']['x'], plot_data['B']['res'], color='skyblue', s=70, zorder=1, label='B-Frame', edgecolors='darkblue')
        ax1.set_title('Motion Compensation Residuals (Prediction Error)', fontsize=14)
        ax1.set_ylabel('Mean Absolute Error (Luma)', fontsize=12)
        ax1.legend()
        ax1.set_ylim(bottom=0)

        # Plot 2: PSNR
        ax2 = axes[1]
        ax2.plot(frame_indices, psnr_values[:num_frames_to_plot], color='gray', linestyle='--', lw=0.8, zorder=0)
        ax2.scatter(plot_data['I']['x'], plot_data['I']['psnr'], color='gold', s=150, zorder=3, label='I-Frame', edgecolors='black', marker='*')
        ax2.scatter(plot_data['P']['x'], plot_data['P']['psnr'], color='salmon', s=70, zorder=2, label='P-Frame', edgecolors='darkred')
        ax2.scatter(plot_data['B']['x'], plot_data['B']['psnr'], color='skyblue', s=70, zorder=1, label='B-Frame', edgecolors='darkblue')
        ax2.set_title('Reconstruction Quality', fontsize=14)
        ax2.set_ylabel('PSNR (dB)', fontsize=12)
        ax2.legend()
        
        # Plot 3: Compression Ratio
        ax3 = axes[2]
        ax3.plot(frame_indices, compression_ratios[:num_frames_to_plot], color='gray', linestyle='--', lw=0.8, zorder=0)
        ax3.scatter(plot_data['I']['x'], plot_data['I']['ratio'], color='gold', s=150, zorder=3, label='I-Frame', edgecolors='black', marker='*')
        ax3.scatter(plot_data['P']['x'], plot_data['P']['ratio'], color='salmon', s=70, zorder=2, label='P-Frame', edgecolors='darkred')
        ax3.scatter(plot_data['B']['x'], plot_data['B']['ratio'], color='skyblue', s=70, zorder=1, label='B-Frame', edgecolors='darkblue')
        ax3.set_title('Per-Frame Compression Efficiency', fontsize=14)
        ax3.set_ylabel('Compression Ratio (X:1)', fontsize=12)
        ax3.set_xlabel('Frame Number', fontsize=12)
        ax3.legend()
        
        ax3.set_ylim(bottom=1) 
        ax3.set_yscale('log')
        
        ax3.grid(True, which="both", ls="--", linewidth=0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()

    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"\nERROR: Could not execute a required command. Please ensure FFmpeg/FFprobe are installed and in your system's PATH.")
        if isinstance(e, subprocess.CalledProcessError):
            print(f"FFmpeg/FFprobe Stderr: {e.stderr}")
    finally:
        # Cleanup temporary files
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
        print("\nCleaned up temporary files.")


def save_video_as_jpeg_sequence(video_tensor, quality=85, output_folder='jpeg_sequence'):
    """
    Saves each frame of a video tensor as an individual JPEG image and calculates
    the resulting compression ratio.

    This serves as a baseline using only intra-frame (spatial) compression,
    ignoring all temporal relationships between frames (Motion-JPEG).

    Args:
        video_tensor (torch.Tensor): The input video tensor with shape 
                                     (num_frames, height, width, channels) and
                                     dtype torch.uint8.
        quality (int): The quality for the JPEG compression, from 1 (worst) to 
                       95 (best). Default is 85.
        output_folder (str): The name of the folder where JPEG images will be saved.

    Returns:
        tuple: A tuple containing (total_compressed_size, compression_ratio).
    """
    # --- 1. Input Validation and Uncompressed Size ---
    if not isinstance(video_tensor, torch.Tensor) or video_tensor.dtype != torch.uint8:
        raise TypeError("Input must be a torch.uint8 tensor.")
    if not (1 <= quality <= 95):
        raise ValueError("JPEG quality must be between 1 and 95.")

    num_frames, height, width, channels = video_tensor.shape
    uncompressed_size = num_frames * height * width * channels
    print(f"Uncompressed Size: {uncompressed_size / 1e6:.2f} MB")

    # --- 2. Create Output Directory ---
    if not os.path.exists(output_folder):
        print(f"Creating output directory: '{output_folder}'")
        os.makedirs(output_folder)

    # Ensure the tensor is on the CPU before converting to numpy
    if video_tensor.device.type != 'cpu':
        video_tensor = video_tensor.cpu()

    print(f"\nProcessing {num_frames} frames with JPEG quality={quality}...")

    # --- 3. Iterate and Save Each Frame ---
    for i in range(num_frames):
        frame_np = video_tensor[i].numpy()
        image = Image.fromarray(frame_np)
        filepath = os.path.join(output_folder, f'frame_{i:04d}.jpg')
        image.save(filepath, 'JPEG', quality=quality)

    print(f"Successfully saved {num_frames} frames as JPEGs in '{output_folder}'.")
    
    # --- 4. Calculate Total Compressed Size and Ratio ---
    total_compressed_size = 0
    for filename in os.listdir(output_folder):
        if filename.endswith('.jpg'):
            total_compressed_size += os.path.getsize(os.path.join(output_folder, filename))
    
    print(f"\nTotal Compressed Size (all JPEGs): {total_compressed_size / 1e6:.2f} MB")

    compression_ratio = 0
    if total_compressed_size > 0:
        compression_ratio = uncompressed_size / total_compressed_size
        print(f"-> MJPEG Compression Ratio: {compression_ratio:.1f} : 1")
    else:
        print("Warning: Compressed size is 0. Cannot calculate ratio.")

    return total_compressed_size, compression_ratio
