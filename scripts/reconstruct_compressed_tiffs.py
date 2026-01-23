#!/usr/bin/env python3
"""Reconstruct TIFF sequences from compressed HEVC outputs.

Given a batch processing results CSV (produced by the tomography compression pipeline),
this script decodes the compressed videos for darks, flats, and projections separately
back into TIFF frames. Each type is decompressed using its own dynamic range.

The reconstructed frames mirror the original folder and filename structure,
but are written under a sibling folder whose name ends with ``_compressed_q{quality}``.

Example:
    python reconstruct_compressed_tiffs.py --quality 100 --csv-path results.csv
    
    # Skip decompressing darks and flats (copy from original instead)
    python reconstruct_compressed_tiffs.py --quality 100 --copy-darks-flats

The script expects ``ffmpeg`` to be available on the system PATH.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


def load_normalization_metadata(npz_path: Union[str, Path]) -> Dict:
    """
    Load normalization metadata from a .npz file saved during compression.
    
    Parameters:
    -----------
    npz_path : str or Path
        Path to the .npz file containing normalization metadata
        
    Returns:
    --------
    metadata : dict
        Dictionary containing normalization parameters:
        - use_per_frame: bool indicating if per-frame normalization was used
        - global_min: float minimum value used for normalization
        - If use_per_frame is True:
            - per_frame_max: ndarray of per-frame max values
            - percentile: float percentile value used (e.g., 99.0)
        - If use_per_frame is False:
            - global_max: float maximum value used for normalization
        - use_attenuation: bool indicating if attenuation mode was used
        - If use_attenuation is True:
            - dark_mean: ndarray of mean dark-field image
            - flat_mean: ndarray of mean flat-field image
    """
    npz_path = Path(npz_path)
    
    if not npz_path.exists():
        raise FileNotFoundError(f"Normalization metadata file not found: {npz_path}")
    
    data = np.load(npz_path)
    metadata = {key: data[key] for key in data.files}
    
    # Convert scalar arrays to Python types for convenience
    for key in ['use_per_frame', 'global_min', 'global_max', 'percentile', 'use_attenuation']:
        if key in metadata and metadata[key].ndim == 0:
            metadata[key] = metadata[key].item()
    
    return metadata


def attenuation_to_raw_projection(
    attenuation: np.ndarray,
    dark_mean: np.ndarray,
    flat_mean: np.ndarray,
) -> np.ndarray:
    """
    Convert attenuation values back to raw projection intensity.
    
    This reverses the attenuation formula: μ = -ln((I - dark) / (flat - dark))
    Solving for I: I = exp(-μ) * (flat - dark) + dark
    
    Parameters:
    -----------
    attenuation : ndarray
        Attenuation values (μ)
    dark_mean : ndarray
        Mean dark-field image
    flat_mean : ndarray
        Mean flat-field image
        
    Returns:
    --------
    raw_projection : ndarray
        Reconstructed raw projection intensity as uint16
    """
    # Compute transmission from attenuation: T = exp(-μ)
    transmission = np.exp(-attenuation)
    
    # Compute raw projection: I = T * (flat - dark) + dark
    raw_projection = transmission * (flat_mean - dark_mean) + dark_mean
    
    # Clip and convert to uint16
    raw_projection = np.clip(np.rint(raw_projection), 0, 65535).astype(np.uint16)
    
    return raw_projection


def find_npz_for_video(video_path: Path) -> Optional[Path]:
    """
    Find the corresponding .npz metadata file for a video file.
    
    Parameters:
    -----------
    video_path : Path
        Path to the compressed video file
        
    Returns:
    --------
    npz_path : Path or None
        Path to the .npz file if it exists, None otherwise
    """
    npz_path = video_path.with_suffix('.npz')
    if npz_path.exists():
        return npz_path
    return None


@dataclass
class TomographyEntry:
    """Entry representing one folder's tomographic compression results."""
    folder_name: str
    quality: int
    num_darks: int
    num_flats: int
    num_projections: int
    width: int
    height: int
    
    # Darks
    darks_output_file: str
    darks_range_min: float
    darks_range_max: float
    
    # Flats
    flats_output_file: str
    flats_range_min: float
    flats_range_max: float
    
    # Projections
    projections_output_file: str
    projections_range_min: float
    projections_range_max: float


def parse_args() -> argparse.Namespace:
    repo_root = Path("/das/home/barbaf_l/p22274/compression_paper").resolve()
    default_csv = repo_root / "streaming_output_test" / "tomography_compression_results_20251120_172833.csv"
    default_ct_base = repo_root

    parser = argparse.ArgumentParser(description="Reconstruct TIFF sequences from compressed tomographic videos")
    parser.add_argument("--quality", type=int, required=True, help="Quality level (e.g. 90, 95, 100) to reconstruct")
    parser.add_argument("--csv-path", type=Path, default=default_csv, help="Path to tomography_compression_results.csv")
    parser.add_argument(
        "--ct-base-path",
        type=Path,
        default=default_ct_base,
        help="Base directory containing original file_*_extracted folders",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_compressed",
        help="Suffix to append when creating reconstructed folders",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing TIFF files if reconstruction folder already exists",
    )
    parser.add_argument(
        "--limit-folders",
        nargs="*",
        help="Optional list of folder names to restrict reconstruction",
    )
    parser.add_argument(
        "--copy-darks-flats",
        action="store_true",
        help="Copy darks and flats from original folder instead of decompressing",
    )
    parser.add_argument(
        "--compute-metrics",
        action="store_true",
        help="Compute PSNR and SSIM metrics comparing decompressed to original TIFFs",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of parallel workers for metric computation",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="hvec-projections",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity (optional)",
    )
    parser.add_argument(
        "--wandb-disable",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--ffmpeg-binary",
        type=str,
        default="ffmpeg",
        help="ffmpeg executable to use for decoding",
    )
    return parser.parse_args()


def load_entries(csv_path: Path, quality: int) -> List[TomographyEntry]:
    """Load tomography compression entries from CSV for a specific quality level."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    entries: List[TomographyEntry] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_quality = int(row["quality"])
            if row_quality != quality:
                continue

            entry = TomographyEntry(
                folder_name=row["folder_name"],
                quality=row_quality,
                num_darks=int(row["num_darks"]),
                num_flats=int(row["num_flats"]),
                num_projections=int(row["projections_frames"]),
                width=int(row["width"]),
                height=int(row["height"]),
                darks_output_file=row["darks_output_file"],
                darks_range_min=float(row["darks_range_min"]),
                darks_range_max=float(row["darks_range_max"]),
                flats_output_file=row["flats_output_file"],
                flats_range_min=float(row["flats_range_min"]),
                flats_range_max=float(row["flats_range_max"]),
                projections_output_file=row["projections_output_file"],
                projections_range_min=float(row["projections_range_min"]),
                projections_range_max=float(row["projections_range_max"]),
            )
            entries.append(entry)

    if not entries:
        raise ValueError(f"No entries found in {csv_path} for quality {quality}")

    return entries


def compute_destination_folder(
    original_folder_name: str,
    base_path: Path,
    suffix: str,
    quality: int,
) -> Path:
    """Compute the destination folder for reconstructed TIFFs."""
    suffix = suffix or ""
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix

    quality_tag = f"_q{quality}"

    if original_folder_name.endswith("_extracted"):
        dest_folder_name = original_folder_name[: -len("_extracted")] + suffix + quality_tag
    else:
        dest_folder_name = original_folder_name + suffix + quality_tag

    return base_path / dest_folder_name


def list_original_tiffs(original_dir: Path) -> List[Path]:
    """Get list of TIFF files from the original folder."""
    if not original_dir.exists():
        raise FileNotFoundError(f"Original folder missing: {original_dir}")

    tiff_patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    files: List[Path] = []
    for pattern in tiff_patterns:
        files.extend(sorted(original_dir.glob(pattern)))

    if not files:
        raise FileNotFoundError(f"No TIFF files found in {original_dir}")

    return files


def decode_video_to_frames(
    ffmpeg_binary: str,
    video_path: Path,
    width: int,
    height: int,
    num_frames: int,
    range_min: float,
    range_max: float,
    per_frame_max: Optional[np.ndarray] = None,
    use_attenuation: bool = False,
    dark_mean: Optional[np.ndarray] = None,
    flat_mean: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """Decode a single video file and denormalize frames using the given range.
    
    Parameters:
    -----------
    ffmpeg_binary : str
        Path to ffmpeg executable
    video_path : Path
        Path to the compressed video file
    width : int
        Frame width in pixels
    height : int
        Frame height in pixels
    num_frames : int
        Number of frames to decode
    range_min : float
        Global minimum value for denormalization
    range_max : float
        Global maximum value for denormalization (used when per_frame_max is None)
    per_frame_max : ndarray, optional
        Array of per-frame maximum values. If provided, uses per-frame
        normalization where each frame has its own range [range_min, per_frame_max[i]].
        If None, uses global range [range_min, range_max] for all frames.
    use_attenuation : bool
        If True, the video contains attenuation values. After denormalization,
        convert back to raw projection using dark_mean and flat_mean.
    dark_mean : ndarray, optional
        Mean dark-field image (required if use_attenuation=True)
    flat_mean : ndarray, optional
        Mean flat-field image (required if use_attenuation=True)
        
    Returns:
    --------
    frames : List[np.ndarray]
        List of denormalized frames as uint16 arrays (raw projections if use_attenuation)
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if use_attenuation and (dark_mean is None or flat_mean is None):
        raise ValueError("dark_mean and flat_mean are required when use_attenuation=True")
    
    use_per_frame = per_frame_max is not None
    if not use_per_frame:
        span = range_max - range_min
    
    frame_size_bytes = width * height * 2

    cmd = [
        ffmpeg_binary,
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray16le",
        "pipe:1",
    ]

    frames = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
        assert proc.stdout is not None
        for frame_idx in range(num_frames):
            chunk = proc.stdout.read(frame_size_bytes)
            if len(chunk) == 0:
                break
            if len(chunk) < frame_size_bytes:
                raise RuntimeError(
                    f"Unexpected EOF while decoding {video_path}; expected more frame data"
                )

            frame = np.frombuffer(chunk, dtype=np.uint16).reshape(height, width)

            # Denormalize using per-frame or global range
            normalized = frame.astype(np.float32) / 65535.0
            
            if use_per_frame:
                # Per-frame denormalization: each frame has its own max
                frame_max = per_frame_max[frame_idx]
                span = frame_max - range_min
                if span > 0:
                    restored = normalized * span + range_min
                else:
                    restored = np.full_like(frame, fill_value=range_min, dtype=np.float32)
            else:
                # Global denormalization
                if span > 0:
                    restored = normalized * span + range_min
                else:
                    restored = np.full_like(frame, fill_value=range_min, dtype=np.float32)

            # If attenuation mode, convert back to raw projection
            if use_attenuation:
                # restored contains attenuation values (μ), convert to raw projection
                restored = attenuation_to_raw_projection(restored, dark_mean, flat_mean)
            else:
                # Standard mode: clip and convert to uint16
                restored = np.clip(np.rint(restored), 0, 65535).astype(np.uint16)
            
            frames.append(restored)

        proc.stdout.close()
        return_code = proc.wait()
        if return_code not in (0, None):
            raise RuntimeError(
                f"ffmpeg exited with code {return_code} while decoding {video_path}"
            )

    return frames


def copy_tiff_files(
    source_dir: Path,
    dest_dir: Path,
    start_idx: int,
    count: int,
    overwrite: bool,
) -> None:
    """Copy TIFF files from source to destination directory."""
    source_files = list_original_tiffs(source_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(count):
        if start_idx + i >= len(source_files):
            raise IndexError(f"Not enough files in source directory (need {count}, have {len(source_files)})")
        
        source_file = source_files[start_idx + i]
        dest_file = dest_dir / source_file.name
        
        if dest_file.exists() and not overwrite:
            raise FileExistsError(
                f"Destination file already exists (use --overwrite to replace): {dest_file}"
            )
        
        shutil.copy2(source_file, dest_file)


def _compute_tiff_pair_metrics(args: Tuple[Path, Path, float]) -> Dict[str, Any]:
    """Worker function to compute SSIM and MSE for a pair of TIFF files."""
    file1, file2, data_range = args
    
    try:
        # Read both images
        img1 = np.array(Image.open(file1))
        img2 = np.array(Image.open(file2))
        
        # Handle different image formats
        if img1.ndim == 3:
            img1 = img1[:, :, 0]
        if img2.ndim == 3:
            img2 = img2[:, :, 0]
        
        # Ensure same shape
        if img1.shape != img2.shape:
            raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
        
        # Convert to float for computation
        img1_f = img1.astype(np.float32)
        img2_f = img2.astype(np.float32)
        
        # Compute SSIM
        ssim_value = ssim(
            img1_f, img2_f,
            data_range=data_range,
            K1=0.01,
            K2=0.03,
            sigma=1.5,
            use_sample_covariance=False,
            gaussian_weights=True,
        )
        
        # Compute MSE
        diff = img1_f - img2_f
        mse_value = float(np.mean(diff ** 2))
        
        # Compute PSNR
        if mse_value > 0:
            psnr_value = 10.0 * np.log10((data_range ** 2) / mse_value)
        else:
            psnr_value = float('inf')
        
        return {
            'filename': file1.name,
            'ssim': float(ssim_value),
            'mse': mse_value,
            'psnr': float(psnr_value),
            'success': True,
        }
    except Exception as e:
        return {
            'filename': file1.name,
            'error': str(e),
            'success': False,
        }


def compute_reconstruction_metrics(
    original_dir: Path,
    reconstructed_dir: Path,
    n_workers: int = 4,
) -> Dict[str, Any]:
    """Compute PSNR and SSIM metrics between original and reconstructed TIFF sequences."""
    
    print(f"\n   📊 Computing reconstruction metrics...")
    
    # Get file lists
    original_files = list_original_tiffs(original_dir)
    reconstructed_files = list_original_tiffs(reconstructed_dir)
    
    if len(original_files) != len(reconstructed_files):
        raise ValueError(
            f"File count mismatch: {len(original_files)} original vs {len(reconstructed_files)} reconstructed"
        )
    
    n_files = len(original_files)
    
    # Determine data range by reading a sample of original files
    print(f"      Analyzing {n_files} file pairs with {n_workers} workers...")
    sample_size = min(10, n_files)
    sample_indices = np.linspace(0, n_files - 1, sample_size, dtype=int)
    
    min_vals = []
    max_vals = []
    for idx in sample_indices:
        img = np.array(Image.open(original_files[idx]))
        if img.ndim == 3:
            img = img[:, :, 0]
        min_vals.append(float(np.min(img)))
        max_vals.append(float(np.max(img)))
    
    data_range = float(np.max(max_vals) - np.min(min_vals))
    print(f"      Data range: {data_range:.2f}")
    
    # Prepare worker arguments
    worker_args = [
        (original_files[i], reconstructed_files[i], data_range)
        for i in range(n_files)
    ]
    
    # Process in parallel
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_compute_tiff_pair_metrics, args) for args in worker_args]
        for fut in tqdm(as_completed(futures), total=n_files, desc="      Computing metrics", unit="file"):
            result = fut.result()
            if result['success']:
                results.append(result)
            else:
                print(f"      ⚠️  Failed to process {result['filename']}: {result.get('error')}")
    
    if len(results) == 0:
        raise RuntimeError("No metrics computed successfully")
    
    # Aggregate results
    ssim_values = [r['ssim'] for r in results]
    mse_values = [r['mse'] for r in results]
    psnr_values = [r['psnr'] for r in results if r['psnr'] != float('inf')]
    
    metrics = {
        'n_files': len(results),
        'mean_ssim': float(np.mean(ssim_values)),
        'std_ssim': float(np.std(ssim_values)),
        'min_ssim': float(np.min(ssim_values)),
        'max_ssim': float(np.max(ssim_values)),
        'mean_mse': float(np.mean(mse_values)),
        'std_mse': float(np.std(mse_values)),
        'mean_psnr': float(np.mean(psnr_values)) if psnr_values else float('inf'),
        'std_psnr': float(np.std(psnr_values)) if psnr_values else 0.0,
        'min_psnr': float(np.min(psnr_values)) if psnr_values else float('inf'),
        'max_psnr': float(np.max(psnr_values)) if psnr_values else float('inf'),
        'data_range': data_range,
    }
    
    # Save detailed results to CSV
    metrics_csv = reconstructed_dir / "_reconstruction_metrics.csv"
    with open(metrics_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'ssim', 'mse', 'psnr'])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'filename': r['filename'],
                'ssim': r['ssim'],
                'mse': r['mse'],
                'psnr': r['psnr'],
            })
    
    # Save summary to JSON
    metrics_json = reconstructed_dir / "_reconstruction_metrics_summary.json"
    with open(metrics_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"      ✅ Metrics computed:")
    print(f"         Mean SSIM: {metrics['mean_ssim']:.6f} ± {metrics['std_ssim']:.6f}")
    print(f"         Mean PSNR: {metrics['mean_psnr']:.2f} dB ± {metrics['std_psnr']:.2f} dB")
    print(f"         Mean MSE:  {metrics['mean_mse']:.4f} ± {metrics['std_mse']:.4f}")
    print(f"         Results saved to:")
    print(f"           {metrics_csv}")
    print(f"           {metrics_json}")
    
    return metrics


def try_import_wandb():
    """Try to import wandb, return None if not available."""
    try:
        import wandb  # type: ignore
        return wandb
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    entries = load_entries(args.csv_path, args.quality)

    limit_set = set(args.limit_folders) if args.limit_folders else None

    # Initialize wandb if metrics are being computed and not disabled
    wandb = None
    if args.compute_metrics and not args.wandb_disable:
        wandb = try_import_wandb()
        if wandb is None:
            print("⚠️  wandb not available. Proceeding without online logging.")

    for entry in entries:
        if limit_set and entry.folder_name not in limit_set:
            continue

        original_dir = args.ct_base_path / entry.folder_name
        dest_dir = compute_destination_folder(
            entry.folder_name,
            args.ct_base_path,
            args.output_suffix,
            entry.quality,
        )

        print(f"\n📁 Processing {entry.folder_name} (quality {entry.quality})")
        print(f"   Original folder: {original_dir}")
        print(f"   Destination folder: {dest_dir}")
        print(f"   Structure: {entry.num_darks} darks + {entry.num_flats} flats + {entry.num_projections} projections")

        # Get original file names
        original_files = list_original_tiffs(original_dir)
        if len(original_files) < entry.num_darks + entry.num_flats + entry.num_projections:
            raise ValueError(
                f"Not enough files in original folder: found {len(original_files)}, "
                f"need {entry.num_darks + entry.num_flats + entry.num_projections}"
            )

        dest_dir.mkdir(parents=True, exist_ok=True)

        # Process darks
        if args.copy_darks_flats:
            print(f"   📋 Copying {entry.num_darks} dark frames from original...")
            copy_tiff_files(original_dir, dest_dir, 0, entry.num_darks, args.overwrite)
        else:
            print(f"   🎬 Decompressing {entry.num_darks} dark frames...")
            print(f"      Video: {entry.darks_output_file}")
            
            # Check for per-frame normalization metadata
            darks_npz_path = find_npz_for_video(Path(entry.darks_output_file))
            darks_per_frame_max = None
            if darks_npz_path is not None:
                darks_metadata = load_normalization_metadata(darks_npz_path)
                if darks_metadata.get('use_per_frame', False):
                    darks_per_frame_max = darks_metadata['per_frame_max']
                    print(f"      Using per-frame normalization from: {darks_npz_path.name}")
                else:
                    print(f"      Using global range: [{entry.darks_range_min}, {entry.darks_range_max}]")
            else:
                print(f"      Range: [{entry.darks_range_min}, {entry.darks_range_max}]")
            
            dark_frames = decode_video_to_frames(
                args.ffmpeg_binary,
                Path(entry.darks_output_file),
                entry.width,
                entry.height,
                entry.num_darks,
                entry.darks_range_min,
                entry.darks_range_max,
                per_frame_max=darks_per_frame_max,
            )
            for i, frame in enumerate(dark_frames):
                dest_path = dest_dir / original_files[i].name
                if dest_path.exists() and not args.overwrite:
                    raise FileExistsError(
                        f"Destination file already exists (use --overwrite): {dest_path}"
                    )
                Image.fromarray(frame).save(dest_path)

        # Process flats
        flat_start_idx = entry.num_darks
        if args.copy_darks_flats:
            print(f"   📋 Copying {entry.num_flats} flat frames from original...")
            copy_tiff_files(original_dir, dest_dir, flat_start_idx, entry.num_flats, args.overwrite)
        else:
            print(f"   🎬 Decompressing {entry.num_flats} flat frames...")
            print(f"      Video: {entry.flats_output_file}")
            
            # Check for per-frame normalization metadata
            flats_npz_path = find_npz_for_video(Path(entry.flats_output_file))
            flats_per_frame_max = None
            if flats_npz_path is not None:
                flats_metadata = load_normalization_metadata(flats_npz_path)
                if flats_metadata.get('use_per_frame', False):
                    flats_per_frame_max = flats_metadata['per_frame_max']
                    print(f"      Using per-frame normalization from: {flats_npz_path.name}")
                else:
                    print(f"      Using global range: [{entry.flats_range_min}, {entry.flats_range_max}]")
            else:
                print(f"      Range: [{entry.flats_range_min}, {entry.flats_range_max}]")
            
            flat_frames = decode_video_to_frames(
                args.ffmpeg_binary,
                Path(entry.flats_output_file),
                entry.width,
                entry.height,
                entry.num_flats,
                entry.flats_range_min,
                entry.flats_range_max,
                per_frame_max=flats_per_frame_max,
            )
            for i, frame in enumerate(flat_frames):
                dest_path = dest_dir / original_files[flat_start_idx + i].name
                if dest_path.exists() and not args.overwrite:
                    raise FileExistsError(
                        f"Destination file already exists (use --overwrite): {dest_path}"
                    )
                Image.fromarray(frame).save(dest_path)

        # Process projections
        proj_start_idx = entry.num_darks + entry.num_flats
        print(f"   🎬 Decompressing {entry.num_projections} projection frames...")
        print(f"      Video: {entry.projections_output_file}")
        
        # Check for per-frame normalization metadata and attenuation mode
        proj_npz_path = find_npz_for_video(Path(entry.projections_output_file))
        proj_per_frame_max = None
        proj_use_attenuation = False
        proj_dark_mean = None
        proj_flat_mean = None
        
        if proj_npz_path is not None:
            proj_metadata = load_normalization_metadata(proj_npz_path)
            if proj_metadata.get('use_per_frame', False):
                proj_per_frame_max = proj_metadata['per_frame_max']
                print(f"      Using per-frame normalization from: {proj_npz_path.name}")
            else:
                print(f"      Using global range: [{entry.projections_range_min}, {entry.projections_range_max}]")
            
            # Check for attenuation mode
            proj_use_attenuation = proj_metadata.get('use_attenuation', False)
            if proj_use_attenuation:
                proj_dark_mean = proj_metadata.get('dark_mean')
                proj_flat_mean = proj_metadata.get('flat_mean')
                if proj_dark_mean is not None and proj_flat_mean is not None:
                    print(f"      📐 ATTENUATION MODE: Converting μ back to raw projections")
                else:
                    print(f"      ⚠️  Attenuation mode but missing dark/flat means - outputting attenuation values")
                    proj_use_attenuation = False
        else:
            print(f"      Range: [{entry.projections_range_min}, {entry.projections_range_max}]")
        
        projection_frames = decode_video_to_frames(
            args.ffmpeg_binary,
            Path(entry.projections_output_file),
            entry.width,
            entry.height,
            entry.num_projections,
            entry.projections_range_min,
            entry.projections_range_max,
            per_frame_max=proj_per_frame_max,
            use_attenuation=proj_use_attenuation,
            dark_mean=proj_dark_mean,
            flat_mean=proj_flat_mean,
        )
        for i, frame in enumerate(projection_frames):
            dest_path = dest_dir / original_files[proj_start_idx + i].name
            if dest_path.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Destination file already exists (use --overwrite): {dest_path}"
                )
            Image.fromarray(frame).save(dest_path)

        # Handle extra files if they exist
        extra_start_idx = entry.num_darks + entry.num_flats + entry.num_projections
        if extra_start_idx < len(original_files):
            num_extra = len(original_files) - extra_start_idx
            print(f"   📋 Copying {num_extra} extra frames from original...")
            copy_tiff_files(original_dir, dest_dir, extra_start_idx, num_extra, args.overwrite)

        print(f"   ✅ Reconstruction complete: {dest_dir}")
        print(f"      Total files: {len(list(dest_dir.glob('*.tif*')))}")

        # Compute metrics if requested
        if args.compute_metrics:
            wb_run = None
            if wandb is not None:
                run_name = f"{entry.folder_name}_q{entry.quality}"
                wb_run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=run_name,
                    config={
                        'folder_name': entry.folder_name,
                        'quality': entry.quality,
                        'num_darks': entry.num_darks,
                        'num_flats': entry.num_flats,
                        'num_projections': entry.num_projections,
                        'width': entry.width,
                        'height': entry.height,
                        'copy_darks_flats': args.copy_darks_flats,
                        'darks_range': [entry.darks_range_min, entry.darks_range_max],
                        'flats_range': [entry.flats_range_min, entry.flats_range_max],
                        'projections_range': [entry.projections_range_min, entry.projections_range_max],
                    },
                    reinit=True,
                )
            
            try:
                metrics = compute_reconstruction_metrics(
                    original_dir=original_dir,
                    reconstructed_dir=dest_dir,
                    n_workers=args.n_workers,
                )
                
                # Log to wandb
                if wb_run is not None:
                    log_payload: Dict[str, Any] = {
                        'reconstruction/n_files': metrics['n_files'],
                        'reconstruction/mean_ssim': metrics['mean_ssim'],
                        'reconstruction/std_ssim': metrics['std_ssim'],
                        'reconstruction/min_ssim': metrics['min_ssim'],
                        'reconstruction/max_ssim': metrics['max_ssim'],
                        'reconstruction/mean_mse': metrics['mean_mse'],
                        'reconstruction/std_mse': metrics['std_mse'],
                        'reconstruction/mean_psnr': metrics['mean_psnr'],
                        'reconstruction/std_psnr': metrics['std_psnr'],
                        'reconstruction/min_psnr': metrics['min_psnr'],
                        'reconstruction/max_psnr': metrics['max_psnr'],
                        'reconstruction/data_range': metrics['data_range'],
                    }
                    wandb.log(log_payload)
                    
                    # Attach CSV artifact
                    try:
                        metrics_csv = dest_dir / "_reconstruction_metrics.csv"
                        metrics_json = dest_dir / "_reconstruction_metrics_summary.json"
                        art = wandb.Artifact(name=f"{run_name}-reconstruction-metrics", type="reconstruction")
                        art.add_file(str(metrics_csv))
                        art.add_file(str(metrics_json))
                        wandb.log_artifact(art)
                    except Exception as e:
                        print(f"      ⚠️  Failed to log artifacts to wandb: {e}")
                    
            except Exception as e:
                print(f"   ❌ Error computing metrics: {e}")
            finally:
                if wb_run is not None:
                    wb_run.finish()

    print("\n🎉 All requested reconstructions completed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        sys.exit(130)
