#!/usr/bin/env python3
"""Reconstruct TIFF sequences from compressed HEVC outputs.

Given compressed video files and their normalization metadata (produced by
batch_compress_tomography), this script decodes the videos back into TIFF
frames. The reconstructed frames mirror the original folder and filename
structure, but are written under a sibling folder.

The script supports two normalization modes:
1. Global min/max normalization (legacy)
2. Per-frame percentile normalization (new, recommended)

Example:
    # Reconstruct from a specific compressed folder
    python reconstruct_compressed_tiffs.py --compressed-dir data/streaming_output/file_3_extracted_q90
    
    # Reconstruct with original filenames from source folder
    python reconstruct_compressed_tiffs.py --compressed-dir data/streaming_output/file_3_extracted_q90 \\
        --original-dir data/ct_files/file_3_extracted

The script expects ``ffmpeg`` to be available on the system PATH.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Add project root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sdate.utils.normalization import load_normalization_metadata


@dataclass
class VideoInfo:
    """Information about a compressed video file and its normalization."""
    video_path: Path
    npz_path: Path
    data_type: str  # 'darks', 'flats', or 'projections'
    width: int
    height: int
    num_frames: int
    metadata: Dict


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_ct_base = repo_root / "data" / "ct_files"

    parser = argparse.ArgumentParser(
        description="Reconstruct TIFF sequences from compressed videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic reconstruction
  python reconstruct_compressed_tiffs.py --compressed-dir data/streaming_output/file_3_extracted_q90

  # With original folder for filenames
  python reconstruct_compressed_tiffs.py --compressed-dir data/streaming_output/file_3_extracted_q90 \\
      --original-dir data/ct_files/file_3_extracted

  # Reconstruct only projections
  python reconstruct_compressed_tiffs.py --compressed-dir data/streaming_output/file_3_extracted_q90 \\
      --data-types projections
        """
    )
    
    parser.add_argument(
        "--compressed-dir",
        type=Path,
        required=True,
        help="Directory containing compressed video files and .npz metadata"
    )
    parser.add_argument(
        "--original-dir",
        type=Path,
        default=None,
        help="Original TIFF directory (for matching filenames). If not provided, uses sequential names."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for reconstructed TIFFs. Default: <compressed-dir>_reconstructed"
    )
    parser.add_argument(
        "--data-types",
        nargs="*",
        choices=["darks", "flats", "projections"],
        default=["darks", "flats", "projections"],
        help="Which data types to reconstruct (default: all)"
    )
    parser.add_argument(
        "--ct-base-path",
        type=Path,
        default=default_ct_base,
        help="Base directory containing original ct_files folders (for auto-detecting original-dir)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing TIFF files if reconstruction folder already exists"
    )
    parser.add_argument(
        "--ffmpeg-binary",
        type=str,
        default="ffmpeg",
        help="ffmpeg executable to use for decoding"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output"
    )
    
    return parser.parse_args()


def find_video_and_metadata(compressed_dir: Path, data_type: str) -> Optional[Tuple[Path, Path]]:
    """Find the video file and corresponding .npz metadata for a data type."""
    # Look for video files matching the data type
    video_patterns = [
        f"*_{data_type}.mov",
        f"*_{data_type}.mp4",
        f"*_{data_type}.mkv",
        f"*_{data_type}.hevc",
    ]
    
    video_files = []
    for pattern in video_patterns:
        video_files.extend(list(compressed_dir.glob(pattern)))
    
    if not video_files:
        return None
    
    # Use the first matching video
    video_path = video_files[0]
    
    # Find corresponding .npz file (same base name)
    npz_path = video_path.with_suffix('.npz')
    
    if not npz_path.exists():
        print(f"   ⚠️  Warning: Normalization metadata not found: {npz_path}")
        return None
    
    return video_path, npz_path


def get_video_dimensions(ffmpeg_binary: str, video_path: Path) -> Tuple[int, int, int]:
    """Get video dimensions and frame count using ffprobe."""
    try:
        # Get dimensions
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        width, height = map(int, result.stdout.strip().split(','))
        
        # Get frame count
        cmd = [
            "ffprobe", "-v", "error",
            "-count_frames",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        num_frames = int(result.stdout.strip())
        
        return width, height, num_frames
    except Exception as e:
        raise RuntimeError(f"Failed to get video info for {video_path}: {e}")


def discover_videos(compressed_dir: Path, data_types: List[str], ffmpeg_binary: str) -> List[VideoInfo]:
    """Discover all video files and their metadata in the compressed directory."""
    videos = []
    
    for data_type in data_types:
        result = find_video_and_metadata(compressed_dir, data_type)
        if result is None:
            continue
        
        video_path, npz_path = result
        
        # Load metadata
        try:
            metadata = load_normalization_metadata(npz_path)
        except Exception as e:
            print(f"   ⚠️  Error loading metadata for {data_type}: {e}")
            continue
        
        # Get video dimensions
        try:
            width, height, num_frames = get_video_dimensions(ffmpeg_binary, video_path)
        except Exception as e:
            print(f"   ⚠️  Error getting video info for {data_type}: {e}")
            continue
        
        videos.append(VideoInfo(
            video_path=video_path,
            npz_path=npz_path,
            data_type=data_type,
            width=width,
            height=height,
            num_frames=num_frames,
            metadata=metadata
        ))
    
    return videos


def list_original_tiffs(original_dir: Path, num_darks: int, num_flats: int, num_projections: int) -> Dict[str, List[str]]:
    """List original TIFF filenames organized by data type."""
    if not original_dir.exists():
        raise FileNotFoundError(f"Original folder missing: {original_dir}")
    
    tiff_patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    files: List[Path] = []
    for pattern in tiff_patterns:
        files.extend(sorted(original_dir.glob(pattern)))
    
    if not files:
        raise FileNotFoundError(f"No TIFF files found in {original_dir}")
    
    filenames = [f.name for f in files]
    
    return {
        'darks': filenames[:num_darks],
        'flats': filenames[num_darks:num_darks + num_flats],
        'projections': filenames[num_darks + num_flats:num_darks + num_flats + num_projections]
    }


def generate_sequential_names(data_type: str, num_frames: int) -> List[str]:
    """Generate sequential filenames for a data type."""
    return [f"{data_type}_{i:06d}.tif" for i in range(num_frames)]


def decode_and_restore(
    ffmpeg_binary: str,
    video_info: VideoInfo,
    frame_names: List[str],
    dest_dir: Path,
    overwrite: bool,
    verbose: bool = False
) -> int:
    """Decode video and restore frames using normalization metadata."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = video_info.metadata
    use_per_frame = metadata.get('use_per_frame', False)
    global_min = metadata['global_min']
    
    if use_per_frame:
        per_frame_max = metadata['per_frame_max']
        if len(per_frame_max) != video_info.num_frames:
            print(f"   ⚠️  Warning: Metadata has {len(per_frame_max)} values but video has {video_info.num_frames} frames")
    else:
        global_max = metadata['global_max']
        span = global_max - global_min
    
    frame_size_bytes = video_info.width * video_info.height * 2
    
    cmd = [
        ffmpeg_binary,
        "-loglevel", "error",
        "-i", str(video_info.video_path),
        "-f", "rawvideo",
        "-pix_fmt", "gray16le",
        "pipe:1",
    ]
    
    frames_written = 0
    
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
        assert proc.stdout is not None
        
        for frame_idx in range(video_info.num_frames):
            chunk = proc.stdout.read(frame_size_bytes)
            if len(chunk) == 0:
                break
            if len(chunk) < frame_size_bytes:
                raise RuntimeError(
                    f"Unexpected EOF while decoding {video_info.video_path}; expected more frame data"
                )
            
            # Decode to 16-bit grayscale (0-65535 range)
            frame = np.frombuffer(chunk, dtype=np.uint16).reshape(
                video_info.height, video_info.width
            )
            
            # Denormalize: convert from 0-65535 to original range
            normalized = frame.astype(np.float32) / 65535.0
            
            if use_per_frame:
                # Per-frame denormalization
                frame_max = per_frame_max[frame_idx]
                span = frame_max - global_min
                if span > 0:
                    restored = normalized * span + global_min
                else:
                    restored = np.full_like(frame, fill_value=global_min, dtype=np.float32)
            else:
                # Global denormalization
                if span > 0:
                    restored = normalized * span + global_min
                else:
                    restored = np.full_like(frame, fill_value=global_min, dtype=np.float32)
            
            # Clip and convert to uint16
            restored = np.clip(np.rint(restored), 0, 65535).astype(np.uint16)
            
            # Save frame
            if frame_idx < len(frame_names):
                dest_filename = frame_names[frame_idx]
            else:
                dest_filename = f"{video_info.data_type}_{frame_idx:06d}.tif"
            
            dest_path = dest_dir / dest_filename
            
            if dest_path.exists() and not overwrite:
                raise FileExistsError(
                    f"Destination file already exists (use --overwrite to replace): {dest_path}"
                )
            
            Image.fromarray(restored).save(dest_path)
            frames_written += 1
            
            if verbose and frame_idx % 100 == 0:
                print(f"      Processed frame {frame_idx + 1}/{video_info.num_frames}")
        
        proc.stdout.close()
        return_code = proc.wait()
        if return_code not in (0, None):
            raise RuntimeError(
                f"ffmpeg exited with code {return_code} while decoding {video_info.video_path}"
            )
    
    return frames_written


def infer_original_folder(compressed_dir: Path, ct_base_path: Path) -> Optional[Path]:
    """Try to infer the original folder from the compressed directory name."""
    # Compressed folder naming convention: <original_name>_q<quality>
    dirname = compressed_dir.name
    
    # Try to find matching folder in ct_base_path
    # Strip _q<number> suffix
    if '_q' in dirname:
        potential_name = dirname.rsplit('_q', 1)[0]
        if potential_name:
            # Try with _extracted suffix
            candidate = ct_base_path / f"{potential_name}"
            if candidate.exists():
                return candidate
            
            # Also try adding _extracted if not present
            if not potential_name.endswith('_extracted'):
                candidate = ct_base_path / f"{potential_name}_extracted"
                if candidate.exists():
                    return candidate
    
    return None


def main() -> int:
    args = parse_args()
    
    compressed_dir = args.compressed_dir.resolve()
    if not compressed_dir.exists():
        print(f"❌ Compressed directory not found: {compressed_dir}")
        return 1
    
    print("=" * 80)
    print("TIFF RECONSTRUCTION FROM COMPRESSED VIDEOS")
    print("=" * 80)
    print(f"\n📁 Compressed directory: {compressed_dir}")
    
    # Discover video files and metadata
    print(f"\n🔍 Discovering video files for: {args.data_types}")
    videos = discover_videos(compressed_dir, args.data_types, args.ffmpeg_binary)
    
    if not videos:
        print("❌ No valid video files with metadata found")
        return 1
    
    print(f"   Found {len(videos)} video(s) to reconstruct:")
    total_frames = 0
    for v in videos:
        norm_type = "per-frame" if v.metadata.get('use_per_frame', False) else "global"
        print(f"   - {v.data_type}: {v.num_frames} frames, {v.width}x{v.height}, {norm_type} normalization")
        total_frames += v.num_frames
    
    # Determine original folder for filenames
    original_dir = args.original_dir
    if original_dir is None:
        original_dir = infer_original_folder(compressed_dir, args.ct_base_path)
        if original_dir:
            print(f"\n📂 Auto-detected original folder: {original_dir}")
    
    # Determine frame names for each data type
    frame_names_by_type: Dict[str, List[str]] = {}
    
    if original_dir and original_dir.exists():
        try:
            # Calculate frame counts from videos
            num_darks = next((v.num_frames for v in videos if v.data_type == 'darks'), 0)
            num_flats = next((v.num_frames for v in videos if v.data_type == 'flats'), 0)
            num_projections = next((v.num_frames for v in videos if v.data_type == 'projections'), 0)
            
            frame_names_by_type = list_original_tiffs(
                original_dir, num_darks, num_flats, num_projections
            )
            print(f"   Using original filenames from {original_dir.name}")
        except Exception as e:
            print(f"   ⚠️  Could not read original filenames: {e}")
            print("   Using sequential naming instead")
            for v in videos:
                frame_names_by_type[v.data_type] = generate_sequential_names(v.data_type, v.num_frames)
    else:
        print("\n📝 Using sequential filenames (no original folder specified)")
        for v in videos:
            frame_names_by_type[v.data_type] = generate_sequential_names(v.data_type, v.num_frames)
    
    # Determine output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = compressed_dir.parent / f"{compressed_dir.name}_reconstructed"
    output_dir = output_dir.resolve()
    
    print(f"\n💾 Output directory: {output_dir}")
    
    # Reconstruct each video
    print(f"\n🎬 Reconstructing {total_frames} frames from {len(videos)} videos...")
    
    all_frames_written = 0
    for video_info in videos:
        print(f"\n   ▶️  {video_info.data_type.capitalize()}: {video_info.num_frames} frames")
        print(f"      Source: {video_info.video_path.name}")
        print(f"      Metadata: {video_info.npz_path.name}")
        
        norm_info = "per-frame" if video_info.metadata.get('use_per_frame', False) else "global"
        percentile = video_info.metadata.get('percentile', 'N/A')
        print(f"      Normalization: {norm_info}" + (f" ({percentile}th percentile)" if norm_info == "per-frame" else ""))
        
        frame_names = frame_names_by_type.get(video_info.data_type, [])
        if not frame_names:
            frame_names = generate_sequential_names(video_info.data_type, video_info.num_frames)
        
        try:
            frames_written = decode_and_restore(
                ffmpeg_binary=args.ffmpeg_binary,
                video_info=video_info,
                frame_names=frame_names,
                dest_dir=output_dir,
                overwrite=args.overwrite,
                verbose=args.verbose
            )
            all_frames_written += frames_written
            print(f"      ✅ Wrote {frames_written} TIFF files")
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    print(f"\n{'=' * 80}")
    print(f"🎉 Reconstruction complete!")
    print(f"   Total frames written: {all_frames_written}")
    print(f"   Output directory: {output_dir}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
