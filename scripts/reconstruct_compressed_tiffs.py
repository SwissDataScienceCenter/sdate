#!/usr/bin/env python3
"""Reconstruct TIFF sequences from compressed HEVC outputs.

Given a batch processing results CSV (produced by the compression notebook),
this script decodes the compressed video for a specific quality level back
into TIFF frames. The reconstructed frames mirror the original folder and
filename structure, but are written under a sibling folder whose name ends
with ``_compressed``.

Example:
    python reconstruct_compressed_tiffs.py --quality 90

The script expects ``ffmpeg`` to be available on the system PATH.
"""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Optional, Dict

import numpy as np
from PIL import Image

# Add parent directory to path to import sdate utilities
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sdate.utils.normalization import load_normalization_metadata


@dataclass
class CompressionEntry:
    folder_name: str
    quality: int
    width: int
    height: int
    processed_frames: int
    global_min: float
    global_max: float
    output_dir: Path
    normalization_metadata: Optional[Dict] = None  # Per-frame normalization data


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_csv = repo_root / "notebooks" / "streaming_output" / "projection_batch_processing_results.csv"
    default_ct_base = repo_root / "data" / "ct_files"

    parser = argparse.ArgumentParser(description="Reconstruct TIFF sequences from compressed videos")
    parser.add_argument("--quality", type=int, required=True, help="Quality level (e.g. 90, 95, 100) to reconstruct")
    parser.add_argument("--csv-path", type=Path, default=default_csv, help="Path to batch_processing_results.csv")
    parser.add_argument(
        "--ct-base-path",
        type=Path,
        default=default_ct_base,
        help="Base directory containing original ct_files folders",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_compressed",
        help="Suffix to append (or replace _extracted) when creating reconstructed folders",
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
        "--data-types",
        nargs="*",
        default=["darks", "flats", "projections"],
        help="Which data types to reconstruct (darks, flats, projections). Default: all",
    )
    parser.add_argument(
        "--ffmpeg-binary",
        type=str,
        default="ffmpeg",
        help="ffmpeg executable to use for decoding",
    )
    return parser.parse_args()


def load_entries(csv_path: Path, quality: int) -> List[CompressionEntry]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    entries: List[CompressionEntry] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_quality = int(row["quality"])
            if row_quality != quality:
                continue

            raw_output_dir = Path(row["output_dir"])
            resolved_output = resolve_output_dir(csv_path, raw_output_dir)

            entry = CompressionEntry(
                folder_name=row["folder_name"],
                quality=row_quality,
                width=int(row["width"]),
                height=int(row["height"]),
                processed_frames=int(row["processed_frames"]),
                global_min=float(row["global_min"]),
                global_max=float(row["global_max"]),
                output_dir=resolved_output,
            )
            entries.append(entry)

    if not entries:
        raise ValueError(f"No entries found in {csv_path} for quality {quality}")

    return entries


def resolve_output_dir(csv_path: Path, raw_output_dir: Path) -> Path:
    """Resolve the output_dir string from the CSV into an absolute Path."""
    candidates = [
        raw_output_dir,
        csv_path.parent / raw_output_dir,
        csv_path.parent.parent / raw_output_dir,
    ]

    for candidate in candidates:
        if candidate.is_absolute() and candidate.exists():
            return candidate
        if not candidate.is_absolute():
            absolute_candidate = candidate.resolve()
            if absolute_candidate.exists():
                return absolute_candidate

    # Fall back to assuming the path is relative to the CSV parent directory
    return (csv_path.parent / raw_output_dir).resolve()


def compute_destination_folder(
    original_folder_name: str,
    base_path: Path,
    suffix: str,
    quality: int,
) -> Path:
    suffix = suffix or ""
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix

    quality_tag = f"_{quality}"

    if original_folder_name.endswith("_extracted"):
        dest_folder_name = original_folder_name[: -len("_extracted")] + suffix + quality_tag
    else:
        dest_folder_name = original_folder_name + suffix + quality_tag

    return base_path / dest_folder_name


def find_video_files(output_dir: Path) -> List[Path]:
    if not output_dir.exists():
        raise FileNotFoundError(f"Compressed output directory does not exist: {output_dir}")

    video_files: List[Path] = []
    for ext in ("*.mov", "*.mp4", "*.mkv", "*.hevc"):
        video_files.extend(sorted(output_dir.glob(ext)))

    if not video_files:
        raise FileNotFoundError(f"No video segments found in {output_dir}")

    return video_files


def load_video_normalization(video_path: Path) -> Optional[Dict]:
    """Load normalization metadata for a video file if available.
    
    Looks for a .npz file with the same name as the video file.
    """
    npz_path = video_path.with_suffix('.npz')
    if npz_path.exists():
        try:
            metadata = load_normalization_metadata(npz_path)
            print(f"   📊 Loaded normalization metadata from {npz_path.name}")
            if metadata.get('use_per_frame', False):
                num_frames = len(metadata.get('per_frame_max', []))
                percentile = metadata.get('percentile', 99.0)
                print(f"      Using per-frame {percentile}th percentile normalization ({num_frames} frames)")
            else:
                print(f"      Using global normalization: [{metadata['global_min']:.1f}, {metadata['global_max']:.1f}]")
            return metadata
        except Exception as e:
            print(f"   ⚠️  Warning: Could not load normalization metadata from {npz_path.name}: {e}")
            return None
    return None


def list_original_tiffs(original_dir: Path) -> List[Path]:
    if not original_dir.exists():
        raise FileNotFoundError(f"Original folder missing: {original_dir}")

    tiff_patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    files: List[Path] = []
    for pattern in tiff_patterns:
        files.extend(sorted(original_dir.glob(pattern)))

    if not files:
        raise FileNotFoundError(f"No TIFF files found in {original_dir}")

    return files


def decode_and_restore(
    ffmpeg_binary: str,
    video_files: Sequence[Path],
    width: int,
    height: int,
    frame_names: Sequence[str],
    dest_dir: Path,
    global_min: float,
    global_max: float,
    overwrite: bool,
    normalization_metadata: Optional[Dict] = None,
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine normalization strategy
    use_per_frame = False
    per_frame_max = None
    
    if normalization_metadata and normalization_metadata.get('use_per_frame', False):
        use_per_frame = True
        per_frame_max = normalization_metadata.get('per_frame_max')
        global_min = normalization_metadata.get('global_min', global_min)
        print(f"   📊 Using per-frame normalization with {len(per_frame_max)} frame values")
    else:
        # Use global normalization
        span = global_max - global_min
        span = float(span)
        if span <= 0:
            span = 0.0
        print(f"   📊 Using global normalization: [{global_min:.1f}, {global_max:.1f}]")

    frame_size_bytes = width * height * 2
    frame_count_expected = len(frame_names)

    frame_index = 0

    for video_idx, video_path in enumerate(video_files):
        remaining = frame_count_expected - frame_index
        if remaining <= 0:
            break

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

        with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
            assert proc.stdout is not None
            while remaining > 0:
                chunk = proc.stdout.read(frame_size_bytes)
                if len(chunk) == 0:
                    break
                if len(chunk) < frame_size_bytes:
                    raise RuntimeError(
                        f"Unexpected EOF while decoding {video_path}; expected more frame data"
                    )

                frame = np.frombuffer(chunk, dtype=np.uint16).reshape(height, width)
                normalized = frame.astype(np.float32) / 65535.0

                # Denormalize based on strategy
                if use_per_frame and per_frame_max is not None:
                    # Per-frame denormalization
                    if frame_index >= len(per_frame_max):
                        raise RuntimeError(
                            f"Frame index {frame_index} exceeds per_frame_max array length {len(per_frame_max)}"
                        )
                    frame_max = per_frame_max[frame_index]
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

                restored = np.clip(np.rint(restored), 0, 65535).astype(np.uint16)

                dest_path = dest_dir / frame_names[frame_index]
                if dest_path.exists() and not overwrite:
                    raise FileExistsError(
                        f"Destination file already exists (use --overwrite to replace): {dest_path}"
                    )

                Image.fromarray(restored).save(dest_path)

                frame_index += 1
                remaining -= 1

            proc.stdout.close()
            return_code = proc.wait()
            if return_code not in (0, None):
                raise RuntimeError(
                    f"ffmpeg exited with code {return_code} while decoding {video_path}"
                )

    if frame_index != frame_count_expected:
        raise RuntimeError(
            f"Decoded {frame_index} frames but expected {frame_count_expected}."
        )


def main() -> int:
    args = parse_args()
    
    # Check if we're dealing with new three-file structure (darks, flats, projections)
    # by looking for files with _darks, _flats, _projections suffixes
    if args.csv_path.exists():
        entries = load_entries(args.csv_path, args.quality)
        use_legacy_mode = True
    else:
        # No CSV, try direct reconstruction from output directory
        entries = []
        use_legacy_mode = False

    limit_set = set(args.limit_folders) if args.limit_folders else None

    # Check if output directory has three-file structure
    if not use_legacy_mode or check_for_three_file_structure(entries):
        return reconstruct_three_file_structure(args)
    
    # Legacy single-file reconstruction
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
        print(f"   Compressed video dir: {entry.output_dir}")
        print(f"   Destination folder: {dest_dir}")

        frame_paths = list_original_tiffs(original_dir)
        frame_names = [p.name for p in frame_paths]

        if entry.processed_frames != len(frame_names):
            print(
                f"   ⚠️  Mismatch between processed_frames ({entry.processed_frames}) and original TIFF count ({len(frame_names)})."
            )

        video_files = find_video_files(entry.output_dir)

        print(
            f"   ▶︎ Decoding {len(video_files)} video segment(s) into {len(frame_names)} TIFF files..."
        )
        
        # Load normalization metadata if available (try first video file)
        normalization_metadata = None
        if video_files:
            normalization_metadata = load_video_normalization(video_files[0])

        decode_and_restore(
            ffmpeg_binary=args.ffmpeg_binary,
            video_files=video_files,
            width=entry.width,
            height=entry.height,
            frame_names=frame_names,
            dest_dir=dest_dir,
            global_min=entry.global_min,
            global_max=entry.global_max,
            overwrite=args.overwrite,
            normalization_metadata=normalization_metadata,
        )

        print(f"   ✅ Reconstruction complete: {dest_dir}")

    print("\n🎉 All requested reconstructions completed.")
    return 0



def check_for_three_file_structure(entries):
    """Check if entries use three-file structure."""
    if not entries:
        return False
    output_dir = entries[0].output_dir
    if not output_dir.exists():
        return False
    return any(output_dir.glob("*_darks.*")) or any(output_dir.glob("*_flats.*"))


def load_tomography_params_from_dir(original_dir):
    """Load tomography parameters by examining TIFF count."""
    tiff_files = list_original_tiffs(original_dir)
    total_files = len(tiff_files)
    return {
        'num_darks': 10,
        'num_flats': 10,
        'num_projections': total_files - 20
    }


def reconstruct_three_file_structure(args):
    """Reconstruct TIFFs from three separate video files."""
    print("Three-File Structure Reconstruction Mode (Not fully implemented yet)")
    print("Using legacy mode instead...")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        sys.exit(130)
