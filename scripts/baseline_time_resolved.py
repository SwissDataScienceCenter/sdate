#!/usr/bin/env python3
"""
Baseline movie generation for time-resolved 4D CT.

Computes ground-truth (GT) and FBP-based baseline reconstructions for every
frame in a timesteps directory and writes HEVC movies in a fully streaming
fashion — volumes are never accumulated in memory across frames.

Baselines generated per frame
------------------------------
- ``gt``      : normalised ground-truth volume loaded directly from the tif
- ``gt_fbp``  : FBP reconstruction from full-angle sinogram (quality ceiling)
- ``sw_fbp``  : FBP from the combined sliding-window limited-angle sinogram
                (uses ``file_range`` neighbouring frames when available)
- ``la_fbp``  : FBP from the single-frame limited-angle sinogram only

Each baseline produces one ``.mov`` file per selected slice showing that
slice evolving over all processed time frames.

Usage example
-------------
    python scripts/baseline_time_resolved.py --data_path /myhome/data/sdate/shared/time_resolved/149_ASM_SP_1ktps/timesteps/149_ASM_SP_1ktps_rotate_35001.tif --output_dir /myhome/data/sdate/shared/time_resolved/149_ASM_SP_1ktps/baseline_output --norm_config /myhome/sdate/checkpoints/ASM_SP_1ktps_time_resolved_norm.json --image_size 256 --angle_start 0 --angle_range 36 --num_full_projs 1000 --file_range 5 --num_movie_slices 10
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, "/myhome/astra-torch")
sys.path.insert(0, "/myhome/chip-project")

import wandb
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from ladiff.datasets import TifVolumeSliceDataset
from astra_torch.lamino import (
    build_lamino_projector,
    fbp_reconstruction_masked,
)
from sdate.stream_hvec import HevcGray10Streamer
from sdate.stream_hvec.stream_gray10 import EncoderParams

LAMINO_ANGLE_DEG = 0.0

# Labels for the four baseline movies and the volume key they correspond to.
_LABELS = ["gt", "gt_fbp", "sw_fbp", "la_fbp"]


# ---------------------------------------------------------------------------
# Helpers shared with inference_time_resolved
# ---------------------------------------------------------------------------

def extract_frame_index(filepath: Path) -> int:
    m = re.search(r"_(\d+)\.tif$", filepath.name)
    if m is None:
        raise ValueError(f"Cannot extract frame index from {filepath.name}")
    return int(m.group(1))


def find_frames_after(
    data_path: Path,
    max_frames: Optional[int] = None,
    frame_stride: int = 1,
) -> list[Path]:
    parent = data_path.parent
    start_idx = extract_frame_index(data_path)
    frames = [
        p for p in sorted(parent.glob("*.tif"))
        if extract_frame_index(p) > start_idx
    ]
    if frame_stride > 1:
        frames = frames[::frame_stride]
    if max_frames is not None:
        frames = frames[:max_frames]
    return frames


def make_circular_mask(h: int, w: int, radius: Optional[float] = None) -> np.ndarray:
    if radius is None:
        radius = w / 2.0
    cy, cx = h / 2.0, w / 2.0
    Y, X = np.ogrid[:h, :w]
    return ((X - cx) ** 2 + (Y - cy) ** 2) <= radius**2


def load_norm_config(norm_config_path: str) -> tuple[float, float]:
    p = Path(norm_config_path)
    if p.exists():
        with open(p) as f:
            nc = json.load(f)
        return float(nc.get("norm_min", 0.0)), float(nc.get("norm_max", 46204.0))
    print(f"  Warning: norm config not found at {p}, using defaults [0, 46204]")
    return 0.0, 46204.0


def masked_psnr(
    gt: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor, data_range: float
) -> float:
    mse = torch.mean((gt[:, mask] - pred[:, mask]) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(data_range**2) / mse).item()


def masked_ssim(
    gt: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor, data_range: float
) -> float:
    vals = []
    for i in range(gt.shape[0]):
        val, _ = compute_ssim(
            gt[i, mask].cpu().numpy(),
            pred[i, mask].cpu().numpy(),
            data_range=data_range,
            full=True,
        )
        vals.append(val)
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Per-frame baseline computation
# ---------------------------------------------------------------------------

def compute_baselines(
    frame_path: Path,
    *,
    all_frames: list[Path],
    global_idx: int,
    half: int,
    file_range: int,
    image_size: int,
    norm_min: float,
    norm_max: float,
    num_full_projs: int,
    full_angle_start: float,
    full_angle_end: float,
    angle_start_deg: float,
    angle_range_deg: float,
    num_la_projs: int,
    det_spacing_mm: float,
    device: torch.device,
    compute_gt_fbp: bool,
    compute_la_fbp: bool,
) -> dict[str, torch.Tensor]:
    """Load one frame and compute all FBP baselines.  Returns CPU tensors."""

    # ---- load window of file_range volumes centred on the target frame ----
    window_start_idx = max(0, global_idx - half)
    window_end_idx = min(len(all_frames) - 1, global_idx + half)
    window_paths = all_frames[window_start_idx : window_end_idx + 1]
    mid_in_window = global_idx - window_start_idx  # index of target within window

    dataset = TifVolumeSliceDataset(
        data_path=window_paths[0],
        file_range=len(window_paths),
        resize=image_size,
        normalize_range=(norm_min, norm_max),
        augment=False,
    )

    num_window = dataset.num_files
    volumes = [dataset.get_volume(k).to(device) for k in range(num_window)]
    gt = volumes[mid_in_window]  # (D, H, W) — target frame
    D, H, W = gt.shape

    gt_fbp: Optional[torch.Tensor] = None
    if compute_gt_fbp:
        # ---- full-angle projections for GT FBP ----
        full_angles_deg = torch.linspace(full_angle_start, full_angle_end, num_full_projs).numpy()
        full_proj = build_lamino_projector(
            vol_shape=gt.shape, det_shape=(D, W), angles_deg=full_angles_deg,
            lamino_angle_deg=LAMINO_ANGLE_DEG, det_spacing_mm=det_spacing_mm, device=device,
        )
        with torch.no_grad():
            full_sino = full_proj(gt.unsqueeze(0).unsqueeze(0)).squeeze(0)
        gt_fbp = fbp_reconstruction_masked(
            projs_vrc=full_sino, angles_deg=full_angles_deg,
            lamino_angle_deg=LAMINO_ANGLE_DEG, vol_shape=gt.shape,
            det_spacing_mm=det_spacing_mm, filter_type="hann", device=device,
        ).squeeze(0)
        del full_proj, full_sino

    # ---- per-frame limited-angle projections ----
    frame_la_angles: list[np.ndarray] = []
    frame_la_sinos: list[torch.Tensor] = []
    for k, vol in enumerate(volumes):
        # global angle window for this volume
        a_start = angle_start_deg + (window_start_idx + k - global_idx) * angle_range_deg
        a_end = a_start + angle_range_deg
        angles = torch.linspace(a_start, a_end, num_la_projs).numpy()
        frame_la_angles.append(angles)
        proj = build_lamino_projector(
            vol_shape=vol.shape, det_shape=(D, W), angles_deg=angles,
            lamino_angle_deg=LAMINO_ANGLE_DEG, det_spacing_mm=det_spacing_mm, device=device,
        )
        with torch.no_grad():
            sino = proj(vol.unsqueeze(0).unsqueeze(0)).squeeze(0)
        frame_la_sinos.append(sino)
        del proj

    la_sino = frame_la_sinos[mid_in_window]
    la_angles = frame_la_angles[mid_in_window]

    la_fbp: Optional[torch.Tensor] = None
    if compute_la_fbp:
        la_fbp = fbp_reconstruction_masked(
            projs_vrc=la_sino, angles_deg=la_angles,
            lamino_angle_deg=LAMINO_ANGLE_DEG, vol_shape=gt.shape,
            det_spacing_mm=det_spacing_mm, filter_type="hann", device=device,
        ).squeeze(0)

    # ---- sliding-window FBP ----
    if num_window > 1:
        sw_sino = torch.cat(frame_la_sinos, dim=0)
        sw_angles = np.concatenate(frame_la_angles, axis=0)
        sw_fbp = fbp_reconstruction_masked(
            projs_vrc=sw_sino, angles_deg=sw_angles,
            lamino_angle_deg=LAMINO_ANGLE_DEG, vol_shape=gt.shape,
            det_spacing_mm=det_spacing_mm, filter_type="hann", device=device,
        ).squeeze(0)
        del sw_sino
    else:
        # file_range == 1: SW baseline is equivalent to LA baseline.
        # Keep this independent so SW can be generated even if LA is skipped.
        sw_fbp = fbp_reconstruction_masked(
            projs_vrc=la_sino, angles_deg=la_angles,
            lamino_angle_deg=LAMINO_ANGLE_DEG, vol_shape=gt.shape,
            det_spacing_mm=det_spacing_mm, filter_type="hann", device=device,
        ).squeeze(0)

    del frame_la_sinos, volumes

    out: dict[str, torch.Tensor] = {
        "gt": gt.cpu(),
        "sw_fbp": sw_fbp.cpu(),
    }
    if gt_fbp is not None:
        out["gt_fbp"] = gt_fbp.cpu()
    if la_fbp is not None:
        out["la_fbp"] = la_fbp.cpu()
    return out


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Streaming baseline movie generation for time-resolved 4D CT"
    )
    p.add_argument("--data_path", required=True,
                   help="Starting .tif file; frames after this index will be processed")
    p.add_argument("--output_dir", required=True,
                   help="Directory to save movies (and optional metrics)")
    p.add_argument("--norm_config", default=None,
                   help="Path to _norm.json file (e.g. from a ddpm checkpoint). "
                        "If omitted, uses --norm_min / --norm_max directly.")
    p.add_argument("--norm_min", type=float, default=0.0,
                   help="Normalisation minimum (used when --norm_config is not provided)")
    p.add_argument("--norm_max", type=float, default=46204.0,
                   help="Normalisation maximum (used when --norm_config is not provided)")
    p.add_argument("--image_size", type=int, default=256)

    # Geometry
    p.add_argument("--num_full_projs", type=int, default=1000)
    p.add_argument("--full_angle_start", type=float, default=0.0)
    p.add_argument("--full_angle_end", type=float, default=180.0)
    p.add_argument("--angle_start", type=float, default=0.0,
                   help="Starting angle for the first frame's limited-angle window")
    p.add_argument("--angle_range", type=float, default=45.0,
                   help="Angular range per frame (degrees)")
    p.add_argument("--det_spacing_mm", type=float, default=1.0)

    # Frame selection
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--frame_stride", type=int, default=1)

    # Sliding-window neighbours
    p.add_argument("--file_range", type=int, default=1,
                   help="Number of consecutive tif files per reconstruction window "
                        "(1 = single-frame, 3/5/… = with neighbours)")

    # Baselines to generate
    p.add_argument("--labels", nargs="+", default=_LABELS,
                   choices=_LABELS,
                   help="Which baseline movies to produce (default: all four)")
    p.add_argument("--skip_gt_fbp", action="store_true",
                   help="Skip the expensive full-angle FBP (gt_fbp); useful for quick runs")
    p.add_argument("--skip_la_fbp", action="store_true",
                   help="Skip LA FBP (la_fbp); useful when only GT/SW baselines are needed")

    # Movie
    p.add_argument("--num_movie_slices", type=int, default=5)
    p.add_argument("--movie_fps", type=int, default=5)
    p.add_argument("--movie_crf", type=int, default=18)

    # Metrics
    p.add_argument("--save_metrics", action="store_true",
                   help="Compute and save PSNR/SSIM metrics to metrics_baseline.json")
    p.add_argument("--wandb_project", type=str, default="time_resolved_baseline")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    movie_dir = output_dir / "movies"
    slices_dir = output_dir / "frame_slices"
    movie_dir.mkdir(parents=True, exist_ok=True)
    slices_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Normalisation range
    if args.norm_config is not None:
        norm_min, norm_max = load_norm_config(args.norm_config)
    else:
        norm_min, norm_max = args.norm_min, args.norm_max
    print(f"Norm range: [{norm_min:.4g}, {norm_max:.4g}]")

    # Effective labels
    labels = [
        l for l in args.labels
        if not (
            (args.skip_gt_fbp and l == "gt_fbp")
            or (args.skip_la_fbp and l == "la_fbp")
        )
    ]
    print(f"Labels: {labels}")

    num_la_projs = int(
        args.angle_range * args.num_full_projs
        / (args.full_angle_end - args.full_angle_start)
    )

    # Find all frames to process
    all_frames = find_frames_after(
        data_path, max_frames=args.max_frames, frame_stride=args.frame_stride
    )
    if not all_frames:
        print(f"No frames after index {extract_frame_index(data_path)} in {data_path.parent}")
        return
    print(f"Found {len(all_frames)} frames")

    half = args.file_range // 2

    # -----------------------------------------------------------------------
    # Determine number of slices D from the first frame
    # -----------------------------------------------------------------------
    import tifffile
    with tifffile.TiffFile(str(all_frames[0])) as tif:
        D_native = len(tif.pages)
    # After resize the slice count stays the same (resize is spatial H,W only)
    D = D_native

    if args.num_movie_slices >= D:
        slice_indices = list(range(D))
    else:
        slice_indices = np.linspace(0, D - 1, args.num_movie_slices, dtype=int).tolist()
    print(f"Movie slices: {slice_indices}")

    # -----------------------------------------------------------------------
    # wandb
    # -----------------------------------------------------------------------
    use_wandb = not args.no_wandb and args.save_metrics
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "data_path": str(data_path),
                "image_size": args.image_size,
                "num_full_projs": args.num_full_projs,
                "angle_start_deg": args.angle_start,
                "angle_range_deg": args.angle_range,
                "num_la_projs": num_la_projs,
                "file_range": args.file_range,
                "num_frames": len(all_frames),
                "labels": labels,
            },
        )

    all_metrics: list[dict] = []

    # -----------------------------------------------------------------------
    # Phase 1: For every frame compute baselines and persist only the needed
    # 2-D slices to disk.  GPU memory is freed after each frame.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 1: Computing baselines and saving slices")
    print("=" * 60)

    for i, fp in enumerate(all_frames):
        fidx = extract_frame_index(fp)
        slices_file = slices_dir / f"slices_{fidx:05d}.pt"

        print(f"\n{'='*60}")
        print(f"Frame {i+1}/{len(all_frames)}: {fp.name} (index {fidx})")
        print(f"{'='*60}")

        frame_angle_start = args.angle_start + i * args.angle_range
        frame_angle_end = frame_angle_start + args.angle_range
        print(f"  LA window: {frame_angle_start:.1f}° – {frame_angle_end:.1f}°")

        if slices_file.exists():
            print("  Already processed — skipping.")
            continue

        t0 = time.time()
        results = compute_baselines(
            frame_path=fp,
            all_frames=all_frames,
            global_idx=i,
            half=half,
            file_range=args.file_range,
            image_size=args.image_size,
            norm_min=norm_min,
            norm_max=norm_max,
            num_full_projs=args.num_full_projs,
            full_angle_start=args.full_angle_start,
            full_angle_end=args.full_angle_end,
            angle_start_deg=frame_angle_start,
            angle_range_deg=args.angle_range,
            num_la_projs=num_la_projs,
            det_spacing_mm=args.det_spacing_mm,
            device=device,
            compute_gt_fbp=("gt_fbp" in labels),
            compute_la_fbp=("la_fbp" in labels),
        )

        # Persist only the 2-D slices we will need for movies.
        slices: dict[str, torch.Tensor] = {}
        for lbl in labels:
            vol = results[lbl]  # (D, H, W) CPU
            for s_idx in slice_indices:
                slices[f"{lbl}_{s_idx}"] = vol[s_idx].float().clamp(0.0, 1.0)
        torch.save(slices, slices_file)

        elapsed = time.time() - t0
        print(f"  Saved slices to {slices_file.name}  ({elapsed:.1f}s)")

        # Optionally compute metrics against gt
        if args.save_metrics:
            gt = results["gt"]
            D_, H_, W_ = gt.shape
            circ_mask = torch.tensor(
                make_circular_mask(H_, W_, radius=W_ / 2.0)
            )
            data_range = (gt[:, circ_mask].max() - gt[:, circ_mask].min()).item()
            metrics: dict = {
                "frame_index": fidx,
                "frame_name": fp.name,
                "la_angle_start_deg": frame_angle_start,
                "la_angle_end_deg": frame_angle_end,
                "elapsed_s": elapsed,
            }
            for lbl in ("gt_fbp", "sw_fbp", "la_fbp"):
                if lbl in results:
                    metrics[f"psnr_{lbl}"] = masked_psnr(
                        gt, results[lbl], circ_mask, data_range
                    )
                    metrics[f"ssim_{lbl}"] = masked_ssim(
                        gt, results[lbl], circ_mask, data_range
                    )
                    print(f"  {lbl}: PSNR={metrics[f'psnr_{lbl}']:.2f} dB  "
                          f"SSIM={metrics[f'ssim_{lbl}']:.4f}")
            all_metrics.append(metrics)
            if use_wandb:
                wandb.log(metrics, step=i)

        del results, slices
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Phase 2: Encode HEVC movies — one streamer at a time, reading saved
    # slices frame by frame so full volumes are never held in memory.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 2: Encoding HEVC movies")
    print("=" * 60)

    params = EncoderParams(
        fps=args.movie_fps, crf_sw=args.movie_crf,
        preset_sw="medium", force_software=True,
    )

    for lbl in labels:
        for s_idx in slice_indices:
            key = f"{lbl}_{s_idx}"
            out_name = f"{lbl}_slice_{s_idx:03d}.mov"
            print(f"  Encoding {out_name} …")
            streamer = HevcGray10Streamer(
                base_path=movie_dir,
                segment_prefix=f"{lbl}_s{s_idx:03d}",
                params=params,
            )
            n_written = 0
            with streamer.start_segment(outfile=out_name):
                for fp in all_frames:
                    fidx = extract_frame_index(fp)
                    slices_file = slices_dir / f"slices_{fidx:05d}.pt"
                    if not slices_file.exists():
                        continue
                    slices = torch.load(slices_file, map_location="cpu")
                    streamer.append_frame(slices[key])
                    n_written += 1
                    del slices
            print(f"    -> {out_name} ({n_written} frames)")

    # -----------------------------------------------------------------------
    # Phase 3: Clean up temporary slice files
    # -----------------------------------------------------------------------
    shutil.rmtree(slices_dir)
    print(f"\nCleaned up temporary slice files ({slices_dir})")

    # -----------------------------------------------------------------------
    # Save metrics
    # -----------------------------------------------------------------------
    if args.save_metrics and all_metrics:
        metrics_path = output_dir / "metrics_baseline.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

        if use_wandb:
            for lbl in ("gt_fbp", "sw_fbp", "la_fbp"):
                key_psnr = f"psnr_{lbl}"
                key_ssim = f"ssim_{lbl}"
                vals_psnr = [m[key_psnr] for m in all_metrics if key_psnr in m]
                vals_ssim = [m[key_ssim] for m in all_metrics if key_ssim in m]
                if vals_psnr:
                    wandb.run.summary[f"avg_{key_psnr}"] = float(np.mean(vals_psnr))
                    wandb.run.summary[f"avg_{key_ssim}"] = float(np.mean(vals_ssim))

    if use_wandb:
        wandb.finish()

    print(f"\nDone. Movies in {movie_dir}")


if __name__ == "__main__":
    main()
