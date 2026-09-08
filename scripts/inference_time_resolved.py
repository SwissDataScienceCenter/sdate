#!/usr/bin/env python3
"""
Batch inference for time-resolved 4D CT using guided DDIM reconstruction.

Processes all frames in a timesteps directory whose frame index is larger than
the starting DATA_PATH, runs DDIM with sinogram guidance, saves reconstructions
locally, logs metrics to wandb, and generates HEVC movies of selected slices.

Each frame uses a rotating limited-angle window to simulate a time-resolved
acquisition: frame i covers [angle_start + i*angle_range, angle_start + (i+1)*angle_range].
ASTRA handles angles beyond 180° transparently.

Usage:
    python scripts/inference_time_resolved.py --data_path /myhome/data/sdate/shared/time_resolved/212_Wunderkerze2/timesteps/212_Wunderkerze2_rotate_04200.tif --checkpoint /myhome/sdate/checkpoints/ddpm_ladiff_time_resolved.pt --output_dir /myhome/data/sdate/shared/time_resolved/212_Wunderkerze2/recon_output --image_size 256 --angle_start 0 --angle_range 36 --num_full_projs 1000 --num_inference_steps 50 --num_movie_slices 5 --wandb_project time_resolved_inference --file_range=5 --sw_guidance_threshold=100 --max_frames=20 --frame_stride=2
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

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
from diffusers.models import UNet2DModel
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from ladiff.datasets import TifVolumeSliceDataset
from ladiff.schedulers.pipeline_ddim import DDIMPipeline
from ladiff.schedulers.scheduling_ddim import GuidedDDIMScheduler
from astra_torch.lamino import (
    build_lamino_projector,
    fbp_reconstruction_masked,
    gd_reconstruction_masked,
)
from sdate.stream_hvec import HevcGray10Streamer, concat_hevc_segments
from sdate.stream_hvec.stream_gray10 import EncoderParams

LAMINO_ANGLE_DEG = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_frame_index(filepath: Path) -> int:
    """Extract the trailing numeric frame index from a tif filename."""
    m = re.search(r"_(\d+)\.tif$", filepath.name)
    if m is None:
        raise ValueError(f"Cannot extract frame index from {filepath.name}")
    return int(m.group(1))


def find_frames_after(
    data_path: Path,
    max_frames: int | None = None,
    frame_stride: int = 1,
) -> list[Path]:
    """Return sorted list of .tif files in the same directory with frame index > data_path's.

    Parameters
    ----------
    max_frames : int | None
        If set, return at most this many frames (applied after striding).
    frame_stride : int
        Take every frame_stride-th frame from the sorted list (default 1 = all).
    """
    parent = data_path.parent
    start_idx = extract_frame_index(data_path)
    frames = []
    for p in sorted(parent.glob("*.tif")):
        idx = extract_frame_index(p)
        if idx > start_idx:
            frames.append(p)
    if frame_stride > 1:
        frames = frames[::frame_stride]
    if max_frames is not None:
        frames = frames[:max_frames]
    return frames


def make_circular_mask(h: int, w: int, radius: float | None = None) -> np.ndarray:
    if radius is None:
        radius = w / 2.0
    cy, cx = h / 2.0, w / 2.0
    Y, X = np.ogrid[:h, :w]
    return ((X - cx) ** 2 + (Y - cy) ** 2) <= radius**2


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
    ssim_values = []
    for i in range(gt.shape[0]):
        val, _ = compute_ssim(
            gt[i, mask].cpu().numpy(),
            pred[i, mask].cpu().numpy(),
            data_range=data_range,
            full=True,
        )
        ssim_values.append(val)
    return float(np.mean(ssim_values))


# ---------------------------------------------------------------------------
# TV regularization helpers
# ---------------------------------------------------------------------------
def tv_z(vol: torch.Tensor) -> torch.Tensor:
    """1D anisotropic total variation along the slice axis (dim 0)."""
    return torch.mean(torch.abs(vol[1:] - vol[:-1]))


def make_tv_regularizer(weight_z: float = 1e-3, weight_xy: float = 0.0):
    """Factory returning a differentiable regularization function.

    Parameters
    ----------
    weight_z : float
        Weight for inter-slice (z-direction) TV.
    weight_xy : float
        Weight for in-plane (x,y) anisotropic TV.  Keep at 0 to preserve
        in-plane sharpness.

    Returns
    -------
    reg_fn : callable
        reg_fn(vol: Tensor[D, H, W]) -> scalar
    """
    def reg_fn(vol: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=vol.device)
        if weight_z > 0.0:
            loss = loss + weight_z * tv_z(vol)
        if weight_xy > 0.0:
            dy = vol[:, 1:, :] - vol[:, :-1, :]
            dx = vol[:, :, 1:] - vol[:, :, :-1]
            loss = loss + weight_xy * (torch.mean(torch.abs(dy)) + torch.mean(torch.abs(dx)))
        return loss
    return reg_fn


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: str, image_size: int, device: torch.device) -> UNet2DModel:
    channels = (64, 64, 128, 128, 256, 256)
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=channels,
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D",
            "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
            "UpBlock2D", "UpBlock2D", "UpBlock2D",
        ),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_norm_config(checkpoint_path: str) -> tuple[float, float]:
    norm_path = Path(checkpoint_path.replace(".pt", "_norm.json"))
    if norm_path.exists():
        with open(norm_path) as f:
            nc = json.load(f)
        return float(nc.get("norm_min", 0.0)), float(nc.get("norm_max", 46204.0))
    print(f"  Warning: no norm config at {norm_path}, using defaults [0, 46204]")
    return 0.0, 46204.0


# ---------------------------------------------------------------------------
# Reconstruction for a single frame (with optional sliding-window neighbors)
# ---------------------------------------------------------------------------
def reconstruct_frame(
    data_path: Path,
    *,
    model: UNet2DModel,
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
    num_inference_steps: int,
    guidance_epochs: int,
    guidance_lr: float,
    guidance_batch_size: int,
    slice_batch_size: int,
    tv_weight_z: float,
    tv_weight_xy: float,
    device: torch.device,
    initial_guess: torch.Tensor | None = None,
    start_step_pct: float = 0.8,
    file_range: int = 1,
    sw_guidance_threshold: int = 100,
) -> dict:
    """Run full pipeline for one frame. Returns dict with results and metrics.

    When *file_range* > 1, loads that many consecutive tif files centred on
    *data_path* and builds per-frame limited-angle sinograms.  The middle
    volume is the ground-truth frame.  A combined sliding-window (SW)
    sinogram is used for guidance at high noise levels (timestep >
    *sw_guidance_threshold*); at low noise levels only the middle frame's
    projections are used.

    With *file_range* = 1 the behaviour is identical to the original
    single-frame pipeline.
    """
    normalize_fn = lambda x: x  # noqa: E731
    denormalize_fn = lambda x: x  # noqa: E731

    # ------------------------------------------------------------------
    # Load data — file_range volumes centred on data_path
    # ------------------------------------------------------------------
    half = file_range // 2
    dataset = TifVolumeSliceDataset(
        data_path=data_path,
        file_range=file_range,
        resize=image_size,
        normalize_range=(norm_min, norm_max) if (norm_min != 0.0 or norm_max != 1.0) else None,
        augment=False,
    )
    mid_idx = dataset.num_files // 2  # middle file = GT
    num_frames = dataset.num_files

    volumes = [dataset.get_volume(k).to(device) for k in range(num_frames)]
    gt = volumes[mid_idx]  # (D, H, W)
    D, H, W = gt.shape

    # ------------------------------------------------------------------
    # Full-angle projections (GT frame only, for reference FBP)
    # ------------------------------------------------------------------
    full_angles_deg = torch.linspace(full_angle_start, full_angle_end, num_full_projs).numpy()
    full_projector = build_lamino_projector(
        vol_shape=gt.shape, det_shape=(D, W), angles_deg=full_angles_deg,
        lamino_angle_deg=LAMINO_ANGLE_DEG, det_spacing_mm=det_spacing_mm, device=device,
    )
    gt_vol = gt.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        full_sino = full_projector(gt_vol).squeeze(0)

    gt_fbp = fbp_reconstruction_masked(
        projs_vrc=full_sino, angles_deg=full_angles_deg,
        lamino_angle_deg=LAMINO_ANGLE_DEG, vol_shape=gt.shape,
        det_spacing_mm=det_spacing_mm, filter_type="hann", device=device,
    ).squeeze(0)

    # ------------------------------------------------------------------
    # Per-frame limited-angle projections
    # Frame k (within the local window) gets angles:
    #   [angle_start_deg + (k - half) * angle_range_deg,
    #    angle_start_deg + (k - half + 1) * angle_range_deg]
    # So k=mid_idx → [angle_start_deg, angle_start_deg + angle_range_deg]
    # ------------------------------------------------------------------
    frame_la_angles: list[np.ndarray] = []
    frame_la_sinos: list[torch.Tensor] = []

    for k in range(num_frames):
        a_start = angle_start_deg + (k - half) * angle_range_deg
        a_end = a_start + angle_range_deg
        angles = torch.linspace(a_start, a_end, num_la_projs).numpy()
        frame_la_angles.append(angles)

        proj = build_lamino_projector(
            vol_shape=volumes[k].shape, det_shape=(D, W), angles_deg=angles,
            lamino_angle_deg=LAMINO_ANGLE_DEG, det_spacing_mm=det_spacing_mm, device=device,
        )
        vol_k = volumes[k].unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            sino_k = proj(vol_k).squeeze(0)
        frame_la_sinos.append(sino_k)

    # Middle frame's LA sinogram
    la_sino = frame_la_sinos[mid_idx]
    la_angles_deg = frame_la_angles[mid_idx]

    # Combined sliding-window sinogram
    sw_sino = torch.cat(frame_la_sinos, dim=0)
    sw_angles_deg = np.concatenate(frame_la_angles, axis=0)

    # LA FBP (middle frame only)
    la_fbp = fbp_reconstruction_masked(
        projs_vrc=la_sino, angles_deg=la_angles_deg,
        lamino_angle_deg=LAMINO_ANGLE_DEG, vol_shape=gt.shape,
        det_spacing_mm=det_spacing_mm, filter_type="hann", device=device,
    ).squeeze(0)

    # SW FBP (all frames combined)
    if num_frames > 1:
        sw_fbp = fbp_reconstruction_masked(
            projs_vrc=sw_sino, angles_deg=sw_angles_deg,
            lamino_angle_deg=LAMINO_ANGLE_DEG, vol_shape=gt.shape,
            det_spacing_mm=det_spacing_mm, filter_type="hann", device=device,
        ).squeeze(0)
    else:
        sw_fbp = la_fbp

    # ------------------------------------------------------------------
    # Guidance function — switches between SW and LA based on timestep
    # ------------------------------------------------------------------
    meas_projs = la_sino.to(device)
    meas_angles = la_angles_deg.copy()
    meas_sw_projs = sw_sino.to(device)
    meas_sw_angles = sw_angles_deg.copy()

    reg_fn = make_tv_regularizer(weight_z=tv_weight_z, weight_xy=tv_weight_xy)
    use_reg = tv_weight_z > 0.0 or tv_weight_xy > 0.0

    def guidance_fn(x, t):
        t_val = int(t)
        use_sw = (num_frames > 1 and t_val > sw_guidance_threshold)
        active_projs = meas_sw_projs if use_sw else meas_projs
        active_angles = meas_sw_angles if use_sw else meas_angles
        return gd_reconstruction_masked(
            projs_vrc=active_projs, angles_deg=active_angles,
            lamino_angle_deg=LAMINO_ANGLE_DEG, vol_shape=gt.shape,
            det_spacing_mm=det_spacing_mm, vol_init=x.to(device),
            max_epochs=guidance_epochs, lr=guidance_lr,
            batch_size=guidance_batch_size, clamp_min=0.0,
            regularization_fn=reg_fn if use_reg else None,
            device=device, verbose=False,
        )

    # ------------------------------------------------------------------
    # DDIM pipeline
    # ------------------------------------------------------------------
    scheduler = GuidedDDIMScheduler(
        num_train_timesteps=1000, guidance_function=guidance_fn,
    )
    pipeline = DDIMPipeline(
        unet=model.to(device), scheduler=scheduler,
        fdk_prior=None, normalize_fn=normalize_fn, denormalize_fn=denormalize_fn,
        slice_batch_size=slice_batch_size,
    )

    if initial_guess is None:
        guess = gt.clone()
    else:
        guess = initial_guess.to(device)
    start_step = int(start_step_pct * num_inference_steps)

    torch.cuda.empty_cache()
    t0 = time.time()
    result = pipeline.truncated_pipeline(
        initial_guess=guess,
        start_step=start_step,
        num_inference_steps=num_inference_steps,
    )
    diffusion_recon = result.images.squeeze(0)  # (D, H, W)
    recon_time = time.time() - t0

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    circ_mask = torch.tensor(
        make_circular_mask(H, W, radius=W / 2.0), device=device
    )
    data_range = (gt[:, circ_mask].max() - gt[:, circ_mask].min()).cpu().item()

    psnr_ddim = masked_psnr(gt, diffusion_recon, circ_mask, data_range)
    ssim_ddim = masked_ssim(gt, diffusion_recon, circ_mask, data_range)
    psnr_la = masked_psnr(gt, la_fbp, circ_mask, data_range)
    ssim_la = masked_ssim(gt, la_fbp, circ_mask, data_range)
    psnr_sw = masked_psnr(gt, sw_fbp, circ_mask, data_range)
    ssim_sw = masked_ssim(gt, sw_fbp, circ_mask, data_range)
    psnr_full = masked_psnr(gt, gt_fbp, circ_mask, data_range)
    ssim_full = masked_ssim(gt, gt_fbp, circ_mask, data_range)

    return {
        "gt": gt.cpu(),
        "diffusion_recon": diffusion_recon.cpu(),
        "la_fbp": la_fbp.cpu(),
        "sw_fbp": sw_fbp.cpu(),
        "gt_fbp": gt_fbp.cpu(),
        "psnr_ddim": psnr_ddim,
        "ssim_ddim": ssim_ddim,
        "psnr_la": psnr_la,
        "ssim_la": ssim_la,
        "psnr_sw": psnr_sw,
        "ssim_sw": ssim_sw,
        "psnr_full": psnr_full,
        "ssim_full": ssim_full,
        "recon_time_s": recon_time,
    }


# ---------------------------------------------------------------------------
# Movie generation
# ---------------------------------------------------------------------------
# Map from movie label to the filename prefix used for saved .pt volumes.
_LABEL_TO_PREFIX = {
    "recon": "recon",
    "gt": "gt",
    "sw_fbp": "sw_fbp",
    "la_fbp": "la_fbp",
    "gt_fbp": "gt_fbp",
}


def generate_movies(
    recon_dir: Path,
    movie_dir: Path,
    frame_paths: list[Path],
    num_movie_slices: int,
    labels: list[str] | None = None,
    fps: int = 10,
    crf: int = 18,
):
    """Generate HEVC movies for k evenly distributed slices.

    Creates movies for each *label* (default: ``["recon", "gt", "sw_fbp"]``).
    Each movie shows a single slice evolving over all time frames.
    Frames whose .pt file is missing are silently skipped so that GT movies
    (which cover all frames) can be longer than recon movies (which may
    cover only a subset when file_range > 1).
    """
    if labels is None:
        labels = ["recon", "gt", "sw_fbp"]
    movie_dir.mkdir(parents=True, exist_ok=True)

    # Load first available recon volume to get slice count
    D = None
    for fp in frame_paths:
        fidx = extract_frame_index(fp)
        candidate = recon_dir / f"recon_{fidx:05d}.pt"
        if candidate.exists():
            D = torch.load(candidate, map_location="cpu").shape[0]
            break
    if D is None:
        print("  No recon volumes found — skipping movie generation.")
        return

    # Choose evenly spaced slice indices
    if num_movie_slices >= D:
        slice_indices = list(range(D))
    else:
        slice_indices = np.linspace(0, D - 1, num_movie_slices, dtype=int).tolist()

    print(f"\nGenerating movies for slices {slice_indices} across {len(frame_paths)} frames")

    params = EncoderParams(fps=fps, crf_sw=crf, preset_sw="medium", force_software=True)

    for label in labels:
        prefix = _LABEL_TO_PREFIX.get(label, label)
        for s_idx in slice_indices:
            out_path = movie_dir / f"{label}_slice_{s_idx:03d}.mov"
            print(f"  {label} slice {s_idx} -> {out_path.name}")

            streamer = HevcGray10Streamer(
                base_path=movie_dir, segment_prefix=f"{label}_s{s_idx:03d}",
                params=params,
            )
            n_written = 0
            with streamer.start_segment(outfile=out_path.name):
                for fp in frame_paths:
                    fidx = extract_frame_index(fp)
                    vol_path = recon_dir / f"{prefix}_{fidx:05d}.pt"
                    if not vol_path.exists():
                        continue  # skip missing frames (e.g. edge frames)
                    vol = torch.load(vol_path, map_location="cpu")
                    frame = vol[s_idx].float().clamp(0.0, 1.0)
                    streamer.append_frame(frame)
                    n_written += 1

            print(f"    -> {out_path} ({n_written} frames)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Batch time-resolved DDIM inference")
    p.add_argument("--data_path", type=str, required=True,
                    help="Starting .tif file (frames after this index will be processed)")
    p.add_argument("--checkpoint", type=str, default=str(PROJECT_ROOT / "checkpoints" / "ddpm_ladiff_time_resolved.pt"),
                    help="Path to diffusion model checkpoint")
    p.add_argument("--output_dir", type=str, required=True,
                    help="Directory to save reconstructions and movies")
    p.add_argument("--image_size", type=int, default=256)

    # Projection geometry
    p.add_argument("--num_full_projs", type=int, default=1000)
    p.add_argument("--full_angle_start", type=float, default=0.0)
    p.add_argument("--full_angle_end", type=float, default=180.0)
    p.add_argument("--angle_start", type=float, default=0.0,
                    help="Starting angle (degrees) for the first frame's limited-angle window")
    p.add_argument("--angle_range", type=float, default=45.0,
                    help="Angular range (degrees) per frame; frame i covers [start+i*range, start+(i+1)*range]")
    p.add_argument("--det_spacing_mm", type=float, default=1.0)

    # Frame selection
    p.add_argument("--max_frames", type=int, default=None,
                    help="Maximum number of frames to process after the starting frame (default: all).")
    p.add_argument("--frame_stride", type=int, default=1,
                    help="Process every frame_stride-th frame (default 1 = every frame).")

    # Sliding-window neighbors
    p.add_argument("--file_range", type=int, default=1,
                    help="Number of consecutive tif files to load per reconstruction. "
                         "1 = single frame (original behaviour), 3 or 5 = with neighbors.")
    p.add_argument("--sw_guidance_threshold", type=int, default=100,
                    help="Diffusion timestep threshold: t > threshold uses sliding-window "
                         "projections, t <= threshold uses middle frame only.")

    # Inference
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_epochs", type=int, default=1)
    p.add_argument("--guidance_lr", type=float, default=0.01)
    p.add_argument("--guidance_batch_size", type=int, default=100)
    p.add_argument("--slice_batch_size", type=int, default=32)

    # TV regularization
    p.add_argument("--tv_weight_z", type=float, default=0.0,
                    help="Inter-slice TV regularization weight (z-axis). 0 = disabled.")
    p.add_argument("--tv_weight_xy", type=float, default=0.0,
                    help="In-plane TV regularization weight. 0 = disabled (keeps in-plane sharpness).")
    p.add_argument("--start_step_pct", type=float, default=0.8,
                    help="Fraction of num_inference_steps to use as start_step for subsequent frames (time coherence).")

    # Movie
    p.add_argument("--num_movie_slices", type=int, default=5,
                    help="Number of evenly spaced slices for movie generation after reconstructions are done.")
    p.add_argument("--movie_fps", type=int, default=5)
    p.add_argument("--movie_crf", type=int, default=18)

    # wandb
    p.add_argument("--wandb_project", type=str, default="time_resolved_inference")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")

    return p.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    recon_dir = output_dir / "volumes"
    movie_dir = output_dir / "movies"
    recon_dir.mkdir(parents=True, exist_ok=True)
    movie_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Compute number of limited-angle projections (same count for every frame)
    num_la_projs = int(
        args.angle_range
        * args.num_full_projs
        / (args.full_angle_end - args.full_angle_start)
    )

    # Find frames to process
    all_frames = find_frames_after(data_path, max_frames=args.max_frames, frame_stride=args.frame_stride)
    if not all_frames:
        print(f"No frames with index > {extract_frame_index(data_path)} found in {data_path.parent}")
        return
    print(f"Found {len(all_frames)} total frames (indices > {extract_frame_index(data_path)})")

    # With file_range > 1 we need 'half' neighbors on each side, so we can
    # only reconstruct the interior frames.  The GT movies will still cover
    # all frames, but recon/sw_fbp movies only the reconstructed subset.
    half = args.file_range // 2
    if half > 0 and len(all_frames) > 2 * half:
        frames = all_frames[half:-half]
        print(f"  file_range={args.file_range} → reconstructing {len(frames)} "
              f"interior frames (skipping {half} on each end)")
    else:
        frames = all_frames
    for f in frames:
        print(f"  {f.name}")

    # Load model
    norm_min, norm_max = load_norm_config(args.checkpoint)
    print(f"Norm range: [{norm_min:.4g}, {norm_max:.4g}]")
    model = load_model(args.checkpoint, args.image_size, device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Init wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "data_path": str(data_path),
                "checkpoint": args.checkpoint,
                "image_size": args.image_size,
                "num_full_projs": args.num_full_projs,
                "full_angle_start": args.full_angle_start,
                "full_angle_end": args.full_angle_end,
                "angle_start_deg": args.angle_start,
                "angle_range_deg": args.angle_range,
                "num_la_projs": num_la_projs,
                "num_inference_steps": args.num_inference_steps,
                "guidance_epochs": args.guidance_epochs,
                "guidance_lr": args.guidance_lr,
                "guidance_batch_size": args.guidance_batch_size,
                "tv_weight_z": args.tv_weight_z,
                "tv_weight_xy": args.tv_weight_xy,
                "file_range": args.file_range,
                "sw_guidance_threshold": args.sw_guidance_threshold,
                "num_frames": len(frames),
            },
        )

    # Load existing metrics if present (for resume)
    metrics_path = output_dir / "metrics.json"
    existing_metrics: dict[int, dict] = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            for m in json.load(f):
                existing_metrics[m["frame_index"]] = m
        print(f"Loaded {len(existing_metrics)} existing metric entries from {metrics_path}")

    # Process each frame
    all_metrics = []
    prev_recon: torch.Tensor | None = None
    for i, fp in enumerate(frames):
        fidx = extract_frame_index(fp)

        # Rotating limited-angle window: the *global* frame index in
        # all_frames determines its angle window.  Find this frame's
        # position in the original sequence.
        global_idx = all_frames.index(fp)
        frame_angle_start = args.angle_start + global_idx * args.angle_range
        frame_angle_end = args.angle_start + (global_idx + 1) * args.angle_range

        recon_file = recon_dir / f"recon_{fidx:05d}.pt"
        gt_file    = recon_dir / f"gt_{fidx:05d}.pt"

        print(f"\n{'='*60}")
        print(f"Frame {i+1}/{len(frames)}: {fp.name} (index {fidx})")
        print(f"  LA window: {frame_angle_start:.1f}° – {frame_angle_end:.1f}°")
        print(f"{'='*60}")

        if recon_file.exists() and gt_file.exists() and fidx in existing_metrics:
            print(f"  Skipping — reconstruction already exists.")
            metrics = existing_metrics[fidx]
            all_metrics.append(metrics)
            print(f"  DDIM:     PSNR={metrics['psnr_ddim']:.2f} dB  SSIM={metrics['ssim_ddim']:.4f}")
            print(f"  LA FBP:   PSNR={metrics['psnr_la_fbp']:.2f} dB  SSIM={metrics['ssim_la_fbp']:.4f}")
            print(f"  Full FBP: PSNR={metrics['psnr_full_fbp']:.2f} dB  SSIM={metrics['ssim_full_fbp']:.4f}")
            prev_recon = torch.load(recon_file, map_location="cpu")
            continue

        # TifVolumeSliceDataset loads file_range files *starting from*
        # data_path, so pass the first file in the window (half before fp).
        window_start_path = all_frames[global_idx - half] if half > 0 else fp
        results = reconstruct_frame(
            data_path=window_start_path,
            model=model,
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
            num_inference_steps=args.num_inference_steps,
            guidance_epochs=args.guidance_epochs,
            guidance_lr=args.guidance_lr,
            guidance_batch_size=args.guidance_batch_size,
            slice_batch_size=args.slice_batch_size,
            tv_weight_z=args.tv_weight_z,
            tv_weight_xy=args.tv_weight_xy,
            device=device,
            initial_guess=prev_recon,
            start_step_pct=args.start_step_pct,
            file_range=args.file_range,
            sw_guidance_threshold=args.sw_guidance_threshold,
        )
        prev_recon = results["diffusion_recon"]

        # Save volumes locally
        torch.save(results["diffusion_recon"], recon_file)
        torch.save(results["gt"], gt_file)
        torch.save(results["la_fbp"], recon_dir / f"la_fbp_{fidx:05d}.pt")
        torch.save(results["sw_fbp"], recon_dir / f"sw_fbp_{fidx:05d}.pt")
        torch.save(results["gt_fbp"], recon_dir / f"gt_fbp_{fidx:05d}.pt")

        metrics = {
            "frame_index": fidx,
            "frame_name": fp.name,
            "la_angle_start_deg": frame_angle_start,
            "la_angle_end_deg": frame_angle_end,
            "psnr_ddim": results["psnr_ddim"],
            "ssim_ddim": results["ssim_ddim"],
            "psnr_la_fbp": results["psnr_la"],
            "ssim_la_fbp": results["ssim_la"],
            "psnr_sw_fbp": results["psnr_sw"],
            "ssim_sw_fbp": results["ssim_sw"],
            "psnr_full_fbp": results["psnr_full"],
            "ssim_full_fbp": results["ssim_full"],
            "recon_time_s": results["recon_time_s"],
        }
        all_metrics.append(metrics)

        print(f"  DDIM:     PSNR={results['psnr_ddim']:.2f} dB  SSIM={results['ssim_ddim']:.4f}")
        print(f"  LA FBP:   PSNR={results['psnr_la']:.2f} dB  SSIM={results['ssim_la']:.4f}")
        print(f"  SW FBP:   PSNR={results['psnr_sw']:.2f} dB  SSIM={results['ssim_sw']:.4f}")
        print(f"  Full FBP: PSNR={results['psnr_full']:.2f} dB  SSIM={results['ssim_full']:.4f}")
        print(f"  Time: {results['recon_time_s']:.1f}s")

        if use_wandb:
            wandb.log(metrics, step=i)

    # Save metrics summary
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Log summary to wandb
    if use_wandb and all_metrics:
        avg_psnr_ddim = np.mean([m["psnr_ddim"] for m in all_metrics])
        avg_ssim_ddim = np.mean([m["ssim_ddim"] for m in all_metrics])
        avg_psnr_la = np.mean([m["psnr_la_fbp"] for m in all_metrics])
        avg_ssim_la = np.mean([m["ssim_la_fbp"] for m in all_metrics])
        avg_psnr_sw = np.mean([m["psnr_sw_fbp"] for m in all_metrics if "psnr_sw_fbp" in m])
        avg_ssim_sw = np.mean([m["ssim_sw_fbp"] for m in all_metrics if "ssim_sw_fbp" in m])
        wandb.run.summary["avg_psnr_ddim"] = avg_psnr_ddim
        wandb.run.summary["avg_ssim_ddim"] = avg_ssim_ddim
        wandb.run.summary["avg_psnr_la_fbp"] = avg_psnr_la
        wandb.run.summary["avg_ssim_la_fbp"] = avg_ssim_la
        wandb.run.summary["avg_psnr_sw_fbp"] = avg_psnr_sw
        wandb.run.summary["avg_ssim_sw_fbp"] = avg_ssim_sw
        wandb.run.summary["num_frames_processed"] = len(all_metrics)

    # Generate movies
    print("\n" + "=" * 60)
    print("Generating HEVC movies")
    print("=" * 60)
    # GT movies use all_frames (full sequence); recon/sw_fbp use only
    # the reconstructed subset (frames), so missing .pt files are skipped.
    movie_labels = ["recon", "sw_fbp", "gt"]
    generate_movies(
        recon_dir=recon_dir,
        movie_dir=movie_dir,
        frame_paths=all_frames,
        num_movie_slices=args.num_movie_slices,
        labels=movie_labels,
        fps=args.movie_fps,
        crf=args.movie_crf,
    )

    if use_wandb:
        wandb.finish()

    print(f"\nDone. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
