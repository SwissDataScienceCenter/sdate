#!/usr/bin/env python3
"""reconstruct_ladiff_slices.py – Iterative limited-angle diffusion reconstruction.

Reconstructs every slice of a GT-FBP volume one at a time using the iterative
DDIM refinement approach from the *test_large_ladiff_conditional* notebook.

Crash-safe: if the run is killed, simply re-launch with the same arguments and
it will skip slices whose output files already exist in the reconstruction folder.

Usage example
-------------
python /myhome/sdate/scripts/reconstruct_ladiff_slices.py --file_idx 1 --angular_range_frac 0.6 --num_outer_iters 6 --num_slices_batch 32 --num_inference_steps 40 --start_step_frac 0.8 --wandb_project cond_ladiff_reconstruction --checkpoint /myhome/sdate/checkpoints/ddpm_ladiff_cond_ladiff_f1.pt

runai training standard submit recon-large-ladiff-f10 -i lfbarba/sdsc_image:1.0.0 -p sdate-luisb --node-type A100 --gpu-devices-request 1 --large-shm --cpu-core-request 8 --cpu-core-limit 8 --cpu-memory-request 32G --cpu-memory-limit 32G --command -- bash /myhome/sdate/scripts/sdate_launcher.sh python /myhome/sdate/scripts/reconstruct_ladiff_slices.py --file_idx 10 --angular_range_frac 0.6 --num_outer_iters 6 --num_slices_batch 32 --num_inference_steps 50 --start_step_frac 0.2 --wandb_project cond_ladiff_reconstruction --checkpoint /myhome/sdate/checkpoints/ddpm_ladiff_cond_ladiff_f10.pt
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ── Project paths ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent   # .../sdate/
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, "/myhome/astra-torch")
sys.path.insert(0, "/myhome/chip-project")

from ladiff.fourier_wedge import apply_circle_mask
from ladiff.recon_utils import (
    load_norm_json,
    make_norm_fns,
    load_unet,
    compute_la_fbp,
    iterative_refine_slice,
)
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Iterative LA-Diffusion reconstruction – all slices, crash-safe."
    )
    # ── Data / paths ──────────────────────────────────────────────────────
    p.add_argument("--file_idx", type=int, default=1,
                   help="File index; used to build paths (default: 1)")
    p.add_argument("--data_root", type=str,
                   default="/myhome/data/sdate/shared/compression_paper",
                   help="Root directory for data files")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to UNet checkpoint (.pt). Defaults to "
                        "<project_root>/checkpoints/ddpm_ladiff_cond_ladiff_f<file_idx>.pt")

    # ── Missing-wedge geometry ────────────────────────────────────────────
    p.add_argument("--angular_range_frac", type=float, default=0.6,
                   help="Fraction of projections used as LA range (default: 0.6)")
    p.add_argument("--tilt_axis", type=int, default=0,
                   help="Tilt axis index: 0=kz, 1=ky, 2=kx (default: 0)")

    # ── Model ─────────────────────────────────────────────────────────────
    p.add_argument("--tiny_model", action="store_true",
                   help="Use the tiny UNet variant (reduced channels)")

    # ── Reconstruction hyper-parameters ──────────────────────────────────
    p.add_argument("--num_outer_iters", type=int, default=30,
                   help="Number of outer refinement iterations per slice (default: 30)")
    p.add_argument("--num_slices_batch", type=int, default=32,
                   help="Number of stochastic copies of a slice per batch (default: 32)")
    p.add_argument("--num_inference_steps", type=int, default=40,
                   help="DDIM inference steps per outer iteration (default: 40)")
    p.add_argument("--start_step_frac", type=float, default=0.8,
                   help="Fraction of inference steps for noise injection (default: 0.8)")
    p.add_argument("--average_gradient_steps", type=bool, default=True,
                   help="Whether to average gradient steps over the batch (default: True)")

    # ── W&B ──────────────────────────────────────────────────────────────
    p.add_argument("--wandb_project", type=str, default="ladiff_reconstruction",
                   help="W&B project name")
    p.add_argument("--wandb_entity", type=str, default=None,
                   help="W&B entity/team (optional)")
    p.add_argument("--no_wandb", action="store_true",
                   help="Disable W&B logging")

    # ── Misc ──────────────────────────────────────────────────────────────
    p.add_argument("--slice_indices", type=int, nargs="*", default=None,
                   help="Subset of slice indices to reconstruct (default: all)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_slice_image(recon_np: np.ndarray, path: Path,
                      vmin: float, vmax: float) -> None:
    """Save a float32 2-D array as a grayscale PNG, clipped to [vmin, vmax]."""
    arr = np.clip(recon_np, vmin, vmax)
    arr = ((arr - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _slice_npy_path(recon_dir: Path, slice_idx: int) -> Path:
    return recon_dir / f"slice_{slice_idx:05d}.npy"


def _slice_img_path(recon_dir: Path, slice_idx: int) -> Path:
    return recon_dir / f"slice_{slice_idx:05d}.png"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  PyTorch {torch.__version__}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # ── Derived paths ─────────────────────────────────────────────────────
    data_root   = Path(args.data_root)
    file_dir    = data_root / f"file_{args.file_idx}_extracted"
    recon_root  = file_dir / "reconstruction"

    gt_fbp_path = recon_root / f"gt_fbp_{args.file_idx}.npy"
    norm_json   = recon_root / f"la_fourier_{args.file_idx}_norm.json"

    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint
        else _PROJECT_ROOT / "checkpoints" / f"ddpm_ladiff_cond_ladiff_f{args.file_idx}.pt"
    )

    # Output sub-folder for per-slice files
    slices_dir = recon_root / "slices_iterative"
    slices_dir.mkdir(parents=True, exist_ok=True)

    # ── Load GT volume ────────────────────────────────────────────────────
    print(f"Loading GT volume from {gt_fbp_path} …")
    gt_fbp_np = np.load(gt_fbp_path)              # (N, H, W) float32
    gt_fbp    = torch.tensor(gt_fbp_np)
    num_slices, slice_h, slice_w = gt_fbp.shape
    print(f"  Volume shape : {gt_fbp.shape}")

    # Percentile range from the *full* volume (used for metrics & image saving)
    metric_min = float(np.percentile(gt_fbp_np, 1))
    metric_max = float(np.percentile(gt_fbp_np, 99))
    print(f"  Metric range : [{metric_min:.4f}, {metric_max:.4f}]")

    # ── Normalization ─────────────────────────────────────────────────────
    norm_min, norm_max = load_norm_json(norm_json)
    normalize_fn, denormalize_fn = make_norm_fns(norm_min, norm_max)
    print(f"  Norm range   : [{norm_min:.4f}, {norm_max:.4f}]")

    # ── Missing-wedge geometry ────────────────────────────────────────────
    angular_range_deg = int(args.angular_range_frac * 180)
    start_angle       = (180 - angular_range_deg) // 2
    tilt_axis         = args.tilt_axis
    print(f"  Angular range: {angular_range_deg}°  start={start_angle}°  axis={tilt_axis}")

    # ── Compute LA-FBP for the full volume ────────────────────────────────
    print("Computing LA-FBP …")
    la_fbp = compute_la_fbp(gt_fbp, angular_range_deg, start_angle, tilt_axis, device)
    gt_fbp = apply_circle_mask(gt_fbp)
    print(f"  LA-FBP shape : {la_fbp.shape}")

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading model from {checkpoint_path} …")
    model = load_unet(checkpoint_path, slice_w, device, tiny=args.tiny_model)
    print(f"  Parameters   : {sum(p.numel() for p in model.parameters()):,}")

    # ── W&B setup ─────────────────────────────────────────────────────────
    run = None
    if not args.no_wandb:
        import wandb  # type: ignore
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "file_idx":            args.file_idx,
                "angular_range_frac":  args.angular_range_frac,
                "angular_range_deg":   angular_range_deg,
                "start_angle":         start_angle,
                "tilt_axis":           tilt_axis,
                "num_outer_iters":     args.num_outer_iters,
                "num_slices_batch":    args.num_slices_batch,
                "num_inference_steps": args.num_inference_steps,
                "start_step_frac":     args.start_step_frac,
                "tiny_model":          args.tiny_model,
                "num_slices_total":    num_slices,
                "checkpoint":          str(checkpoint_path),
            },
            resume="allow",
        )
        print(f"W&B run: {run.url if run else 'N/A'}")

    # ── Determine which slices to process ─────────────────────────────────
    if args.slice_indices is not None:
        all_slices = sorted(args.slice_indices)
    else:
        all_slices = list(range(num_slices))

    # Crash-safe: skip slices already completed
    pending_slices = [
        s for s in all_slices
        if not _slice_npy_path(slices_dir, s).exists()
    ]
    skipped = len(all_slices) - len(pending_slices)
    if skipped:
        print(f"Skipping {skipped} already-completed slices.")
    print(f"Processing {len(pending_slices)} slices …\n")

    dr = metric_max - metric_min

    # ── Per-slice reconstruction loop ─────────────────────────────────────
    for progress_idx, slice_idx in enumerate(pending_slices):
        t0 = time.time()
        print(f"[{progress_idx + 1}/{len(pending_slices)}]  Slice {slice_idx} …")

        best_recon_np, metrics_history = iterative_refine_slice(
            slice_idx=slice_idx,
            gt_fbp=gt_fbp,
            la_fbp=la_fbp,
            model=model,
            normalize_fn=normalize_fn,
            denormalize_fn=denormalize_fn,
            angular_range_deg=angular_range_deg,
            start_angle=start_angle,
            tilt_axis=tilt_axis,
            device=device,
            num_slices_batch=args.num_slices_batch,
            num_outer_iters=args.num_outer_iters,
            start_step_frac=args.start_step_frac,
            num_inference_steps=args.num_inference_steps,
            average_gradient_steps=args.average_gradient_steps
        )

        elapsed = time.time() - t0
        gt_slice_np = gt_fbp[slice_idx].numpy()
        psnr_best = compute_psnr(gt_slice_np, best_recon_np, data_range=dr)
        ssim_best = compute_ssim(gt_slice_np, best_recon_np, data_range=dr)

        # Local (per-slice) data range from 1st/99th percentile of the GT slice
        local_dr = float(np.percentile(gt_slice_np, 99) - np.percentile(gt_slice_np, 1))
        local_psnr_best = compute_psnr(gt_slice_np, best_recon_np, data_range=local_dr)
        local_ssim_best = compute_ssim(gt_slice_np, best_recon_np, data_range=local_dr)

        # LA-FBP baseline for reference
        la_slice_np = la_fbp[slice_idx].numpy()
        psnr_la = compute_psnr(gt_slice_np, la_slice_np, data_range=dr)
        ssim_la = compute_ssim(gt_slice_np, la_slice_np, data_range=dr)
        local_psnr_la = compute_psnr(gt_slice_np, la_slice_np, data_range=local_dr)
        local_ssim_la = compute_ssim(gt_slice_np, la_slice_np, data_range=local_dr)

        print(f"  Best PSNR={psnr_best:.2f} dB  SSIM={ssim_best:.4f}  "
              f"local_PSNR={local_psnr_best:.2f} dB  local_SSIM={local_ssim_best:.4f}  "
              f"(LA baseline: PSNR={psnr_la:.2f}  SSIM={ssim_la:.4f}  "
              f"local_PSNR={local_psnr_la:.2f}  local_SSIM={local_ssim_la:.4f})  "
              f"elapsed={elapsed:.1f}s")

        # ── Save per-slice npy + PNG ──────────────────────────────────────
        np.save(_slice_npy_path(slices_dir, slice_idx), best_recon_np)
        _save_slice_image(
            best_recon_np,
            _slice_img_path(slices_dir, slice_idx),
            vmin=metric_min,
            vmax=metric_max,
        )

        # ── Log to W&B ───────────────────────────────────────────────────
        if run is not None:
            for m in metrics_history:
                run.log({
                    f"slice_{slice_idx}/psnr_full":  m["psnr_full"],
                    f"slice_{slice_idx}/ssim_full":  m["ssim_full"],
                    f"slice_{slice_idx}/psnr_mean":  m["psnr_mean"],
                    f"slice_{slice_idx}/ssim_mean":  m["ssim_mean"],
                    f"slice_{slice_idx}/outer_iter": m["outer_iter"],
                })
            run.log({
                "slice_idx":        slice_idx,
                "psnr_best":        psnr_best,
                "ssim_best":        ssim_best,
                "local_psnr_best":  local_psnr_best,
                "local_ssim_best":  local_ssim_best,
                "psnr_la":          psnr_la,
                "ssim_la":          ssim_la,
                "local_psnr_la":    local_psnr_la,
                "local_ssim_la":    local_ssim_la,
                "elapsed_s":        elapsed,
            })

    # ── Assemble volume from per-slice npy files ──────────────────────────
    print("\nAssembling final volume …")
    all_done = [
        s for s in all_slices
        if _slice_npy_path(slices_dir, s).exists()
    ]
    if len(all_done) < len(all_slices):
        missing = set(all_slices) - set(all_done)
        print(f"WARNING: {len(missing)} slices are still missing: {sorted(missing)}")
        print("Skipping volume assembly.")
    else:
        # Load and stack in order
        sample_shape = np.load(_slice_npy_path(slices_dir, all_done[0])).shape
        volume = np.zeros((num_slices, *sample_shape), dtype=np.float32)
        for s in all_slices:
            volume[s] = np.load(_slice_npy_path(slices_dir, s))

        out_path = recon_root / f"diffusion_{args.file_idx}_iterative.npy"
        np.save(out_path, volume)
        print(f"Volume saved to {out_path}  shape={volume.shape}")

        # Log final aggregate metrics to W&B
        if run is not None:
            gt_np = gt_fbp.numpy()
            psnr_vol = compute_psnr(gt_np, volume, data_range=dr)
            ssim_vol = float(np.mean([
                compute_ssim(gt_np[s], volume[s], data_range=dr)
                for s in all_slices
            ]))
            local_ssim_vol = float(np.mean([
                compute_ssim(
                    gt_np[s], volume[s],
                    data_range=float(np.percentile(gt_np[s], 99) - np.percentile(gt_np[s], 1)),
                )
                for s in all_slices
            ]))
            local_psnr_vol = float(np.mean([
                compute_psnr(
                    gt_np[s], volume[s],
                    data_range=float(np.percentile(gt_np[s], 99) - np.percentile(gt_np[s], 1)),
                )
                for s in all_slices
            ]))
            run.summary["volume_psnr"] = psnr_vol
            run.summary["volume_ssim"] = ssim_vol
            run.summary["volume_local_psnr"] = local_psnr_vol
            run.summary["volume_local_ssim"] = local_ssim_vol
            print(f"Volume PSNR={psnr_vol:.2f} dB  SSIM={ssim_vol:.4f}  "
                  f"local_PSNR={local_psnr_vol:.2f} dB  local_SSIM={local_ssim_vol:.4f}")

        # Clean up individual slice files
        print("Removing per-slice .npy files …")
        for s in all_slices:
            p = _slice_npy_path(slices_dir, s)
            if p.exists():
                p.unlink()
        print("Done.")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
