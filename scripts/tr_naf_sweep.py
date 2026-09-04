#!/usr/bin/env python
"""Train a sweep of TR-NAF configs on a time-resolved dataset; save results + metrics + movies.

Fully argument-driven so it runs on ANY time-resolved tif dataset without editing
code.  Builds ONE shared acquisition, computes the (dynamic) SW-FBP baseline
once, trains a field per value of K, and writes everything under a per-dataset
output directory so multiple datasets never collide.

Outputs (in --out, default checkpoints/tr_naf/<dataset_name>):
  acquisition.pt   GT volumes, times, angles, dynamic SW-FBP baseline, meta
  K{K}.pt          per-run coefficients + config + losses
  summary.json     metrics table
  movies/          per-slice HEVC movies (gt + sw_fbp once, recon per --movie-k)

Examples
--------
  # 212 Wunderkerze (uint16), skip=5
  python scripts/tr_naf_sweep.py \
      --data-path /myhome/data/sdate/shared/time_resolved/212_Wunderkerze2/timesteps/212_Wunderkerze2_rotate_04001.tif \
      --skip 5

  # 149 ASM (uint8) — same command, different path; normalization is automatic
  python scripts/tr_naf_sweep.py \
      --data-path /myhome/data/sdate/shared/time_resolved/149_ASM_SP_1ktps/timesteps/149_ASM_SP_1ktps_rotate_35001.tif \
      --skip 5
"""

import sys, json, argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")
sys.path.insert(0, "/myhome/chip-project")

from sdate.tr_naf import (
    build_acquisition, sliding_window_fbp, tr_naf_reconstruction,
    reconstruct_volume_at, evaluate_frames, make_circular_mask, save_result,
    load_result, generate_slice_movies, build_noise_acquisition, per_frame_fbp,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def dataset_name_from_path(data_path: Path) -> str:
    """.../<dataset>/timesteps/<file>.tif -> <dataset> (falls back to file stem)."""
    p = Path(data_path)
    if p.parent.name == "timesteps":
        return p.parent.parent.name
    return p.parent.name or p.stem


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", type=str,
                    default="/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2/timesteps/212_Wunderkerze2_rotate_04001.tif",
                    help="Starting tif file of the dataset (5-digit numeric suffix).")
    ap.add_argument("--out", type=str, default=None,
                    help="Output dir (default: checkpoints/tr_naf/<dataset_name>).")
    # acquisition geometry
    ap.add_argument("--cube", type=int, default=128)
    ap.add_argument("--num-frames", type=int, default=25, help="time frames (=sweeps*180/angle_range)")
    ap.add_argument("--angle-range", type=float, default=None,
                    help="wedge width per frame (deg). Default: 36 (limited_angle) / 180 (noise).")
    ap.add_argument("--num-full-projs", type=int, default=360, help="proj density over 180 deg")
    ap.add_argument("--skip", type=int, default=5, help="tif timesteps skipped between frames")
    ap.add_argument("--full-boundary-frames", type=int, default=1,
                    help="first/last N frames use the full 0-180 range (static object "
                         "before/after dynamics). 0 disables.")
    ap.add_argument("--norm-min", type=float, default=None,
                    help="normalization min; omit for auto per-dataset min/max (recommended).")
    ap.add_argument("--norm-max", type=float, default=None, help="normalization max; omit for auto.")
    # training
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--k-values", type=int, nargs="+", default=[1, 4, 6, 10])
    ap.add_argument("--n-levels", type=int, default=8, help="hash-encoding levels")
    ap.add_argument("--max-resolution", type=int, default=None,
                    help="finest hash-grid resolution. Default = cube (grid==voxels). "
                         "Set BELOW cube (e.g. 64 for a 128 volume) to suppress "
                         "limited-angle speckle in hard/high-motion cases.")
    ap.add_argument("--reg-tv", type=float, default=1e-3,
                    help="spatial TV weight (mean-normalized). ~5e-2 for hard cases.")
    ap.add_argument("--reg-temporal-max", type=float, default=1e-2,
                    help="L2 temporal curvature weight (annealed stiff->0). Set 0 to disable.")
    ap.add_argument("--reg-temporal-tv", type=float, default=0.0,
                    help="L1 temporal TV weight (edge-preserving; constant). Set 0 to disable.")
    ap.add_argument("--temporal-anneal-frac", type=float, default=0.6)
    ap.add_argument("--retrain", action="store_true",
                    help="retrain even if a checkpoint already exists (default: load & skip).")
    # regime: limited-angle (line-integral/MSE) vs noise (count-space Poisson/Anscombe)
    ap.add_argument("--regime", choices=["limited_angle", "noise"], default="limited_angle")
    ap.add_argument("--photons", type=float, default=1e4, help="[noise] flat-field I0 (dose knob)")
    ap.add_argument("--p-max", type=float, default=2.5, help="[noise] max line integral (contrast)")
    ap.add_argument("--read-noise", type=float, default=0.0, help="[noise] Gaussian read-noise sigma")
    ap.add_argument("--gain", type=float, default=1.0, help="[noise] detector gain")
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--data-fidelity", choices=["mse", "poisson_nll", "anscombe"], default=None,
                    help="loss; default mse for limited_angle, poisson_nll for noise.")
    # movies
    ap.add_argument("--movie-k", type=int, nargs="*", default=[4, 10],
                    help="K values to render recon movies for ([] to disable).")
    ap.add_argument("--num-movie-slices", type=int, default=5)
    ap.add_argument("--movie-fps", type=int, default=5)
    return ap.parse_args()


def main():
    args = parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.data_path)
    dset = dataset_name_from_path(data_path)
    default_root = PROJECT_ROOT / "checkpoints" / "tr_naf"
    if args.out:
        out = Path(args.out)
    else:
        out = (default_root / "noise" / dset) if args.regime == "noise" else (default_root / dset)
    out.mkdir(parents=True, exist_ok=True)

    fidelity = args.data_fidelity or ("poisson_nll" if args.regime == "noise" else "mse")
    angle_range = args.angle_range if args.angle_range is not None else (
        180.0 if args.regime == "noise" else 36.0)
    norm_range = None
    if args.norm_min is not None and args.norm_max is not None:
        norm_range = (args.norm_min, args.norm_max)

    print(f"Dataset: {dset}\nRegime:  {args.regime} (fidelity={fidelity})\nOutput:  {out}\nDevice:  {dev}",
          flush=True)

    # ---- build acquisition + baseline (regime-specific; model/loss are shared) ----
    if args.regime == "noise":
        frames, meta = build_noise_acquisition(
            data_path, num_frames=args.num_frames, angle_range_deg=angle_range,
            num_full_projs=args.num_full_projs, timestep_skip=args.skip, cube_size=args.cube,
            photons=args.photons, p_max=args.p_max, gain=args.gain, read_noise=args.read_noise,
            dark=0.0, normalize_range=norm_range, noise_seed=args.noise_seed, device=dev,
        )
        print(f"acquisition: {len(frames)} frames | {meta['num_la_projs']} projs/frame "
              f"(angle={meta['angle_range_deg']:.0f} deg) | I0={meta['photons']:.0e} "
              f"read={meta['read_noise']} | cube={args.cube}", flush=True)
        base_vols = per_frame_fbp(frames, meta, device=dev)
        base_name = "per-frame FBP"
    else:
        frames, meta = build_acquisition(
            data_path, num_frames=args.num_frames, angle_range_deg=angle_range,
            num_full_projs=args.num_full_projs, timestep_skip=args.skip, cube_size=args.cube,
            full_boundary_frames=args.full_boundary_frames,
            normalize_range=norm_range, device=dev,
        )
        print(f"acquisition: {len(frames)} frames | {meta['num_sweeps']:.1f} sweeps | "
              f"{meta['num_la_projs']} projs/frame | cube={args.cube}", flush=True)
        base_vols = sliding_window_fbp(frames, meta, device=dev)
        base_name = f"SW-FBP (window={max(1, round(180.0/meta['angle_range_deg']))})"

    true_vols = [f.true_volume for f in frames]
    times = [f.t_norm for f in frames]
    mask = make_circular_mask(args.cube, args.cube, device=dev)
    scale = float(meta.get("scale", 1.0))   # noise regime learns mu in scaled units

    sw = base_vols  # kept name for the saved bundle
    sw_metrics = evaluate_frames(base_vols, true_vols, mask=mask)
    print(f"{base_name}: mean PSNR {sw_metrics['psnr'].mean():.2f} dB | "
          f"SSIM {sw_metrics['ssim'].mean():.4f} | rel-err {sw_metrics['rel_err'].mean():.4f}", flush=True)

    torch.save({
        "true_volumes": torch.stack(true_vols).cpu(),
        "times": times,
        "angles": [f.angles_deg for f in frames],
        "sw_fbp": torch.stack(sw).cpu(),   # baseline volumes (SW-FBP or per-frame FBP)
        "baseline_name": base_name,
        "meta": meta,
        "scale": scale,
        "data_range": sw_metrics["data_range"],
        "dataset": dset,
    }, out / "acquisition.pt")

    summary = {"dataset": dset, "regime": args.regime, "fidelity": fidelity,
               "baseline": base_name,
               "config": dict(cube=args.cube, num_frames=args.num_frames, angle_range=args.angle_range,
                              num_full_projs=args.num_full_projs, skip=args.skip, iters=args.iters,
                              n_levels=args.n_levels, max_resolution=args.max_resolution,
                              reg_tv=args.reg_tv, reg_temporal_tv=args.reg_temporal_tv,
                              photons=args.photons, p_max=args.p_max, read_noise=args.read_noise,
                              full_boundary_frames=args.full_boundary_frames,
                              norm=(meta["norm_min"], meta["norm_max"])),
               "sw_fbp": {"psnr": float(sw_metrics["psnr"].mean()),
                          "ssim": float(sw_metrics["ssim"].mean()),
                          "rel_err": float(sw_metrics["rel_err"].mean())},
               "runs": {}}

    for K in args.k_values:
        ckpt = out / f"K{K}.pt"
        if ckpt.exists() and not args.retrain:
            print(f"\n=== K={K}: loading existing checkpoint (skip training) ===", flush=True)
            result, _ = load_result(ckpt, device=dev)
        else:
            print(f"\n=== Training K={K} ===", flush=True)
            result = tr_naf_reconstruction(
                frames, meta, K=K, n_iterations=args.iters, lr=args.lr,
                reg_tv=args.reg_tv, reg_temporal_max=args.reg_temporal_max,
                reg_temporal_tv=args.reg_temporal_tv,
                temporal_anneal_frac=args.temporal_anneal_frac,
                data_fidelity=fidelity,
                field_kwargs=dict(n_levels=args.n_levels, base_resolution=8,
                                  max_resolution=args.max_resolution,
                                  hidden_dim=128, n_hidden_layers=3),
                device=dev, seed=0, verbose=False,
            )
            save_result(result, meta, ckpt)
        recon = [reconstruct_volume_at(result, t) / scale for t in times]
        m = evaluate_frames(recon, true_vols, mask=mask, data_range=sw_metrics["data_range"])
        summary["runs"][str(K)] = {
            "psnr": float(m["psnr"].mean()), "ssim": float(m["ssim"].mean()),
            "rel_err": float(m["rel_err"].mean()), "time_s": result["time"],
            "final_loss": result["losses"][-1], "psnr_per_frame": m["psnr"].tolist(),
        }
        print(f"K={K}: PSNR {m['psnr'].mean():.2f} dB | SSIM {m['ssim'].mean():.4f} | "
              f"rel-err {m['rel_err'].mean():.4f} | {result['time']:.0f}s", flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n==== SUMMARY ====", flush=True)
    print(f"{'config':>14} {'PSNR':>8} {'SSIM':>8} {'rel-err':>8}")
    print(f"{base_name:>14} {summary['sw_fbp']['psnr']:>8.2f} "
          f"{summary['sw_fbp']['ssim']:>8.4f} {summary['sw_fbp']['rel_err']:>8.4f}")
    for K in args.k_values:
        r = summary["runs"][str(K)]
        print(f"{'TR-NAF K=' + str(K):>12} {r['psnr']:>8.2f} {r['ssim']:>8.4f} {r['rel_err']:>8.4f}")
    print(f"\nSaved to {out}")

    # ---- movies: GT + dynamic SW-FBP once, recon per --movie-k ----
    if args.movie_k:
        movie_dir = out / "movies"
        print(f"\nGenerating movies for K={args.movie_k} (+ GT + SW-FBP) ...", flush=True)
        try:
            generate_slice_movies(movie_dir, times, gt_volumes=true_vols, sw_fbp=sw,
                                  num_movie_slices=args.num_movie_slices, fps=args.movie_fps, crf=18)
            for K in args.movie_k:
                result, _ = load_result(out / f"K{K}.pt", device=dev)
                recon = [reconstruct_volume_at(result, t) / scale for t in times]
                generate_slice_movies(movie_dir, times, recon_volumes=recon,
                                      num_movie_slices=args.num_movie_slices, fps=args.movie_fps,
                                      crf=18, prefix=f"K{K}_")
            print(f"Movies saved to {movie_dir}")
        except Exception as e:  # e.g. ffmpeg not installed
            print(f"[movies skipped] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
