#!/usr/bin/env python3
"""Annealed-N2V: iterative renoising/annealing denoiser, round 0 = an EXISTING
baseline N2V+Huber+context checkpoint (wunderkerze2, dose=0.05).

Pivot from tr_diffusion_anneal_n2n.py (see conversation/memory): the plain-MSE
N2N round denoiser let the network route noise straight through via skip
connections since nothing forced it to ignore the directly-visible noisy input
pixel. N2V's blind-spot masking removes that shortcut structurally, and this
project already has a validated N2V+Huber+context checkpoint that produces a
genuinely blurred-but-noiseless E[x|context] estimate -- so round 0 here is
FREE (load that checkpoint, no training), and every round after warm-starts
the SAME architecture via the SAME BaselineN2VLoss, but self-supervising
against the current iterate x_hat_k instead of a fresh video read. The
renoise blend always mixes back in the ORIGINAL fixed noisy measurement
(never the iterate), directly in the checkpoint's own normalised [-1,1]
raw-count space -- no Anscombe transform this time.

See sdate/tr_diffusion/CONTEXT.md / docs/adr/0001,0002 for the design
decisions this still carries over (fixed measurement seed, cosine anneal
schedule w/ the alphas[k+1] shift, fixed per-round epoch budget monitored via
the PSNR/SSIM trajectory, final reconstruction eval on the k=21-baseline's
window).

Needs a real GPU (UNet training/inference) + ASTRA (astra_torch, for the
final reconstruction eval): submit via RunAI, e.g.

  runai workspace submit sdate-anneal-n2v \\
    -i lfbarba/sdsc_image:1.0.1 -p sdate-luisb \\
    --gpu-request-type portion --gpu-portion-request 0.5 \\
    --node-type A100 --large-shm --cpu-core-request 4 --cpu-core-limit 10 \\
    --cpu-memory-limit 64G --preemptibility preemptible \\
    --command -- bash /myhome/sdate/scripts/sdate_launcher.sh \\
      python scripts/tr_diffusion_anneal_n2v.py \\
      --run_dir /myhome/data/sdate/shared/time_resolved/212_Wunderkerze2/tr_anneal_n2v/run1

If preempted, resubmit the IDENTICAL command -- --run_dir's state.json makes
it resume from the last completed round automatically.
"""
from __future__ import annotations

import os
import subprocess
import sys as _sys

os.environ["PATH"] = "/myhome/bin:" + os.environ.get("PATH", "")

try:
    import pytorch_base  # noqa: F401
except ImportError:
    # `pip install -e /myhome/BaseTraining` (tried first) registered the
    # package but it still wasn't importable in THIS process -- rather than
    # chase why, pytorch_base is a plain importable package directory with no
    # compiled/build step, so just put its parent on sys.path directly.
    _sys.path.insert(0, "/myhome/BaseTraining")
    import pytorch_base  # noqa: F401

# `sdate`'s setup.py doesn't pin diffusers; a freshly `pip install -e .`'d
# container can pick up a diffusers release newer than this image's torch
# build supports -- diffusers.utils.torch_utils accesses torch.xpu.empty_cache
# at IMPORT TIME to build a per-backend dispatch dict, which raises
# AttributeError on a torch build with no xpu (Intel GPU) submodule. Pinning
# diffusers to an older version cascaded into pip downgrading torch itself
# (worse). Stubbing the one missing attribute is far more surgical: nothing
# on this CUDA-only job ever calls torch.xpu.* for real, the dispatch dict
# just needs the reference to exist at import time.
class _DummyTorchBackend:
    """No-op stand-in for an accelerator backend submodule (torch.xpu/mps/...):
    ANY attribute access resolves to a no-op callable, so it doesn't matter
    exactly which method(s) diffusers.utils.torch_utils happens to reference
    at import time (empty_cache, device_count, ...)."""

    def __getattr__(self, name):
        return lambda *a, **kw: None


import torch as _torch
for _backend in ("xpu", "mps", "npu", "mtia"):
    if not hasattr(_torch, _backend):
        setattr(_torch, _backend, _DummyTorchBackend())

# The above stub isn't enough on its own: this image's baked-in torch (2.0.0,
# confirmed via pip's own dependency-conflict warnings -- unrelated to sdate's
# install step, --no-deps there changed nothing) predates dtypes diffusers'
# LATEST release needs just to import (torch.float8_e4m3fn, for its torchao
# quantizer support this project never uses). Not fixable by stubbing a
# missing dtype the way a missing method can be stubbed -- pin diffusers down
# to a version that predates that requirement instead, WITH --no-deps this
# time (an earlier unscoped pin cascaded into downgrading torch itself, which
# was worse). UNet2DModel's API is stable across this range, so this is safe
# for what this project actually uses diffusers for.
try:
    from diffusers.models import UNet2DModel  # noqa: F401
except Exception:
    # diffusers' own lazy-import machinery wraps the underlying AttributeError
    # in a RuntimeError at this layer -- catch broadly rather than guess the
    # exact wrapper type.
    subprocess.run([_sys.executable, "-m", "pip", "install", "diffusers==0.27.2",
                    "--no-deps", "--force-reinstall", "-q"], check=True)

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")

from sdate.tr_diffusion.anneal import (
    AnnealN2VDataset, _IndexedWrapper, alpha_schedule, create_counts_memmap, create_x_hat_memmap,
    infer_round_n2v, load_state, pick_trajectory_indices, save_state, train_round_n2v,
    write_counts_meta,
)
from sdate.tr_diffusion.data import TimeResolvedFrameDataset
from sdate.tr_diffusion.load import load_denoiser
from sdate.tr_diffusion.losses import BaselineN2VLoss
from sdate.tr_diffusion.profiles import DatasetProfile

EVAL_NOISE_SEED = 12345   # matches reconstruct.denoise_sequence's default -- same "noisy" realization


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="wunderkerze2")
    p.add_argument("--ckpt", default="checkpoints/tr_denoise_baseline_k1_dose005.pt",
                   help="existing baseline N2V+Huber+context checkpoint -- round 0's free denoiser")
    p.add_argument("--run_dir", required=True, help="where checkpoints/x_hat memmap/trajectory/metrics live")

    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--final_round", type=int, default=None,
                   help="stop after this round (0-indexed) instead of --rounds-1, e.g. to cash out at an "
                        "SSIM peak seen in a prior run -- the alpha schedule is still computed over the "
                        "full --rounds so the per-round alphas match that prior run exactly")
    p.add_argument("--schedule", choices=["cosine", "linear"], default="cosine")
    p.add_argument("--round_epochs", type=int, default=2, help="epochs for every round after round 0 (which is free)")
    p.add_argument("--epoch_overrides", default="", help="JSON dict {round_index: epochs} to bump specific rounds")

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--data_seed", type=int, default=0, help="fixes the noisy measurement/context for the whole run")

    p.add_argument("--traj_frames", type=int, default=6)
    p.add_argument("--traj_seed", type=int, default=0)

    p.add_argument("--eval_frame_start", type=int, default=402_149)
    p.add_argument("--eval_frame_end", type=int, default=497_749)
    p.add_argument("--det_bin", type=int, default=2)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    epoch_overrides = json.loads(args.epoch_overrides) if args.epoch_overrides else {}
    final_round = args.rounds - 1 if args.final_round is None else args.final_round
    assert 0 <= final_round <= args.rounds - 1

    prof = DatasetProfile.load(args.profile)
    model, ckpt_cfg = load_denoiser(args.ckpt, device=device)
    k = int(ckpt_cfg["k"])
    crop = tuple(ckpt_cfg["crop"])
    norm_min, norm_max = float(ckpt_cfg["norm_min"]), float(ckpt_cfg["norm_max"])
    extra_noise_dose = float(ckpt_cfg["extra_noise_dose"])
    n2v_ratio = float(ckpt_cfg.get("n2v_ratio", 0.02))
    n2v_window = int(ckpt_cfg.get("n2v_window", 5))
    conditioning_probability = float(ckpt_cfg.get("conditioning_probability", 0.5))
    loss_type = ckpt_cfg.get("loss_type", "huber")
    assert not ckpt_cfg.get("poisson_head", False), "this pipeline assumes a plain single-channel checkpoint"
    print(f"loaded {args.ckpt}: k={k} crop={crop} norm=({norm_min:.2f},{norm_max:.2f}) "
          f"dose={extra_noise_dose} n2v_ratio={n2v_ratio} n2v_window={n2v_window} "
          f"conditioning_probability={conditioning_probability} loss_type={loss_type}", flush=True)

    base_ds = TimeResolvedFrameDataset(
        mov_path=prof.mov_path, memmap_path=prof.memmap_path, k=k,
        frame_start=prof.frame_start, frame_end=prof.frame_end, crop=crop,
        axis_col=prof.rot_axis_col, deg_per_frame=prof.deg_per_frame,
        norm_range=(norm_min, norm_max),
        extra_noise_dose=extra_noise_dose, noise_seed=args.data_seed,
    )
    n = len(base_ds)
    first_index = int(base_ds.indices.min())
    data_range = norm_max - norm_min
    print(f"dataset: {n} usable frames, crop={crop}, k={k}", flush=True)

    state_path = run_dir / "model"
    x_hat_path = run_dir / "x_hat.f16"
    state = load_state(state_path)
    if state is not None:
        start_round = int(state["last_completed_round"]) + 1
        ckpt = torch.load(str(state_path) + ".pt", map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"resuming from round {start_round}", flush=True)
    else:
        start_round = 0

    alphas = alpha_schedule(args.rounds, kind=args.schedule)
    print(f"alpha schedule ({args.schedule}): {np.round(alphas, 3).tolist()}", flush=True)

    traj_indices = pick_trajectory_indices(base_ds, args.eval_frame_start, args.eval_frame_end,
                                           n=args.traj_frames, seed=args.traj_seed)
    traj_frame_ids = base_ds.indices[traj_indices]
    print(f"trajectory frames: {traj_frame_ids.tolist()}", flush=True)

    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else []
    (run_dir / "trajectory").mkdir(exist_ok=True)
    mmse_anchor_path = run_dir / "mmse_anchor.f16"
    final_estimate_path = run_dir / "final_estimate.f16"

    if start_round == 0:
        fresh_mm = create_x_hat_memmap(x_hat_path, n, crop)  # created empty; round 0 only WRITES into it

        print("\n=== round 0/{}  (free -- existing checkpoint, no training) ===".format(final_round), flush=True)
        loader = torch.utils.data.DataLoader(_IndexedWrapper(base_ds), batch_size=args.eval_batch_size,
                                             shuffle=False, num_workers=args.num_workers, pin_memory=True)
        out_mm = create_counts_memmap(mmse_anchor_path, n, crop)
        is_last = (final_round == 0)
        alpha_k = None if is_last else float(alphas[1])
        result = infer_round_n2v(model, loader, device, alpha_k, data_range, norm_min, norm_max,
                                 x_hat_mm=(None if is_last else fresh_mm),
                                 trajectory_indices=traj_indices, out_counts_mm=out_mm)
        print(f"round 0: PSNR {result['psnr']:.2f}  SSIM {result['ssim']:.3f}"
              + ("" if alpha_k is None else f"  (renoised with alpha={alpha_k:.3f})"), flush=True)
        metrics.append({"round": 0, "epochs": 0, "alpha": alpha_k, "psnr": result["psnr"], "ssim": result["ssim"]})
        metrics_path.write_text(json.dumps(metrics, indent=2))
        np.savez(run_dir / "trajectory" / "round_00.npz", frame_ids=traj_frame_ids,
                 x_hat=np.stack([result["trajectory"][int(i)]["x_hat"] for i in traj_indices]),
                 D=np.stack([result["trajectory"][int(i)]["D"] for i in traj_indices]))
        write_counts_meta(mmse_anchor_path, first_index, n, crop, extra_noise_dose, EVAL_NOISE_SEED,
                          ckpt="round0_mmse_anchor_existing_checkpoint")
        if is_last:
            write_counts_meta(final_estimate_path, first_index, n, crop, extra_noise_dose, EVAL_NOISE_SEED,
                              ckpt="round0_final_estimate")
            import shutil
            shutil.copyfile(mmse_anchor_path, final_estimate_path)
        save_state(state_path, model, 0, {"rounds": args.rounds, "schedule": args.schedule, "k": k})
        del fresh_mm
        start_round = 1

    dataset = AnnealN2VDataset(base_ds, x_hat_path)
    for k_round in range(start_round, final_round + 1):
        is_last = (k_round == final_round)
        epochs = int(epoch_overrides.get(str(k_round), args.round_epochs))
        print(f"\n=== round {k_round}/{final_round}  epochs={epochs}  {'(final)' if is_last else ''} ===", flush=True)

        loss_fn = BaselineN2VLoss(device, ratio=n2v_ratio, window=n2v_window,
                                  conditioning_probability=conditioning_probability,
                                  loss_type=loss_type, poisson_head=False)
        train_round_n2v(model, dataset, loss_fn, epochs, device, batch_size=args.batch_size,
                        lr=args.learning_rate, weight_decay=args.weight_decay, num_workers=args.num_workers)

        loader = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)
        out_mm = create_counts_memmap(final_estimate_path, n, crop) if is_last else None
        alpha_k = None if is_last else float(alphas[k_round + 1])
        x_hat_mm = None if is_last else dataset._memmap()
        result = infer_round_n2v(model, loader, device, alpha_k, data_range, norm_min, norm_max,
                                 x_hat_mm=x_hat_mm, trajectory_indices=traj_indices, out_counts_mm=out_mm)
        print(f"round {k_round}: PSNR {result['psnr']:.2f}  SSIM {result['ssim']:.3f}"
              + ("" if alpha_k is None else f"  (renoised with alpha={alpha_k:.3f})"), flush=True)

        metrics.append({"round": k_round, "epochs": epochs, "alpha": alpha_k,
                        "psnr": result["psnr"], "ssim": result["ssim"]})
        metrics_path.write_text(json.dumps(metrics, indent=2))
        np.savez(run_dir / "trajectory" / f"round_{k_round:02d}.npz", frame_ids=traj_frame_ids,
                 x_hat=np.stack([result["trajectory"][int(i)]["x_hat"] for i in traj_indices]),
                 D=np.stack([result["trajectory"][int(i)]["D"] for i in traj_indices]))
        if is_last:
            write_counts_meta(final_estimate_path, first_index, n, crop, extra_noise_dose, EVAL_NOISE_SEED,
                              ckpt=f"round{k_round}_final_estimate")
        save_state(state_path, model, k_round, {"rounds": args.rounds, "schedule": args.schedule, "k": k})

    print("\n=== round loop complete; reconstructing MMSE anchor + final estimate ===", flush=True)
    from sdate.tr_diffusion import reconstruct as R
    variants = {"annealed_n2v_final": str(final_estimate_path), "mmse_anchor": str(mmse_anchor_path)}
    res = R.run_windows(prof.mov_path, prof.memmap_path, variants, det_bin=args.det_bin, method="fbp",
                        frame_start=args.eval_frame_start, frame_end=args.eval_frame_end,
                        axis_col=prof.rot_axis_col, deg_per_frame=prof.deg_per_frame,
                        device=device, log_every=25)
    summary = {"n_windows": len(res["window_starts"]), "eval_frame_range": [args.eval_frame_start, args.eval_frame_end]}
    for arm in list(variants) + ["noisy"]:
        m = res["metrics"][arm]
        summary[f"{arm}_psnr"] = float(m["psnr"].mean())
        summary[f"{arm}_ssim"] = float(m["ssim"].mean())
        summary[f"{arm}_sharpness"] = float(m["sharpness"].mean())
        print(f"  {arm:20s} PSNR {m['psnr'].mean():6.2f}  SSIM {m['ssim'].mean():.3f}  "
              f"sharpness {m['sharpness'].mean():.3f}", flush=True)
    (run_dir / "recon_summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
