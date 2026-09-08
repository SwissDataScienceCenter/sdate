"""Sliding-window CT reconstruction from (denoised) time-resolved projections.

End-to-end test of the denoiser: reconstruct the rotating ``212_Wunderkerze2``
sample from its projection stream and measure how close the **denoised**
reconstruction is to the **GT (native, low-noise)** reconstruction.

Pipeline
--------
1. ``denoise_sequence`` — run the baseline denoiser over the full usable frame
   range (dose-0.05-noised input) and cache the denoised projections to a memmap.
2. Angles: each frame ``f`` is a projection at ``f * DEG_PER_FRAME`` degrees
   (calibration; see ``project-wunderkerze2-rotation``).
3. ``sliding_windows`` — a window spanning ``window_deg`` (~180°, ~100 frames)
   slides by ``stride`` frames (default = disjoint 180°).
4. Projections are raw detector **counts**; converted to attenuation line
   integrals ``p = -ln(count / I0)`` (``I0`` = bright/air level) before recon.
5. ``reconstruct`` — parallel-beam (``lamino_angle=0``) FBP (fast, whole sweep)
   or GD (iterative, the hexplane method; on a subset). Detector binned by
   ``det_bin`` (in-plane resolution knob).

Three arms, identical geometry, compared with circular-masked PSNR/SSIM vs GT:
``GT`` (native), ``denoised`` (baseline output), ``noisy`` (un-denoised floor).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .data import _center_crop, resolve_warped_context_dir
from .frames import MemmapFrameSource
from .geometry import DEG_PER_FRAME, PERIOD_180, ROT_AXIS_COL
from .noise import add_poisson_noise

DEFAULT_GD = dict(max_epochs=300, lr=1e-1, clamp_min=0.0)
ARMS = ("GT", "denoised", "noisy")


# --------------------------------------------------------------------------- #
# stage 1: denoise the full sequence -> cached memmap of denoised counts
# --------------------------------------------------------------------------- #
@torch.no_grad()
def denoise_sequence(ckpt, mov_path, memmap_path, out_path,
                     frame_start: int = 400_000, frame_end: int = 500_000,
                     dose: float = 0.05, native_noise: bool = False, noise_seed: int = 12345, ss_timestep: int = 500,
                     num_samples: int = 1, n2n_q: float = 1.0,
                     diffusion_inference: str = "single_shot", ancestral_num_steps: int = 50,
                     batch: int = 64, num_workers: int = 8,
                     deg_per_frame: float = DEG_PER_FRAME, axis_col: float = ROT_AXIS_COL,
                     poisson_posterior: bool = True, var_out_path: Optional[str] = None,
                     device: Optional[torch.device] = None, log_every: int = 50):
    """Denoise every usable frame and cache counts to a memmap.

    ``poisson_posterior`` (poisson_head checkpoints only): ``True`` (default)
    -- the exact Gamma-Poisson posterior-mean combination (sharper); ``False``
    -- ``mu`` alone, the single-head-equivalent point estimate. See
    :func:`sdate.tr_diffusion.pipeline.denoise_frames_baseline`.

    ``var_out_path`` (poisson_head + ``poisson_posterior=True`` checkpoints
    only): if given, ALSO writes the exact posterior variance (``var_post``
    from :func:`sdate.tr_diffusion.nb_head.posterior_mean`, raw count² units,
    NOT renormalised -- see ``return_uncertainty`` in
    :func:`sdate.tr_diffusion.pipeline.denoise_frames_baseline`) to a second
    float16 memmap, same ``first_index``/``num_frames``/``crop`` convention as
    the primary (mean) output. This is the per-pixel heteroscedastic noise
    map an Ambient-Tweedie-style consumer needs alongside the mean.

    Model type is auto-detected from ``ckpt``'s config (``mode``: baseline/diffusion,
    ``denoise_mode``: n2v/n2n, default n2v for older checkpoints):

    * **n2v baseline** — ``denoise_frames_baseline`` (one pass, blind-spot input).
    * **n2v diffusion** — single-shot ``pred_x0`` at ``ss_timestep`` (or its
      posterior mean over ``num_samples`` draws), blind-spot input.
    * **n2n baseline/diffusion** — same shapes, but the central input is the
      measured dose-thinned frame itself (optionally further thinned to fraction
      ``n2n_q``; ``1.0`` = the full measurement, expected best — see
      :func:`sdate.tr_diffusion.pipeline.denoise_frames_n2n_baseline`), and the
      class-label conditioning carries the (discretised) input fraction instead
      of the N2V present/absent flag. For n2n diffusion, ``diffusion_inference``
      selects ``"single_shot"`` (one forward pass / posterior mean over
      ``num_samples`` draws at ``ss_timestep``) or ``"ancestral"`` (full DDIM
      sampling from ``ss_timestep`` down to 0 over ``ancestral_num_steps`` steps,
      initialised from the noisy measurement — see
      :func:`sdate.tr_diffusion.pipeline.partial_diffusion_n2n`).

    Input is the dose-noised frame (deterministic per-frame noise), UNLESS
    ``native_noise=True`` -- then the central/context frames are the raw
    measured counts as-is (no synthetic Poisson thinning at all), matching a
    checkpoint trained with ``--extra_noise_dose`` omitted (see
    scripts/tr_diffusion_jointfbp_context_cache.py's own ``--native_noise``
    convention). ``dose`` still controls ``poisson_dose`` for the posterior-
    mean combination (pass ``1.0`` for native, the correct rate-relationship
    regardless of ``native_noise``) and is recorded in the output meta either
    way. Writes ``out_path`` float16 ``(n_usable, H, W)`` + ``out_path + '.meta.npz'``.
    Returns ``(first_index, num_frames, config)``.
    """
    from torch.utils.data import DataLoader

    from .data import TimeResolvedFrameDataset
    from .load import load_denoiser
    from .pipeline import (
        denoise_frames_baseline, denoise_frames_bootstrap, denoise_frames_bootstrap_reverse,
        denoise_frames_n2n_baseline, denoise_frames_noise2clean, denoise_frames_refine,
        denoise_frames_sinogram, partial_diffusion_n2n, pred_x0_ensemble, pred_x0_n2n_ensemble,
        pred_x0_n2n_swap_ensemble,
    )
    from .sino_transform import SinoTransform

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_denoiser(ckpt, device=device)
    mode = cfg.get("mode", "baseline")
    denoise_mode = cfg.get("denoise_mode", "n2v")

    # --mode bootstrap, direction="forward": `cfg` describes the (k=0, context-free)
    # bootstrap model itself, but the DATASET below must match the FROZEN base
    # checkpoint's own shape (k, crop, neighborhoods, ...) since it's the base model
    # that actually consumes the multi-frame context -- see train.py's --mode bootstrap
    # geometry-inheritance block. direction="reverse" needs neither the base model nor
    # its context at inference (the model consumes the raw measurement directly), so it
    # uses `cfg`'s own (k=0) geometry -- a plain single-frame dataset, no base checkpoint load.
    # --mode refine (angular-resolution-gap experiment, "Leg 1"): unlike bootstrap, `cfg`
    # describes the SAME geometry the frozen base checkpoint has (train.py's --mode refine
    # geometry-inheritance block copies it verbatim), so `ds_cfg = cfg` works directly --
    # it just ALSO needs `base_model` loaded and `warped_context_dir` resolved.
    base_model = None
    bootstrap_direction = cfg.get("bootstrap_direction", "forward")
    if mode == "bootstrap" and bootstrap_direction == "forward":
        base_model, base_cfg = load_denoiser(cfg["base_checkpoint"], device=device)
        ds_cfg = base_cfg
        bootstrap_input_mode = cfg.get("bootstrap_input", "mean")
        bootstrap_present = float(base_cfg.get("conditioning_probability", 1.0)) > 0.0
    elif mode == "refine":
        base_model, base_cfg = load_denoiser(cfg["base_checkpoint"], device=device)
        ds_cfg = cfg
        refine_present = float(base_cfg.get("conditioning_probability", 1.0)) > 0.0
    else:
        ds_cfg = cfg

    crop = tuple(ds_cfg["crop"])
    lo_n, hi_n = float(cfg["norm_min"]), float(cfg["norm_max"])
    sino_transform = None
    if mode == "sinogram":
        sino_transform = SinoTransform(
            crop, num_angles=int(cfg["sino_num_angles"]), det_cols=int(cfg["sino_det_cols"]),
            angle_max_deg=float(cfg["sino_angle_max_deg"]), device=device,
        )
        sino_norm_min, sino_norm_max = float(cfg["sino_norm_min"]), float(cfg["sino_norm_max"])
    cond_angle_time = bool(ds_cfg.get("cond_angle_time", False))
    warped_temporal_memmaps = (
        resolve_warped_context_dir(ds_cfg["warped_context_dir"], int(ds_cfg["k"]),
                                   bool(ds_cfg.get("include_mirror", False)),
                                   ds_cfg.get("neighborhoods", "both"),
                                   bool(ds_cfg.get("temporal_raw_pairs", False)), deg_per_frame)
        if ds_cfg.get("warped_context_dir") else None
    )
    ds = TimeResolvedFrameDataset(
        mov_path=mov_path, memmap_path=memmap_path, k=int(ds_cfg["k"]),
        frame_start=frame_start, frame_end=frame_end, crop=crop,
        neighborhoods=ds_cfg.get("neighborhoods", "both"),
        norm_range=(lo_n, hi_n), extra_noise_dose=(None if native_noise else dose), noise_seed=noise_seed,
        axis_col=axis_col, deg_per_frame=deg_per_frame,
        cond_angle_time=cond_angle_time,
        warped_temporal_memmaps=warped_temporal_memmaps,
        temporal_raw_pairs=bool(ds_cfg.get("temporal_raw_pairs", False)),
        # the (base, for bootstrap) checkpoint's OWN training range, NOT this call's
        # (possibly different) eval frame_start/frame_end -- the model's time conditioning
        # is calibrated to the training range only.
        cond_frame_start=ds_cfg.get("frame_start"), cond_frame_end=ds_cfg.get("frame_end"),
        aux_channel_memmap=ds_cfg.get("aux_channel_memmap"),
    )
    first = int(ds.indices.min())
    n = int(ds.indices.max()) - first + 1
    assert n == len(ds.indices), "usable indices must be contiguous"
    # A conditioning_probability=0 checkpoint (the "context-only"/no-N2V ablation)
    # was EXCLUSIVELY trained with the central channel zeroed and class_label=0;
    # calling it with the default present=True regime is out-of-distribution.
    present = float(cfg.get("conditioning_probability", 1.0)) > 0.0
    p_bins = int(cfg.get("p_bins", 100))
    n2n_prediction_type = cfg.get("n2n_prediction_type", "epsilon")
    print(f"denoise_sequence mode={mode} denoise_mode={denoise_mode}" +
          (f" q={n2n_q}" if denoise_mode == "n2n" else f" present={present}") +
          (f" t={ss_timestep} num_samples={num_samples}" if mode != "baseline" else "") +
          (f" diffusion_inference={diffusion_inference} pred_type={n2n_prediction_type}"
           if (denoise_mode == "n2n" and mode != "baseline") else "") +
          (f" bootstrap_direction={bootstrap_direction} bootstrap_input={bootstrap_input_mode} "
           f"base_checkpoint={cfg.get('base_checkpoint')}"
           if mode == "bootstrap" and bootstrap_direction == "forward" else
           f" bootstrap_direction={bootstrap_direction}" if mode == "bootstrap" else
           f" base_checkpoint={cfg.get('base_checkpoint')} warped_context_dir={cfg.get('warped_context_dir')}"
           if mode == "refine" else "") +
          (f" poisson_posterior={poisson_posterior}" if cfg.get("poisson_head", False) else "") +
          (f" sino_shape=({sino_transform.num_angles},{sino_transform.det_cols}) "
           f"sino_norm=({sino_norm_min:.2f},{sino_norm_max:.2f})" if mode == "sinogram" else ""), flush=True)

    if var_out_path is not None and not (mode in ("baseline", "bootstrap", "refine") and cfg.get("poisson_head", False)
                                         and poisson_posterior):
        raise ValueError("var_out_path requires a poisson_head baseline/bootstrap/refine checkpoint with "
                         "poisson_posterior=True (return_uncertainty is only meaningful for the exact "
                         "posterior combination)")
    mm = np.memmap(out_path, dtype=np.float16, mode="w+", shape=(n, crop[0], crop[1]))
    var_mm = (np.memmap(var_out_path, dtype=np.float16, mode="w+", shape=(n, crop[0], crop[1]))
             if var_out_path is not None else None)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=num_workers,
                        pin_memory=True)
    done = 0
    for bi, item in enumerate(loader):
        central = item["central"].to(device, non_blocking=True)
        context = item["context"].to(device, non_blocking=True)
        fidx = item["frame_index"].numpy()
        if denoise_mode == "n2n":
            if mode == "baseline":
                den = denoise_frames_n2n_baseline(model, central, context, q=n2n_q, p_bins=p_bins,
                                                  norm_min=lo_n, norm_max=hi_n)
            elif diffusion_inference == "ancestral":
                den, _ = partial_diffusion_n2n(model, central, context, q=n2n_q, t_start=ss_timestep,
                                               t_end=0, num_steps=ancestral_num_steps,
                                               num_samples=num_samples, p_bins=p_bins, chunk_size=32,
                                               prediction_type=n2n_prediction_type,
                                               norm_min=lo_n, norm_max=hi_n)
            elif diffusion_inference == "swap_avg":
                den, _ = pred_x0_n2n_swap_ensemble(model, central, context, lo_n, hi_n,
                                                   timestep=ss_timestep,
                                                   p_split=0.5, p_bins=p_bins,
                                                   prediction_type=n2n_prediction_type,
                                                   num_samples=num_samples, chunk_size=32)
            else:
                den, _ = pred_x0_n2n_ensemble(model, central, context, q=n2n_q, timestep=ss_timestep,
                                              num_samples=num_samples, p_bins=p_bins, chunk_size=64,
                                              prediction_type=n2n_prediction_type,
                                              norm_min=lo_n, norm_max=hi_n)
        elif mode in ("baseline", "context_only"):
            # a --warped_context_dir baseline checkpoint (Leg 2, angular-resolution-gap
            # experiment) was trained on context_warped, NOT the raw context every other
            # baseline checkpoint uses -- see losses.py's BaselineN2VLoss.compute_loss,
            # which does this exact swap at training time.
            baseline_context = (item["context_warped"].to(device, non_blocking=True)
                               if ds_cfg.get("warped_context_dir") else context)
            cond_channels = item["cond_channels"].to(device, non_blocking=True) if cond_angle_time else None
            aux_channels = item["aux_channel"].to(device, non_blocking=True) if "aux_channel" in item else None
            out = denoise_frames_baseline(model, central, baseline_context, present=present,
                                          cond_channels=cond_channels, aux_channels=aux_channels,
                                          poisson_head=bool(cfg.get("poisson_head", False)),
                                          poisson_mean_only=(cfg.get("loss_type") == "poisson"),
                                          poisson_posterior=poisson_posterior,
                                          norm_min=lo_n, norm_max=hi_n, poisson_dose=dose,
                                          return_uncertainty=var_mm is not None)  # normalized [-1,1] (+ raw var_post)
            den, var_post = out if var_mm is not None else (out, None)
        elif mode == "noise2clean":
            den = denoise_frames_noise2clean(model, central, context)
            var_post = None
        elif mode == "bootstrap" and bootstrap_direction == "reverse":
            out = denoise_frames_bootstrap_reverse(model, central, norm_min=lo_n, norm_max=hi_n,
                                                    poisson_posterior=poisson_posterior, poisson_dose=dose,
                                                    return_uncertainty=var_mm is not None)
            den, var_post = out if var_mm is not None else (out, None)
        elif mode == "bootstrap":
            # NOTE: input_mode='sample' with a large num_samples (e.g. 64-draw averaging)
            # multiplies the bootstrap model's forward-pass batch (num_samples * dataloader
            # batch); chunk_size caps how many of those go through the UNet at once. Do NOT
            # raise this casually -- even a single-channel input still runs through the full
            # 6-stage/256-channel UNet, and a chunk_size=256 attempt OOM'd a 40GB GPU portion
            # (needed >8GB for one allocation alone) at the function's own default of 32.
            out = denoise_frames_bootstrap(model, base_model, central, context,
                                           input_mode=bootstrap_input_mode, num_samples=num_samples,
                                           present=bootstrap_present, norm_min=lo_n, norm_max=hi_n,
                                           poisson_posterior=poisson_posterior, poisson_dose=dose,
                                           return_uncertainty=var_mm is not None)
            den, var_post = out if var_mm is not None else (out, None)
        elif mode == "refine":
            refine_context = item["context_warped"].to(device, non_blocking=True)
            out = denoise_frames_refine(model, base_model, central, context, refine_context,
                                        input_mode="sample", num_samples=num_samples,
                                        present=refine_present, norm_min=lo_n, norm_max=hi_n,
                                        poisson_posterior=poisson_posterior, poisson_dose=dose,
                                        return_uncertainty=var_mm is not None)
            den, var_post = out if var_mm is not None else (out, None)
        elif mode == "sinogram":
            den = denoise_frames_sinogram(model, sino_transform, central, context,
                                          norm_min=lo_n, norm_max=hi_n,
                                          sino_norm_min=sino_norm_min, sino_norm_max=sino_norm_max,
                                          present=present)
            var_post = None
        else:
            den, _ = pred_x0_ensemble(model, central, context, timestep=ss_timestep,
                                      num_samples=num_samples, chunk_size=64,
                                      ratio=cfg["n2v_ratio"], window=cfg["n2v_window"])
            var_post = None
        counts = ((den.clamp(-1, 1) + 1) * 0.5 * (hi_n - lo_n) + lo_n)[:, 0]
        counts = counts.float().cpu().numpy().astype(np.float16)
        if var_post is not None:
            var_np = var_post[:, 0].float().cpu().numpy().astype(np.float16)
        for j, f in enumerate(fidx):
            mm[int(f) - first] = counts[j]
            if var_mm is not None:
                var_mm[int(f) - first] = var_np[j]
        done += len(fidx)
        if log_every and bi % log_every == 0:
            print(f"  denoised {done}/{n}", flush=True)
    mm.flush()
    meta = dict(first_index=first, num_frames=n, crop=list(crop), dose=dose, native_noise=native_noise,
                noise_seed=noise_seed, norm_min=lo_n, norm_max=hi_n,
                ckpt=str(ckpt), mode=mode, denoise_mode=denoise_mode,
                ss_timestep=ss_timestep, num_samples=num_samples, n2n_q=n2n_q,
                diffusion_inference=diffusion_inference, poisson_posterior=poisson_posterior,
                # only present for --mode bootstrap -- omitted (not None) for every other mode so
                # a plain np.load(...).meta.npz reader never has to allow_pickle for an object-dtype
                # None field it was never going to look up anyway.
                **({"bootstrap_direction": bootstrap_direction}
                   if mode == "bootstrap" and bootstrap_direction == "reverse" else
                   {"base_checkpoint": str(cfg["base_checkpoint"]), "bootstrap_input": bootstrap_input_mode,
                    "bootstrap_direction": bootstrap_direction}
                   if mode == "bootstrap" else {}))
    np.savez(str(out_path) + ".meta.npz", **meta)
    print(f"wrote denoised memmap {out_path}  ({n} frames {first}..{first+n-1})", flush=True)
    if var_mm is not None:
        var_mm.flush()
        np.savez(str(var_out_path) + ".meta.npz", **{**meta, "content": "posterior_variance_raw_counts_sq"})
        print(f"wrote posterior-variance memmap {var_out_path}  ({n} frames {first}..{first+n-1})", flush=True)
    return first, n, meta


@torch.no_grad()
def cascade_sequence(baseline_ckpt, src_memmap_path, out_path, batch: int = 64,
                     deg_per_frame: float = DEG_PER_FRAME, poisson_posterior: bool = True,
                     device: Optional[torch.device] = None, log_every: int = 50):
    """Cascade denoiser: run the baseline on an already-denoised (e.g. diffusion
    single-shot) memmap, building the baseline's context from that same denoised
    sequence. Writes a new counts memmap + meta. The usable range shrinks by the
    temporal-context margin relative to the source.
    """
    from .geometry import build_context_layout, usable_frame_range
    from .load import load_denoiser
    from .pipeline import denoise_frames_baseline

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_denoiser(baseline_ckpt, device=device)
    assert cfg["mode"] == "baseline", "cascade second stage must be a baseline model"
    k, crop = int(cfg["k"]), tuple(cfg["crop"])
    lo_n, hi_n = float(cfg["norm_min"]), float(cfg["norm_max"])
    temporal_raw_pairs = bool(cfg.get("temporal_raw_pairs", False))
    layout = build_context_layout(k, period_360=360.0 / deg_per_frame, temporal_raw_pairs=temporal_raw_pairs)

    sm = np.load(str(src_memmap_path) + ".meta.npz")
    s_first, s_n = int(sm["first_index"]), int(sm["num_frames"])
    src = np.memmap(src_memmap_path, dtype=np.float16, mode="r", shape=(s_n, crop[0], crop[1]))

    def get(f):
        return src[int(f) - s_first].astype(np.float32)

    def get_interp(fidx):
        lo = int(np.floor(fidx)); frac = float(fidx - lo)
        return get(lo) if frac <= 1e-6 else (1 - frac) * get(lo) + frac * get(lo + 1)

    lo_u, hi_u = usable_frame_range(s_first, s_first + s_n, k, period_360=360.0 / deg_per_frame,
                                    temporal_raw_pairs=temporal_raw_pairs)
    frames = np.arange(lo_u, hi_u)
    n = len(frames)
    mm = np.memmap(out_path, dtype=np.float16, mode="w+", shape=(n, crop[0], crop[1]))
    norm = lambda c: 2.0 * (c - lo_n) / (hi_n - lo_n) - 1.0
    for b0 in range(0, n, batch):
        fb = frames[b0:b0 + batch]
        cen = np.stack([get(f) for f in fb])
        ctx = np.stack([[get_interp(f + t.frame_offset) if t.interp
                         else get(f + int(t.frame_offset)) for t in layout] for f in fb])
        cen_t = norm(torch.from_numpy(cen).to(device)).unsqueeze(1)
        ctx_t = norm(torch.from_numpy(ctx).to(device))
        den = denoise_frames_baseline(model, cen_t, ctx_t,
                                      poisson_head=bool(cfg.get("poisson_head", False)),
                                      poisson_mean_only=(cfg.get("loss_type") == "poisson"),
                                      poisson_posterior=poisson_posterior,
                                      norm_min=lo_n, norm_max=hi_n)
        counts = ((den.clamp(-1, 1) + 1) * 0.5 * (hi_n - lo_n) + lo_n)[:, 0].float().cpu().numpy().astype(np.float16)
        for j, f in enumerate(fb):
            mm[int(f) - lo_u] = counts[j]
        if log_every and (b0 // batch) % log_every == 0:
            print(f"  cascade {b0 + len(fb)}/{n}", flush=True)
    mm.flush()
    np.savez(str(out_path) + ".meta.npz", first_index=lo_u, num_frames=n, crop=list(crop),
             dose=float(sm["dose"]), noise_seed=int(sm["noise_seed"]), norm_min=lo_n, norm_max=hi_n,
             mode="cascade", src=str(src_memmap_path))
    print(f"wrote cascade memmap {out_path}  ({n} frames {lo_u}..{lo_u+n-1})", flush=True)
    return lo_u, n


@torch.no_grad()
def temporal_average_sequence(mov_path, memmap_path, out_path, k: int,
                              frame_start: int, frame_end: int,
                              dose: float = 0.05, noise_seed: int = 12345,
                              crop: Tuple[int, int] = (128, 512),
                              deg_per_frame: float = DEG_PER_FRAME, axis_col: float = ROT_AXIS_COL,
                              batch: int = 200, device: Optional[torch.device] = None,
                              log_every: int = 50):
    """Classical non-learned baseline ("sliding-window FBP" input prep).

    For each output frame ``f``, average ``2k+1`` **independent** single
    measurements of the SAME projection angle: ``f`` itself plus ``f ± m*period``
    for ``m = 1..k``. Each is a genuinely different physical acquisition from a
    different turn (interpolated at its exact sub-frame position via the same
    ``get_interp`` convention as the model's temporal context taps, then
    independently Poisson-noised at ``dose``). Averaging reduces noise
    ``~1/sqrt(2k+1)`` at the cost of blurring dynamics across the ``k`` turns
    spanned — the projections being averaged are literally earlier/later in
    time. ``k=0`` is the plain single noisy measurement (matches the "noisy"
    floor used elsewhere; not usually needed as its own memmap).

    Writes ``out_path`` float16 ``(n, H, W)`` + ``out_path + '.meta.npz'``.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    period_360 = 360.0 / deg_per_frame
    src = MemmapFrameSource(memmap_path, mov_path)
    lo = max(int(frame_start), src.first_index)
    hi = min(int(frame_end), src.last_index)
    margin = int(np.ceil(k * period_360)) + 1
    lo_u, hi_u = lo + margin, hi - margin
    if hi_u <= lo_u:
        raise ValueError(f"frame range [{lo}, {hi}) too small for k={k} (margin {margin})")
    n = hi_u - lo_u

    mm = np.memmap(out_path, dtype=np.float16, mode="w+", shape=(n, crop[0], crop[1]))
    # One persistent generator per tap offset m, advanced sequentially across the
    # whole sequence -> every (frame, m) pair gets an independent Poisson draw
    # (a real repeated-turn acquisition would never share noise across turns).
    gens = {m: torch.Generator(device=device).manual_seed(int(noise_seed) + 1_000_003 * (m + k))
            for m in range(-k, k + 1)}
    for b0 in range(0, n, batch):
        f = np.arange(lo_u + b0, min(lo_u + b0 + batch, hi_u))
        acc = torch.zeros(len(f), crop[0], crop[1], device=device)
        for m in range(-k, k + 1):
            pos = f + m * period_360
            lo_idx = np.floor(pos).astype(np.int64)
            frac = torch.from_numpy((pos - lo_idx).astype(np.float32)).to(device).view(-1, 1, 1)
            f0 = native_window_gpu(src, lo_idx, crop, axis_col, device)
            f1 = native_window_gpu(src, lo_idx + 1, crop, axis_col, device)
            interp = (1.0 - frac) * f0 + frac * f1
            acc += add_poisson_noise(interp, dose, generator=gens[m])
        avg = acc / (2 * k + 1)
        mm[b0:b0 + len(f)] = avg.float().cpu().numpy().astype(np.float16)
        if log_every and (b0 // batch) % log_every == 0:
            print(f"  swfbp k={k} {b0 + len(f)}/{n}", flush=True)
    mm.flush()
    np.savez(str(out_path) + ".meta.npz", first_index=lo_u, num_frames=n, crop=list(crop),
             dose=dose, noise_seed=noise_seed, mode="swfbp", k=k, deg_per_frame=deg_per_frame)
    print(f"wrote swfbp(k={k}) memmap {out_path}  ({n} frames {lo_u}..{lo_u+n-1})", flush=True)
    return lo_u, n


# --------------------------------------------------------------------------- #
# geometry / windows
# --------------------------------------------------------------------------- #
def projection_angles(frame_indices, ref: int = 0, mod: float = 360.0,
                      deg_per_frame: float = DEG_PER_FRAME) -> np.ndarray:
    a = (np.asarray(frame_indices, dtype=np.float64) - ref) * deg_per_frame
    return a % mod if mod else a


def window_length_frames(window_deg: float = 180.0, deg_per_frame: float = DEG_PER_FRAME) -> int:
    """Number of consecutive frames whose angular span ≈ ``window_deg``."""
    return int(round(window_deg / deg_per_frame))


def sliding_windows(start: int, end: int, stride: int, window_deg: float = 180.0,
                    deg_per_frame: float = DEG_PER_FRAME) -> List[Tuple[int, np.ndarray]]:
    """List of ``(window_start, frame_index_array)`` covering ``[start, end)``."""
    win = window_length_frames(window_deg, deg_per_frame)
    out, s = [], int(start)
    while s + win <= end:
        out.append((s, np.arange(s, s + win)))
        s += int(stride)
    return out


# --------------------------------------------------------------------------- #
# projection domain
# --------------------------------------------------------------------------- #
def estimate_I0(source: MemmapFrameSource, frames: Sequence[int],
                crop: Tuple[int, int] = (128, 512), axis_col: float = ROT_AXIS_COL,
                pct: float = 99.5) -> float:
    """Estimate the flat/air level I0 as a high percentile of the counts."""
    vals = np.concatenate([
        _center_crop(source.get(int(f)), crop[0], crop[1], axis_col).ravel() for f in frames
    ])
    return float(np.percentile(vals, pct))


def counts_to_attenuation(counts: torch.Tensor, I0: float, eps: float = 1.0) -> torch.Tensor:
    """Raw counts -> attenuation line integrals ``p = -ln(count / I0)`` (>= 0)."""
    return (-torch.log(counts.clamp_min(eps) / I0)).clamp_min(0.0)


def load_calibration_average(mov_path, crop: Tuple[int, int], axis_col: float,
                             drop_first: int = 1, ffmpeg: str = "/myhome/bin/ffmpeg",
                             height: Optional[int] = None, width: Optional[int] = None) -> np.ndarray:
    """Average a dark/flat calibration ``.mov`` to a low-noise ``(H, W)`` counts map.

    Drops the first ``drop_first`` frame(s) before averaging -- frame 0 is a
    clear decoder/settling-transient outlier in both the wunderkerze2 darks and
    flats streams (mean far off the rest of the sequence) -- then applies the
    same ``crop``/``axis_col`` window as the projection stream so the map lines
    up pixel-for-pixel with denoised/native projection tensors.

    ``height``/``width`` are the calibration stream's NATIVE (pre-crop) decoded
    frame size -- defaults to :data:`sdate.tr_diffusion.geometry.FRAME_H`/``FRAME_W``
    (the wunderkerze2 native size) for backward compatibility; pass the
    dataset's own profile ``height``/``width`` for any other dataset (they are
    NOT necessarily the same as ``crop``, which is the post-crop target size).
    """
    import subprocess

    from .frames import denormalize, load_norm_sidecar
    from .geometry import FRAME_H, FRAME_W

    height = FRAME_H if height is None else int(height)
    width = FRAME_W if width is None else int(width)
    side = load_norm_sidecar(mov_path)
    n = side["per_frame_min"].shape[0]
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(mov_path), "-pix_fmt", "gray16le", "-f", "rawvideo", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    arr = np.frombuffer(proc.stdout, np.uint16).reshape(-1, height, width)
    assert arr.shape[0] == n, f"{mov_path}: decoded {arr.shape[0]} frames, sidecar has {n}"
    counts = np.stack([
        denormalize(arr[k], side["per_frame_min"][k], side["per_frame_max"][k]) for k in range(n)
    ])
    avg = counts[drop_first:].mean(axis=0)
    return _center_crop(avg, crop[0], crop[1], axis_col)


def counts_to_attenuation_flatdark(counts: torch.Tensor, dark: torch.Tensor, flat: torch.Tensor,
                                   eps: float = 1e-3) -> torch.Tensor:
    """Raw counts -> attenuation via real per-pixel flat/dark correction.

    ``p = -ln((count - dark) / (flat - dark))`` (clamped >= 0). ``dark``/``flat``
    are ``(H, W)`` maps (or broadcastable), pre-cropped/aligned the same way as
    ``counts`` (see :func:`load_calibration_average`). Unlike the single-scalar
    ``I0`` approximation in :func:`counts_to_attenuation`, this captures the
    real per-pixel flat-field response (beam profile, detector non-uniformity)
    instead of assuming every pixel sees the same unattenuated flux.
    """
    span = (flat - dark).clamp_min(eps)
    transmission = ((counts - dark) / span).clamp_min(eps)
    return (-torch.log(transmission)).clamp_min(0.0)


def attenuation_to_counts_flatdark(atten: torch.Tensor, dark: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`counts_to_attenuation_flatdark`: ``count = dark + (flat-dark)*exp(-p)``.

    Used to turn a SIMULATED/reprojected line-integral sinogram (e.g. the
    forward-projection of a clean joint-FBP reconstruction, see
    ``scripts/tr_diffusion_jointfbp_context_cache.py``) back into raw-counts
    space, matching the convention every other cached projection memmap in
    this project uses (so it can be consumed by the same normalisation /
    dataset machinery as a real measurement, e.g. as an ``aux_channel_memmap``
    context tap) -- not a real measurement, no noise is added.
    """
    return dark + (flat - dark) * torch.exp(-atten)


def destripe_sinogram(atten: torch.Tensor, kernel: int = 31) -> torch.Tensor:
    """Suppress ring artifacts by removing the angle-invariant per-column bias.

    A genuine per-detector-column calibration error (whether from residual
    flat/dark noise or a flat/dark-vs-sample count-rate mismatch) is constant
    across all projection angles, so it survives averaging ``atten`` over
    views as a high-spatial-frequency (column-to-column) component -- real
    object/geometry content is smooth in the view average since the object
    rotates through all angles in the window. Subtracting the view-averaged
    profile's residual-above-a-smoothed-trend from every view removes the
    bias while leaving genuine structure untouched.

    ``atten``: ``(V, R, C)`` attenuation projections (views, rows, columns).
    ``kernel``: column-axis smoothing width (odd recommended); trend below
    this spatial frequency is kept, anything sharper is treated as bias.
    """
    col_mean = atten.mean(dim=0, keepdim=True)  # (1, R, C)
    pad = kernel // 2
    padded = F.pad(col_mean, (pad, pad, 0, 0), mode="reflect")
    smoothed = F.avg_pool2d(padded.unsqueeze(0), (1, kernel), stride=1).squeeze(0)
    residual = col_mean - smoothed
    return atten - residual


def bin_detector(x_vrc: torch.Tensor, b: int) -> torch.Tensor:
    """Average-pool the (R, C) detector plane of a ``(V, R, C)`` stack by ``b``."""
    if b <= 1:
        return x_vrc
    return F.avg_pool2d(x_vrc.unsqueeze(1), int(b)).squeeze(1)


# --------------------------------------------------------------------------- #
# reconstruction
# --------------------------------------------------------------------------- #
def reconstruct(p_vrc: torch.Tensor, angles_deg: np.ndarray, det_bin: int = 2,
                method: str = "fbp", filter_type: str = "hann",
                gd_kwargs: Optional[dict] = None, vol_shape: Optional[Tuple[int, int, int]] = None,
                device: Optional[torch.device] = None, clamp: bool = True) -> torch.Tensor:
    """Reconstruct one window's volume from attenuation projections ``(V, R, C)``.

    ``det_bin`` bins the detector (and thus the in-plane resolution): ``2`` gives
    a ``(R/2, C/2, C/2)`` volume, ``1`` full resolution. ``method`` is ``"fbp"``
    (fast analytic) or ``"gd"`` (iterative gradient descent).

    ``clamp`` (default ``True``, the long-standing behaviour) floors the volume at
    0. Pass ``clamp=False`` whenever reconstructions are going to be LINEARLY
    COMBINED -- e.g. averaging Noise2Inverse angular sub-reconstructions (see
    :mod:`sdate.tr_diffusion.n2i`) or blending two arms: FBP itself is linear in
    the projections, but ``clamp_min(0)`` is not, and a sparse-view
    sub-reconstruction swings far more negative than a full-view one, so
    clamping each one first biases their mean upward (measured: +12% relative L1
    at K=2, +36% at K=4, versus the full-view reconstruction). Clamp once at the
    end instead.
    """
    from astra_torch.lamino import fbp_reconstruction_masked, gd_reconstruction_masked

    device = device or p_vrc.device
    p = bin_detector(p_vrc.to(device), det_bin)
    v, r, c = p.shape
    vs = vol_shape or (r, c, c)
    angles = np.asarray(angles_deg)
    if method == "fbp":
        vol = fbp_reconstruction_masked(p, angles, lamino_angle_deg=0.0, vol_shape=vs,
                                        det_spacing_mm=1.0, filter_type=filter_type, device=device)
    elif method == "gd":
        gd = {**DEFAULT_GD, **(gd_kwargs or {})}
        gd.setdefault("batch_size", int(v))
        vol = gd_reconstruction_masked(p, angles, 0.0, vol_shape=vs, det_spacing_mm=1.0,
                                       device=device, verbose=False, **gd)
    else:
        raise ValueError(f"method must be 'fbp' or 'gd', got {method!r}")
    if vol.dim() == 4:
        vol = vol.squeeze(0)
    return vol.clamp_min(0.0) if clamp else vol


# --------------------------------------------------------------------------- #
# per-arm window projections
# --------------------------------------------------------------------------- #
def native_window(source: MemmapFrameSource, idx: np.ndarray, crop, axis_col) -> torch.Tensor:
    """GT native counts for a window -> ``(V, R, C)`` float32 (CPU, per-frame loop)."""
    return torch.from_numpy(
        np.stack([_center_crop(source.get(int(f)), crop[0], crop[1], axis_col) for f in idx])
    ).float()


def noisy_window(native: torch.Tensor, idx: np.ndarray, dose: float, noise_seed: int) -> torch.Tensor:
    """Reproduce the dose-noised counts (matches the dataset's per-frame RNG)."""
    out = torch.empty_like(native)
    for j, f in enumerate(idx):
        g = torch.Generator().manual_seed(int(noise_seed) + int(f))
        out[j] = add_poisson_noise(native[j], dose, generator=g)
    return out


# --- GPU-vectorised window loaders (fast path: no per-frame Python/CPU loop) --
def _crop_gpu(x: torch.Tensor, out_h: int, out_w: int, axis_col: float) -> torch.Tensor:
    h, w = x.shape[-2:]
    top = (h - out_h) // 2
    left = max(0, min(int(round(axis_col - out_w / 2.0)), w - out_w))
    return x[..., top:top + out_h, left:left + out_w]


def native_window_gpu(source: MemmapFrameSource, idx: np.ndarray, crop, axis_col,
                      device: torch.device) -> torch.Tensor:
    """GT native counts for a contiguous window, denormalised + cropped on GPU."""
    lo, hi = int(idx[0]), int(idx[-1]) + 1
    off = source.first_index
    raw = np.asarray(source._mm[lo - off:hi - off])          # (V,H,W) uint16, one contiguous read
    x = torch.from_numpy(raw).to(device).float()
    fmin = torch.as_tensor(source.per_frame_min[lo:hi], device=device).view(-1, 1, 1)
    fmax = torch.as_tensor(source.per_frame_max[lo:hi], device=device).view(-1, 1, 1)
    x = x / 65535.0 * (fmax - fmin).clamp_min(1e-6) + fmin
    return _crop_gpu(x, crop[0], crop[1], axis_col)


def denoised_window_gpu(den_mm, first: int, idx: np.ndarray, device: torch.device) -> torch.Tensor:
    """Denoised counts for a contiguous window straight to GPU."""
    lo, hi = int(idx[0]), int(idx[-1]) + 1
    return torch.from_numpy(np.asarray(den_mm[lo - first:hi - first]).astype(np.float32)).to(device)


def noisy_window_gpu(native_counts: torch.Tensor, dose: float,
                     generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Dose-noised counts for the whole window in one vectorised GPU Poisson draw.

    Distribution-matched to the denoiser's dose (not the identical per-frame
    realisation used at training — fine for the noisy-floor arm).
    """
    return add_poisson_noise(native_counts, dose, generator=generator)


# --------------------------------------------------------------------------- #
# metrics / masking (reuse tr_naf convention)
# --------------------------------------------------------------------------- #
def masked_scores(gt_vol: torch.Tensor, pred_vol: torch.Tensor,
                  mask: torch.Tensor, data_range: float, slice_stride: int = 4):
    from ..tr_naf.metrics import masked_psnr, masked_ssim
    return (masked_psnr(gt_vol, pred_vol, mask, data_range),
            masked_ssim(gt_vol, pred_vol, mask, data_range, slice_stride=slice_stride))


def masked_sharpness(gt_vol: torch.Tensor, pred_vol: torch.Tensor, mask: torch.Tensor) -> float:
    """Gradient-energy ratio vs GT: 1.0 = matched sharpness, <1 = blurred, >1 = over-sharp/noisy."""
    from ..tr_naf.metrics import masked_sharpness_ratio
    return masked_sharpness_ratio(gt_vol, pred_vol, mask)


def make_mask(h: int, w: int, radius_frac: float = 0.95) -> torch.Tensor:
    from ..tr_naf.metrics import make_circular_mask
    return make_circular_mask(h, w, radius=radius_frac * min(h, w) / 2.0)


# --------------------------------------------------------------------------- #
# movies (HevcGray10Streamer, same as other notebooks)
# --------------------------------------------------------------------------- #
def write_slice_movie(frames: List[torch.Tensor], out_path, vmin: float, vmax: float,
                      q: int = 90, preset_sw: str = "medium") -> Path:
    """Write a list of ``(H, W)`` slices as a gray10 HEVC ``.mov`` (windowed to [0,1]).

    ``preset_sw`` defaults to "medium" (not the encoder's own "veryslow" default)
    -- these are inspection/result movies generated after long GPU sweeps, where
    encode time matters more than squeezing out the last bit of compression.
    """
    from ..stream_hvec.stream_gray10 import EncoderParams, HevcGray10Streamer, concat_hevc_segments

    out_path = Path(out_path)
    # Force software libx265 (this Linux ffmpeg has no hevc_videotoolbox hardware encoder).
    st = HevcGray10Streamer(out_path.parent, segment_prefix=out_path.stem,
                            params=EncoderParams(force_software=True, preset_sw=preset_sw))
    span = max(vmax - vmin, 1e-6)
    with st.start_segment(q=q):
        for s in frames:
            f = ((s.float() - vmin) / span).clamp(0.0, 1.0).contiguous()
            st.append_frame(f)
    concat_hevc_segments(st.segments, str(out_path))
    return out_path


def write_projection_movie(mov_path, memmap_path, denoised, out_path,
                           frame_start: int, frame_end: int, dose: float,
                           crop: Optional[Tuple[int, int]] = None, axis_col: float = ROT_AXIS_COL,
                           noise_seed: int = 12345, chunk: int = 200, q: int = 90,
                           preset_sw: str = "medium",
                           device: Optional[torch.device] = None) -> Tuple[Path, List[str]]:
    """Render a GT | denoised... | noisy movie of raw 2D projection FRAMES over time.

    Distinct from :func:`run_windows`'s reconstruction-slice movies: this shows
    the detector frames themselves (post crop/denorm), not reconstructed volume
    slices. ``denoised`` is a dict ``{name: memmap_path}`` of one or more cached
    :func:`denoise_sequence` outputs (e.g. baseline / diffusion posterior-mean);
    each is horizontally concatenated with a synthesised ``GT`` and ``noisy`` arm,
    frame by frame, over ``[frame_start, frame_end)``.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = denoised if isinstance(denoised, dict) else {"denoised": denoised}
    src = MemmapFrameSource(memmap_path, mov_path)

    mms, firsts = {}, {}
    for name, path in variants.items():
        m = np.load(str(path) + ".meta.npz")
        fi, ni = int(m["first_index"]), int(m["num_frames"])
        crop = (int(m["crop"][0]), int(m["crop"][1])) if crop is None else crop
        mms[name] = np.memmap(path, dtype=np.float16, mode="r", shape=(ni, crop[0], crop[1]))
        firsts[name] = fi

    lo = max(int(frame_start), src.first_index, *(firsts.values()))
    hi = min(int(frame_end), src.first_index + src.num_frames,
             *(firsts[n] + mms[n].shape[0] for n in variants))
    arms = ["GT"] + list(variants) + ["noisy"]

    # Stream straight to the HEVC encoder chunk-by-chunk instead of buffering every
    # frame of the whole movie in a Python list first: at full resolution this is
    # ~1 MB/frame (4 panels x 128x512 float32), and a full wunderkerze2-range render
    # is ~50k frames -- buffering the lot is a ~50 GB RAM spike (confirmed: this
    # caused a real OOM/crash on the shared server), vs. a few hundred MB for one
    # `chunk`-sized batch at a time.
    from ..stream_hvec.stream_gray10 import EncoderParams, HevcGray10Streamer, concat_hevc_segments

    out_path = Path(out_path)
    st = HevcGray10Streamer(out_path.parent, segment_prefix=out_path.stem,
                            params=EncoderParams(force_software=True, preset_sw=preset_sw))
    vmin = vmax = span = None
    with st.start_segment(q=q):
        for c0 in range(lo, hi, chunk):
            idx = np.arange(c0, min(c0 + chunk, hi))
            gt = native_window_gpu(src, idx, crop, axis_col, device)
            if vmin is None:
                lo_p, hi_p = np.percentile(gt.detach().cpu().numpy(), [1, 99])
                vmin, vmax = float(lo_p), float(hi_p)
                span = max(vmax - vmin, 1e-6)
            g = torch.Generator(device=device).manual_seed(int(noise_seed) + int(c0))
            noisy = noisy_window_gpu(gt, dose, generator=g)
            dens = [denoised_window_gpu(mms[name], firsts[name], idx, device) for name in variants]
            combo = torch.cat([gt] + dens + [noisy], dim=-1)
            normed = ((combo.float() - vmin) / span).clamp(0.0, 1.0)
            for i in range(normed.shape[0]):
                st.append_frame(normed[i].detach().cpu().contiguous())
    concat_hevc_segments(st.segments, str(out_path))
    return out_path, arms


def run_windows(mov_path, memmap_path, denoised,
                window_starts: Optional[Sequence[int]] = None, stride: Optional[int] = None,
                window_deg: float = 180.0, det_bin: int = 2, method: str = "fbp",
                gd_kwargs: Optional[dict] = None, movie_rows: Optional[Sequence[int]] = None,
                axis_col: float = ROT_AXIS_COL, I0_pct: float = 99.5, gpu_load: bool = True,
                frame_start: Optional[int] = None, frame_end: Optional[int] = None,
                deg_per_frame: float = DEG_PER_FRAME,
                dark_map: Optional[torch.Tensor] = None, flat_map: Optional[torch.Tensor] = None,
                destripe_k: Optional[int] = None,
                device: Optional[torch.device] = None, log_every: int = 50) -> dict:
    """Reconstruct sliding windows for one or more denoised variants and score vs GT.

    ``denoised`` is either a memmap path (single ``"denoised"`` arm) or a dict
    ``{name: memmap_path}`` (e.g. ``{"baseline":..., "diffusion_ss":..., "cascade":...}``).
    Arms = ``GT`` + variants + ``noisy``; PSNR/SSIM computed for every variant and
    ``noisy`` vs the GT reconstruction. Windows are limited to the intersection of
    all variants' cached frame ranges (and ``frame_start/frame_end``). Returns
    per-window metrics + a few slice rows per arm for movies (no full volumes).

    ``dark_map``/``flat_map`` (optional, ``(H, W)`` pre-cropped/aligned counts
    maps -- see :func:`load_calibration_average`): if given, attenuation uses
    the real per-pixel flat/dark correction (:func:`counts_to_attenuation_flatdark`)
    instead of the single-scalar ``I0`` approximation (:func:`counts_to_attenuation`).

    ``destripe_k`` (optional): if given, applies :func:`destripe_sinogram` with
    this kernel width to every arm's attenuation sinogram before FBP -- removes
    ring artifacts (angle-invariant per-column bias), which the real flat/dark
    correction above can otherwise introduce or sharpen (a per-pixel flat/dark
    mismatch against the sample's actual count-rate regime is angle-invariant,
    exactly the ring signature).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = denoised if isinstance(denoised, dict) else {"denoised": denoised}

    mms, firsts = {}, {}
    dose = seed = None
    crop = None
    for name, path in variants.items():
        m = np.load(str(path) + ".meta.npz")
        fi, ni = int(m["first_index"]), int(m["num_frames"])
        crop = (int(m["crop"][0]), int(m["crop"][1])) if crop is None else crop
        dose = float(m["dose"]) if dose is None else dose
        seed = int(m["noise_seed"]) if seed is None else seed
        mms[name] = np.memmap(path, dtype=np.float16, mode="r", shape=(ni, crop[0], crop[1]))
        firsts[name] = (fi, ni)
    src = MemmapFrameSource(memmap_path, mov_path)
    arms = ["GT"] + list(variants) + ["noisy"]

    win = window_length_frames(window_deg, deg_per_frame)
    stride = int(stride or win)
    gen_lo = max(fi for fi, _ in firsts.values())
    gen_hi = min(fi + ni for fi, ni in firsts.values())
    if frame_start is not None:
        gen_lo = max(gen_lo, int(frame_start))
    if frame_end is not None:
        gen_hi = min(gen_hi, int(frame_end))
    all_windows = sliding_windows(gen_lo, gen_hi, stride, window_deg, deg_per_frame)
    if window_starts is not None:
        want = set(int(s) for s in window_starts)
        all_windows = [(s, idx) for (s, idx) in all_windows if s in want]

    use_flatdark = dark_map is not None and flat_map is not None
    if use_flatdark:
        I0 = None
        dark_t, flat_t = dark_map.to(device), flat_map.to(device)
    else:
        I0 = estimate_I0(src, np.arange(gen_lo, gen_hi, max(1, (gen_hi - gen_lo) // 64)), crop, axis_col, I0_pct)
    nslices, hplane = crop[0] // det_bin, crop[1] // det_bin
    if movie_rows is None:
        movie_rows = [nslices // 4, nslices // 2, (3 * nslices) // 4]
    mask = make_mask(hplane, hplane).to(device)

    score_arms = list(variants) + ["noisy"]
    metrics = {a: {"psnr": [], "ssim": [], "sharpness": []} for a in score_arms}
    movie = {a: [] for a in arms}
    starts = []
    for wi, (s, idx) in enumerate(all_windows):
        gt = native_window_gpu(src, idx, crop, axis_col, device)
        g = torch.Generator(device=device).manual_seed(int(seed) + int(s))
        counts = {"GT": gt, "noisy": noisy_window_gpu(gt, dose, generator=g)}
        for name in variants:
            counts[name] = denoised_window_gpu(mms[name], firsts[name][0], idx, device)
        ang = projection_angles(idx, deg_per_frame=deg_per_frame)
        if use_flatdark:
            atten = {a: counts_to_attenuation_flatdark(counts[a].to(device), dark_t, flat_t) for a in arms}
        else:
            atten = {a: counts_to_attenuation(counts[a].to(device), I0) for a in arms}
        if destripe_k:
            atten = {a: destripe_sinogram(atten[a], destripe_k) for a in arms}
        vols = {a: reconstruct(atten[a], ang, det_bin=det_bin, method=method,
                               gd_kwargs=gd_kwargs, device=device)
                for a in arms}
        gtv = vols["GT"]
        dr = float(gtv[..., mask].max() - gtv[..., mask].min())
        for a in score_arms:
            ps, ss = masked_scores(gtv, vols[a], mask, dr)
            metrics[a]["psnr"].append(ps)
            metrics[a]["ssim"].append(ss)
            metrics[a]["sharpness"].append(masked_sharpness(gtv, vols[a], mask))
        for a in arms:
            movie[a].append(vols[a][list(movie_rows)].detach().cpu())
        starts.append(s)
        if log_every and wi % log_every == 0:
            lead = list(variants)[0]
            print(f"  [{method}] window {wi + 1}/{len(all_windows)} start {s}  "
                  f"{lead} {metrics[lead]['psnr'][-1]:.2f}dB", flush=True)

    return dict(method=method, arms=arms, crop=crop, det_bin=det_bin, I0=I0,
                used_flat_dark=use_flatdark, destripe_k=destripe_k,
                window_deg=window_deg, stride=stride, window_starts=starts,
                movie_rows=list(movie_rows),
                metrics={a: {k: np.array(v) for k, v in d.items()} for a, d in metrics.items()},
                movie=movie)


def most_dynamic_window(gt_slice_movie: List[torch.Tensor]) -> Tuple[int, np.ndarray]:
    """Index of the window with the largest change vs the previous window.

    ``gt_slice_movie`` is the per-window GT slice stack (list over windows of a
    ``(rows, H, W)`` tensor). Returns ``(argmax_index, per_window_change)`` where
    change[i] = mean |frame_i - frame_{i-1}| — the same "how much did the movie
    move" signal as the fixed-angle time-lapse in the calibration notebook.
    """
    stack = torch.stack([s.float().mean(0) for s in gt_slice_movie])  # (W, H, Wd)
    diff = torch.zeros(stack.shape[0])
    diff[1:] = (stack[1:] - stack[:-1]).abs().mean(dim=(1, 2))
    return int(diff.argmax().item()), diff.numpy()
