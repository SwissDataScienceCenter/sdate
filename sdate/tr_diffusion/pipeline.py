"""Conditional DDIM inference for the time-resolved frame denoiser (phase 2).

Adapted from the ideas in ``isodiffusion/schedulers/{scheduling_ddim,
pipeline_ddim_2d}.py`` but specialised to flat 2D frames and the N2V contract:
the corrupted central channel is **resampled with a fresh blind-spot mask on
every denoising step**, so no single mask biases the result.

Sampling (per DDIM step ``t``):

1. ``corrupted = blind_spot_corrupt(central_measured)``  (fresh mask each step)
2. ``eps = unet([x_t, corrupted, context], t, class_label=1)``
3. ``x_{t-1} = scheduler.step(eps, t, x_t)``

The measured central frame is provided as conditioning throughout; the network
only ever fills the blind spots (with denoised estimates) and diffuses toward a
coherent clean frame.  This module is wired but only lightly exercised — the
project scope so far is training; treat it as the drop-in inference path.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler

from .n2v import blind_spot_corrupt
from .nb_head import gamma_sample, posterior_mean, split_mu_var
from .noise import binomial_complementary_split, binomial_thin


def build_ddim_scheduler(num_train_timesteps: int = 1000, **kw) -> DDIMScheduler:
    return DDIMScheduler(num_train_timesteps=num_train_timesteps, **kw)


@torch.no_grad()
def denoise_frames(
    model,
    central: torch.Tensor,
    context: torch.Tensor,
    scheduler: Optional[DDIMScheduler] = None,
    num_inference_steps: int = 50,
    start_step: int = 0,
    eta: float = 0.0,
    ratio: float = 0.02,
    window: int = 5,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Denoise a batch of central frames given their context.

    Parameters
    ----------
    central, context:
        ``(B, 1, H, W)`` and ``(B, 4k, H, W)`` normalised tensors on the model's
        device (as produced by :class:`~sdate.tr_diffusion.data.TimeResolvedFrameDataset`).
    start_step:
        Truncation index into the schedule.  ``0`` starts from pure noise; a
        larger value starts from the measured central frame noised to that
        timestep (cheaper, stays closer to the measurement).

    Returns the denoised ``x_0`` estimate, ``(B, 1, H, W)``.
    """
    device = central.device
    dtype = next(model.parameters()).dtype
    central = central.to(device=device, dtype=dtype)
    context = context.to(device=device, dtype=dtype)
    b = central.shape[0]

    scheduler = scheduler or build_ddim_scheduler()
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps[start_step:]

    noise = torch.randn(central.shape, generator=generator, device=device, dtype=dtype)
    if start_step == 0:
        x_t = noise
    else:
        t0 = timesteps[0].expand(b)
        x_t = scheduler.add_noise(central, noise, t0)

    class_labels = torch.ones(b, device=device, dtype=torch.long)
    for t in timesteps:
        corrupted, _ = blind_spot_corrupt(central, ratio=ratio, window=window, generator=generator)
        model_input = torch.cat([x_t, corrupted, context], dim=1)
        eps = model(model_input, timestep=t.expand(b), class_labels=class_labels, return_dict=False)[0]
        x_t = scheduler.step(eps, t, x_t, eta=eta, generator=generator).prev_sample
    return x_t


@torch.no_grad()
def denoise_frames_ensemble(
    model,
    central: torch.Tensor,
    context: torch.Tensor,
    num_samples: int = 8,
    scheduler: Optional[DDIMScheduler] = None,
    num_inference_steps: int = 50,
    start_step: int = 40,
    eta: float = 0.0,
    ratio: float = 0.02,
    window: int = 5,
    generator: Optional[torch.Generator] = None,
    chunk_size: int = 16,
):
    """Draw ``num_samples`` stochastic denoised samples per frame and average them.

    Each sample sees independent initial noise and an independent per-step
    blind-spot mask, so the samples are diverse (even at ``eta=0``, because the
    truncated start noises each replicate differently). Their mean is a Monte
    Carlo estimate of the posterior mean ``E[x0 | y]`` — the MMSE denoiser — which
    generally beats any single sample in PSNR.

    ``chunk_size`` bounds how many (frame, sample) replicates are denoised at
    once, so peak memory stays fixed no matter how large ``B * num_samples`` is.

    Returns ``(mean, samples)`` with shapes ``(B, 1, H, W)`` and
    ``(B, num_samples, 1, H, W)``.
    """
    b = central.shape[0]
    rep_central = central.repeat_interleave(num_samples, dim=0)
    rep_context = context.repeat_interleave(num_samples, dim=0)
    total = rep_central.shape[0]
    cs = max(1, int(chunk_size)) if chunk_size else total

    outs = []
    for i in range(0, total, cs):
        outs.append(denoise_frames(
            model, rep_central[i : i + cs], rep_context[i : i + cs], scheduler=scheduler,
            num_inference_steps=num_inference_steps, start_step=start_step, eta=eta,
            ratio=ratio, window=window, generator=generator,
        ))
    out = torch.cat(outs, dim=0)
    samples = out.view(b, num_samples, *out.shape[1:])
    return samples.mean(dim=1), samples


@torch.no_grad()
def pred_x0_ensemble(
    model,
    central: torch.Tensor,
    context: torch.Tensor,
    timestep: int = 300,
    num_samples: int = 64,
    num_train_timesteps: int = 1000,
    ratio: float = 0.02,
    window: int = 5,
    chunk_size: int = 32,
    generator: Optional[torch.Generator] = None,
):
    """Posterior-mean denoiser: average ``num_samples`` one-shot ``pred_x0`` estimates.

    Each estimate noises the measurement to a fixed ``timestep``, corrupts the
    central frame with a fresh blind-spot mask, does ONE forward pass, and reads
    off ``x0 = (x_t - sqrt(1-a_t) eps) / sqrt(a_t)``. Every draw is
    ``E[x0 | x_t, conditioning]`` for a different noise/mask realisation, so their
    mean approximates ``E[x0 | y]`` — the MMSE estimate — far more directly than
    the ancestral DDIM loop (which, with the corrupted-central conditioning,
    tends to reconverge to the noisy measurement at low timesteps).

    ``chunk_size`` bounds peak memory so ``num_samples`` can be made large.
    Returns ``(mean, samples)`` shaped ``(B, 1, H, W)`` and ``(B, num_samples, 1, H, W)``.
    """
    from diffusers import DDPMScheduler

    device = central.device
    dtype = next(model.parameters()).dtype
    central = central.to(device=device, dtype=dtype)
    context = context.to(device=device, dtype=dtype)
    b = central.shape[0]

    sched = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    acp = sched.alphas_cumprod.to(device)[int(timestep)]
    sa, sb = acp ** 0.5, (1.0 - acp) ** 0.5

    rep_central = central.repeat_interleave(num_samples, dim=0)
    rep_context = context.repeat_interleave(num_samples, dim=0)
    total = rep_central.shape[0]
    cs = max(1, int(chunk_size)) if chunk_size else total

    outs = []
    for i in range(0, total, cs):
        cc, xx = rep_central[i : i + cs], rep_context[i : i + cs]
        n = cc.shape[0]
        t = torch.full((n,), int(timestep), device=device, dtype=torch.long)
        noise = torch.randn(cc.shape, generator=generator, device=device, dtype=dtype)
        x_t = sched.add_noise(cc, noise, t)
        corrupted, _ = blind_spot_corrupt(cc, ratio=ratio, window=window, generator=generator)
        eps = model(
            torch.cat([x_t, corrupted, xx], dim=1), timestep=t,
            class_labels=torch.ones(n, device=device, dtype=torch.long), return_dict=False,
        )[0]
        outs.append((x_t - sb * eps) / sa)
    samples = torch.cat(outs, dim=0).view(b, num_samples, *central.shape[1:])
    return samples.mean(dim=1), samples


@torch.no_grad()
def partial_diffusion(
    model,
    central: torch.Tensor,
    context: torch.Tensor,
    t_start: int = 500,
    t_end: int = 400,
    num_steps: int = 10,
    num_train_timesteps: int = 1000,
    eta: float = 0.0,
    ratio: float = 0.02,
    window: int = 5,
    num_samples: int = 1,
    chunk_size: int = 32,
    generator: Optional[torch.Generator] = None,
):
    """Short DDIM refinement over the interval ``[t_end, t_start]``.

    Noise the measurement to ``t_start``, run ``num_steps`` DDIM updates down to
    ``t_end`` (resampling the blind-spot mask each step), then read off ``pred_x0``
    at ``t_end``. This sits between single-shot ``pred_x0`` (num_steps -> the
    interval collapses) and full ancestral sampling to 0 (which reconverges to the
    noisy input). Manual DDIM update over an arbitrary integer sub-schedule so any
    interval / step count works. ``num_samples`` draws are averaged (posterior mean).

    Returns ``(mean, samples)`` shaped ``(B, 1, H, W)`` and ``(B, num_samples, 1, H, W)``.
    """
    from diffusers import DDPMScheduler

    device = central.device
    dtype = next(model.parameters()).dtype
    central = central.to(device=device, dtype=dtype)
    context = context.to(device=device, dtype=dtype)
    b = central.shape[0]
    num_steps = max(1, int(num_steps))

    sched = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    acp = sched.alphas_cumprod.to(device)
    ts = torch.linspace(float(t_start), float(t_end), num_steps + 1).round().long()
    ts = ts.clamp(0, num_train_timesteps - 1)

    rep_central = central.repeat_interleave(num_samples, dim=0)
    rep_context = context.repeat_interleave(num_samples, dim=0)
    total = rep_central.shape[0]
    cs = max(1, int(chunk_size)) if chunk_size else total

    def corrupt_and_eps(cc, xx, x, t):
        n = cc.shape[0]
        corrupted, _ = blind_spot_corrupt(cc, ratio=ratio, window=window, generator=generator)
        return model(
            torch.cat([x, corrupted, xx], dim=1),
            timestep=torch.full((n,), int(t), device=device, dtype=torch.long),
            class_labels=torch.ones(n, device=device, dtype=torch.long), return_dict=False,
        )[0]

    outs = []
    for i0 in range(0, total, cs):
        cc, xx = rep_central[i0 : i0 + cs], rep_context[i0 : i0 + cs]
        n = cc.shape[0]
        noise = torch.randn(cc.shape, generator=generator, device=device, dtype=dtype)
        x = sched.add_noise(cc, noise, torch.full((n,), int(ts[0]), device=device))
        for j in range(num_steps):
            t_cur, t_nxt = int(ts[j]), int(ts[j + 1])
            a_cur, a_nxt = acp[t_cur], acp[t_nxt]
            eps = corrupt_and_eps(cc, xx, x, t_cur)
            x0 = (x - (1 - a_cur).sqrt() * eps) / a_cur.sqrt()
            sigma = torch.zeros((), device=device)
            if eta > 0 and t_cur > t_nxt:
                sigma = eta * ((1 - a_nxt) / (1 - a_cur)).clamp_min(0).sqrt() \
                    * (1 - a_cur / a_nxt).clamp_min(0).sqrt()
            x = a_nxt.sqrt() * x0 + (1 - a_nxt - sigma ** 2).clamp_min(0).sqrt() * eps
            if eta > 0 and float(sigma) > 0:
                x = x + sigma * torch.randn(x.shape, generator=generator, device=device, dtype=dtype)
        # final x0 estimate read at t_end
        t_end_i = int(ts[-1])
        a_end = acp[t_end_i]
        eps = corrupt_and_eps(cc, xx, x, t_end_i)
        outs.append((x - (1 - a_end).sqrt() * eps) / a_end.sqrt())
    samples = torch.cat(outs, dim=0).view(b, num_samples, *central.shape[1:])
    return samples.mean(dim=1), samples


def _n2n_input_and_label(central: torch.Tensor, q: float, p_bins: int,
                         norm_min: Optional[float] = None, norm_max: Optional[float] = None,
                         generator: Optional[torch.Generator] = None):
    """Inference-time N2N conditioning: thin ``central`` by fraction ``q`` (>=1 -> no
    thinning, the full measurement) and the matching discretised p-bin class label.

    ``central`` arrives normalised to ``[-1, 1]`` (the dataset/reconstruct.py
    convention) but :func:`~sdate.tr_diffusion.noise.binomial_thin` needs
    non-negative raw detector counts, so for ``q < 1`` this denormalises with
    ``norm_min``/``norm_max`` (the checkpoint's own training range) before
    thinning and renormalises after. At ``q >= 1`` thinning is a no-op (returns
    ``central`` unchanged) so no norm range is needed there.
    """
    b = central.shape[0]
    p_bin = torch.full((b,), int(round(min(q, 1.0) * p_bins)), device=central.device, dtype=torch.long)
    if q >= 1.0:
        return central, p_bin
    if norm_min is None or norm_max is None:
        raise ValueError("_n2n_input_and_label needs norm_min/norm_max to denormalise `central` "
                          "to raw counts before binomial thinning for q < 1.0")
    span = float(norm_max - norm_min)
    counts = (central.clamp(-1, 1) + 1) * 0.5 * span + norm_min
    thinned_counts = binomial_thin(counts, q, generator=generator)
    thinned = (thinned_counts - norm_min) / span * 2 - 1
    return thinned, p_bin


@torch.no_grad()
def denoise_frames_n2n_baseline(
    model, central: torch.Tensor, context: torch.Tensor,
    q: float = 1.0, p_bins: int = 100,
    norm_min: Optional[float] = None, norm_max: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Single forward pass of the N2N baseline regressor (predicts the split target).

    ``central`` is the actual measured (dose-thinned) frame; ``q`` is the assumed
    input fraction of that measurement to feed the model (``q=1`` = no further
    thinning -> the full, least-noisy available input — the expected best choice
    per the "large fraction wins" hypothesis, but ``q`` should be ablated).
    ``norm_min``/``norm_max`` are required whenever ``q < 1`` (see
    :func:`_n2n_input_and_label`).
    """
    device = central.device
    dtype = next(model.parameters()).dtype
    central = central.to(device=device, dtype=dtype)
    context = context.to(device=device, dtype=dtype)
    inp, p_bin = _n2n_input_and_label(central, q, p_bins, norm_min, norm_max, generator=generator)
    model_input = torch.cat([inp, context], dim=1)
    timesteps = torch.zeros(central.shape[0], device=device, dtype=torch.long)
    return model(model_input, timestep=timesteps, class_labels=p_bin, return_dict=False)[0]


@torch.no_grad()
def pred_x0_n2n_ensemble(
    model, central: torch.Tensor, context: torch.Tensor,
    q: float = 1.0, timestep: int = 500, num_samples: int = 1, p_bins: int = 100,
    num_train_timesteps: int = 1000, chunk_size: int = 32,
    prediction_type: str = "epsilon",
    norm_min: Optional[float] = None, norm_max: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
):
    """N2N analogue of :func:`pred_x0_ensemble`: single-shot / posterior-mean
    ``pred_x0`` from the diffusion N2N model, conditioned on the input fraction
    ``q`` (thinned from the measured frame) instead of a blind-spot mask + label=1.

    Each of the ``num_samples`` draws uses fresh diffusion noise (and, if ``q<1``,
    a fresh binomial thinning draw); their mean is the posterior-mean estimate.
    Returns ``(mean, samples)`` shaped ``(B, 1, H, W)`` and ``(B, num_samples, 1, H, W)``.
    """
    from diffusers import DDPMScheduler

    device = central.device
    dtype = next(model.parameters()).dtype
    central = central.to(device=device, dtype=dtype)
    context = context.to(device=device, dtype=dtype)
    b = central.shape[0]

    sched = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    acp = sched.alphas_cumprod.to(device)[int(timestep)]
    sa, sb = acp ** 0.5, (1.0 - acp) ** 0.5

    rep_central = central.repeat_interleave(num_samples, dim=0)
    rep_context = context.repeat_interleave(num_samples, dim=0)
    total = rep_central.shape[0]
    cs = max(1, int(chunk_size)) if chunk_size else total

    outs = []
    for i in range(0, total, cs):
        cc, xx = rep_central[i : i + cs], rep_context[i : i + cs]
        n = cc.shape[0]
        t = torch.full((n,), int(timestep), device=device, dtype=torch.long)
        noise = torch.randn(cc.shape, generator=generator, device=device, dtype=dtype)
        x_t = sched.add_noise(cc, noise, t)
        inp, p_bin = _n2n_input_and_label(cc, q, p_bins, norm_min, norm_max, generator=generator)
        raw = model(torch.cat([x_t, inp, xx], dim=1), timestep=t, class_labels=p_bin, return_dict=False)[0]
        x0 = raw if prediction_type == "sample" else (x_t - sb * raw) / sa
        outs.append(x0)
    samples = torch.cat(outs, dim=0).view(b, num_samples, *central.shape[1:])
    return samples.mean(dim=1), samples


@torch.no_grad()
def pred_x0_n2n_swap_ensemble(
    model, central: torch.Tensor, context: torch.Tensor,
    norm_min: float, norm_max: float,
    timestep: int = 500, p_split: float = 0.5, p_bins: int = 100,
    prediction_type: str = "sample",
    num_samples: int = 1, chunk_size: int = 32,
    generator: Optional[torch.Generator] = None,
):
    """Swap-averaged single-shot N2N prediction: split the REAL measurement into
    two complementary halves (:func:`~sdate.tr_diffusion.noise.binomial_complementary_split`,
    no fresh Poisson redraw), predict ``x0`` conditioned on EACH half in turn, and
    average the two estimates.

    This is the inference-time counterpart of the swap-consistency training
    objective (:class:`sdate.tr_diffusion.losses.DiffusionN2NLoss` with
    ``consistency_weight > 0``): training asked the two directions to agree,
    this exploits that by averaging them, which should reduce variance versus a
    single-direction prediction if the consistency training actually worked.

    ``central``/``context`` arrive normalised to ``[-1, 1]`` (the dataset/
    reconstruct.py convention); binomial splitting needs non-negative raw
    detector counts, so ``central`` is denormalised (``norm_min``/``norm_max``,
    the SAME range the checkpoint was trained with) before splitting, and each
    half is renormalised before being fed to the model.

    ``num_samples > 1`` adds POSTERIOR-MEAN averaging on top of the swap
    averaging: each of the ``num_samples`` draws gets a fresh binomial split
    realisation and fresh diffusion noise for both branches, and all
    ``2 * num_samples`` resulting estimates (both halves, all draws) are
    averaged together -- combining the two independent variance-reduction
    mechanisms (swap + posterior sampling).

    Returns ``(mean, samples)`` shaped ``(B, 1, H, W)`` and
    ``(B, 2*num_samples, 1, H, W)``, normalised.
    """
    from diffusers import DDPMScheduler

    device = central.device
    dtype = next(model.parameters()).dtype
    central = central.to(device=device, dtype=dtype)
    context = context.to(device=device, dtype=dtype)
    b = central.shape[0]

    span = float(norm_max - norm_min)
    def denorm(x): return (x.clamp(-1, 1) + 1) * 0.5 * span + norm_min
    def norm(x): return (x - norm_min) / span * 2 - 1

    sched = DDPMScheduler(num_train_timesteps=1000)
    acp = sched.alphas_cumprod.to(device)[int(timestep)]
    sa, sb = acp ** 0.5, (1.0 - acp) ** 0.5

    rep_central = central.repeat_interleave(num_samples, dim=0)
    rep_context = context.repeat_interleave(num_samples, dim=0)
    total = rep_central.shape[0]
    cs = max(1, int(chunk_size)) if chunk_size else total

    p_bin_a_val = int(round(p_split * p_bins))
    p_bin_b_val = int(round((1.0 - p_split) * p_bins))

    def _predict(inp, p_bin_val, xx, n, t):
        p_bin = torch.full((n,), p_bin_val, device=device, dtype=torch.long)
        noise = torch.randn(inp.shape, generator=generator, device=device, dtype=dtype)
        x_t = sched.add_noise(inp, noise, t)
        raw = model(torch.cat([x_t, inp, xx], dim=1), timestep=t, class_labels=p_bin,
                    return_dict=False)[0]
        return raw if prediction_type == "sample" else (x_t - sb * raw) / sa

    outs_a, outs_b = [], []
    for i in range(0, total, cs):
        cc, xx = rep_central[i : i + cs], rep_context[i : i + cs]
        n = cc.shape[0]
        t = torch.full((n,), int(timestep), device=device, dtype=torch.long)
        cc_counts = denorm(cc)
        half_a_counts, half_b_counts = binomial_complementary_split(cc_counts, p_split, generator=generator)
        half_a, half_b = norm(half_a_counts), norm(half_b_counts)
        outs_a.append(_predict(half_a, p_bin_a_val, xx, n, t))
        outs_b.append(_predict(half_b, p_bin_b_val, xx, n, t))
    x0_a = torch.cat(outs_a, dim=0).view(b, num_samples, *central.shape[1:])
    x0_b = torch.cat(outs_b, dim=0).view(b, num_samples, *central.shape[1:])
    samples = torch.cat([x0_a, x0_b], dim=1)
    return samples.mean(dim=1), samples


@torch.no_grad()
def partial_diffusion_n2n(
    model,
    central: torch.Tensor,
    context: torch.Tensor,
    q: float = 1.0,
    t_start: int = 500,
    t_end: int = 0,
    num_steps: int = 50,
    p_bins: int = 100,
    num_train_timesteps: int = 1000,
    eta: float = 0.0,
    num_samples: int = 1,
    chunk_size: int = 32,
    prediction_type: str = "epsilon",
    norm_min: Optional[float] = None, norm_max: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
):
    """N2N analogue of :func:`partial_diffusion`: DDIM refinement over ``[t_end,
    t_start]`` conditioned on the (optionally further-thinned) measured frame + its
    p-bin label, instead of a blind-spot mask + present=1. Defaults to the full
    ancestral range ``t_start=500 -> t_end=0``: "initialise with the noisy
    measurement" (noise it to ``t_start``) and use the N2N conditioning at every
    step down to 0 — exactly the trajectory N2V found counterproductive there (the
    blind-spot conditioning reconverges to the noisy input); worth re-testing here
    since the N2N conditioning mechanism is different (nothing is deliberately
    hidden from the model, so it may not reconverge the same way).

    Same manual-DDIM machinery as :func:`partial_diffusion` (arbitrary integer
    sub-schedule, ``num_samples`` posterior-mean averaging); set ``t_end=t_start``
    (``num_steps=1``) to recover single-shot behaviour equivalent to
    :func:`pred_x0_n2n_ensemble`.

    Returns ``(mean, samples)`` shaped ``(B, 1, H, W)`` and ``(B, num_samples, 1, H, W)``.
    """
    from diffusers import DDPMScheduler

    device = central.device
    dtype = next(model.parameters()).dtype
    central = central.to(device=device, dtype=dtype)
    context = context.to(device=device, dtype=dtype)
    b = central.shape[0]
    num_steps = max(1, int(num_steps))

    sched = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    acp = sched.alphas_cumprod.to(device)
    ts = torch.linspace(float(t_start), float(t_end), num_steps + 1).round().long()
    ts = ts.clamp(0, num_train_timesteps - 1)

    rep_central = central.repeat_interleave(num_samples, dim=0)
    rep_context = context.repeat_interleave(num_samples, dim=0)
    total = rep_central.shape[0]
    cs = max(1, int(chunk_size)) if chunk_size else total

    def cond_predict(cc, xx, x, t):
        n = cc.shape[0]
        inp, p_bin = _n2n_input_and_label(cc, q, p_bins, norm_min, norm_max, generator=generator)
        raw = model(
            torch.cat([x, inp, xx], dim=1),
            timestep=torch.full((n,), int(t), device=device, dtype=torch.long),
            class_labels=p_bin, return_dict=False,
        )[0]
        a_t = acp[int(t)]
        if prediction_type == "sample":
            x0 = raw
            eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt().clamp_min(1e-6)
        else:
            eps = raw
            x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt().clamp_min(1e-6)
        return x0, eps

    outs = []
    for i0 in range(0, total, cs):
        cc, xx = rep_central[i0 : i0 + cs], rep_context[i0 : i0 + cs]
        n = cc.shape[0]
        noise = torch.randn(cc.shape, generator=generator, device=device, dtype=dtype)
        x = sched.add_noise(cc, noise, torch.full((n,), int(ts[0]), device=device))
        for j in range(num_steps):
            t_cur, t_nxt = int(ts[j]), int(ts[j + 1])
            a_cur, a_nxt = acp[t_cur], acp[t_nxt]
            x0, eps = cond_predict(cc, xx, x, t_cur)
            sigma = torch.zeros((), device=device)
            if eta > 0 and t_cur > t_nxt:
                sigma = eta * ((1 - a_nxt) / (1 - a_cur)).clamp_min(0).sqrt() \
                    * (1 - a_cur / a_nxt).clamp_min(0).sqrt()
            x = a_nxt.sqrt() * x0 + (1 - a_nxt - sigma ** 2).clamp_min(0).sqrt() * eps
            if eta > 0 and float(sigma) > 0:
                x = x + sigma * torch.randn(x.shape, generator=generator, device=device, dtype=dtype)
        t_end_i = int(ts[-1])
        x0, _ = cond_predict(cc, xx, x, t_end_i)
        outs.append(x0)
    samples = torch.cat(outs, dim=0).view(b, num_samples, *central.shape[1:])
    return samples.mean(dim=1), samples


@torch.no_grad()
def denoise_frames_baseline(model, central: torch.Tensor, context: torch.Tensor,
                            present: bool = True, cond_channels: Optional[torch.Tensor] = None,
                            aux_channels: Optional[torch.Tensor] = None,
                            poisson_head: bool = False, poisson_mean_only: bool = False,
                            norm_min: Optional[float] = None, norm_max: Optional[float] = None,
                            poisson_dose: float = 1.0, poisson_posterior: bool = True,
                            return_uncertainty: bool = False):
    """Single forward pass of the baseline regressor (predicts ``x_0``).

    ``present=False`` runs the "context-only" pathway (corrupted-central channel
    zeroed, ``class_label=0``) — the regime a ``conditioning_probability=0``
    ablation checkpoint was EXCLUSIVELY trained on. Calling such a checkpoint
    with the default ``present=True`` is out-of-distribution (it never saw
    ``class_label=1`` in training), so pass ``present=False`` for those models.

    ``cond_channels`` (optional) — the ``(B, 3, H, W)`` angle/time conditioning
    planes (see :func:`sdate.tr_diffusion.geometry.angle_time_cond_array`), for
    checkpoints trained with ``cond_angle_time=True``. Must be given for those
    checkpoints (their ``in_channels`` includes the extra planes) and omitted
    for every other checkpoint.

    ``aux_channels`` (optional) — the ``(B, T, H, W)`` cached extra-channel
    tensor (``T>=1``; e.g. ``data.py``'s ``aux_channel_memmap``/
    ``instance["aux_channel"]``), for checkpoints trained with
    ``--aux_channel_memmap``. Must be given for those checkpoints and omitted
    otherwise; appended AFTER ``cond_channels`` in the input stack, matching
    training's fixed channel order (see
    :class:`sdate.tr_diffusion.losses.BaselineN2VLoss`).

    ``poisson_head`` — for checkpoints trained with the two-head Gamma-Poisson
    NB-NLL loss (:class:`sdate.tr_diffusion.losses.BaselineN2VLoss` with
    ``poisson_head=True`` — the STANDARD baseline architecture, see
    :mod:`sdate.tr_diffusion.nb_head`): the model output is a ``(mu, var)``
    belief rather than a point estimate. Two inference modes off this SAME
    checkpoint (no separate single-head model needed):

    * ``poisson_posterior=True`` (default) — the value returned is the exact
      Gamma-Poisson POSTERIOR MEAN combining that belief with the real
      observed count at every pixel (``central``, denormalised via
      ``norm_min``/``norm_max`` — required in this mode), not the raw ``mu``.
      Sharper (recovers detail the blind-spot architecture otherwise blurs
      away) at a small PSNR/SSIM cost. ``poisson_dose`` MUST match the actual
      thinning fraction of ``central`` (see
      :func:`sdate.tr_diffusion.nb_head.posterior_mean`) — 1.0 for a
      native/non-thinned measurement, or the synthetic ``dose`` used to
      generate it (e.g. via ``extra_noise_dose`` in
      :mod:`sdate.tr_diffusion.reconstruct`). ``return_uncertainty=True``
      additionally returns the posterior variance (raw count units, NOT
      renormalised).
    * ``poisson_posterior=False`` — returns ``mu`` alone (context-only, no
      observation combination): the single-head-equivalent point estimate.
      Empirically identical to the superseded single-channel
      ``loss_type="poisson"`` model's output (pixelwise correlation 0.999) --
      use this when the old single-head behaviour (or its slightly better
      whole-frame PSNR/SSIM) is what's wanted, still needs ``norm_min``/``norm_max``.

    ``poisson_mean_only`` — for checkpoints trained with the single-channel
    ``loss_type="poisson"`` ablation (same point-estimate head as the default
    Huber/MAE/MSE path, but the correct Poisson likelihood instead of a
    homoscedastic one -- see :class:`sdate.tr_diffusion.losses.BaselineN2VLoss`):
    the raw output is a pre-softplus value, not the denoised pixel directly, so
    it's passed through ``softplus`` (recovering the Poisson mean in raw counts)
    and renormalised. No posterior combination (no variance head to combine
    with). Mutually exclusive with ``poisson_head``.
    """
    device = central.device
    dtype = next(model.parameters()).dtype
    central = central.to(device=device, dtype=dtype)
    context = context.to(device=device, dtype=dtype)
    b = central.shape[0]
    corrupted = blind_spot_corrupt(central)[0] if present else torch.zeros_like(central)
    parts = [corrupted, context]
    if cond_channels is not None:
        parts.append(cond_channels.to(device=device, dtype=dtype))
    if aux_channels is not None:
        parts.append(aux_channels.to(device=device, dtype=dtype))
    model_input = torch.cat(parts, dim=1)
    timesteps = torch.zeros(b, device=device, dtype=torch.long)
    class_labels = torch.full((b,), int(present), device=device, dtype=torch.long)
    raw = model(model_input, timestep=timesteps, class_labels=class_labels, return_dict=False)[0]

    if poisson_mean_only:
        if norm_min is None or norm_max is None:
            raise ValueError("poisson_mean_only=True needs norm_min/norm_max to recover raw counts "
                             "(the checkpoint's own training normalisation range).")
        span = float(norm_max - norm_min)
        mu = F.softplus(raw) + 1e-6
        return (mu - norm_min) / span * 2 - 1

    if not poisson_head:
        return raw

    if norm_min is None or norm_max is None:
        raise ValueError("poisson_head=True needs norm_min/norm_max to recover raw counts "
                         "(the checkpoint's own training normalisation range).")
    mu, var = split_mu_var(raw)
    span = float(norm_max - norm_min)
    if not poisson_posterior:
        return (mu - norm_min) / span * 2 - 1
    y = (central.clamp(-1, 1) + 1) * 0.5 * span + norm_min
    x_hat, var_post = posterior_mean(y, mu, var, dose=poisson_dose)
    x_hat_norm = (x_hat - norm_min) / span * 2 - 1
    if return_uncertainty:
        return x_hat_norm, var_post
    return x_hat_norm


@torch.no_grad()
def denoise_frames_sinogram(model, sino_transform, central: torch.Tensor, context: torch.Tensor,
                            norm_min: float, norm_max: float, sino_norm_min: float, sino_norm_max: float,
                            present: bool = True) -> torch.Tensor:
    """Inference for a :class:`sdate.tr_diffusion.losses.SinogramN2VLoss` checkpoint.

    Denoises in per-frame Radon (sinogram) space (see
    :mod:`sdate.tr_diffusion.sino_transform`), then inverse-transforms (FBP)
    the result back to a denoised PROJECTION -- same ``[-1, 1]``-normalised,
    ``(B, 1, ny, nx)`` convention as :func:`denoise_frames_baseline`, so it
    plugs directly into the rest of the eval toolkit (``run_windows``,
    ``write_slice_movie``, ...) unchanged.

    ``norm_min``/``norm_max`` are the checkpoint's own PROJECTION (raw-count)
    normalisation range; ``sino_norm_min``/``sino_norm_max`` are the SEPARATE
    sinogram-domain range fit at training time (a sinogram's line-integral
    values live on a very different scale than raw projection counts).
    """
    device = central.device
    dtype = next(model.parameters()).dtype

    def denorm(x, lo, hi):
        return (x.to(dtype=torch.float32) + 1.0) * 0.5 * (hi - lo) + lo

    def sino_norm(s, lo, hi):
        return 2.0 * (s - lo) / (hi - lo) - 1.0

    central_counts = denorm(central, norm_min, norm_max)
    context_counts = denorm(context, norm_min, norm_max)
    sino_central = sino_norm(sino_transform.forward(central_counts), sino_norm_min, sino_norm_max)
    sino_context = (sino_norm(sino_transform.forward(context_counts), sino_norm_min, sino_norm_max)
                    if context_counts.shape[1] > 0 else sino_transform.forward(context_counts))

    b = sino_central.shape[0]
    corrupted = blind_spot_corrupt(sino_central)[0] if present else torch.zeros_like(sino_central)
    model_input = torch.cat([corrupted, sino_context], dim=1).to(dtype=dtype)
    timesteps = torch.zeros(b, device=device, dtype=torch.long)
    class_labels = torch.full((b,), int(present), device=device, dtype=torch.long)
    pred_sino_norm = model(model_input, timestep=timesteps, class_labels=class_labels, return_dict=False)[0]

    pred_sino_counts = (pred_sino_norm.to(dtype=torch.float32) + 1.0) * 0.5 * (sino_norm_max - sino_norm_min) + sino_norm_min
    proj_counts = sino_transform.inverse(pred_sino_counts)
    return (proj_counts - norm_min) / (norm_max - norm_min) * 2 - 1


@torch.no_grad()
def denoise_frames_noise2clean(model, central: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """Single forward pass for a ``--mode noise2clean`` checkpoint.

    Unlike :func:`denoise_frames_baseline`, this does NOT apply N2V blind-spot
    corruption or conditioning-dropout to ``central`` -- :class:`sdate.tr_diffusion.
    losses.NoiseToCleanLoss` never masked its input at training time (there's a
    real independent ground truth, so no identity-shortcut to guard against),
    so corrupting it here would be an inference-only train/eval mismatch.
    Always a plain single-channel point estimate (``poisson_head=False`` for
    every checkpoint of this mode).
    """
    device = central.device
    dtype = next(model.parameters()).dtype
    central = central.to(device=device, dtype=dtype)
    context = context.to(device=device, dtype=dtype)
    b = central.shape[0]
    model_input = torch.cat([central, context], dim=1)
    timesteps = torch.zeros(b, device=device, dtype=torch.long)
    class_labels = torch.ones(b, device=device, dtype=torch.long)
    return model(model_input, timestep=timesteps, class_labels=class_labels, return_dict=False)[0]


@torch.no_grad()
def bootstrap_belief(base_model, central: torch.Tensor, context: torch.Tensor,
                     present: bool = True, cond_channels: Optional[torch.Tensor] = None,
                     eps: float = 1e-6) -> tuple:
    """The FROZEN base ``poisson_head`` checkpoint's own context-only Gamma-Poisson
    belief ``(mu, var)`` for a frame, in RAW COUNT units (undoes the head's softplus
    split -- see :func:`sdate.tr_diffusion.nb_head.split_mu_var`).

    Exactly the base model's own forward pass (blind-spot-corrupted central +
    multi-frame context, same as :func:`denoise_frames_baseline`), just stopping
    one step earlier -- before combining with any observation. Shared by
    :class:`sdate.tr_diffusion.losses.BootstrapPoissonLoss` (training) and
    :func:`denoise_frames_bootstrap` (inference) below, so the two see the
    identical belief for the identical input.
    """
    device = central.device
    dtype = next(base_model.parameters()).dtype
    central = central.to(device=device, dtype=dtype)
    context = context.to(device=device, dtype=dtype)
    b = central.shape[0]
    corrupted = blind_spot_corrupt(central)[0] if present else torch.zeros_like(central)
    parts = [corrupted, context]
    if cond_channels is not None:
        parts.append(cond_channels.to(device=device, dtype=dtype))
    model_input = torch.cat(parts, dim=1)
    timesteps = torch.zeros(b, device=device, dtype=torch.long)
    class_labels = torch.full((b,), int(present), device=device, dtype=torch.long)
    raw = base_model(model_input, timestep=timesteps, class_labels=class_labels, return_dict=False)[0]
    return split_mu_var(raw, eps=eps)


def bootstrap_input_from_belief(mu_raw: torch.Tensor, var_raw: torch.Tensor, input_mode: str,
                                norm_min: float, norm_max: float, eps: float = 1e-6) -> torch.Tensor:
    """``mu_raw``/``var_raw`` (raw counts, from :func:`bootstrap_belief`) -> the
    NORMALISED ``[-1, 1]`` single-channel input for the bootstrap model.
    ``input_mode='mean'`` uses ``mu_raw`` directly (deterministic); ``'sample'``
    draws a fresh :func:`sdate.tr_diffusion.nb_head.gamma_sample` every call.
    """
    if input_mode not in ("mean", "sample"):
        raise ValueError("input_mode must be 'mean' or 'sample'")
    value_raw = mu_raw if input_mode == "mean" else gamma_sample(mu_raw, var_raw, eps=eps)
    span = float(norm_max - norm_min)
    return (value_raw - norm_min) / span * 2 - 1


@torch.no_grad()
def denoise_frames_bootstrap(bootstrap_model, base_model, central: torch.Tensor, context: torch.Tensor,
                             input_mode: str = "mean", num_samples: int = 1,
                             present: bool = True, cond_channels: Optional[torch.Tensor] = None,
                             norm_min: Optional[float] = None, norm_max: Optional[float] = None,
                             poisson_posterior: bool = False, poisson_dose: float = 1.0,
                             return_uncertainty: bool = False,
                             eps: float = 1e-6, chunk_size: int = 32):
    """Second-stage self-distillation denoiser (see
    :class:`sdate.tr_diffusion.losses.BootstrapPoissonLoss`): reads the frozen
    ``base_model``'s own context-only belief for this frame (:func:`bootstrap_belief`),
    turns it into the bootstrap model's single input channel
    (:func:`bootstrap_input_from_belief`, ``input_mode`` -- 'mean' or 'sample'),
    and runs the bootstrap model. Returns its result in the SAME normalised
    ``[-1, 1]`` convention as :func:`denoise_frames_baseline`.

    ``poisson_posterior`` (default ``False``, the original behaviour --
    reproduces the "bootstrap does NOT come out sharper" result in the README):
    the bootstrap model ALSO has a two-head ``poisson_head`` architecture, but
    its own ``var`` was, until now, discarded entirely -- its final answer was
    just its ``mu`` (a point estimate of a point estimate, never re-touching the
    real observed pixel). Passing ``poisson_posterior=True`` combines the
    bootstrap model's own ``(mu, var)`` belief with the REAL observed pixel
    (``central``, denormalised) via the exact Gamma-Poisson posterior
    (:func:`sdate.tr_diffusion.nb_head.posterior_mean`) -- the SAME Bayes step
    that gives the base model's ``poisson_head`` architecture its sharpness
    (see README "Baseline loss correction"). ``poisson_dose`` MUST match the
    thinning fraction ``central`` was actually measured/synthesised at (1.0 =
    native; e.g. 0.05 for the ``extra_noise_dose`` regime used everywhere else
    in this project). ``return_uncertainty=True`` additionally returns the
    posterior variance (raw count units, not renormalised) -- only meaningful
    together with ``poisson_posterior=True``.

    ``input_mode='sample'`` with ``num_samples > 1`` averages over that many
    FRESH Gamma draws (each run through the bootstrap model independently) --
    a posterior-mean-style MMSE estimate over the base model's own local
    uncertainty, the same role :func:`pred_x0_ensemble` plays for the diffusion
    model. ``input_mode='mean'`` is deterministic, so ``num_samples`` is
    ignored (forced to 1) -- repeating an identical forward pass wastes compute.

    ``norm_min``/``norm_max`` must be the SAME range both checkpoints were
    trained with (verified identical at bootstrap training time -- see
    ``train.py``'s ``--mode bootstrap`` geometry inheritance).
    """
    if norm_min is None or norm_max is None:
        raise ValueError("denoise_frames_bootstrap needs norm_min/norm_max (the shared "
                         "normalisation range both checkpoints were trained with)")
    mu_raw, var_raw = bootstrap_belief(base_model, central, context, present=present,
                                       cond_channels=cond_channels, eps=eps)
    n = 1 if input_mode == "mean" else max(1, int(num_samples))
    b = mu_raw.shape[0]

    rep_mu_raw = mu_raw.repeat_interleave(n, dim=0)
    rep_var_raw = var_raw.repeat_interleave(n, dim=0)
    inputs_norm = bootstrap_input_from_belief(rep_mu_raw, rep_var_raw, input_mode, norm_min, norm_max, eps=eps)

    device = inputs_norm.device
    dtype = next(bootstrap_model.parameters()).dtype
    inputs_norm = inputs_norm.to(dtype=dtype)
    total = inputs_norm.shape[0]
    cs = max(1, int(chunk_size)) if chunk_size else total
    span = float(norm_max - norm_min)

    rep_y = None
    if poisson_posterior:
        y = (central.to(device=device, dtype=torch.float32).clamp(-1, 1) + 1) * 0.5 * span + norm_min
        rep_y = y.repeat_interleave(n, dim=0)

    outs, var_outs = [], ([] if (poisson_posterior and return_uncertainty) else None)
    for i in range(0, total, cs):
        chunk = inputs_norm[i : i + cs]
        m = chunk.shape[0]
        t = torch.zeros(m, device=device, dtype=torch.long)
        cls = torch.ones(m, device=device, dtype=torch.long)
        raw = bootstrap_model(chunk, timestep=t, class_labels=cls, return_dict=False)[0]
        mu2, var2 = split_mu_var(raw, eps=eps)
        if poisson_posterior:
            y_chunk = rep_y[i : i + cs].to(dtype=mu2.dtype)
            x_hat, var_post = posterior_mean(y_chunk, mu2, var2, dose=poisson_dose, eps=eps)
            outs.append((x_hat - norm_min) / span * 2 - 1)
            if var_outs is not None:
                var_outs.append(var_post)
        else:
            outs.append((mu2 - norm_min) / span * 2 - 1)
    samples = torch.cat(outs, dim=0).view(b, n, *mu_raw.shape[1:])
    result = samples.mean(dim=1)
    if var_outs is not None:
        var_samples = torch.cat(var_outs, dim=0).view(b, n, *mu_raw.shape[1:])
        return result, var_samples.mean(dim=1)
    return result


@torch.no_grad()
def denoise_frames_refine(refine_model, base_model, central: torch.Tensor, base_context: torch.Tensor,
                          refine_context: torch.Tensor, input_mode: str = "sample", num_samples: int = 1,
                          present: bool = True, cond_channels: Optional[torch.Tensor] = None,
                          norm_min: Optional[float] = None, norm_max: Optional[float] = None,
                          poisson_posterior: bool = True, poisson_dose: float = 1.0,
                          return_uncertainty: bool = False,
                          eps: float = 1e-6, chunk_size: int = 32):
    """Inference for a ``RefinementLoss`` checkpoint (angular-resolution-gap
    experiment, "Leg 1"): :func:`denoise_frames_bootstrap`'s exact mechanism
    (frozen ``base_model``'s own context-only belief -> a single sampled/mean
    input channel), with ONE addition -- ``refine_context`` (the motion-
    compensated ``context_warped`` channels, see ``data.py``) concatenated
    alongside that channel before calling ``refine_model``, matching training's
    channel order (:class:`sdate.tr_diffusion.losses.RefinementLoss`).

    ``base_context`` is the RAW (unwarped) multi-frame context fed to the
    FROZEN ``base_model`` -- must be exactly what that checkpoint was trained
    on, same as :func:`denoise_frames_bootstrap`'s ``context`` argument.
    ``refine_context`` is the SECOND-stage ``refine_model``'s own context
    (warped taps + unchanged rotation taps) and has nothing to do with the
    base model's forward pass.

    See :func:`denoise_frames_bootstrap`'s docstring for the meaning of
    ``input_mode``/``num_samples``/``poisson_posterior``/``poisson_dose``/
    ``return_uncertainty`` -- identical here, just with ``refine_context``
    threaded through the multi-sample expansion alongside ``mu_raw``/``var_raw``.
    """
    if norm_min is None or norm_max is None:
        raise ValueError("denoise_frames_refine needs norm_min/norm_max (the shared "
                         "normalisation range both checkpoints were trained with)")
    mu_raw, var_raw = bootstrap_belief(base_model, central, base_context, present=present,
                                       cond_channels=cond_channels, eps=eps)
    n = 1 if input_mode == "mean" else max(1, int(num_samples))
    b = mu_raw.shape[0]

    rep_mu_raw = mu_raw.repeat_interleave(n, dim=0)
    rep_var_raw = var_raw.repeat_interleave(n, dim=0)
    inputs_norm = bootstrap_input_from_belief(rep_mu_raw, rep_var_raw, input_mode, norm_min, norm_max, eps=eps)
    rep_context = refine_context.repeat_interleave(n, dim=0)

    device = inputs_norm.device
    dtype = next(refine_model.parameters()).dtype
    inputs_norm = torch.cat([inputs_norm, rep_context.to(device=device, dtype=inputs_norm.dtype)], dim=1)
    inputs_norm = inputs_norm.to(dtype=dtype)
    total = inputs_norm.shape[0]
    cs = max(1, int(chunk_size)) if chunk_size else total
    span = float(norm_max - norm_min)

    rep_y = None
    if poisson_posterior:
        y = (central.to(device=device, dtype=torch.float32).clamp(-1, 1) + 1) * 0.5 * span + norm_min
        rep_y = y.repeat_interleave(n, dim=0)

    outs, var_outs = [], ([] if (poisson_posterior and return_uncertainty) else None)
    for i in range(0, total, cs):
        chunk = inputs_norm[i : i + cs]
        m = chunk.shape[0]
        t = torch.zeros(m, device=device, dtype=torch.long)
        cls = torch.ones(m, device=device, dtype=torch.long)
        raw = refine_model(chunk, timestep=t, class_labels=cls, return_dict=False)[0]
        mu2, var2 = split_mu_var(raw, eps=eps)
        if poisson_posterior:
            y_chunk = rep_y[i : i + cs].to(dtype=mu2.dtype)
            x_hat, var_post = posterior_mean(y_chunk, mu2, var2, dose=poisson_dose, eps=eps)
            outs.append((x_hat - norm_min) / span * 2 - 1)
            if var_outs is not None:
                var_outs.append(var_post)
        else:
            outs.append((mu2 - norm_min) / span * 2 - 1)
    samples = torch.cat(outs, dim=0).view(b, n, *mu_raw.shape[1:])
    result = samples.mean(dim=1)
    if var_outs is not None:
        var_samples = torch.cat(var_outs, dim=0).view(b, n, *mu_raw.shape[1:])
        return result, var_samples.mean(dim=1)
    return result


@torch.no_grad()
def denoise_frames_bootstrap_reverse(bootstrap_model, central: torch.Tensor,
                                     norm_min: Optional[float] = None, norm_max: Optional[float] = None,
                                     poisson_posterior: bool = False, poisson_dose: float = 1.0,
                                     return_uncertainty: bool = False, eps: float = 1e-6):
    """Inference for a ``BootstrapPoissonLoss(direction="reverse")`` checkpoint.

    Unlike :func:`denoise_frames_bootstrap` (the ``forward`` direction), this
    model was trained to predict a Gamma-posterior draw of the base model's
    belief FROM the real measurement -- so at inference it needs neither the
    frozen base model nor any multi-frame context: ``central`` (the real,
    single-channel measurement, already normalised ``[-1, 1]``) is fed
    directly as the model's own input, and its output ``mu`` is the answer.

    ``poisson_posterior=True`` additionally combines the model's own
    ``(mu, var)`` with ``central`` itself (denormalised) via
    :func:`sdate.tr_diffusion.nb_head.posterior_mean` -- note this re-uses the
    SAME pixel that was already the model's input (unlike the base model's own
    posterior step, which combines a context-only belief with an
    INDEPENDENT observation), so it is a self-referential Bayes update, not a
    second independent evidence source; included for completeness/exploration
    only, default off.
    """
    if norm_min is None or norm_max is None:
        raise ValueError("denoise_frames_bootstrap_reverse needs norm_min/norm_max (the shared "
                         "normalisation range the checkpoint was trained with)")
    device = central.device
    dtype = next(bootstrap_model.parameters()).dtype
    model_input = central.to(device=device, dtype=dtype)
    b = model_input.shape[0]
    t = torch.zeros(b, device=device, dtype=torch.long)
    cls = torch.ones(b, device=device, dtype=torch.long)
    raw = bootstrap_model(model_input, timestep=t, class_labels=cls, return_dict=False)[0]
    mu, var = split_mu_var(raw, eps=eps)

    span = float(norm_max - norm_min)
    if poisson_posterior:
        y = (central.to(device=device, dtype=torch.float32).clamp(-1, 1) + 1) * 0.5 * span + norm_min
        x_hat, var_post = posterior_mean(y, mu, var, dose=poisson_dose, eps=eps)
        result = (x_hat - norm_min) / span * 2 - 1
        if return_uncertainty:
            return result, var_post
        return result
    result = (mu - norm_min) / span * 2 - 1
    if return_uncertainty:
        return result, var
    return result
