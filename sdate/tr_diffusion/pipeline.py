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
from diffusers import DDIMScheduler

from .n2v import blind_spot_corrupt
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
                            present: bool = True, cond_channels: Optional[torch.Tensor] = None) -> torch.Tensor:
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
    model_input = torch.cat(parts, dim=1)
    timesteps = torch.zeros(b, device=device, dtype=torch.long)
    class_labels = torch.full((b,), int(present), device=device, dtype=torch.long)
    return model(model_input, timestep=timesteps, class_labels=class_labels, return_dict=False)[0]
