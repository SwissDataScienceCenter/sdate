"""Training losses: conditional-diffusion (ε) and baseline (x_0) — both N2V-masked.

Shared recipe per batch:

1. ``x_0`` = the (normalised) central frame.
2. Blind-spot-corrupt ``x_0`` -> ``(corrupted, mask)`` (:mod:`sdate.tr_diffusion.n2v`).
3. Conditioning dropout: with probability ``1 - conditioning_probability`` a
   sample is trained *without* the central frame (corrupted channel zeroed,
   ``class_label = 0``); otherwise *with* it (``class_label = 1``).
4. Assemble the model input in the fixed channel order and predict.
5. **Masked loss.** For *with-central* samples the loss is evaluated only at the
   blind-spot pixels (the rest leak ``x_0`` through the corrupted channel and
   carry no denoising signal).  For *without-central* samples nothing leaks, so
   the loss is over the full frame (standard conditional objective on the
   neighbours alone).

The only difference between the two losses is what is predicted: the diffusion
model predicts the added noise from ``[x_t, corrupted, context]``; the baseline
predicts ``x_0`` directly from ``[corrupted, context]`` in one pass.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_base.base_loss import BaseLoss

from .n2v import blind_spot_corrupt
from .nb_head import beta_nll_weight, gamma_sample, nb_nll, nb_nll_gaussian, poisson_nll, split_mu_var
from .noise import add_poisson_noise
from .pipeline import bootstrap_belief, bootstrap_input_from_belief

# --------------------------------------------------------------------------- #
# Noise2Noise (binomial-split) losses — exploratory alternative to N2V.
#
# The dataset splits the fixed dose measurement into two independent views
# (``central_input`` fraction p, ``central_target`` fraction 1-p); the model
# predicts the target from the input + full-dose neighbours, conditioned on the
# discretised fraction ``p_bin`` (via class_embed_type="timestep"). Because the
# two splits are independent, the loss is over the FULL frame (no blind-spot
# mask, no conditioning-dropout — those were N2V devices).
# --------------------------------------------------------------------------- #


def _make_loss(loss_type: str) -> Optional[nn.Module]:
    lt = loss_type.lower()
    if lt == "mae":
        return nn.L1Loss(reduction="none")
    if lt == "mse":
        return nn.MSELoss(reduction="none")
    if lt == "huber":
        return nn.HuberLoss(reduction="none")
    if lt == "poisson":
        # Handled specially (BaselineN2VLoss only, see poisson_mean_only below) --
        # needs a positivity-constrained, counts-scale prediction, not a plain
        # nn loss module comparing raw output to the normalised target.
        return None
    raise ValueError("loss_type must be one of: mae, mse, huber, poisson")


def _to_device(instance: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in instance.items()}


def _masked_mean(per_pixel: torch.Tensor, loss_mask: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    if weight is not None:
        per_pixel = per_pixel * weight
    return (per_pixel * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)


def _edge_weight_map(x0: torch.Tensor, edge_weight: float) -> torch.Tensor:
    """``1 + edge_weight * normalized local-gradient-magnitude(x0)``, same shape as ``x0``.

    A static (detached, GT-derived) per-pixel weight that up-weights the loss at
    high-gradient (edge) pixels relative to flat regions, so the regressor is
    penalised more for smoothing edges away than for small errors in flat areas —
    this only reweights the SAME masked pixel-loss terms N2V already restricts
    to (it never looks past the blind-spot mask), so it can't leak the identity
    shortcut the masking exists to prevent.
    """
    dy = F.pad((x0[..., 1:, :] - x0[..., :-1, :]).abs(), (0, 0, 0, 1))
    dx = F.pad((x0[..., :, 1:] - x0[..., :, :-1]).abs(), (0, 1, 0, 0))
    grad = (dy + dx).detach()
    grad = grad / grad.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return 1.0 + edge_weight * grad


class _N2VLossBase(BaseLoss):
    def __init__(self, device, ratio=0.02, window=5, conditioning_probability=0.5, loss_type="huber",
                edge_weight=0.0):
        super().__init__(["loss"])
        if not 0.0 <= conditioning_probability <= 1.0:
            raise ValueError("conditioning_probability must be in [0, 1]")
        self.device = device
        self.ratio = float(ratio)
        self.window = int(window)
        self.conditioning_probability = float(conditioning_probability)
        self.loss_type = loss_type.lower()
        self.loss = _make_loss(loss_type)
        self.edge_weight = float(edge_weight)

    def _prepare(self, instance) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``x0, context, corrupted, present, loss_mask, weight``."""
        instance = _to_device(instance, self.device)
        x0 = instance["central"].float()
        context = instance["context"].float()
        bsz = x0.shape[0]

        corrupted, mask = blind_spot_corrupt(x0, ratio=self.ratio, window=self.window)
        present = (torch.rand(bsz, device=self.device) < self.conditioning_probability)
        present_map = present.view(bsz, 1, 1, 1)

        corrupted = corrupted * present_map.to(corrupted.dtype)
        # with-central -> loss only at blind spots; without-central -> full frame.
        loss_mask = torch.where(present_map, mask, torch.ones_like(mask)).float()
        weight = _edge_weight_map(x0, self.edge_weight) if self.edge_weight > 0 else None
        return x0, context, corrupted, present.long(), loss_mask, weight


class DiffusionN2VLoss(_N2VLossBase):
    def __init__(self, noise_scheduler, device, **kw):
        super().__init__(device, **kw)
        if self.loss_type == "poisson":
            raise ValueError("loss_type='poisson' (mean-only Poisson NLL, see BaselineN2VLoss) is "
                             "only meaningful for the baseline (x0-prediction) loss, not epsilon-prediction.")
        self.noise_scheduler = noise_scheduler

    def compute_loss(self, instance, model):
        x0, context, corrupted, present, loss_mask, weight = self._prepare(instance)
        bsz = x0.shape[0]

        noise = torch.randn_like(x0)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device
        ).long()
        x_t = self.noise_scheduler.add_noise(x0, noise, timesteps)

        model_input = torch.cat([x_t, corrupted, context], dim=1)
        noise_pred = model(model_input, timestep=timesteps, class_labels=present, return_dict=False)[0]

        # NOTE: edge_weight is derived from x0 (the clean target), but the diffusion
        # model predicts NOISE, not x0 -- so weighting by x0's edge map here would
        # reweight the noise-prediction loss by an unrelated signal. edge_weight is
        # only meaningful for the baseline (x0-prediction) loss below.
        loss = _masked_mean(self.loss(noise_pred, noise), loss_mask)
        return loss, {"loss": loss.item()}


class BaselineN2VLoss(_N2VLossBase):
    """Single-pass regression baseline: predict denoised ``x_0`` directly.

    Three mutually exclusive prediction modes, all sharing the same masking/
    conditioning-dropout recipe above:

    * ``poisson_head=True`` (the STANDARD/default -- see README "Key
      experimental findings" + project memory project-tr-diffusion) -- the
      2-output-channel head (see :func:`sdate.tr_diffusion.model.create_baseline_unet`)
      trained with the exact Negative-Binomial NLL. Supersedes both modes
      below: its own ``mu`` behaves identically to the single-channel
      ``loss_type="poisson"`` point estimate (measured pixelwise correlation
      0.999 -- both fit the same context-conditional mean), and additionally
      supports the exact posterior-mean combination with the real observation
      at inference (see :mod:`sdate.tr_diffusion.nb_head`), which recovers
      sharpness neither single-channel point estimate can. Pass
      ``poisson_posterior=False`` at inference (see
      :func:`sdate.tr_diffusion.pipeline.denoise_frames_baseline`) to read off
      the (superseding) single-head-equivalent ``mu`` from this SAME checkpoint
      -- no need to train the single-channel variant separately.
    * ``loss_type="poisson"`` (legacy ablation, superseded by the above --
      requires ``--no-poisson_head``) -- a single-channel point estimate (no
      variance head) trained with the correct Poisson NLL instead of a
      homoscedastic one. Kept for loading/comparing against already-trained
      checkpoints; prefer ``poisson_head=True`` with ``poisson_posterior=False``
      for new work.
    * default (``loss_type in {huber, mae, mse}``, requires ``--no-poisson_head``)
      -- the original single-channel point estimate against a homoscedastic
      loss. Legacy; kept for loading old checkpoints.

    The first two both require ``norm_min``/``norm_max`` (the dataset's own
    normalisation range) to recover raw detector counts for the Poisson
    likelihood.
    """

    def __init__(self, device, ratio=0.02, window=5, conditioning_probability=0.5, loss_type="huber",
                edge_weight=0.0, poisson_head=True, norm_min=None, norm_max=None,
                poisson_warmup_steps=0, poisson_beta_nll_power=0.0, poisson_eps=1e-6, poisson_dose=1.0,
                gaussian_floor=False, sigma_read2=None):
        super().__init__(device, ratio=ratio, window=window,
                         conditioning_probability=conditioning_probability, loss_type=loss_type,
                         edge_weight=edge_weight)
        self.poisson_head = bool(poisson_head)
        self.poisson_mean_only = (self.loss_type == "poisson")
        if self.poisson_head and self.poisson_mean_only:
            raise ValueError("poisson_head=True and loss_type='poisson' are mutually exclusive "
                             "(two-head NB-NLL vs single-channel Poisson-mean-only ablation).")
        if self.poisson_head or self.poisson_mean_only:
            if norm_min is None or norm_max is None:
                raise ValueError("poisson_head=True / loss_type='poisson' need norm_min/norm_max to "
                                 "recover raw counts for the Poisson likelihood (the dataset's own "
                                 "normalisation range).")
            self.norm_min = float(norm_min)
            self.norm_max = float(norm_max)
        # Steps to fit mu alone via plain Poisson NLL (var unused -> no gradient to
        # the var head) before switching on the full heteroscedastic NB-NLL below.
        # Set by the caller once the total step count is known (see train.py).
        self.poisson_warmup_steps = int(poisson_warmup_steps)
        self.poisson_beta_nll_power = float(poisson_beta_nll_power)
        self.poisson_eps = float(poisson_eps)
        # KNOWN thinning fraction the target ``x0`` was synthesised at (see
        # sdate.tr_diffusion.noise.add_poisson_noise / train.py's --extra_noise_dose).
        # 1.0 = native/non-thinned target (the default, plain-Poisson case). Only
        # affects poisson_head's NB-NLL (dose changes the argmin); the mean-only
        # plain Poisson NLL below is dose-invariant up to a constant factor, so it
        # doesn't need this.
        self.poisson_dose = float(poisson_dose)
        self._poisson_step = 0
        # Poisson+Gaussian read-noise-floor NLL (nb_nll_gaussian) instead of the exact
        # Gamma-Poisson NB marginal (nb_nll) -- see nb_nll_gaussian's docstring for why:
        # the plain NB-NLL is numerically unstable at native/high-count regimes once the
        # var head is near its untrained init (alpha = mu^2/var explodes). sigma_read2 is
        # a fixed (not learned) per-pixel Gaussian read-noise variance map, estimated from
        # real dark-frame variance (scripts/tr_diffusion_estimate_read_noise.py).
        self.gaussian_floor = bool(gaussian_floor)
        if self.gaussian_floor:
            if sigma_read2 is None:
                raise ValueError("gaussian_floor=True requires sigma_read2 (a per-pixel dark-variance "
                                 "map or a global scalar float).")
            if not torch.is_tensor(sigma_read2):
                sigma_read2 = torch.tensor(sigma_read2, dtype=torch.float32)
            self.sigma_read2 = sigma_read2.to(device=device, dtype=torch.float32)

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        span = self.norm_max - self.norm_min
        return (x + 1.0) * 0.5 * span + self.norm_min

    def compute_loss(self, instance, model):
        x0, context, corrupted, present, loss_mask, weight = self._prepare(instance)
        bsz = x0.shape[0]

        # Angular-resolution-gap experiment: if the dataset was built with
        # warped_temporal_memmaps, prefer the motion-compensated context over the
        # raw one _prepare() returned -- everything else about this loss (the
        # blind-spot-corrupted CENTRAL channel, masking, present-dropout) is
        # unchanged, isolating the context data source as the only variable.
        if "context_warped" in instance:
            context = instance["context_warped"].to(self.device, non_blocking=True).float()

        parts = [corrupted, context]
        if "cond_channels" in instance:
            parts.append(instance["cond_channels"].to(self.device, non_blocking=True).float())
        if "aux_channel" in instance:
            parts.append(instance["aux_channel"].to(self.device, non_blocking=True).float())
        model_input = torch.cat(parts, dim=1)
        timesteps = torch.zeros(bsz, device=self.device, dtype=torch.long)
        raw = model(model_input, timestep=timesteps, class_labels=present, return_dict=False)[0]

        if self.poisson_mean_only:
            mu = F.softplus(raw) + self.poisson_eps
            y = self._denormalize(x0)
            per_pixel = poisson_nll(y, mu, eps=self.poisson_eps)
            loss = _masked_mean(per_pixel, loss_mask, weight)
            return loss, {"loss": loss.item()}

        if not self.poisson_head:
            loss = _masked_mean(self.loss(raw, x0), loss_mask, weight)
            return loss, {"loss": loss.item()}

        mu, var = split_mu_var(raw, eps=self.poisson_eps)
        y = self._denormalize(x0)
        warm = self._poisson_step < self.poisson_warmup_steps
        self._poisson_step += 1
        if warm:
            per_pixel = poisson_nll(y, mu, eps=self.poisson_eps)
        elif self.gaussian_floor:
            per_pixel = nb_nll_gaussian(y, mu, var, self.sigma_read2, dose=self.poisson_dose,
                                        eps=self.poisson_eps)
            if self.poisson_beta_nll_power > 0:
                per_pixel = per_pixel * beta_nll_weight(mu, var, power=self.poisson_beta_nll_power,
                                                        eps=self.poisson_eps)
        else:
            per_pixel = nb_nll(y, mu, var, dose=self.poisson_dose, eps=self.poisson_eps)
            if self.poisson_beta_nll_power > 0:
                per_pixel = per_pixel * beta_nll_weight(mu, var, power=self.poisson_beta_nll_power,
                                                        eps=self.poisson_eps)
        loss = _masked_mean(per_pixel, loss_mask, weight)
        return loss, {"loss": loss.item()}


class _N2NLossBase(BaseLoss):
    def __init__(self, device, loss_type="huber", stats_names=None):
        super().__init__(stats_names or ["loss"])
        self.device = device
        self.loss = _make_loss(loss_type)

    def _prep(self, instance):
        inst = _to_device(instance, self.device)
        return (inst["central_input"].float(), inst["central_target"].float(),
                inst["context"].float(), inst["p_bin"].long())


class DiffusionN2NLoss(_N2NLossBase):
    """Conditioning = input-split + neighbours + p; predicts either the noise
    added to the target-split (``prediction_type="epsilon"``, the original
    behaviour) or the target-split itself directly (``prediction_type="sample"``).

    Optional **swap-consistency** term (``consistency_weight > 0``, exploratory):
    since ``central_input``/``central_target`` are two independent halves of the
    SAME underlying signal (binomial split), the model is also asked to predict
    input-from-target (at an independently sampled timestep) and the two
    resulting ``x0`` estimates are pushed together with an extra MSE term. This
    is the diffusion analogue of the Noise2Noise argument applied per-step:
    regressing (in x0-space) to an independent noisy target already recovers the
    same minimiser as regressing to the clean signal, so forcing the two
    conditioning directions to agree discourages the model from depending on
    which split half or timestep it happened to see.
    """

    def __init__(self, noise_scheduler, device, prediction_type="epsilon",
                 consistency_weight=0.0, p_bins=100, **kw):
        if prediction_type not in ("epsilon", "sample"):
            raise ValueError("prediction_type must be 'epsilon' or 'sample'")
        stats = ["loss"] if consistency_weight <= 0 else ["loss", "loss_recon", "loss_consistency"]
        super().__init__(device, stats_names=stats, **kw)
        self.noise_scheduler = noise_scheduler
        self.prediction_type = prediction_type
        self.consistency_weight = float(consistency_weight)
        self.p_bins = int(p_bins)

    def _branch(self, model, x0_target, cond_input, ctx, p_bin, t):
        noise = torch.randn_like(x0_target)
        x_t = self.noise_scheduler.add_noise(x0_target, noise, t)
        model_input = torch.cat([x_t, cond_input, ctx], dim=1)
        pred = model(model_input, timestep=t, class_labels=p_bin, return_dict=False)[0]
        if self.prediction_type == "sample":
            recon_loss = self.loss(pred, x0_target).mean()
            x0_est = pred
        else:
            recon_loss = self.loss(pred, noise).mean()
            acp = self.noise_scheduler.alphas_cumprod.to(self.device)[t].view(-1, 1, 1, 1)
            x0_est = (x_t - (1 - acp).sqrt() * pred) / acp.sqrt().clamp_min(1e-6)
        return recon_loss, x0_est

    def compute_loss(self, instance, model):
        inp, tgt, ctx, p_bin = self._prep(instance)
        bsz = inp.shape[0]
        num_t = self.noise_scheduler.config.num_train_timesteps
        t_a = torch.randint(0, num_t, (bsz,), device=self.device).long()
        loss_a, x0_a = self._branch(model, tgt, inp, ctx, p_bin, t_a)

        if self.consistency_weight <= 0:
            return loss_a, {"loss": loss_a.item()}

        # branch B: condition on the OTHER split half -> its actual fraction is
        # (1-p), i.e. p_bins - p_bin (not p_bin, which describes branch A's input).
        p_bin_b = (self.p_bins - p_bin).clamp(0, self.p_bins)
        t_b = torch.randint(0, num_t, (bsz,), device=self.device).long()
        loss_b, x0_b = self._branch(model, inp, tgt, ctx, p_bin_b, t_b)

        recon_loss = loss_a + loss_b
        consistency = torch.nn.functional.mse_loss(x0_a, x0_b)
        loss = recon_loss + self.consistency_weight * consistency
        return loss, {"loss": loss.item(), "loss_recon": recon_loss.item(),
                      "loss_consistency": consistency.item()}


class DiffusionN2NCleanTargetLoss(DiffusionN2NLoss):
    """Like :class:`DiffusionN2NLoss`, but regresses toward a fixed EXTERNAL
    clean reference (``instance["clean_target"]``, e.g. a Bayesian two-head
    ``poisson_head`` reconstruction -- see :mod:`sdate.tr_diffusion.nb_head`
    and the README's "two-head Gamma-Poisson" section) instead of the other
    binomial-split half.

    Conditioning is unchanged: the two independent binomial-split views of
    the SAME raw dose-thinned measurement (fractions ``p`` and ``1-p``) are
    still generated and still each condition one branch. What changes is the
    regression target -- both branches now regress toward the SAME external
    ``clean_target`` instead of toward each other's split half. The
    swap-consistency term is unchanged: the two branches' ``x0`` estimates
    (from independent split-halves and independent timesteps) are still
    pushed to agree, which remains meaningful here since they're two noisy
    views of the same underlying frame predicting the same clean target.
    """

    def _prep(self, instance):
        inst = _to_device(instance, self.device)
        return (inst["central_input"].float(), inst["central_target"].float(),
                inst["clean_target"].float(), inst["context"].float(), inst["p_bin"].long())

    def compute_loss(self, instance, model):
        inp, tgt, clean, ctx, p_bin = self._prep(instance)
        bsz = inp.shape[0]
        num_t = self.noise_scheduler.config.num_train_timesteps
        t_a = torch.randint(0, num_t, (bsz,), device=self.device).long()
        loss_a, x0_a = self._branch(model, clean, inp, ctx, p_bin, t_a)

        if self.consistency_weight <= 0:
            return loss_a, {"loss": loss_a.item()}

        p_bin_b = (self.p_bins - p_bin).clamp(0, self.p_bins)
        t_b = torch.randint(0, num_t, (bsz,), device=self.device).long()
        loss_b, x0_b = self._branch(model, clean, tgt, ctx, p_bin_b, t_b)

        recon_loss = loss_a + loss_b
        consistency = torch.nn.functional.mse_loss(x0_a, x0_b)
        loss = recon_loss + self.consistency_weight * consistency
        return loss, {"loss": loss.item(), "loss_recon": recon_loss.item(),
                      "loss_consistency": consistency.item()}


class BaselineN2NLoss(_N2NLossBase):
    """Single-pass regression: predict the target-split from input-split + neighbours + p."""

    def compute_loss(self, instance, model):
        inp, tgt, ctx, p_bin = self._prep(instance)
        bsz = inp.shape[0]
        model_input = torch.cat([inp, ctx], dim=1)
        t = torch.zeros(bsz, device=self.device, dtype=torch.long)
        pred = model(model_input, timestep=t, class_labels=p_bin, return_dict=False)[0]
        loss = self.loss(pred, tgt).mean()
        return loss, {"loss": loss.item()}


class NoiseToCleanLoss(BaseLoss):
    """Supervised noise2clean regression on synthetic phantom data with REAL
    ground truth (``instance["reference"]``, the pre-thinning clean phantom
    frame produced by ``TimeResolvedFrameDataset``'s ``extra_noise_dose``
    branch -- see ``data.py``). Predicts the clean frame directly from the
    noisy central + noisy context in one pass, plain MSE, no N2V masking and
    no conditioning dropout: unlike self-supervised training, there is a real
    independent ground truth here, so there's no identity-shortcut/leakage
    risk for masking to guard against (same reasoning as :class:`BaselineN2NLoss`).

    Uses a plain single-channel regression head (``poisson_head=False`` on the
    model) rather than the two-head Gamma-Poisson NB-NLL used elsewhere in
    this module -- the point of this model is a dumb, sharp-feature detector
    trained with a real target, not another dose-aware Bayes estimator.
    """

    def __init__(self, device, loss_type="mse"):
        super().__init__(["loss"])
        self.device = device
        self.loss = _make_loss(loss_type)

    def compute_loss(self, instance, model):
        inst = _to_device(instance, self.device)
        central = inst["central"].float()
        context = inst["context"].float()
        target = inst["reference"].float()
        bsz = central.shape[0]

        model_input = torch.cat([central, context], dim=1)
        t = torch.zeros(bsz, device=self.device, dtype=torch.long)
        cls = torch.ones(bsz, device=self.device, dtype=torch.long)
        pred = model(model_input, timestep=t, class_labels=cls, return_dict=False)[0]
        loss = self.loss(pred, target).mean()
        return loss, {"loss": loss.item()}


class ContextOnlyNoiseToCleanLoss(BaseLoss):
    """Native-noise super-time-resolution baseline: predict the REAL measured
    central projection frame from JOINT-FBP CONTEXT TAPS ALONE -- the central
    frame is NEVER shown to the network, not even corrupted/masked (matching
    the ``conditioning_probability=0`` / ``present=False`` convention used
    elsewhere in this module, so the resulting checkpoint is directly
    consumable by ``pipeline.denoise_frames_baseline(..., present=False)``
    unmodified). No synthetic noise anywhere in this pipeline (dataset built
    with ``extra_noise_dose=None``, and the aux taps themselves were cached
    from native-noise joint-FBP reconstructions via
    ``tr_diffusion_jointfbp_context_cache.py --native_noise``), so like
    :class:`NoiseToCleanLoss` there is a real independent target with no
    identity-shortcut/leakage risk -- plain MSE, no N2V masking, no
    conditioning dropout. See project memory
    ``project-tr-diffusion-jointfbpctx`` for the full experiment design: this
    is the training half of a scheme meant to synthesise NEVER-MEASURED
    projection angles at inference time, by reprojecting the same context taps
    at an angle nothing was actually recorded at.
    """

    def __init__(self, device, loss_type="mse"):
        super().__init__(["loss"])
        self.device = device
        self.loss = _make_loss(loss_type)

    def compute_loss(self, instance, model):
        inst = _to_device(instance, self.device)
        central = inst["central"].float()
        context = inst["context"].float()
        bsz = central.shape[0]

        corrupted = torch.zeros_like(central)
        parts = [corrupted, context]
        if "aux_channel" in inst:
            parts.append(inst["aux_channel"].float())
        model_input = torch.cat(parts, dim=1)
        t = torch.zeros(bsz, device=self.device, dtype=torch.long)
        cls = torch.zeros(bsz, device=self.device, dtype=torch.long)  # class_label=0 == without-central
        pred = model(model_input, timestep=t, class_labels=cls, return_dict=False)[0]
        loss = self.loss(pred, central).mean()
        return loss, {"loss": loss.item()}


class SinogramN2VLoss(_N2VLossBase):
    """N2V + context, but denoising in per-frame 2D-Radon ("sinogram") space
    instead of raw projection space (see :mod:`sdate.tr_diffusion.sino_transform`).

    Every channel of the usual central+context recipe (central AND every
    rotation/temporal tap) is independently Radon-transformed (each is its own
    standalone 2D image, unrelated to the real 3D acquisition geometry) BEFORE
    anything else happens -- the blind-spot corruption/masking then applies to
    the resulting CENTRAL SINOGRAM, not the raw projection. This order matters:
    corrupting the raw projection first and transforming afterward would smear
    that corruption across many sinogram pixels via the line integrals,
    making "the loss only applies at these masked pixels" meaningless. Target
    = the (uncorrupted) sinogram of the SAME noisy central projection --
    plain N2V, no independent second noise draw the way
    :class:`BootstrapPoissonLoss` needs.

    Loss is plain Huber/MAE/MSE (``poisson_head``/``loss_type="poisson"`` are
    NOT available here): a sinogram value is a line-integral sum over many raw
    pixels, so it's no longer cleanly Poisson-distributed -- exactly why this
    domain is expected to have much better per-pixel SNR than raw projection
    space, but also why the NB-NLL machinery used elsewhere in this project
    doesn't apply.

    ``cond_angle_time`` conditioning is NOT wired up for this mode (those
    channels are constant-valued angle/time planes, not images -- transforming
    them would be meaningless, and passing them through unentransformed would
    need a separate broadcast-to-sinogram-shape path not built here).
    """

    def __init__(self, device, sino_transform, norm_min, norm_max, sino_norm_min, sino_norm_max,
                ratio=0.02, window=5, conditioning_probability=0.5, loss_type="huber",
                edge_weight=0.0):
        super().__init__(device, ratio=ratio, window=window,
                         conditioning_probability=conditioning_probability, loss_type=loss_type,
                         edge_weight=edge_weight)
        if self.loss_type == "poisson":
            raise ValueError("loss_type='poisson' is not meaningful in sinogram space (line-integral "
                             "values are no longer Poisson-distributed) -- use mae/mse/huber")
        self.sino = sino_transform
        self.norm_min = float(norm_min)
        self.norm_max = float(norm_max)
        self.sino_norm_min = float(sino_norm_min)
        self.sino_norm_max = float(sino_norm_max)

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 1.0) * 0.5 * (self.norm_max - self.norm_min) + self.norm_min

    def _sino_norm(self, s: torch.Tensor) -> torch.Tensor:
        span = self.sino_norm_max - self.sino_norm_min
        return 2.0 * (s - self.sino_norm_min) / span - 1.0

    def compute_loss(self, instance, model):
        instance = _to_device(instance, self.device)
        central_counts = self._denorm(instance["central"].float())
        context_counts = self._denorm(instance["context"].float())
        bsz = central_counts.shape[0]

        sino_central = self._sino_norm(self.sino.forward(central_counts))  # (B,1,V,C)
        sino_context = self._sino_norm(self.sino.forward(context_counts)) if context_counts.shape[1] > 0 \
            else self.sino.forward(context_counts)  # already-empty (B,0,V,C), skip renorm of nothing

        corrupted, mask = blind_spot_corrupt(sino_central, ratio=self.ratio, window=self.window)
        present = (torch.rand(bsz, device=self.device) < self.conditioning_probability)
        present_map = present.view(bsz, 1, 1, 1)
        corrupted = corrupted * present_map.to(corrupted.dtype)
        loss_mask = torch.where(present_map, mask, torch.ones_like(mask)).float()
        weight = _edge_weight_map(sino_central, self.edge_weight) if self.edge_weight > 0 else None

        model_input = torch.cat([corrupted, sino_context], dim=1)
        timesteps = torch.zeros(bsz, device=self.device, dtype=torch.long)
        pred = model(model_input, timestep=timesteps, class_labels=present.long(), return_dict=False)[0]

        loss = _masked_mean(self.loss(pred, sino_central), loss_mask, weight)
        return loss, {"loss": loss.item()}


class BootstrapPoissonLoss(BaseLoss):
    """Second-stage self-distillation: "sharpen the baseline's own output".

    Instead of the usual multi-frame + N2V-corrupted-central conditioning, the
    input to THIS model is a single channel derived from a FROZEN,
    already-trained ``poisson_head`` baseline checkpoint's own context-only
    Gamma-Poisson belief ``(mu, var)`` for the SAME frame (the prior belief
    BEFORE combining with the real observation) -- no multi-frame context, no
    N2V masking. ``input_mode`` selects what exactly is fed in:

    * ``"mean"`` (default) -- the deterministic prior mean ``mu``.
    * ``"sample"`` -- a FRESH draw from the moment-matched Gamma(``mu``,
      ``var``) prior (:func:`sdate.tr_diffusion.nb_head.gamma_sample`), redrawn
      every call. Unlike
      ``mu``, this is never the same value twice for the same frame, which
      pushes the second-stage model to learn a mapping that's robust to the
      base model's own local uncertainty rather than trusting a fixed point
      estimate -- closer in spirit to Noise2Noise (though the input's noise
      mechanism is Gamma, not Poisson-split, so it is not a literal N2N pair).

    Either way, the regression target is the real noisy measurement ``y``,
    scored with the exact same two-head Gamma-Poisson NB-NLL as
    :class:`BaselineN2VLoss`'s ``poisson_head`` branch.

    A network cannot predict independent Poisson noise from a smoothed input,
    so minimising this NLL forces this second model to converge to a genuine
    conditional-mean estimate ``E[y | input]`` -- testing whether that
    estimate is SHARPER than the base model's own output (which was only ever
    trained against context, never scored against a real observation at the
    pixels that matter).

    The target ``y`` is drawn from an INDEPENDENT noise realisation (a fresh
    :func:`sdate.tr_diffusion.noise.add_poisson_noise` draw off
    ``instance["reference"]``, the base dataset's pre-thinning frame) rather
    than reusing ``instance["central"]`` (the draw fed to the frozen model) --
    otherwise the base model's belief would already leak this exact ``y``
    through its own corrupted-central input (present with probability
    ``conditioning_probability``, unmasked at ~98% of pixels -- see
    :mod:`sdate.tr_diffusion.n2v`), and this loss would trivially reward
    copying that leak back out instead of testing anything.

    ``direction="reverse"`` (exploratory, user's N2N-symmetry idea): swaps
    which of the two noisy views is input vs. target. Instead of (base
    model's belief -> fresh measurement), train (the REAL measurement ->
    a fresh Gamma-posterior draw off the base model's belief). The model
    input is simply ``instance["central"]`` itself (no base-model call
    needed to build it); the regression target is
    :func:`sdate.tr_diffusion.nb_head.gamma_sample` of the frozen base
    model's ``(mu, var)`` belief for the SAME frame. Since the target is a
    continuous Gamma draw of the rate, not a further dose-thinned Poisson
    count, it is scored at ``dose=1.0`` regardless of ``base_dose`` (there is
    no known extra thinning layered on top of it -- the mean-value/Bregman
    property of the Poisson/NB NLL still makes ``mu`` converge to
    ``E[target | measurement]`` regardless of the exact noise model assumed
    for ``var``). The point: at INFERENCE this direction needs no frozen
    base model or multi-frame context at all -- the trained network stands
    alone on the raw single-pixel measurement (see
    :func:`sdate.tr_diffusion.pipeline.denoise_frames_bootstrap_reverse`),
    testing whether it distilled the context-rich model's belief into a
    context-free one-shot regressor.
    """

    def __init__(self, base_model, device, norm_min, norm_max, base_dose=1.0, base_present=True,
                input_mode="mean", direction="forward",
                poisson_warmup_steps=0, poisson_beta_nll_power=0.0, poisson_eps=1e-6):
        super().__init__(["loss"])
        if input_mode not in ("mean", "sample"):
            raise ValueError("input_mode must be 'mean' or 'sample'")
        if direction not in ("forward", "reverse"):
            raise ValueError("direction must be 'forward' or 'reverse'")
        self.base_model = base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.norm_min = float(norm_min)
        self.norm_max = float(norm_max)
        self.base_dose = float(base_dose)
        self.base_present = bool(base_present)
        self.input_mode = str(input_mode)
        self.direction = str(direction)
        self.poisson_warmup_steps = int(poisson_warmup_steps)
        self.poisson_beta_nll_power = float(poisson_beta_nll_power)
        self.poisson_eps = float(poisson_eps)
        self._step = 0

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        span = self.norm_max - self.norm_min
        return (x + 1.0) * 0.5 * span + self.norm_min

    @torch.no_grad()
    def _mu_input(self, central_a, context_a, cond_channels_a):
        # Shared with inference (pipeline.denoise_frames_bootstrap) so training and
        # eval see the identical base-model belief / mean-vs-sample recipe.
        mu_raw, var_raw = bootstrap_belief(self.base_model, central_a, context_a,
                                           present=self.base_present, cond_channels=cond_channels_a,
                                           eps=self.poisson_eps)
        return bootstrap_input_from_belief(mu_raw, var_raw, self.input_mode,
                                           self.norm_min, self.norm_max, eps=self.poisson_eps)

    def compute_loss(self, instance, model):
        instance = _to_device(instance, self.device)
        central_a = instance["central"].float()
        context_a = instance["context"].float()
        cond_channels_a = instance.get("cond_channels")
        if cond_channels_a is not None:
            cond_channels_a = cond_channels_a.float()

        if self.direction == "reverse":
            model_input = central_a  # the real measurement itself -- no base-model call needed to build it
            with torch.no_grad():
                mu_raw, var_raw = bootstrap_belief(self.base_model, central_a, context_a,
                                                   present=self.base_present, cond_channels=cond_channels_a,
                                                   eps=self.poisson_eps)
                y = gamma_sample(mu_raw, var_raw, eps=self.poisson_eps)  # a rate draw, not a further-thinned count
            loss_dose = 1.0
        else:
            reference = instance["reference"].float()
            model_input = self._mu_input(central_a, context_a, cond_channels_a)
            y = add_poisson_noise(self._denorm(reference), self.base_dose)  # independent draw B
            loss_dose = self.base_dose

        bsz = model_input.shape[0]
        t = torch.zeros(bsz, device=self.device, dtype=torch.long)
        cls = torch.ones(bsz, device=self.device, dtype=torch.long)
        raw = model(model_input, timestep=t, class_labels=cls, return_dict=False)[0]
        mu, var = split_mu_var(raw, eps=self.poisson_eps)

        warm = self._step < self.poisson_warmup_steps
        self._step += 1
        if warm:
            per_pixel = poisson_nll(y, mu, eps=self.poisson_eps)
        else:
            per_pixel = nb_nll(y, mu, var, dose=loss_dose, eps=self.poisson_eps)
            if self.poisson_beta_nll_power > 0:
                per_pixel = per_pixel * beta_nll_weight(mu, var, power=self.poisson_beta_nll_power,
                                                        eps=self.poisson_eps)
        loss = per_pixel.mean()
        return loss, {"loss": loss.item()}


class RefinementLoss(BaseLoss):
    """Angular-resolution-gap experiment, "Leg 1": ``BootstrapPoissonLoss``'s
    existing self-distillation mechanism (frozen base checkpoint's own
    context-only belief -> a FRESH Gamma-posterior sample as the single-channel
    input, scored against an independent noisy draw) with ONE addition: extra
    context channels (see ``data.py``'s ``context_warped`` -- the motion-
    compensated temporal taps from the warped-context precompute, plus the
    unchanged rotation taps) concatenated alongside the sampled channel, where
    plain bootstrap self-distillation is context-free (k=0).

    Always ``input_mode="sample"`` (never the deterministic ``"mean"``) --
    confirmed with the user this is the same mechanism validated to work well
    in the earlier bootstrap self-distillation experiments, not a fixed proxy.

    The regression target ``y`` is ``instance["central"]`` -- the SINGLE real
    dose-thinned measurement we actually have for that frame, no resampling.
    This deliberately does NOT follow ``BootstrapPoissonLoss``'s "forward"
    direction (target = a FRESH ``add_poisson_noise`` draw off
    ``instance["reference"]``): that mechanism requires being able to draw
    multiple independent noisy realisations of the same underlying signal from
    a cleaner reference, which is not something a real single-shot deployment
    ever has (one measurement, period -- no oracle to redraw from). Repeatedly
    scoring against fresh redraws of ``reference`` pushes the model's ``mu``
    toward ``reference`` itself over training, a leak the user caught. This
    class does not worry about the OTHER leak that motivated that choice (the
    frozen base model's belief already having seen ~98% of ``central``
    directly via its own 2%-ratio blind-spot corruption) -- per explicit user
    direction, no masking / no N2V here; the loss is plain full-frame NB-NLL,
    exactly the standard baseline's ``poisson_head`` branch.
    """

    def __init__(self, base_model, device, norm_min, norm_max, base_dose=1.0, base_present=True,
                poisson_warmup_steps=0, poisson_beta_nll_power=0.0, poisson_eps=1e-6):
        super().__init__(["loss"])
        self.base_model = base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.norm_min = float(norm_min)
        self.norm_max = float(norm_max)
        self.base_dose = float(base_dose)
        self.base_present = bool(base_present)
        self.poisson_warmup_steps = int(poisson_warmup_steps)
        self.poisson_beta_nll_power = float(poisson_beta_nll_power)
        self.poisson_eps = float(poisson_eps)
        self._step = 0

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        span = self.norm_max - self.norm_min
        return (x + 1.0) * 0.5 * span + self.norm_min

    def compute_loss(self, instance, model):
        instance = _to_device(instance, self.device)
        central_a = instance["central"].float()
        context_a = instance["context"].float()  # RAW/unwarped -- what the base model expects
        cond_channels_a = instance.get("cond_channels")
        if cond_channels_a is not None:
            cond_channels_a = cond_channels_a.float()

        with torch.no_grad():
            mu_raw, var_raw = bootstrap_belief(self.base_model, central_a, context_a,
                                               present=self.base_present, cond_channels=cond_channels_a,
                                               eps=self.poisson_eps)
        central_proxy = bootstrap_input_from_belief(mu_raw, var_raw, "sample",
                                                     self.norm_min, self.norm_max, eps=self.poisson_eps)
        context_warped = instance["context_warped"].float()
        model_input = torch.cat([central_proxy, context_warped], dim=1)

        y = self._denorm(central_a)  # the single real dose-thinned measurement -- no resampling

        bsz = model_input.shape[0]
        t = torch.zeros(bsz, device=self.device, dtype=torch.long)
        cls = torch.ones(bsz, device=self.device, dtype=torch.long)
        raw = model(model_input, timestep=t, class_labels=cls, return_dict=False)[0]
        mu, var = split_mu_var(raw, eps=self.poisson_eps)

        warm = self._step < self.poisson_warmup_steps
        self._step += 1
        if warm:
            per_pixel = poisson_nll(y, mu, eps=self.poisson_eps)
        else:
            per_pixel = nb_nll(y, mu, var, dose=self.base_dose, eps=self.poisson_eps)
            if self.poisson_beta_nll_power > 0:
                per_pixel = per_pixel * beta_nll_weight(mu, var, power=self.poisson_beta_nll_power,
                                                        eps=self.poisson_eps)
        loss = per_pixel.mean()
        return loss, {"loss": loss.item()}
