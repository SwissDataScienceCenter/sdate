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


def _make_loss(loss_type: str) -> nn.Module:
    lt = loss_type.lower()
    if lt == "mae":
        return nn.L1Loss(reduction="none")
    if lt == "mse":
        return nn.MSELoss(reduction="none")
    if lt == "huber":
        return nn.HuberLoss(reduction="none")
    raise ValueError("loss_type must be one of: mae, mse, huber")


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
    """Single-pass regression baseline: predict denoised ``x_0`` directly."""

    def compute_loss(self, instance, model):
        x0, context, corrupted, present, loss_mask, weight = self._prepare(instance)
        bsz = x0.shape[0]

        parts = [corrupted, context]
        if "cond_channels" in instance:
            parts.append(instance["cond_channels"].to(self.device, non_blocking=True).float())
        model_input = torch.cat(parts, dim=1)
        timesteps = torch.zeros(bsz, device=self.device, dtype=torch.long)
        x0_pred = model(model_input, timestep=timesteps, class_labels=present, return_dict=False)[0]

        loss = _masked_mean(self.loss(x0_pred, x0), loss_mask, weight)
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
