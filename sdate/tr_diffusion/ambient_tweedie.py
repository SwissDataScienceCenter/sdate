"""Ambient-Tweedie diffusion: train a denoiser that is exact at every noise
level, including BELOW the one noise level ever observed in training.

Adapts "Consistent Diffusion Meets Tweedie" (Daras, Dimakis, Daskalakis,
arXiv:2404.10177) to this project's setting. See
``/root/.claude/plans/radiant-singing-scott.md`` for the full derivation and
design rationale; the summary:

The paper assumes a single noisy observation ``x_tn = x_0 + sigma_tn * z``
with a GLOBAL SCALAR ``sigma_tn``. Here the one observation is
``poisson_head``'s own Bayesian posterior over the clean rate: mean ``mu_hat``
(``clean_target``) and posterior variance ``var_post`` -- approximated as
Gaussian, but with a per-pixel (HETEROSCEDASTIC) ``sigma_tn(x)``, not a
scalar. This module generalises the paper's double-Tweedie identity and
consistency loss to that per-pixel case (re-derived from first principles,
cross-checked against the paper's stated result).

Noise convention is VARIANCE-EXPLODING (VE, ``x_t = y + sigma_t * eta``,
no signal attenuation), deliberately separate from the VP/DDPM convention
the rest of this package's diffusion models use -- ``y`` has coefficient 1
on ``x_0``, which only fits a VE trajectory. The network still parameterises
``h_theta(x_t, t) = E[x_0 | x_t]`` directly (x0-prediction, "sample"
convention, matching ``n2n_prediction_type="sample"`` elsewhere).

Two loss regimes, split at a fixed global floor ``sigma_tn_eff`` (a
conservative upper bound on the per-pixel ``sigma_tn(x)`` map, so the ADSM
decomposition below is valid everywhere without per-pixel masking):

* ``sigma_t > sigma_tn_eff`` -- **ADSM** (Ambient Denoising Score Matching):
  supervised directly against the real observation ``y``.
* ``sigma_t <= sigma_tn_eff`` -- **consistency**: no direct label; anchors to
  the ADSM estimate ``pred = h_theta(x_t, sigma_t)`` ITSELF (at the genuinely-
  supervised ``sigma_t`` just drawn above), renoises it down to a fresh
  ``sigma_t' < sigma_tn_eff``, and regresses ``h_theta(x_t', sigma_t')`` back
  toward that anchor.
  An EARLIER version instead bootstrapped ``x_t'`` by chaining the model's own
  no-grad predictions down from the ``sigma_tn_eff`` BOUNDARY (with an EMA
  target network on the far side, in the spirit of Consistency Models, Song
  et al. 2023) -- confirmed via a sigma-level sweep of raw ``h_theta`` outputs
  to collapse to a flat, structureless constant. Root cause: the ADSM
  residual's ``coef_h = (sigma_t^2-sigma_tn^2)/sigma_t^2`` vanishes exactly at
  ``sigma_t == sigma_tn_eff``, so that boundary point receives ZERO gradient
  from ADSM -- and since the old consistency loss bootstrapped its ENTIRE
  chain starting from that unsupervised point (never touching a genuinely
  ADSM-trained ``sigma_t > sigma_tn_eff`` state), it had no anchor to the
  region that actually learns anything, and settled on a trivial constant
  that satisfies self-consistency for free. Anchoring directly to ``pred``
  (this version) ties every below-floor training signal to a point that IS
  supervised, removing that gap entirely -- see ``AmbientTweedieLoss.compute_loss``.
  **``adsm_sigma_t_max`` (new knob, default = ``sigma_max``, i.e. the original
  full-range behaviour)**: caps how far above ``sigma_tn_eff`` the ADSM branch's
  OWN ``sigma_t`` (and hence the anchor's source noise level) is drawn from. In a
  linear-Gaussian/Tweedie proxy, the optimal below-floor amplification implied by
  the ``n2n`` leg (regressing toward ``x_t`` instead of ``anchor``) scales like
  ``1/kappa(sigma_t) = (sigma_data^2+sigma_t^2)/sigma_data^2`` -- i.e. it GROWS
  the further above ``sigma_tn_eff`` the anchor's ``sigma_t`` happens to have
  been drawn from, and ``pred_t_prime`` has NO way to observe which ``sigma_t``
  that was (not part of its own input). With the full ``[sigma_tn_eff,
  sigma_max]~=[0.185,2.0]`` range and ``sigma_data~=0.49``, this required gain
  ranges from ``~1.14x`` near the boundary to ``~17.5x`` near ``sigma_max`` --
  forcing the network to learn some single compromise averaged over wildly
  different implied amplifications, which manifests as speckle below the floor
  (confirmed as the ``n2n``-only ablation's failure mode). Narrowing
  ``adsm_sigma_t_max`` to e.g. ``1.5*sigma_tn_eff`` bounds this mismatch tightly
  while keeping ``coef_h`` comfortably nonzero (``~0.56`` at the far end, vs. the
  boundary's own ``coef_h=0`` -- see failure (4) below for why sampling
  literally AT ``sigma_tn_eff`` alone would recreate the dead-zone collapse).
  ``sample_below_floor`` never evaluates the network above ``sigma_tn_eff``
  anyway, so this costs nothing at inference -- it only trades off against how
  much nonzero-``coef_h`` ADSM gradient reaches ``pred`` near the boundary.

  **``n2c`` ALSO observed to fail, differently, at full training (2026-08-21)**:
  the ablation's mid-training snapshot (epoch ~26/30) looked like a clean
  winner (+11dB), but the completed checkpoint instead shows a mode collapse --
  below-floor predictions lose all sharpness (blurred/flat), visually much
  worse than even ``n2n``'s AT-the-floor behaviour (which stayed excellent
  right up to the boundary, per the same ablation -- ``n2n``'s failure is
  specifically a below-floor speckle, not a boundary problem). Root cause not
  yet nailed down (candidate: continued training under a numerically-dominant,
  wide-range ADSM loss progressively pulling the shared weights' low-sigma
  behaviour toward the same aggressive shrinkage/blur the ADSM branch correctly
  learns at high sigma -- a shared-weight interference effect in the OTHER
  direction from what made ``n2c`` look good early). Not yet fixed; the
  ``adsm_sigma_t_max`` narrowing above is being tried on ``n2n`` first, since
  ``n2n``'s failure mode is the better-understood one and its above/at-floor
  quality was already the best observed.

  Three consistency-branch variants, selected via
  ``AmbientTweedieLoss(..., consistency_legs=...)`` (``"both"`` default =
  n2c+n2n summed, or ``"n2c"``/``"n2n"``/``"dual"`` alone). n2c/n2n both compare
  against the SAME ``h_theta(x_t', sigma_t')``:
  ``n2c`` (regress toward the anchor itself, standard denoising-score-matching
  style) and ``n2n`` (regress toward ``x_t`` instead, the Noise2Noise-flavoured
  alternative -- an unbiased noisy realization is, in expectation, as good a
  regression target as the clean one). ``dual`` instead draws a SECOND,
  independent above-floor anchor and compares two INDEPENDENT below-floor
  predictions to EACH OTHER (no stop-gradient on either side) -- mutual
  consistency among below-floor estimates, without anchoring either one
  directly to an above-floor value. n2c/n2n are ALWAYS computed and logged
  (free, shared forward pass); ``dual`` needs 2 extra forward passes so it's
  only computed when selected. Run each of n2c/n2n/dual as a fully SEPARATE
  training job (independent models, not a shared multi-head network) to
  isolate which is responsible for an artifact seen in a ``"both"`` run, since
  gradient magnitude alone does not reliably predict this
  (confirmed empirically: despite ``n2n``'s loss value being ~34x larger than
  ``n2c``'s on a converged "both" checkpoint, their gradient NORMS were
  comparable -- ``n2n``'s loss is dominated by ``x_t``'s own irreducible
  injected-noise variance, which isn't all gradient-reducible).

**The network's input is ``x_t`` ALONE (plus ``t``) -- no ``y``, no ``context``,
no ``sigma_tn`` channel.** Two earlier versions of this module fed the network
extra side-channel conditioning (``y`` directly, then just multi-frame
``context``) and both let it learn a shortcut: a function nearly constant in
``(x_t, sigma)`` that reads off the side channel instead trivially satisfies
BOTH loss terms -- ADSM, because the side channel already approximates
``E[x0|x_t]`` reasonably at high sigma, and consistency, because a function
constant in ``(x_t, sigma)`` is trivially self-consistent across every noise
level. Confirmed empirically both times: reverse-sampling trajectories that
should differ came back pixel-identical (``y``-conditioned version) or nearly
so (``context``-conditioned version, measured via a seed-sensitivity
ablation). Matching the paper's literal ``h_theta(x_t, t)`` signature removes
every shortcut at the source. ``y`` and ``sigma_tn`` remain essential to the
LOSS (regression target / forward-process noise scale) and to inference-time
DPS guidance (below) -- they are just never concatenated into the network's
own input. See ``condition_on_measurement=False`` and ``k=0`` (no context
taps) in :func:`sdate.tr_diffusion.model.create_diffusion_unet` /
``sdate.tr_diffusion.train``.

**Getting to a genuinely homoscedastic, scalar ``sigma_tn``** (matching the
paper exactly, rather than generalising its math to a per-pixel map): apply
the Anscombe transform (:func:`sdate.tr_diffusion.noise.anscombe_transform`)
to the RAW dose-thinned measurement before any of this -- for Poisson counts,
``z = 2*sqrt(counts + 3/8)`` has approximately UNIT, GLOBALLY CONSTANT
variance regardless of the local count rate. ``sigma_tn_map`` below is still
named/shaped as a per-pixel tensor for backward compatibility with the
broadcasting arithmetic, but in the Anscombe setting it is simply a constant
filled everywhere with the same calibrated scalar -- see
``data.py``'s ``anscombe=True`` path. Sampling ends with
:func:`sdate.tr_diffusion.noise.inverse_anscombe` to get back to counts.

**DPS (Diffusion Posterior Sampling) guidance.** Since ``y``'s Gaussian
approximation gives an EXACT, known likelihood ``p(y|x_0) = N(y; x_0,
sigma_tn_map^2)``, both :func:`sample_below_floor` and
:func:`sample_below_floor_posterior` can additionally pull each reverse step
toward matching that measurement (Chung et al. 2022, adapted here with an
identity measurement operator, and GENERALISED from their one gradient step
per diffusion iteration to a configurable few via ``dps_steps``): at each
rung, holding the noise level fixed, repeatedly (a) backprop the data-fit
term ``||y - h_theta(x,t)||^2 / sigma_tn_map^2`` through the network to get
its gradient w.r.t. the current state ``x``, and (b) take a normalised
gradient-descent step of magnitude ``dps_scale`` in that direction -- a short
inner optimisation pushing ``x`` toward the MLE of the measurement
(``h_theta(x)=y``), re-evaluating the network's own prior at every inner step
rather than linearising once. ``dps_scale=0`` (default) skips this entirely,
at zero extra forward/backward cost. See :func:`_dps_refine`, shared by both
samplers.
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from pytorch_base.base_loss import BaseLoss


def sigma_to_temb(sigma: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    """Map a continuous VE noise level to a timestep-embedding-friendly scalar.

    ``UNet2DModel``'s built-in sinusoidal time embedding accepts any real
    input, but stays best-conditioned near the range its frequencies were
    designed for (~[0, 1000)) -- so we linearly rescale log(sigma) into
    [0, 999]. This mapping is deterministic and shared between training and
    inference (:func:`sample_below_floor`) -- they MUST agree, since the
    network only ever sees timestep values produced by this function.
    """
    log_lo, log_hi = math.log(sigma_min), math.log(sigma_max)
    frac = (torch.log(sigma.clamp_min(sigma_min)) - log_lo) / (log_hi - log_lo)
    return frac.clamp(0.0, 1.0) * 999.0


def sample_log_uniform(n: int, lo: float, hi: float, device, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    u = torch.rand(n, device=device, generator=generator)
    return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))


def geometric_ladder(lo: float, hi: float, n_rungs: int, device) -> torch.Tensor:
    """``n_rungs + 1`` log-spaced values from ``lo`` to ``hi`` (ascending), rung 0 = lo, rung n_rungs = hi."""
    return torch.exp(torch.linspace(math.log(lo), math.log(hi), n_rungs + 1, device=device))


def sample_log_uniform_elementwise(lo: float, hi: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Like :func:`sample_log_uniform`, but with a PER-ELEMENT upper bound (a tensor, e.g.
    the per-example sigma_t already drawn for the ADSM branch) instead of one shared scalar."""
    u = torch.rand(hi.shape, device=hi.device, generator=generator)
    return torch.exp(math.log(lo) + u * (torch.log(hi) - math.log(lo)))


def _model_input(x_t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    # context is an empty (0-channel) tensor when k=0 (the default for this module,
    # see the docstring) -- concatenating it is then a no-op, leaving x_t alone.
    return torch.cat([x_t, context], dim=1)


def _predict(model, x_t: torch.Tensor, sigma: torch.Tensor, context: torch.Tensor,
            sigma_min: float, sigma_max: float) -> torch.Tensor:
    """One ``h_theta(x_t, t) -> x0-estimate`` forward pass. ``sigma`` is a scalar (shared across the batch).

    Deliberately does NOT take ``y`` or ``sigma_tn_map`` -- see the module docstring."""
    b = x_t.shape[0]
    temb = sigma_to_temb(sigma, sigma_min, sigma_max).expand(b)
    class_labels = torch.ones(b, device=x_t.device, dtype=torch.long)  # no auxiliary label needed here; constant.
    return model(_model_input(x_t, context), timestep=temb,
                 class_labels=class_labels, return_dict=False)[0]


def _dps_refine(model, x: torch.Tensor, y: torch.Tensor, sigma_tn_map: torch.Tensor,
                context: torch.Tensor, sigma_cur: torch.Tensor, sigma_min: float, sigma_max: float,
                dps_scale: float, dps_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run ``dps_steps`` DPS gradient-descent steps on ``x`` at the FIXED noise level
    ``sigma_cur``, each pulling it toward the MLE of the known measurement likelihood
    ``N(y; h_theta(x,sigma_cur), sigma_tn_map^2)`` (Chung et al. 2022's guidance term,
    generalised here from their single correction step per diffusion iteration to a
    configurable few -- see the module docstring's DPS section). The network is
    RE-EVALUATED from the refined ``x`` every inner step (not just linearised once),
    so this is a genuine short inner optimisation, constrained to stay near the
    model's own prior manifold since ``h_theta`` still has to explain whatever ``x``
    becomes. Each step's update direction is the data-fit gradient normalised by its
    own norm (a fixed per-step "distance" ``dps_scale``, since the residual's
    magnitude varies a lot across pixels/rungs), matching the existing single-step
    convention in this module.

    Returns ``(x, pred)``, both DETACHED -- ``pred`` is a fresh forward pass at the
    final ``x`` (one extra no-grad forward beyond the ``dps_steps`` grad-tracked ones),
    so callers can feed it directly into a renoise step without worrying it was
    computed from a stale, pre-correction ``x``.
    """
    for _ in range(dps_steps):
        x = x.detach().requires_grad_(True)
        pred = _predict(model, x, sigma_cur, context, sigma_min, sigma_max)
        data_fit = 0.5 * ((pred - y) ** 2 / sigma_tn_map.clamp_min(1e-6) ** 2).sum()
        (grad,) = torch.autograd.grad(data_fit, x)
        grad_norm = grad.flatten(1).norm(dim=1).clamp_min(1e-8).view(-1, 1, 1, 1)
        with torch.no_grad():
            x = x - (dps_scale / grad_norm) * grad
    x = x.detach()
    with torch.no_grad():
        pred = _predict(model, x, sigma_cur, context, sigma_min, sigma_max)
    return x, pred


class AmbientTweedieLoss(BaseLoss):
    """``n_rungs``/``ema_decay`` are accepted (train.py already passes them, and n_rungs
    is separately needed for :func:`sample_below_floor` at inference) but are NOT used by
    this loss anymore -- see ``compute_loss``'s consistency branch docstring for why the
    ladder-bootstrap + EMA-target design they used to configure was replaced."""

    def __init__(self, device, sigma_tn_eff: float, sigma_min: float = 0.002, sigma_max: float = 2.0,
                n_rungs: int = 40, consistency_weight: float = 1.0, ema_decay: float = 0.999,
                consistency_legs: str = "both", adsm_sigma_t_max: Optional[float] = None):
        super().__init__(["loss", "loss_adsm", "loss_consistency",
                          "loss_consistency_n2c", "loss_consistency_n2n", "loss_consistency_dual"])
        self.device = device
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_tn_eff = float(sigma_tn_eff)
        assert self.sigma_min < self.sigma_tn_eff < self.sigma_max
        # Upper bound for the ADSM branch's OWN sigma_t sampling (and hence the anchor's
        # source sigma) -- defaults to sigma_max (the original, full-range behaviour).
        # Set below sigma_max to cap how far above sigma_tn_eff the anchor can be drawn
        # from: n2n's consistency target regresses toward x_t, whose required
        # amplification RELATIVE TO the anchor grows (roughly like 1/kappa(sigma_t) in a
        # linear-Gaussian proxy) the further sigma_t sits above sigma_tn_eff -- since
        # sigma_t is NOT part of pred_t_prime's own input, a wide range forces it to learn
        # some compromise gain averaged over wildly different implied amplifications,
        # which manifests as speckle below the floor (see project memory
        # project-tr-diffusion-ambient-tweedie for the derivation). sample_below_floor
        # never evaluates the network above sigma_tn_eff anyway (it only walks the ladder
        # DOWN from there), so narrowing this has no inference-time cost -- it only
        # trades off against how much nonzero-coef_h ADSM gradient signal reaches `pred`
        # near the boundary (coef_h -> 0 exactly AT sigma_tn_eff, so don't set this equal
        # to sigma_tn_eff itself, see failure mode (4) in the module docstring).
        self.adsm_sigma_t_max = float(adsm_sigma_t_max) if adsm_sigma_t_max is not None else self.sigma_max
        assert self.sigma_tn_eff < self.adsm_sigma_t_max <= self.sigma_max
        self.consistency_weight = float(consistency_weight)
        assert consistency_legs in ("both", "n2c", "n2n", "dual"), consistency_legs
        # Which leg(s) of the consistency branch actually drive `loss` (see
        # compute_loss's docstring for what each leg means). n2c/n2n are ALWAYS
        # computed and logged regardless of this setting -- cheap, since they
        # share the same pred_t_prime forward pass -- so an n2c-only or n2n-only run
        # still reports the other leg's value for direct comparison against a
        # separate run trained with the other leg (or both), e.g. to isolate which
        # leg is responsible for an artifact seen in a `consistency_legs="both"` run.
        self.consistency_legs = consistency_legs

    def _prep(self, instance) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = self.device
        y = instance["clean_target"].to(dev, non_blocking=True).float()
        sigma_tn_map = instance["sigma_tn_map"].to(dev, non_blocking=True).float()
        context = instance["context"].to(dev, non_blocking=True).float()
        return y, sigma_tn_map, context

    def _adsm_x_t(self, y, sigma_tn_map, sigma_t, generator=None) -> torch.Tensor:
        var_diff = (sigma_t.view(-1, 1, 1, 1) ** 2 - sigma_tn_map ** 2).clamp_min(0.0)
        eta = torch.randn(y.shape, device=y.device, generator=generator)
        return y + var_diff.sqrt() * eta

    def compute_loss(self, instance, model) -> Tuple[torch.Tensor, Dict[str, float]]:
        y, sigma_tn_map, context = self._prep(instance)
        bsz = y.shape[0]

        # --- ADSM branch (sigma_t in [sigma_tn_eff, adsm_sigma_t_max], one draw per example) ---
        sigma_t = sample_log_uniform(bsz, self.sigma_tn_eff, self.adsm_sigma_t_max, y.device)
        x_t = self._adsm_x_t(y, sigma_tn_map, sigma_t)
        pred = _predict(model, x_t, sigma_t, context, self.sigma_min, self.sigma_max)
        sigma_t_sq = (sigma_t.view(-1, 1, 1, 1) ** 2)
        coef_h = (sigma_t_sq - sigma_tn_map ** 2).clamp_min(0.0) / sigma_t_sq
        coef_xt = (sigma_tn_map ** 2) / sigma_t_sq
        resid = coef_h * pred + coef_xt * x_t - y
        loss_adsm = resid.pow(2).mean()

        # --- consistency branch: anchor to the ADSM estimate ITSELF (`pred`, at a
        # genuinely-supervised sigma_t > sigma_tn_eff -- confirmed by direct inspection to
        # already produce a reasonable blurred estimate), then renoise DOWN to a fresh
        # sigma_t_prime < sigma_tn_eff and compare.
        #
        # This replaces an earlier design that bootstrapped x_t_prime by chaining the
        # model's OWN no-grad predictions down from the sigma_tn_eff boundary, using an
        # EMA target network on the far side -- confirmed (via a sigma-level sweep of raw
        # h_theta outputs) to collapse to a flat, structureless constant: the ADSM loss's
        # gradient w.r.t. `pred` vanishes exactly at sigma_t == sigma_tn_eff (coef_h -> 0),
        # so that boundary point -- and everything bootstrapped from it -- received
        # essentially NO training signal from either loss term, and settled on whatever
        # trivial function happened to satisfy the (nearly vacuous) self-consistency
        # objective. Anchoring directly to `pred` fixes this: the below-floor target is
        # now always tied to a point that IS genuinely trained by ADSM.
        #
        # Two legs against the SAME h_theta(x_t_prime, sigma_t_prime) forward pass:
        #   n2c ("noise2clean"): regress toward the anchor itself (`pred`, detached) --
        #     treats the ADSM estimate as a pseudo-clean target, the direct analogue of
        #     standard denoising score matching against a (noisy) label.
        #   n2n ("noise2noise"): regress toward `x_t` instead -- a real but noisier
        #     alternative view, in the spirit of Noise2Noise (regressing toward any
        #     unbiased noisy realization is equivalent in expectation to regressing
        #     toward the clean signal). Both terms are logged separately to compare.
        anchor = pred.detach()
        sigma_t_prime = sample_log_uniform(bsz, self.sigma_min, self.sigma_tn_eff, y.device)
        x_t_prime = anchor + sigma_t_prime.view(-1, 1, 1, 1) * torch.randn_like(anchor)
        pred_t_prime = _predict(model, x_t_prime, sigma_t_prime, context, self.sigma_min, self.sigma_max)

        loss_consistency_n2c = (pred_t_prime - anchor).pow(2).mean()
        loss_consistency_n2n = (pred_t_prime - x_t).pow(2).mean()

        # Third leg ("dual"): a SECOND, fully independent above-floor draw (its own
        # sigma_t_b, its own noise) gives a second ADSM anchor `pred_b`, renoised down to
        # its own independent sigma_t_prime_b < sigma_tn_eff. The loss compares the two
        # resulting below-floor predictions to EACH OTHER -- ||h(x_t',t') - h(x_t'',t'')||^2,
        # no stop-gradient on either side -- rather than anchoring either one to a specific
        # above-floor value. This enforces mutual consistency AMONG below-floor estimates
        # without requiring them to individually match "clean"/"noisy" above-floor targets
        # the way n2c/n2n do. Unlike n2c/n2n (free -- they reuse the ADSM branch's own
        # `pred`/`x_t`/`sigma_t`), this needs 2 extra forward passes, so it's only computed
        # when actually selected (not logged-for-free like n2c/n2n).
        if self.consistency_legs == "dual":
            sigma_t_b = sample_log_uniform(bsz, self.sigma_tn_eff, self.adsm_sigma_t_max, y.device)
            x_t_b = self._adsm_x_t(y, sigma_tn_map, sigma_t_b)
            with torch.no_grad():
                anchor_b = _predict(model, x_t_b, sigma_t_b, context, self.sigma_min, self.sigma_max)
            sigma_t_prime_b = sample_log_uniform(bsz, self.sigma_min, self.sigma_tn_eff, y.device)
            x_t_prime_b = anchor_b + sigma_t_prime_b.view(-1, 1, 1, 1) * torch.randn_like(anchor_b)
            pred_t_prime_b = _predict(model, x_t_prime_b, sigma_t_prime_b, context, self.sigma_min, self.sigma_max)
            loss_consistency_dual = (pred_t_prime - pred_t_prime_b).pow(2).mean()
        else:
            loss_consistency_dual = torch.zeros((), device=y.device)

        if self.consistency_legs == "both":
            loss_consistency = loss_consistency_n2c + loss_consistency_n2n
        elif self.consistency_legs == "n2c":
            loss_consistency = loss_consistency_n2c
        elif self.consistency_legs == "n2n":
            loss_consistency = loss_consistency_n2n
        else:
            loss_consistency = loss_consistency_dual

        loss = loss_adsm + self.consistency_weight * loss_consistency
        return loss, {"loss": loss.item(), "loss_adsm": loss_adsm.item(),
                      "loss_consistency": loss_consistency.item(),
                      "loss_consistency_n2c": loss_consistency_n2c.item(),
                      "loss_consistency_n2n": loss_consistency_n2n.item(),
                      "loss_consistency_dual": loss_consistency_dual.item()}


def sample_below_floor(model, y: torch.Tensor, sigma_tn_map: torch.Tensor, context: torch.Tensor,
                       sigma_min: float, sigma_max: float, sigma_tn_eff: float,
                       n_rungs: int = 40, eta: float = 0.0, dps_scale: float = 0.0,
                       dps_steps: int = 1,
                       chunk_size: Optional[int] = None,
                       generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Jump-and-renoise sampler from ``sigma_tn_eff`` down to ``sigma_min`` (i.e. BELOW
    the observed noise floor) -- the actual "below-floor sampling" this whole module
    exists to make valid.

    The boundary state at ``sigma_tn_eff`` is constructed DIRECTLY via the analytic
    ADSM forward formula (:meth:`AmbientTweedieLoss._adsm_x_t`, same call the training
    loop's consistency-branch bootstrap uses to seed its own chain) rather than by
    iteratively reverse-sampling down from ``sigma_max`` -- the model is never trained
    to self-compose across multiple steps ABOVE sigma_tn_eff (the ADSM loss only
    supervises independent single forward evaluations there), so treating that regime
    as an iterative sampler silently exercises an unsupported code path and compounds
    error before ever reaching the part of the model that's actually trained to be
    self-consistent. ``sigma_max`` is kept as a parameter only because it's needed for
    the shared ``sigma_to_temb`` rescaling, matching training exactly.

    Mirrors this project's existing hand-rolled steppers (``pipeline.partial_diffusion``/
    ``partial_diffusion_n2n``) but with VE sigma algebra instead of VP alpha_bar algebra.
    ``eta > 0`` adds stochasticity (re-noise slightly above the target rung before the
    next jump); ``eta=0`` (default) is the deterministic ODE-like sampler.

    ``dps_scale > 0`` additionally applies Diffusion Posterior Sampling guidance
    (Chung et al. 2022) at every step, pulling the sample toward the known
    measurement ``y`` (Gaussian likelihood ``N(y; x_0, sigma_tn_map^2)``, identity
    measurement operator): backprop the data-fit term through the network to get
    its gradient w.r.t. the current ``x_t``, normalise by the gradient's own norm
    (a fixed per-step "distance" rather than an unscaled step -- residual magnitude
    varies a lot across pixels/steps), and subtract ``dps_scale`` times that from
    the freshly-sampled next state. ``dps_scale=0`` (default) reduces to the plain
    sampler above with no extra network forward/backward cost.

    ``chunk_size``: split the batch and run each sub-batch through the full sampler
    sequentially. The DPS path keeps every step's forward activations for backprop
    (unlike the no-grad plain path), which can OOM a large batch at once; pass e.g.
    16-32 here for a big eval batch. ``None`` (default) runs the whole batch at once.
    """
    if chunk_size is not None and y.shape[0] > chunk_size:
        outs = [
            sample_below_floor(model, y[s:s + chunk_size], sigma_tn_map[s:s + chunk_size], context[s:s + chunk_size],
                               sigma_min, sigma_max, sigma_tn_eff, n_rungs=n_rungs, eta=eta, dps_scale=dps_scale,
                               dps_steps=dps_steps, generator=generator)
            for s in range(0, y.shape[0], chunk_size)
        ]
        return torch.cat(outs, dim=0)

    device = y.device
    ladder = geometric_ladder(sigma_min, sigma_tn_eff, n_rungs, device)  # ascending, sigma_min .. sigma_tn_eff
    bsz = y.shape[0]

    with torch.no_grad():
        var_diff = (torch.as_tensor(sigma_tn_eff, device=device) ** 2 - sigma_tn_map ** 2).clamp_min(0.0)
        x = y + var_diff.sqrt() * torch.randn(y.shape, device=device, generator=generator)

    for m in range(len(ladder) - 1, 0, -1):
        sigma_cur, sigma_nxt = ladder[m], ladder[m - 1]
        sigma_step = sigma_nxt + eta * (sigma_cur - sigma_nxt)
        if dps_scale > 0 and dps_steps > 0:
            x, pred = _dps_refine(model, x, y, sigma_tn_map, context, sigma_cur.expand(bsz),
                                  sigma_min, sigma_max, dps_scale, dps_steps)
        else:
            with torch.no_grad():
                pred = _predict(model, x, sigma_cur.expand(bsz), context, sigma_min, sigma_max)
        with torch.no_grad():
            x = pred + sigma_step * torch.randn(y.shape, device=device, generator=generator)
    return x.detach()


def ve_posterior_step(pred: torch.Tensor, x_cur: torch.Tensor, sigma_cur: torch.Tensor,
                      sigma_next: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Proper VE ancestral/posterior renoise step, REUSING the implied noise direction
    instead of adding fully independent fresh noise -- the mechanism every standard
    diffusion sampler (DDPM/DDIM/EDM) actually uses, and which this project's original
    ``x = pred + sigma_next*randn(...)`` renoise step was missing (confirmed by comparing
    against the official "Consistent Diffusion Meets Tweedie" implementation, see project
    memory project-tr-diffusion-ambient-tweedie for the full derivation and cross-check).

    Derivation: in the VE forward process (``x_t = x_0 + sigma_t*eta``), the conditional
    law of ``x_s | x_t, x_0`` for ``s<t`` is the exact Brownian-bridge conditional (pin a
    Brownian path at 0 and at ``t``, read off its value at ``s``):
        mean = x_0 + (sigma_s^2/sigma_t^2) * (x_t - x_0)
        var  = sigma_s^2 * (1 - sigma_s^2/sigma_t^2)
    Substituting our point estimate ``pred`` for the unknown ``x_0``, and writing the
    implied noise direction ``eta_pred = (x_cur - pred) / sigma_cur`` (recovered from our
    x0-parameterised network's own output -- we have no separate epsilon head), this
    becomes:
        x_next = pred + (sigma_next^2 / sigma_cur) * eta_pred
                      + sigma_next * sqrt(1 - (sigma_next/sigma_cur)^2) * z,   z ~ N(0,I) fresh
    Checked at both limits: sigma_next -> sigma_cur gives back x_cur exactly (identity);
    sigma_next -> 0 gives exactly pred (fully denoised). The MEAN term above matches
    ``ambient_utils.loss.from_x0_pred_to_xnature_pred_ve_to_ve`` in the paper's own
    released code (https://github.com/giannisdaras/ambient_utils) exactly, independently
    re-derived here before that cross-check was made.

    NUMERICS: an earlier version computed this via an intermediate
    ``eta_pred = (x_cur-pred)/sigma_cur`` and then multiplied back by ``sigma_cur`` --
    mathematically cancels exactly, but under fp16 mixed precision this divide-then-
    multiply round trip let a transient large ``(x_cur-pred)`` blow up through a
    division by ``sigma_cur`` (bounded below by ``sigma_tn_eff``, not itself tiny, but
    still small enough to amplify), which is the confirmed cause of a real NaN
    divergence during training on 2026-08-23 (loss_consistency jumped 60x one epoch
    before the run went NaN two epochs later -- see project memory). Fixed by (a)
    algebraically cancelling the division out of the formula entirely
    (``ratio_sq*(x_cur-pred)`` directly, no division anywhere in this function), and
    (b) computing in float32 explicitly regardless of the ambient autocast context,
    casting back to the input dtype only at the very end.
    """
    orig_dtype = pred.dtype
    pred = pred.float()
    x_cur = x_cur.float()
    sigma_cur_b = sigma_cur.float().view(-1, 1, 1, 1)
    sigma_next_b = sigma_next.float().view(-1, 1, 1, 1)
    ratio_sq = (sigma_next_b / sigma_cur_b) ** 2
    mean = pred + ratio_sq * (x_cur - pred)
    fresh_std = sigma_next_b * (1.0 - ratio_sq).clamp_min(0.0).sqrt()
    z = torch.randn(pred.shape, device=pred.device, generator=generator)
    return (mean + fresh_std * z).to(orig_dtype)


def sample_below_floor_posterior(model, y: torch.Tensor, sigma_tn_map: torch.Tensor, context: torch.Tensor,
                                 sigma_min: float, sigma_max: float, sigma_tn_eff: float,
                                 n_rungs: int = 40, dps_scale: float = 0.0, dps_steps: int = 3,
                                 chunk_size: Optional[int] = None,
                                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Same ladder/boundary-construction as :func:`sample_below_floor`, but using
    :func:`ve_posterior_step` for every renoise transition instead of naive independent
    fresh noise. Pure inference-time change -- usable directly on a checkpoint trained
    with the OLD renoise step, no retraining needed to test whether this alone reduces
    the blur/over-smoothing seen in every below-floor result so far.

    ``dps_scale > 0`` additionally runs ``dps_steps`` DPS gradient-refinement steps
    (see :func:`_dps_refine` / the module docstring's DPS section) at EVERY rung,
    fixing up ``x`` to better satisfy the known measurement ``y`` before handing the
    (re-evaluated) ``pred`` to :func:`ve_posterior_step`. ``dps_scale=0`` (default)
    skips this and costs nothing beyond the existing no-grad forward pass.
    """
    if chunk_size is not None and y.shape[0] > chunk_size:
        outs = [
            sample_below_floor_posterior(model, y[s:s + chunk_size], sigma_tn_map[s:s + chunk_size],
                                         context[s:s + chunk_size], sigma_min, sigma_max, sigma_tn_eff,
                                         n_rungs=n_rungs, dps_scale=dps_scale, dps_steps=dps_steps,
                                         generator=generator)
            for s in range(0, y.shape[0], chunk_size)
        ]
        return torch.cat(outs, dim=0)

    device = y.device
    ladder = geometric_ladder(sigma_min, sigma_tn_eff, n_rungs, device)
    bsz = y.shape[0]

    with torch.no_grad():
        var_diff = (torch.as_tensor(sigma_tn_eff, device=device) ** 2 - sigma_tn_map ** 2).clamp_min(0.0)
        x = y + var_diff.sqrt() * torch.randn(y.shape, device=device, generator=generator)

    for m in range(len(ladder) - 1, 0, -1):
        sigma_cur, sigma_nxt = ladder[m].expand(bsz), ladder[m - 1].expand(bsz)
        if dps_scale > 0 and dps_steps > 0:
            x, pred = _dps_refine(model, x, y, sigma_tn_map, context, sigma_cur,
                                  sigma_min, sigma_max, dps_scale, dps_steps)
        else:
            with torch.no_grad():
                pred = _predict(model, x, sigma_cur, context, sigma_min, sigma_max)
        x = ve_posterior_step(pred, x, sigma_cur, sigma_nxt, generator=generator)
    return x.detach()


def sample_below_floor_posterior_bootstrap(model, y: torch.Tensor, sigma_tn_map: torch.Tensor, context: torch.Tensor,
                                           sigma_min: float, sigma_max: float, sigma_tn_eff: float,
                                           n_rungs: int = 40, n_iters: int = 6,
                                           dps_scale: float = 0.0, dps_steps: int = 3,
                                           generator: Optional[torch.Generator] = None) -> list:
    """Self-bootstrapped refinement of :func:`sample_below_floor_posterior`: run the
    below-floor reverse chain once in full (boundary constructed from ``y`` at
    ``sigma_tn_eff``, walked all the way down to ``sigma_min``), then repeatedly treat
    the PREVIOUS round's final estimate as a fresh pseudo-clean sample, re-inject VE
    forward noise up to a rung ONE STEP LOWER than the previous round started from
    (i.e. progressively LESS added noise each round), and re-run the SAME
    (optionally DPS-guided) reverse chain from there back down to ``sigma_min``.

    This is the diffusion analogue of a resampling / "time-travel" refinement pass
    (cf. RePaint, SDEdit-style partial forward-backward loops): each round gets to
    correct the previous round's estimate using the SAME network and (if enabled)
    the SAME DPS measurement-consistency pull, starting from progressively less
    perturbation each time -- so later rounds make smaller, more local corrections
    rather than re-deriving the whole estimate from scratch.

    Returns a list of tensors, one per round (in order), each the full below-floor
    estimate at the END of that round (sigma~``sigma_min``). Round 0 is EXACTLY
    :func:`sample_below_floor_posterior`'s ordinary single-pass output (same
    ``dps_scale``/``dps_steps``). Stops early (fewer than ``n_iters`` entries
    returned) once the re-noising rung would drop below 1 -- nothing meaningful
    left to bootstrap from at that point.
    """
    device = y.device
    ladder = geometric_ladder(sigma_min, sigma_tn_eff, n_rungs, device)
    bsz = y.shape[0]

    with torch.no_grad():
        var_diff = (torch.as_tensor(sigma_tn_eff, device=device) ** 2 - sigma_tn_map ** 2).clamp_min(0.0)
        x = y + var_diff.sqrt() * torch.randn(y.shape, device=device, generator=generator)

    outputs = []
    for it in range(n_iters):
        start_rung = n_rungs - it
        if start_rung < 1:
            break
        if it > 0:
            with torch.no_grad():
                x = x + ladder[start_rung] * torch.randn(x.shape, device=device, generator=generator)
        for m in range(start_rung, 0, -1):
            sigma_cur, sigma_nxt = ladder[m].expand(bsz), ladder[m - 1].expand(bsz)
            if dps_scale > 0 and dps_steps > 0:
                x, pred = _dps_refine(model, x, y, sigma_tn_map, context, sigma_cur,
                                      sigma_min, sigma_max, dps_scale, dps_steps)
            else:
                with torch.no_grad():
                    pred = _predict(model, x, sigma_cur, context, sigma_min, sigma_max)
            x = ve_posterior_step(pred, x, sigma_cur, sigma_nxt, generator=generator)
        outputs.append(x.detach().clone())
    return outputs


class AmbientTweedieFaithfulLoss(BaseLoss):
    """Faithful port of the official "Consistent Diffusion Meets Tweedie" training
    recipe (github.com/giannisdaras/ambient-tweedie, ambient_utils package), adapted
    from their VP/epsilon-parameterised SDXL-LoRA setup to this project's VE +
    x0-parameterised UNet. Cross-checking their actual released code (not just the
    paper text) against ``AmbientTweedieLoss`` surfaced three concrete gaps -- all
    fixed here, none requiring the elaborate per-rung replay-buffer curriculum in
    ``AmbientTweedieCurriculumLoss`` (their method gets away with a single, simple
    mechanism applied over MANY steps with a SMALL weight; see project memory
    project-tr-diffusion-ambient-tweedie for the full derivation/cross-check):

    1. **Renoise step**: their ``move_one_step`` reuses the model's own predicted
       noise DIRECTION (blended with a smaller amount of fresh noise) instead of
       adding fully independent fresh noise -- the standard mechanism every real
       diffusion sampler (DDPM/DDIM/EDM) uses, and which ``AmbientTweedieLoss``'s
       ``anchor + sigma'*randn(...)`` construction (and the original
       ``sample_below_floor``) were missing. See :func:`ve_posterior_step` for the
       VE-adapted formula (independently re-derived via the Brownian-bridge
       conditional, then cross-checked to match their ``ve_to_ve`` mean helper
       exactly).
    2. **Unbiased consistency loss**: theirs is a PRODUCT of two INDEPENDENT
       posterior draws' deviations from the anchor (``(preds_prime_1-x0_pred)*
       (preds_prime_2-x0_pred)``), not one draw's deviation SQUARED. Squaring a
       single sample also penalises the model's own legitimate sampling variance
       (not just its bias), an extra spurious pressure toward an overly
       deterministic/blurry output that the product form avoids.
    3. **Consistency weight**: their production configs use ``consistency_coeff=
       0.015`` -- a light regulariser on top of the dominant ADSM signal, NOT
       equal-weighted at 1.0 as ``AmbientTweedieLoss`` used.

    Also step-count-gated warmup (``consistency_warmup_steps``, their
    ``consistency_kick_in``, default 30000 matching their production configs)
    before the consistency term contributes at all -- gated on ``model.training``
    so eval-time ``compute_loss`` calls don't advance the counter.

    Unlike the curriculum design, there is no rung-by-rung schedule and no replay
    buffer here: `sigma_s` for the consistency check is drawn log-uniformly across
    the ENTIRE ``[sigma_min, sigma_t)`` range every step (single hop, matching their
    simplest ``num_consistency_steps=1`` configuration) -- the paper's own claim is
    that THIS, done with the corrected renoise formula/loss/weight, is already
    enough; testing that claim directly on our data is the point of this class.
    """

    def __init__(self, device, sigma_tn_eff: float, sigma_min: float = 0.002, sigma_max: float = 2.0,
                consistency_weight: float = 0.015, consistency_warmup_steps: int = 30000,
                consistency_ramp_steps: int = 2000, adsm_sigma_t_max: Optional[float] = None):
        super().__init__(["loss", "loss_adsm", "loss_consistency"])
        self.device = device
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_tn_eff = float(sigma_tn_eff)
        assert self.sigma_min < self.sigma_tn_eff < self.sigma_max
        self.consistency_weight = float(consistency_weight)
        self.consistency_warmup_steps = int(consistency_warmup_steps)
        # Linearly ramp the EFFECTIVE consistency weight from 0 up to consistency_weight over
        # this many steps immediately after warmup ends, rather than switching it on at full
        # strength in one step. Added after a NaN divergence when RESUMING training from a
        # checkpoint with --faithful_consistency_warmup_steps 0 (immediate full-strength
        # activation): the fresh optimizer/LR-scheduler state (--load_checkpoint only restores
        # model weights) combined with a sudden full-weight new loss term on an already-mature
        # model caused a fast, severe divergence -- much faster and worse than the original
        # cold-start run's slower-building instability (which only showed a 60x loss_consistency
        # jump after several fully-active epochs before eventually going NaN). A ramp is the
        # standard mitigation for "newly-activated loss term destabilises training" in general;
        # this also softens the ORIGINAL cold-start activation, not just resumes.
        self.consistency_ramp_steps = int(consistency_ramp_steps)
        self.adsm_sigma_t_max = float(adsm_sigma_t_max) if adsm_sigma_t_max is not None else self.sigma_max
        assert self.sigma_tn_eff < self.adsm_sigma_t_max <= self.sigma_max
        self._step = 0

    def _prep(self, instance):
        dev = self.device
        y = instance["clean_target"].to(dev, non_blocking=True).float()
        sigma_tn_map = instance["sigma_tn_map"].to(dev, non_blocking=True).float()
        context = instance["context"].to(dev, non_blocking=True).float()
        return y, sigma_tn_map, context

    def compute_loss(self, instance, model) -> Tuple[torch.Tensor, Dict[str, float]]:
        y, sigma_tn_map, context = self._prep(instance)
        bsz = y.shape[0]
        if model.training:
            self._step += 1

        # --- ADSM branch (unchanged -- already matches the paper's own ve_to_ve mean formula) ---
        sigma_t = sample_log_uniform(bsz, self.sigma_tn_eff, self.adsm_sigma_t_max, y.device)
        sigma_t_sq = sigma_t.view(-1, 1, 1, 1) ** 2
        eta = torch.randn(y.shape, device=y.device)
        var_diff = (sigma_t_sq - sigma_tn_map ** 2).clamp_min(0.0)
        x_t = y + var_diff.sqrt() * eta
        pred = _predict(model, x_t, sigma_t, context, self.sigma_min, self.sigma_max)
        coef_h = (sigma_t_sq - sigma_tn_map ** 2).clamp_min(0.0) / sigma_t_sq
        coef_xt = (sigma_tn_map ** 2) / sigma_t_sq
        resid = coef_h * pred + coef_xt * x_t - y
        loss_adsm = resid.pow(2).mean()

        # --- consistency branch: single posterior hop, two independent draws, unbiased product loss ---
        if self._step < self.consistency_warmup_steps:
            loss_consistency = torch.zeros((), device=y.device)
            effective_weight = 0.0
        else:
            x0_pred = pred.detach()
            sigma_s = sample_log_uniform_elementwise(self.sigma_min, sigma_t)
            x_s_1 = ve_posterior_step(x0_pred, x_t, sigma_t, sigma_s)
            x_s_2 = ve_posterior_step(x0_pred, x_t, sigma_t, sigma_s)
            pred_s_1 = _predict(model, x_s_1, sigma_s, context, self.sigma_min, self.sigma_max)
            pred_s_2 = _predict(model, x_s_2, sigma_s, context, self.sigma_min, self.sigma_max)
            loss_consistency = ((pred_s_1 - x0_pred) * (pred_s_2 - x0_pred)).mean()
            steps_since_warmup = self._step - self.consistency_warmup_steps
            ramp_frac = min(1.0, steps_since_warmup / max(1, self.consistency_ramp_steps))
            effective_weight = self.consistency_weight * ramp_frac

        loss = loss_adsm + effective_weight * loss_consistency
        if not torch.isfinite(loss):
            # Defensive guard added after a real NaN divergence on 2026-08-23 (root cause:
            # see ve_posterior_step's docstring -- fixed there, but keeping this guard so a
            # future rare bad batch contributes zero gradient instead of corrupting the
            # model's weights permanently via a NaN optimizer step.
            print(f"[AmbientTweedieFaithfulLoss] WARNING: non-finite loss at step {self._step} "
                  f"(loss_adsm={loss_adsm.item()}, loss_consistency={loss_consistency.item()}) "
                  "-- skipping this step's gradient contribution.")
            loss = 0.0 * pred.sum()
        return loss, {"loss": loss.item(), "loss_adsm": loss_adsm.item(), "loss_consistency": loss_consistency.item()}


class AmbientTweedieCurriculumLoss(BaseLoss):
    """Anneal the below-floor consistency loss outward from the boundary, one ladder
    rung at a time, using a per-rung REPLAY BUFFER of the model's own recent outputs
    as training seeds -- fixes the root cause found by the single-hop diagnostic (see
    project memory project-tr-diffusion-ambient-tweedie): ``AmbientTweedieLoss``'s
    consistency branch always renoises a single hop DOWN from the live ADSM anchor,
    so the network at a given below-floor sigma never sees, during training, the kind
    of partially-refined state that a REAL multi-rung reverse chain (``sample_below_floor``)
    actually feeds it after several iterations -- an out-of-distribution input at
    inference that the network responds to by amplifying speckle. Confirmed directly:
    a single-hop reproduction of the training-time construction scores 18-30dB (graceful
    degradation) on a checkpoint whose actual multi-rung chain output scores 8.5dB
    (speckle collapse) -- the network itself is fine, the training/inference MISMATCH is not.

    **The fix**: make the training-time construction at every below-floor rung
    STRUCTURALLY IDENTICAL to what ``sample_below_floor`` actually does there --
    always renoise from the model's own output AT THE ADJACENT RUNG ABOVE, never
    from a single top-level anchor. Ladder rungs are shared verbatim with
    :func:`sample_below_floor` (:func:`geometric_ladder(sigma_min, sigma_tn_eff, n_rungs)`)
    so training and inference walk the identical discretisation.

    **Schedule** (epoch-driven via :meth:`log_epoch_summary`'s ``epoch`` argument --
    NOT a self-managed step counter, since that would also count eval-time
    ``compute_loss`` calls; ``log_epoch_summary`` only fires once per true epoch and
    is called with the harness's own authoritative epoch index):
    - Epoch < ``warmup_epochs`` (default 1): ADSM only, consistency weight is
      effectively zero -- lets the boundary stabilise before anything below it opens
      (matches the project's long-standing "phase 1" idea, now actually implemented).
    - Epoch ``warmup_epochs + i`` (i=0,1,2,...): the OPEN below-floor range is
      ``[ladder[max(0, n_rungs - i - 1)], sigma_tn_eff]`` -- one additional rung opens
      per epoch, so with ``n_rungs=20`` the full range down to ``sigma_min`` is open by
      epoch ``warmup_epochs + 19`` (21 epochs total with the default 1-epoch warmup).
    - Once open, a rung's band NEVER closes -- every training step picks ONE band
      uniformly at random among all currently-open bands (not one band per batch
      element -- keeps the buffer bookkeeping simple, and there are thousands of steps
      per epoch so every open band still gets ample exposure). The ADSM branch is
      computed unconditionally every step, exactly as in ``AmbientTweedieLoss``.

    **Per-rung replay buffers** (a per-rung ``collections.deque(maxlen=buffer_size)``,
    default 200, holding individual detached CPU tensors, NOT a single shared pool --
    see project memory for why a shared pool was rejected: it would let a newly-opened
    band train on a mix of qualities/biases from several rungs up, which never happens
    in the real chain and would reintroduce a different distribution mismatch):
    - Band ``k`` = ``[ladder[k], ladder[k+1]]`` for ``k`` in ``[0, n_rungs-1]``.
    - Training band ``k`` seeds its noised input from: the LIVE ADSM anchor
      (``pred.detach()``) if ``k == n_rungs-1`` (the band adjacent to the boundary --
      exactly ``AmbientTweedieLoss``'s existing mechanism, always fresh, no buffer
      needed); otherwise a random draw (with replacement) from ``buffer[k+1]`` --
      i.e. from what band ``k+1``'s OWN recent training actually produced.
    - Every step that trains band ``k`` also pushes its (detached) ``pred_t_prime``
      outputs into ``buffer[k]`` (except ``k == 0``, the bottom band, which has
      nothing below it to seed). By the time a new band opens, the buffer that feeds
      it has already been filling for a full epoch's worth of steps (thousands),
      since the band above it was open and being trained the whole time.
    - ``n2c`` only for now (regress ``pred_t_prime`` toward the seed used to build its
      input) -- ``n2n``'s original justification (regressing toward a genuinely
      unbiased noisy measurement of the true signal) only holds cleanly against the
      live ADSM anchor at the top band; generalising it to buffer-sourced seeds
      several rungs down needs more thought, deferred until this (confirmed-stable)
      leg is validated end to end.
    """

    def __init__(self, device, sigma_tn_eff: float, sigma_min: float = 0.002, sigma_max: float = 2.0,
                n_rungs: int = 20, consistency_weight: float = 1.0, ema_decay: float = 0.999,
                consistency_legs: str = "n2c", adsm_sigma_t_max: Optional[float] = None,
                warmup_epochs: int = 1, buffer_size: int = 200):
        super().__init__(["loss", "loss_adsm", "loss_consistency", "band_idx", "buffer_min_fill"])
        assert consistency_legs == "n2c", "only n2c is implemented for the curriculum design so far"
        self.device = device
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_tn_eff = float(sigma_tn_eff)
        assert self.sigma_min < self.sigma_tn_eff < self.sigma_max
        self.n_rungs = int(n_rungs)
        self.consistency_weight = float(consistency_weight)
        self.adsm_sigma_t_max = float(adsm_sigma_t_max) if adsm_sigma_t_max is not None else self.sigma_max
        assert self.sigma_tn_eff < self.adsm_sigma_t_max <= self.sigma_max
        self.warmup_epochs = int(warmup_epochs)
        self.buffer_size = int(buffer_size)
        # Shared verbatim with sample_below_floor -- training and inference MUST walk
        # the identical discretisation for this fix to actually close the gap.
        self.ladder = geometric_ladder(self.sigma_min, self.sigma_tn_eff, self.n_rungs, device)
        self.buffers = {k: deque(maxlen=self.buffer_size) for k in range(self.n_rungs)}
        self._current_epoch = 0  # advanced by log_epoch_summary using the harness's own epoch index

    def _floor_idx(self) -> int:
        """Lowest OPEN rung index this epoch (n_rungs = fully closed, i.e. warmup)."""
        e = self._current_epoch
        if e < self.warmup_epochs:
            return self.n_rungs
        bands_opened = (e - self.warmup_epochs) + 1
        return max(0, self.n_rungs - bands_opened)

    def log_epoch_summary(self, instance, model, epoch: int) -> None:
        self._current_epoch = epoch + 1
        floor_idx = self._floor_idx()
        fills = [(k, len(v)) for k, v in sorted(self.buffers.items()) if len(v) > 0]
        print(f"[ambient_tweedie_curriculum] epoch {epoch} done -> epoch {epoch + 1} floor_rung={floor_idx} "
              f"(open sigma range=[{self.ladder[floor_idx].item():.5f}, {self.sigma_tn_eff:.5f}]) "
              f"buffer fills: {fills}")

    def _prep(self, instance):
        dev = self.device
        y = instance["clean_target"].to(dev, non_blocking=True).float()
        sigma_tn_map = instance["sigma_tn_map"].to(dev, non_blocking=True).float()
        context = instance["context"].to(dev, non_blocking=True).float()
        return y, sigma_tn_map, context

    def compute_loss(self, instance, model) -> Tuple[torch.Tensor, Dict[str, float]]:
        y, sigma_tn_map, context = self._prep(instance)
        bsz = y.shape[0]

        # --- ADSM branch: unconditional every step, exactly as in AmbientTweedieLoss ---
        sigma_t = sample_log_uniform(bsz, self.sigma_tn_eff, self.adsm_sigma_t_max, y.device)
        sigma_t_sq = sigma_t.view(-1, 1, 1, 1) ** 2
        eta = torch.randn(y.shape, device=y.device)
        var_diff = (sigma_t_sq - sigma_tn_map ** 2).clamp_min(0.0)
        x_t = y + var_diff.sqrt() * eta
        pred = _predict(model, x_t, sigma_t, context, self.sigma_min, self.sigma_max)
        coef_h = (sigma_t_sq - sigma_tn_map ** 2).clamp_min(0.0) / sigma_t_sq
        coef_xt = (sigma_tn_map ** 2) / sigma_t_sq
        resid = coef_h * pred + coef_xt * x_t - y
        loss_adsm = resid.pow(2).mean()

        # --- consistency branch: one band, chosen uniformly among currently-open bands ---
        floor_idx = self._floor_idx()
        if floor_idx >= self.n_rungs:
            loss_consistency = torch.zeros((), device=y.device)
            band_idx, buffer_min_fill = -1, 0
        else:
            k = random.randint(floor_idx, self.n_rungs - 1)
            sigma_lo, sigma_hi = self.ladder[k].item(), self.ladder[k + 1].item()
            sigma_t_prime = sample_log_uniform(bsz, sigma_lo, sigma_hi, y.device)

            if k == self.n_rungs - 1:
                seed = pred.detach()
            else:
                buf = self.buffers[k + 1]
                if len(buf) == 0:
                    # Shouldn't normally happen given the schedule (the feeding buffer has
                    # had a full epoch to fill by the time this band opens) -- fall back to
                    # the live ADSM anchor rather than crash if it somehow does.
                    seed = pred.detach()
                else:
                    picks = [buf[random.randrange(len(buf))] for _ in range(bsz)]
                    seed = torch.stack(picks).to(y.device)

            x_t_prime = seed + sigma_t_prime.view(-1, 1, 1, 1) * torch.randn_like(seed)
            pred_t_prime = _predict(model, x_t_prime, sigma_t_prime, context, self.sigma_min, self.sigma_max)
            loss_consistency = (pred_t_prime - seed).pow(2).mean()

            if k > 0:
                for i in range(bsz):
                    self.buffers[k].append(pred_t_prime[i].detach().cpu())

            band_idx = k
            active_keys = range(max(1, floor_idx + 1), self.n_rungs)
            buffer_min_fill = min((len(self.buffers[j]) for j in active_keys), default=0)

        loss = loss_adsm + self.consistency_weight * loss_consistency
        return loss, {"loss": loss.item(), "loss_adsm": loss_adsm.item(),
                      "loss_consistency": loss_consistency.item(),
                      "band_idx": float(band_idx), "buffer_min_fill": float(buffer_min_fill)}
