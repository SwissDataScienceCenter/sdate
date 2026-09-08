#!/usr/bin/env python3
"""Train the conditional time-resolved frame denoiser (diffusion or baseline).

Both models share the dataset, the N2V masking, and the conditioning-dropout
recipe; ``--mode`` selects which is trained:

* ``diffusion`` — ε-prediction conditional DDPM (in_channels = 2 + 4k).
* ``baseline``  — single-pass x_0 regressor (in_channels = 1 + 4k).

Example (native self-supervised, k=3 default, memmap-backed)::

    python -m sdate.tr_diffusion.train --mode diffusion \
        --mov /myhome/data/sdate/shared/time_resolved/212_Wunderkerze2/212_Wunderkerze2.mov \
        --memmap /myhome/data/sdate/shared/time_resolved/212_Wunderkerze2/frames_400k_600k.u16 \
        --frame_start 400000 --frame_end 600000 \
        --batch_size 16 --epochs 100 --exp_name tr_diff_k3 --wandb

Reuses ``pytorch_base.PyTorchExperiment`` and the diffusers UNet2DModel, matching
``isodiffusion/train_conditional_2d.py``.

**Current defaults (see README "Key experimental findings" + project memory
project-tr-diffusion)**: ``--k 3`` (monotonic gain over k=1/k=2, no real cost)
and, for ``--mode baseline``, ``--poisson_head`` (the two-head Gamma-Poisson
NB-NLL loss -- supersedes both the plain Huber/MAE/MSE point estimate and the
single-channel ``--loss_type poisson`` ablation; pass ``--no-poisson_head`` to
opt back into the legacy single-channel path). Both inference modes (posterior
combination or ``mu`` alone) are available off the SAME poisson_head
checkpoint -- see ``poisson_posterior`` in
:func:`sdate.tr_diffusion.pipeline.denoise_frames_baseline`.
"""

from __future__ import annotations

import json
import os
import random
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
for _c in (Path("/myhome/BaseTraining"), Path("/myhome/sdsc"), Path("/myhome/chip-project")):
    if _c.exists() and str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from pytorch_base.experiment import PyTorchExperiment  # noqa: E402

from sdate.tr_diffusion.ambient_tweedie import (  # noqa: E402
    AmbientTweedieCurriculumLoss, AmbientTweedieFaithfulLoss, AmbientTweedieLoss,
)
from sdate.tr_diffusion.data import TimeResolvedFrameDataset, resolve_warped_context_dir  # noqa: E402
from sdate.tr_diffusion.load import load_config, load_denoiser  # noqa: E402
from sdate.tr_diffusion.losses import (  # noqa: E402
    BaselineN2NLoss, BaselineN2VLoss, BootstrapPoissonLoss, ContextOnlyNoiseToCleanLoss, DiffusionN2NCleanTargetLoss,
    DiffusionN2NLoss, DiffusionN2VLoss, NoiseToCleanLoss, RefinementLoss, SinogramN2VLoss,
)
from sdate.tr_diffusion.model import create_baseline_unet, create_diffusion_unet  # noqa: E402
from sdate.tr_diffusion.sino_transform import SinoTransform, choose_sino_shape, fit_sino_norm_range  # noqa: E402


def parse_args():
    p = ArgumentParser(description="Train time-resolved conditional frame denoiser.")
    p.add_argument("--mode", choices=["diffusion", "baseline", "ambient_tweedie", "bootstrap", "noise2clean",
                                      "refine", "context_only", "sinogram"],
                   default="diffusion")
    p.add_argument("--denoise_mode", choices=["n2v", "n2n"], default="n2v",
                   help="n2v = blind-spot masked loss (default); n2n = binomial-split "
                        "Noise2Noise (requires --extra_noise_dose; conditions on split fraction p).")
    p.add_argument("--p_min", type=float, default=0.1, help="n2n: min split fraction sampled per example.")
    p.add_argument("--p_max", type=float, default=0.9, help="n2n: max split fraction.")
    p.add_argument("--p_bins", type=int, default=100, help="n2n: discretisation of p for the class embedding.")
    p.add_argument("--n2n_prediction_type", choices=["epsilon", "sample"], default="epsilon",
                   help="n2n diffusion only: predict added noise (original) or the target-split x0 directly.")
    p.add_argument("--n2n_consistency_weight", type=float, default=0.0,
                   help="n2n diffusion only: weight on the swap-consistency term (0 = disabled, original behaviour). "
                        "Predicts BOTH directions (input->target, target->input) at independent timesteps and "
                        "penalises disagreement between the two x0 estimates.")
    p.add_argument("--clean_target_memmap", type=str, default=None,
                   help="n2n diffusion only: path to a cached denoised-sequence memmap (e.g. a Bayesian "
                        "two-head poisson_head reconstruction, see reconstruct.denoise_sequence's output "
                        "format) to use as the regression target INSTEAD of the other binomial-split half. "
                        "Conditioning is unchanged (still two independent split-fraction views of the raw "
                        "measurement); both branches now regress toward this external clean reference, and "
                        "the swap-consistency term (--n2n_consistency_weight) still asks their two x0 "
                        "estimates to agree. Training centres are intersected with the memmap's covered "
                        "range (its own first_index/num_frames sidecar).")
    p.add_argument("--aux_channel_memmap", type=str, default=None, nargs="+",
                   help="--mode baseline only: one or more paths to cached extra INPUT channels (e.g. a "
                        "frozen noise2clean checkpoint's own prediction on this dataset's real frames, or "
                        "the T same-angle joint-FBP reprojection taps -- see "
                        "reconstruct.denoise_sequence's/tr_diffusion_jointfbp_context_cache.py's output "
                        "format) appended to the model's input stack (+N in_channels, via "
                        "extra_cond_channels, N = number of paths given). All paths must share the same "
                        "covered frame range/crop. Training centres are intersected with that range.")
    p.add_argument("--warped_context_dir", type=str, default=None,
                   help="--mode baseline or refine only: directory of per-tap motion-compensated "
                        "context caches (scripts/tr_diffusion_warped_context_precompute.py's output, "
                        "one warped_ctx_{tap.name}.f16 per 'temporal' tap in this dataset's layout) -- "
                        "the angular-resolution-gap experiment. --mode baseline: replaces the "
                        "corresponding raw temporal context channels (data.py's additive "
                        "context_warped field) with no other change, isolating the context data "
                        "source as the only variable ('Leg 2'). --mode refine: supplies the SAME "
                        "warped context as extra input alongside the sampled base-model belief "
                        "('Leg 1'); --base_checkpoint is also required in that mode.")
    p.add_argument("--var_target_memmap", type=str, default=None,
                   help="--mode ambient_tweedie only: companion posterior-VARIANCE cache to "
                        "--clean_target_memmap (see reconstruct.denoise_sequence's var_out_path) -- "
                        "the per-pixel heteroscedastic noise map. Must cover the exact same frame "
                        "range as --clean_target_memmap. Mutually exclusive with --anscombe.")
    p.add_argument("--anscombe", action="store_true",
                   help="--mode ambient_tweedie only: variance-stabilize the raw dose-thinned "
                        "measurement itself (Anscombe transform) instead of using poisson_head's "
                        "heteroscedastic posterior as y -- replicates 'Consistent Diffusion Meets "
                        "Tweedie' LITERALLY (a single global scalar sigma_tn, no per-pixel map, no "
                        "poisson_head dependency at all). Mutually exclusive with "
                        "--clean_target_memmap/--var_target_memmap. See sdate.tr_diffusion.noise.")
    p.add_argument("--base_checkpoint", type=str, default=None,
                   help="--mode bootstrap or refine only: path to a trained --mode baseline "
                        "--poisson_head checkpoint (.pt) to distil from -- this run's model predicts "
                        "the noisy measurement from that frozen checkpoint's own context-only belief "
                        "(mu/var), instead of the usual multi-frame context alone (see "
                        "BootstrapPoissonLoss / RefinementLoss). Its geometry/channel-shape config (k, "
                        "crop, neighborhoods, temporal_raw_pairs, cond_angle_time, include_mirror, "
                        "axis_col, deg_per_frame, norm_min/max, extra_noise_dose) is REUSED VERBATIM "
                        "(overriding any conflicting flags below) so the frozen model's forward pass "
                        "stays in-distribution -- only --profile/--mov/--memmap/--frame_start/"
                        "--frame_end may still meaningfully differ. Requires extra_noise_dose to have "
                        "been set on the base checkpoint (both modes redraw an INDEPENDENT second "
                        "noise realisation from its pre-thinning `reference` frame for the target).")
    p.add_argument("--bootstrap_input", choices=["mean", "sample"], default="mean",
                   help="--mode bootstrap --bootstrap_direction forward only: what to feed as the "
                        "single input channel. 'mean' (default) = the frozen base model's "
                        "deterministic prior mean `mu`. 'sample' = a FRESH draw from its "
                        "moment-matched Gamma(mu, var) prior every call (see "
                        "BootstrapPoissonLoss/_gamma_sample) -- injects the base model's own local "
                        "uncertainty as noise instead of collapsing to a point estimate.")
    p.add_argument("--bootstrap_direction", choices=["forward", "reverse"], default="forward",
                   help="--mode bootstrap only: 'forward' (default) = predict a fresh noisy "
                        "measurement FROM the base model's belief (--bootstrap_input selects mean "
                        "or Gamma-sample conditioning; see BootstrapPoissonLoss). 'reverse' "
                        "(exploratory, N2N-symmetry variant) = predict a Gamma-posterior draw of "
                        "the base model's belief FROM the real measurement instead -- swaps which "
                        "noisy view is input vs. target. Ignores --bootstrap_input (the input is "
                        "always the raw measurement). At inference this direction needs neither the "
                        "frozen base model nor multi-frame context -- see "
                        "pipeline.denoise_frames_bootstrap_reverse.")
    p.add_argument("--sino_num_angles", type=int, default=None,
                   help="--mode sinogram only: number of synthetic Radon-transform angles (0 to "
                        "--sino_angle_max_deg). Default: auto -- see "
                        "sino_transform.choose_sino_shape (padded to the crop's diagonal, rounded up "
                        "to a multiple of 32 for the UNet's downsampling, matched to --sino_det_cols "
                        "for a square sinogram).")
    p.add_argument("--sino_det_cols", type=int, default=None,
                   help="--mode sinogram only: number of detector columns in the synthetic sinogram. "
                        "Default: auto (see --sino_num_angles).")
    p.add_argument("--sino_angle_max_deg", type=float, default=180.0,
                   help="--mode sinogram only: angular span of the per-frame virtual Radon transform. "
                        "180 (default) already fully determines a 2D image via FBP (angle+180 is the "
                        "mirror of angle) -- no reason to go higher.")
    p.add_argument("--sino_norm_sample_frames", type=int, default=32,
                   help="--mode sinogram only: number of frames sampled to fit the sinogram-domain "
                        "normalisation range (a sinogram's line-integral values live on a very "
                        "different scale than raw projection counts -- see sino_transform.fit_sino_norm_range).")
    p.add_argument("--sigma_min", type=float, default=0.002, help="ambient_tweedie: VE schedule floor (normalized units).")
    p.add_argument("--sigma_max", type=float, default=2.0, help="ambient_tweedie: VE schedule ceiling (normalized units).")
    p.add_argument("--n_rungs", type=int, default=40,
                   help="ambient_tweedie: log-spaced consistency-ladder rungs between --sigma_min and sigma_tn_eff.")
    p.add_argument("--consistency_weight", type=float, default=None,
                   help="ambient_tweedie: weight on the consistency term. Default 1.0, or 0.015 with --faithful "
                        "(matching the paper's own production configs -- a light regulariser, not equal-weighted).")
    p.add_argument("--ema_decay", type=float, default=0.999, help="ambient_tweedie: EMA decay for the consistency target network.")
    p.add_argument("--curriculum", action="store_true",
                   help="ambient_tweedie: use AmbientTweedieCurriculumLoss instead of AmbientTweedieLoss -- "
                        "anneal the below-floor consistency training outward from the boundary one ladder "
                        "rung per epoch, using a per-rung replay buffer of the model's own recent outputs as "
                        "training seeds (chained rung-to-rung, matching sample_below_floor's own walk exactly) "
                        "instead of always renoising a single hop down from the top-level ADSM anchor. Fixes "
                        "the train/inference mismatch the single-hop diagnostic found (see project memory "
                        "project-tr-diffusion-ambient-tweedie). Only consistency_legs=n2c is implemented for "
                        "this mode -- --curriculum requires --consistency_legs n2c explicitly.")
    p.add_argument("--curriculum_warmup_epochs", type=int, default=1,
                   help="--curriculum only: epochs of ADSM-only training before the first below-floor band opens.")
    p.add_argument("--curriculum_buffer_size", type=int, default=200,
                   help="--curriculum only: max size of each per-rung replay buffer (FIFO, oldest evicted first).")
    p.add_argument("--faithful", action="store_true",
                   help="ambient_tweedie: use AmbientTweedieFaithfulLoss instead of AmbientTweedieLoss -- a "
                        "faithful port of the official ambient-tweedie (github.com/giannisdaras/ambient-tweedie) "
                        "training recipe, adapted to our VE + x0-parameterised setting. Three concrete fixes vs "
                        "AmbientTweedieLoss found by reading their actual released code: (1) the consistency "
                        "renoise step reuses the model's own predicted noise direction (ve_posterior_step) "
                        "instead of adding fully independent fresh noise, (2) the consistency loss is the "
                        "unbiased two-independent-sample product estimator, not one squared difference, (3) a "
                        "much smaller consistency_weight (their 0.015, not 1.0). No replay buffer, no rung "
                        "schedule -- single hop, log-uniform over the whole below-floor range every step, "
                        "matching their simplest num_consistency_steps=1 configuration. Mutually exclusive with "
                        "--curriculum.")
    p.add_argument("--faithful_consistency_warmup_steps", type=int, default=30000,
                   help="--faithful only: training steps of ADSM-only warmup before the consistency term "
                        "contributes at all (their consistency_kick_in).")
    p.add_argument("--faithful_consistency_ramp_steps", type=int, default=2000,
                   help="--faithful only: steps to linearly ramp the effective consistency weight from 0 up "
                        "to --consistency_weight immediately after warmup ends, instead of switching it on at "
                        "full strength in one step -- avoids a fast destabilising divergence, especially when "
                        "resuming from a checkpoint with warmup_steps=0 (fresh optimizer/scheduler state hitting "
                        "an already-mature model). Confirmed necessary: a NaN divergence occurred on resume "
                        "without this ramp (see project memory project-tr-diffusion-ambient-tweedie).")
    p.add_argument("--adsm_upper_mult", type=float, default=None,
                   help="ambient_tweedie: caps the ADSM branch's OWN sigma_t sampling (hence the "
                        "consistency anchor's source noise level) at sigma_tn_eff * this multiplier, "
                        "instead of the default full range up to --sigma_max. Bounds the amplification "
                        "the n2n consistency leg implicitly requires (grows with how far above "
                        "sigma_tn_eff the anchor's sigma_t was drawn from -- see ambient_tweedie.py's "
                        "docstring), which is believed to cause its below-floor speckle. Must stay > 1.0 "
                        "(coef_h -> 0 exactly at sigma_tn_eff, so values too close to 1.0 recreate the "
                        "boundary dead-zone collapse, failure (4) in project memory). None (default) = "
                        "unrestricted (original behaviour).")
    p.add_argument("--consistency_legs", choices=["both", "n2c", "n2n", "dual"], default="both",
                   help="ambient_tweedie: which consistency-branch variant drives training -- 'n2c' (regress "
                        "toward the ADSM anchor), 'n2n' (regress toward x_t instead), 'both' (n2c+n2n summed, "
                        "default), or 'dual' (two INDEPENDENT below-floor predictions regressed toward EACH "
                        "OTHER instead of toward either above-floor anchor). Run each as a SEPARATE job "
                        "(independent models) to isolate which is responsible for an artifact seen in a 'both' "
                        "run -- n2c/n2n values are always logged regardless of this setting ('dual' is not, "
                        "since unlike n2c/n2n it needs 2 extra forward passes so isn't computed for free).")
    p.add_argument("--sigma_tn_eff_percentile", type=float, default=95.0,
                   help="ambient_tweedie: percentile of the per-pixel sigma_tn(x) map (over "
                        "--sigma_tn_eff_sample_frames sampled frames) used as the fixed global ADSM/consistency "
                        "split -- a conservative floor so the ADSM decomposition is valid everywhere.")
    p.add_argument("--sigma_tn_eff_sample_frames", type=int, default=64,
                   help="ambient_tweedie: number of frames sampled to fit sigma_tn_eff.")
    p.add_argument("--profile", type=str, default=None,
                   help="DatasetProfile name (e.g. asc_thixo) or JSON path; supplies "
                        "mov/memmap/crop/frame-range/axis/deg_per_frame (explicit flags override).")
    p.add_argument("--mov", type=str, default=None, help="Path to the .mov (with .norm.npz sidecar).")
    p.add_argument("--memmap", type=str, default=None, help="Optional pre-extracted uint16 memmap for fast access.")
    p.add_argument("--k", type=int, default=3,
                   help="Context radius: in_channels = (2 or 1) + 4k. Default 3: monotonic gain "
                        "over k=1/k=2 with no capacity/inference-FLOPs cost (see README ablation) -- "
                        "only downside is slower data-loading I/O and a slightly smaller usable range.")
    p.add_argument("--neighborhoods", choices=["both", "rotation", "temporal"], default="both",
                   help="Which context taps to keep: both (default), rotation-only (angular "
                        "neighbours, no same-angle-across-turns taps), or temporal-only "
                        "(same-angle-across-turns taps, no angular neighbours). Ablates the "
                        "two conditioning neighbourhoods independently.")
    p.add_argument("--frame_start", type=int, default=None)
    p.add_argument("--frame_end", type=int, default=None)
    p.add_argument("--crop_h", type=int, default=None)
    p.add_argument("--crop_w", type=int, default=None)
    p.add_argument("--deg_per_frame", type=float, default=None, help="Rotation rate (from profile if unset).")
    p.add_argument("--axis_col", type=float, default=None, help="Centre-of-rotation column (from profile if unset).")
    p.add_argument("--include_mirror", action="store_true", help="Add 180-degree half-turn mirror taps (+2 ch).")
    p.add_argument("--temporal_raw_pairs", action=BooleanOptionalAction, default=True,
                   help="Default ON: replace each interpolated same-angle temporal tap (PERIOD_360 is "
                        "never an integer, so a plain blend of the two bracketing frames at a fixed "
                        "ratio was the old behaviour -- confirmed to ghost at moving edges) with BOTH "
                        "bracketing frames as separate, un-blurred channels (temporal context doubles "
                        "2k -> 4k, total context 4k -> 6k; confirmed a small but consistent reconstruction "
                        "PSNR/SSIM improvement over the interpolated version). Pass --no-temporal_raw_pairs "
                        "to reproduce the old interpolated behaviour (e.g. to match an existing checkpoint's "
                        "recipe).")
    p.add_argument("--extra_noise_dose", type=float, default=None,
                   help="If set (0<dose<=1), train on extra-Poisson-noised frames; original kept as reference.")
    p.add_argument("--noise_seed", type=int, default=None,
                   help="Extra-noise RNG seed. Default None = fresh noise each epoch (train); "
                        "set an int for reproducible per-frame noise (eval).")
    p.add_argument("--max_samples", type=int, default=None, help="Subsample this many usable centres for the epoch.")
    p.add_argument("--norm_min", type=float, default=None)
    p.add_argument("--norm_max", type=float, default=None)

    p.add_argument("--n2v_ratio", type=float, default=0.02, help="Blind-spot pixel fraction.")
    p.add_argument("--n2v_window", type=int, default=5, help="Blind-spot neighbour window (odd).")
    p.add_argument("--conditioning_probability", type=float, default=0.5,
                   help="Probability of keeping the corrupted central frame (with/without-central).")
    p.add_argument("--cond_angle_time", action="store_true",
                   help="Baseline only: add 3 extra input channels -- sin(angle), cos(angle) of "
                        "the central frame's rotation angle, and its normalised position in "
                        "[frame_start, frame_end) -- so the model can learn angle/time-dependent "
                        "structure directly instead of only inferring it from the neighbour frames.")
    p.add_argument("--loss_type", choices=["mae", "mse", "huber", "poisson"], default="huber",
                   help="Only used when --no-poisson_head (loss_type is ignored otherwise). "
                        "'poisson': legacy single-channel ablation -- same point estimate as "
                        "mae/mse/huber, but trained with the correct Poisson NLL (weights pixels by "
                        "their actual count-dependent noise level) instead of a homoscedastic loss. "
                        "Superseded by --poisson_head (its own mu output is empirically identical, "
                        "see nb_head.py); kept for loading/comparing already-trained checkpoints.")
    p.add_argument("--edge_weight", type=float, default=0.0,
                   help="Baseline (n2v) only: up-weight the masked pixel loss at high-gradient "
                        "(edge) GT pixels by 1+edge_weight*normalized_grad, to fight regression-to-"
                        "mean blur. 0 = disabled (original behaviour).")
    p.add_argument("--poisson_head", action=BooleanOptionalAction, default=True,
                   help="Baseline (n2v) only, default ON (the STANDARD baseline architecture -- see "
                        "README 'Key experimental findings' + project memory project-tr-diffusion): "
                        "2-output-channel Gamma-Poisson (mu, var) belief trained with the exact "
                        "Negative-Binomial NLL instead of a single-channel point estimate -- see "
                        "sdate.tr_diffusion.nb_head. Supersedes both --loss_type poisson and the "
                        "plain Huber/MAE/MSE point estimate: this same checkpoint supports BOTH "
                        "inference modes going forward (posterior-mean combination, or mu alone -- "
                        "see poisson_posterior in pipeline.denoise_frames_baseline/reconstruct.py), so "
                        "there is no need to also train the single-channel variant. Silently not "
                        "applied for --mode diffusion or --denoise_mode n2n (not wired for either). "
                        "Pass --no-poisson_head for the legacy single-channel path (--loss_type "
                        "mae/mse/huber/poisson).")
    p.add_argument("--poisson_warmup_frac", type=float, default=0.1,
                   help="poisson_head only: fraction of total training steps to fit mu alone via "
                        "plain Poisson NLL (var receives no gradient) before switching on the full "
                        "NB-NLL -- mitigates the heteroscedastic-collapse failure mode (var "
                        "inflating instead of mu improving). 0 disables the warm start.")
    p.add_argument("--poisson_beta_nll_power", type=float, default=0.0,
                   help="poisson_head only: if >0, rescale the (post-warm-start) NB-NLL by "
                        "(alpha+beta).detach()**(-power) (beta-NLL style) to further rebalance "
                        "the mu-gradient against var inflation, if collapse persists past warm "
                        "start. 0 = disabled (default).")
    p.add_argument("--gaussian_floor", action="store_true",
                   help="poisson_head only: use nb_nll_gaussian (Poisson+Gaussian read-noise-floor "
                        "Gaussian-CLT NLL) instead of the exact NB marginal (nb_nll). Needed at "
                        "native/full-dose counts, where nb_nll's alpha=mu^2/var term can explode "
                        "and produce NaN once var is near its untrained init right after warmup "
                        "(see nb_head.nb_nll_gaussian's docstring). Requires --sigma_read2_map.")
    p.add_argument("--sigma_read2_map", type=str, default=None,
                   help="--gaussian_floor only: path to a per-pixel (H, W) .npy float32 map of the "
                        "detector's additive Gaussian read-noise variance, sized to match --crop -- "
                        "see scripts/tr_diffusion_estimate_read_noise.py.")

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--test_fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exp_name", type=str, default="tr_denoise")
    p.add_argument("--save_checkpoint", type=str, default="")
    p.add_argument("--load_checkpoint", type=str, default="")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16", "auto"], default="fp16")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_cfg = None
    if args.mode in ("bootstrap", "refine"):
        if not args.base_checkpoint:
            raise SystemExit(f"--mode {args.mode} requires --base_checkpoint (a trained --mode baseline "
                             "--poisson_head checkpoint)")
        if args.denoise_mode != "n2v":
            raise SystemExit(f"--mode {args.mode} only supports --denoise_mode n2v (the dataset's plain "
                             "extra-noise branch, which supplies the `reference` frame this mode needs)")
        if args.mode == "refine" and not args.warped_context_dir:
            raise SystemExit("--mode refine also requires --warped_context_dir (the motion-compensated "
                             "context this leg adds on top of the base checkpoint's own belief -- see "
                             "scripts/tr_diffusion_warped_context_precompute.py)")
        base_cfg = load_config(args.base_checkpoint)
        if base_cfg.get("mode", "baseline") != "baseline":
            raise SystemExit(f"--base_checkpoint must be a --mode baseline checkpoint, got mode={base_cfg.get('mode')!r}")
        if not base_cfg.get("poisson_head", False):
            raise SystemExit(f"--base_checkpoint must have poisson_head=True ({args.mode} reads off its `mu`/`var` head)")
        if base_cfg.get("extra_noise_dose") is None:
            raise SystemExit(f"--base_checkpoint must have been trained with --extra_noise_dose ({args.mode} "
                             "needs its pre-thinning `reference` frame to draw an INDEPENDENT second noise "
                             "realisation for the target -- otherwise the target would share the base "
                             "model's own noise draw, leaking the answer through its corrupted-central input "
                             "rather than genuinely testing whether this leg sharpens the estimate)")
        # Force the geometry/channel-shape params to match the frozen model exactly -- the dataset built
        # below feeds the FROZEN base model, so its context layout must be pixel-for-pixel what that
        # checkpoint was trained with. Only --profile/--mov/--memmap/--frame_start/--frame_end are free to differ.
        args.profile = args.profile or base_cfg.get("profile")
        args.mov = args.mov or base_cfg.get("mov")
        args.k = int(base_cfg["k"])
        args.crop_h, args.crop_w = int(base_cfg["crop"][0]), int(base_cfg["crop"][1])
        args.include_mirror = bool(base_cfg.get("include_mirror", False))
        args.neighborhoods = base_cfg.get("neighborhoods", "both")
        args.temporal_raw_pairs = bool(base_cfg.get("temporal_raw_pairs", False))
        args.cond_angle_time = bool(base_cfg.get("cond_angle_time", False))
        args.axis_col = base_cfg.get("axis_col", args.axis_col)
        args.deg_per_frame = base_cfg.get("deg_per_frame", args.deg_per_frame)
        args.extra_noise_dose = float(base_cfg["extra_noise_dose"])
        args.norm_min, args.norm_max = float(base_cfg["norm_min"]), float(base_cfg["norm_max"])
        if args.mode == "bootstrap":
            print(f"bootstrap: inherited geometry from {args.base_checkpoint} "
                  f"(k={args.k}, crop={(args.crop_h, args.crop_w)}, dose={args.extra_noise_dose}, "
                  f"norm=({args.norm_min:.3f},{args.norm_max:.3f})) -- the NEW model itself is k=0, "
                  "in_channels=1 (mu-only input)")
        else:
            print(f"refine: inherited geometry from {args.base_checkpoint} "
                  f"(k={args.k}, crop={(args.crop_h, args.crop_w)}, dose={args.extra_noise_dose}, "
                  f"norm=({args.norm_min:.3f},{args.norm_max:.3f})) -- the NEW model keeps k={args.k}'s "
                  "own context (warped), in_channels = 1 + context (same shape as a normal baseline model)")

    # Resolve dataset geometry from a profile (explicit flags override profile values).
    from sdate.tr_diffusion.geometry import ANGLE_TIME_COND_CHANNELS, DEG_PER_FRAME, ROT_AXIS_COL
    from sdate.tr_diffusion.profiles import DatasetProfile
    prof = DatasetProfile.load(args.profile) if args.profile else None
    _pick = lambda v, pv, dv: v if v is not None else (pv if prof is not None else dv)
    args.mov = args.mov or (prof.mov_path if prof else None)
    if not args.mov:
        raise SystemExit("provide --mov or --profile")
    args.memmap = args.memmap or (prof.memmap_path if prof else None)
    args.crop_h = _pick(args.crop_h, prof.crop[0] if prof else None, 128)
    args.crop_w = _pick(args.crop_w, prof.crop[1] if prof else None, 512)
    args.frame_start = _pick(args.frame_start, prof.frame_start if prof else None, 400_000)
    args.frame_end = _pick(args.frame_end, prof.frame_end if prof else None, 600_000)
    args.deg_per_frame = _pick(args.deg_per_frame, prof.deg_per_frame if prof else None, DEG_PER_FRAME)
    args.axis_col = _pick(args.axis_col, prof.rot_axis_col if prof else None, ROT_AXIS_COL)
    print(f"geometry: mov={args.mov}\n  crop={(args.crop_h, args.crop_w)} range=[{args.frame_start},{args.frame_end}] "
          f"deg/frame={args.deg_per_frame} axis={args.axis_col}")

    if args.denoise_mode == "n2n" and args.extra_noise_dose is None:
        raise SystemExit("--denoise_mode n2n requires --extra_noise_dose (the fixed measurement dose to split)")
    if args.clean_target_memmap and not (args.mode == "ambient_tweedie" or (args.mode == "diffusion" and args.denoise_mode == "n2n")):
        raise SystemExit("--clean_target_memmap is only wired up for --mode ambient_tweedie or "
                         "--mode diffusion --denoise_mode n2n")
    if args.var_target_memmap and args.mode != "ambient_tweedie":
        raise SystemExit("--var_target_memmap is only wired up for --mode ambient_tweedie")
    if args.anscombe and args.mode != "ambient_tweedie":
        raise SystemExit("--anscombe is only wired up for --mode ambient_tweedie")
    if args.aux_channel_memmap and args.mode not in ("baseline", "context_only"):
        raise SystemExit("--aux_channel_memmap is only wired up for --mode baseline or context_only")
    if args.warped_context_dir and args.mode not in ("baseline", "refine"):
        raise SystemExit("--warped_context_dir is only wired up for --mode baseline or refine")
    if args.mode == "ambient_tweedie":
        if args.anscombe and (args.clean_target_memmap or args.var_target_memmap):
            raise SystemExit("--anscombe and --clean_target_memmap/--var_target_memmap are alternative "
                             "ways to get a homoscedastic-ish y -- pass only one")
        if not args.anscombe and not (args.clean_target_memmap and args.var_target_memmap):
            raise SystemExit("--mode ambient_tweedie requires either --anscombe or both "
                             "--clean_target_memmap and --var_target_memmap")
    if args.curriculum and args.mode != "ambient_tweedie":
        raise SystemExit("--curriculum is only wired up for --mode ambient_tweedie")
    if args.curriculum and args.consistency_legs != "n2c":
        raise SystemExit("--curriculum only implements consistency_legs=n2c so far -- pass --consistency_legs n2c")
    if args.faithful and args.mode != "ambient_tweedie":
        raise SystemExit("--faithful is only wired up for --mode ambient_tweedie")
    if args.faithful and args.curriculum:
        raise SystemExit("--faithful and --curriculum are mutually exclusive loss designs")
    if args.consistency_weight is None:
        args.consistency_weight = 0.015 if args.faithful else 1.0
    if args.mode == "ambient_tweedie" and args.extra_noise_dose is None:
        raise SystemExit("--mode ambient_tweedie requires --extra_noise_dose -- with --anscombe, it's the "
                         "dose fraction to Gaussianize directly; otherwise it must match the dose used to "
                         "produce --clean_target_memmap/--var_target_memmap, e.g. via "
                         "reconstruct.denoise_sequence's dose= -- context taps are thinned to the SAME dose "
                         "so the model doesn't condition on cleaner side-information than the cached "
                         "posterior actually saw)")
    if args.mode == "noise2clean":
        if args.extra_noise_dose is None:
            raise SystemExit("--mode noise2clean requires --extra_noise_dose -- the dataset's native "
                             "frames are the CLEAN synthetic phantom (see generate_synthetic_phantom.py); "
                             "extra_noise_dose is what synthesises the noisy input/context from it, with "
                             "the clean frame kept as `reference` (the regression target)")
        if args.denoise_mode != "n2v":
            raise SystemExit("--mode noise2clean only supports --denoise_mode n2v (the dataset's plain "
                             "extra-noise branch, which supplies `central`/`context`/`reference`) -- "
                             "n2n's binomial-split branch returns different keys and has no clean target")
    if args.mode == "context_only":
        if args.extra_noise_dose is not None:
            raise SystemExit("--mode context_only is the NATIVE-noise super-time-resolution regime -- "
                             "it must not add synthetic dose thinning on top of the real measurement "
                             "(the aux taps themselves should also be cached with --native_noise, see "
                             "tr_diffusion_jointfbp_context_cache.py)")
        if not args.aux_channel_memmap:
            raise SystemExit("--mode context_only requires --aux_channel_memmap (the joint-FBP "
                             "reprojection context taps) -- the whole point of this mode is regressing "
                             "the measured central frame from context ALONE, with no other input")
        if args.k != 0:
            print(f"context_only: forcing k=0 (was {args.k}) -- rotation/temporal taps would leak real "
                  "nearby-angle measurements, defeating the point of inferring never-measured angles "
                  "from the joint-FBP context alone at inference time")
            args.k = 0
        if args.conditioning_probability != 0.0:
            # denoise_sequence auto-detects present=False from this SAVED config value
            # (reconstruct.py: `present = cfg.get("conditioning_probability", 1.0) > 0.0`) --
            # ContextOnlyNoiseToCleanLoss always zeroes the central slot regardless of this
            # flag's value (it's unused during training), but it MUST be persisted as 0.0 or
            # inference would default to present=True and feed the real central frame in,
            # completely out-of-distribution for this model.
            print(f"context_only: forcing conditioning_probability=0.0 (was {args.conditioning_probability}) "
                  "-- required for denoise_sequence to auto-detect the context-only inference regime")
            args.conditioning_probability = 0.0
    if args.mode == "sinogram":
        if args.denoise_mode != "n2v":
            raise SystemExit("--mode sinogram only supports --denoise_mode n2v")
        if args.loss_type not in ("mae", "mse", "huber"):
            raise SystemExit("--mode sinogram requires --loss_type mae/mse/huber (pass --no-poisson_head "
                             "if needed) -- a sinogram's line-integral values are no longer "
                             "Poisson-distributed, so poisson_head/loss_type=poisson don't apply")
    if args.mode == "ambient_tweedie" and args.k != 0:
        # Replicate "Consistent Diffusion Meets Tweedie" literally: the network there is
        # h_theta(x_t, t) alone -- no side-channel conditioning beyond the noisy sample
        # itself, not even sigma_tn_map (a per-pixel map for the old heteroscedastic
        # path, or just a constant under --anscombe -- either way it's now excluded
        # from the network's input entirely, see ambient_tweedie.py). Two earlier
        # versions of this model conditioned on y directly, then on multi-frame
        # `context`, and BOTH let the network learn a shortcut (a function nearly
        # constant in x_t/sigma that reads off the side channel instead -- confirmed
        # empirically both times: pixel-identical output under one, near-zero
        # seed-sensitivity under the other). Forcing k=0 removes the remaining
        # shortcut at the source, same mechanism the --mode bootstrap branch above
        # already uses for its own "no context" architecture.
        print(f"ambient_tweedie: forcing k=0 (was {args.k}) -- no multi-frame context, "
              "network sees only [x_t]")
        args.k = 0
    norm_range = (args.norm_min, args.norm_max) if (args.norm_min is not None and args.norm_max is not None) else None

    warped_temporal_memmaps = None
    if args.warped_context_dir:
        try:
            warped_temporal_memmaps = resolve_warped_context_dir(
                args.warped_context_dir, args.k, args.include_mirror,
                args.neighborhoods, args.temporal_raw_pairs, args.deg_per_frame,
            )
        except FileNotFoundError as e:
            raise SystemExit(str(e))
        print(f"warped context: {len(warped_temporal_memmaps)} tap(s) from {args.warped_context_dir} "
              f"({sorted(warped_temporal_memmaps)})")

    dataset = TimeResolvedFrameDataset(
        mov_path=args.mov, memmap_path=args.memmap, k=args.k,
        frame_start=args.frame_start, frame_end=args.frame_end,
        crop=(args.crop_h, args.crop_w), include_mirror=args.include_mirror,
        neighborhoods=args.neighborhoods,
        norm_range=norm_range, extra_noise_dose=args.extra_noise_dose,
        max_samples=args.max_samples, seed=args.seed,
        axis_col=args.axis_col, deg_per_frame=args.deg_per_frame,
        n2n=(args.denoise_mode == "n2n"), p_range=(args.p_min, args.p_max), p_bins=args.p_bins,
        cond_angle_time=args.cond_angle_time, temporal_raw_pairs=args.temporal_raw_pairs,
        clean_target_memmap=args.clean_target_memmap, var_target_memmap=args.var_target_memmap,
        aux_channel_memmap=args.aux_channel_memmap,
        warped_temporal_memmaps=warped_temporal_memmaps,
        anscombe=args.anscombe,
    )
    n = len(dataset)
    test_size = max(1, int(args.test_fraction * n)) if n > 1 else 0
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_ds = torch.utils.data.Subset(dataset, idx[:-test_size] if test_size else idx)
    test_ds = torch.utils.data.Subset(dataset, idx[-test_size:] if test_size else idx)
    print(f"Dataset: {n} usable centres -> train={len(train_ds)}, test={len(test_ds)}")
    print(f"norm_min={dataset.norm_min:.3f}, norm_max={dataset.norm_max:.3f}, "
          f"in_channels(diffusion={dataset.in_channels_diffusion}, baseline={dataset.in_channels_baseline})")

    sample_size = (args.crop_h, args.crop_w)
    n2n = args.denoise_mode == "n2n"
    extra_cond_channels = ANGLE_TIME_COND_CHANNELS if args.cond_angle_time else 0
    if args.cond_angle_time and args.mode not in ("baseline", "bootstrap"):
        raise SystemExit("--cond_angle_time is currently only wired up for --mode baseline/bootstrap")
    # --poisson_head defaults to True (the standard architecture -- see README/project
    # memory). It's only wired for --mode baseline --denoise_mode n2v; for every other
    # mode/denoise_mode it's silently NOT applied below rather than raising, since
    # raising would turn the new default into a trap for diffusion/n2n runs that never
    # touch the flag at all.
    if args.loss_type == "poisson" and args.poisson_head:
        raise SystemExit("--loss_type poisson and --poisson_head are mutually exclusive "
                         "(single-channel Poisson-mean ablation vs two-head NB-NLL) -- pass "
                         "--no-poisson_head to use the legacy single-channel ablation")
    if args.loss_type == "poisson" and args.mode != "baseline":
        raise SystemExit("--loss_type poisson is currently only wired up for --mode baseline")
    if args.loss_type == "poisson" and n2n:
        raise SystemExit("--loss_type poisson is currently only wired up for --denoise_mode n2v")
    if args.mode == "diffusion":
        model = create_diffusion_unet(k=args.k, sample_size=sample_size, include_mirror=args.include_mirror,
                                      neighborhoods=args.neighborhoods, temporal_raw_pairs=args.temporal_raw_pairs)
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        n2n_loss_cls = DiffusionN2NCleanTargetLoss if args.clean_target_memmap else DiffusionN2NLoss
        loss_fn = (n2n_loss_cls(noise_scheduler, device, loss_type=args.loss_type,
                                    prediction_type=args.n2n_prediction_type,
                                    consistency_weight=args.n2n_consistency_weight,
                                    p_bins=args.p_bins) if n2n else
                   DiffusionN2VLoss(noise_scheduler, device, ratio=args.n2v_ratio, window=args.n2v_window,
                                    conditioning_probability=args.conditioning_probability, loss_type=args.loss_type))
        assert model.config.in_channels == dataset.in_channels_diffusion
    elif args.mode == "ambient_tweedie":
        # extra_cond_channels=0: sigma_tn_map is never fed to the network (see
        # ambient_tweedie.py's module docstring) -- with k=0 too, in_channels=1 (x_t alone).
        model = create_diffusion_unet(k=args.k, sample_size=sample_size, include_mirror=args.include_mirror,
                                      neighborhoods=args.neighborhoods, temporal_raw_pairs=args.temporal_raw_pairs,
                                      extra_cond_channels=0, condition_on_measurement=False)
        if args.anscombe:
            # A single GLOBAL scalar by construction (see data.py's anscombe_sigma_tn_norm) --
            # no percentile/upper-bound estimation needed, just read it off directly.
            sigma_tn_eff = float(dataset.anscombe_sigma_tn_norm)
            print(f"sigma_tn_eff (anscombe, exact global scalar) = {sigma_tn_eff:.5f} "
                  f"(schedule [{args.sigma_min}, {sigma_tn_eff:.5f}, {args.sigma_max}])")
        else:
            rng = np.random.default_rng(args.seed + 2)
            picks = rng.choice(len(dataset), size=min(args.sigma_tn_eff_sample_frames, len(dataset)), replace=False)
            pooled = np.concatenate([dataset[int(i)]["sigma_tn_map"].numpy().ravel() for i in picks])
            sigma_tn_eff = float(np.percentile(pooled, args.sigma_tn_eff_percentile))
            print(f"sigma_tn_eff (p{args.sigma_tn_eff_percentile} over {len(picks)} frames) = {sigma_tn_eff:.5f} "
                  f"(schedule [{args.sigma_min}, {sigma_tn_eff:.5f}, {args.sigma_max}])")
        adsm_sigma_t_max = sigma_tn_eff * args.adsm_upper_mult if args.adsm_upper_mult is not None else None
        if adsm_sigma_t_max is not None:
            print(f"adsm_sigma_t_max = {adsm_sigma_t_max:.5f} (sigma_tn_eff * {args.adsm_upper_mult}, "
                  f"vs unrestricted sigma_max={args.sigma_max})")
        if args.curriculum:
            print(f"curriculum: warmup_epochs={args.curriculum_warmup_epochs} "
                  f"buffer_size={args.curriculum_buffer_size} n_rungs={args.n_rungs} "
                  f"(1 rung opens per epoch -> full range open by epoch "
                  f"{args.curriculum_warmup_epochs + args.n_rungs - 1})")
            loss_fn = AmbientTweedieCurriculumLoss(device, sigma_tn_eff=sigma_tn_eff, sigma_min=args.sigma_min,
                                                   sigma_max=args.sigma_max, n_rungs=args.n_rungs,
                                                   consistency_weight=args.consistency_weight,
                                                   consistency_legs=args.consistency_legs,
                                                   adsm_sigma_t_max=adsm_sigma_t_max,
                                                   warmup_epochs=args.curriculum_warmup_epochs,
                                                   buffer_size=args.curriculum_buffer_size)
        elif args.faithful:
            print(f"faithful: consistency_weight={args.consistency_weight} "
                  f"consistency_warmup_steps={args.faithful_consistency_warmup_steps} "
                  f"consistency_ramp_steps={args.faithful_consistency_ramp_steps}")
            loss_fn = AmbientTweedieFaithfulLoss(device, sigma_tn_eff=sigma_tn_eff, sigma_min=args.sigma_min,
                                                 sigma_max=args.sigma_max,
                                                 consistency_weight=args.consistency_weight,
                                                 consistency_warmup_steps=args.faithful_consistency_warmup_steps,
                                                 consistency_ramp_steps=args.faithful_consistency_ramp_steps,
                                                 adsm_sigma_t_max=adsm_sigma_t_max)
        else:
            loss_fn = AmbientTweedieLoss(device, sigma_tn_eff=sigma_tn_eff, sigma_min=args.sigma_min,
                                         sigma_max=args.sigma_max, n_rungs=args.n_rungs,
                                         consistency_weight=args.consistency_weight, ema_decay=args.ema_decay,
                                         consistency_legs=args.consistency_legs,
                                         adsm_sigma_t_max=adsm_sigma_t_max)
        assert model.config.in_channels == dataset.in_channels_ambient_tweedie
    elif args.mode == "bootstrap":
        base_model, _ = load_denoiser(args.base_checkpoint, device=device)
        model = create_baseline_unet(k=0, sample_size=sample_size, poisson_head=True)
        base_present = float(base_cfg.get("conditioning_probability", 1.0)) > 0.0
        loss_fn = BootstrapPoissonLoss(
            base_model, device, norm_min=dataset.norm_min, norm_max=dataset.norm_max,
            base_dose=float(args.extra_noise_dose), base_present=base_present,
            input_mode=args.bootstrap_input, direction=args.bootstrap_direction,
            poisson_beta_nll_power=args.poisson_beta_nll_power,
        )
        assert model.config.in_channels == 1
    elif args.mode == "refine":
        base_model, _ = load_denoiser(args.base_checkpoint, device=device)
        base_present = float(base_cfg.get("conditioning_probability", 1.0)) > 0.0
        model = create_baseline_unet(k=args.k, sample_size=sample_size, include_mirror=args.include_mirror,
                                     neighborhoods=args.neighborhoods, temporal_raw_pairs=args.temporal_raw_pairs,
                                     poisson_head=True)
        loss_fn = RefinementLoss(
            base_model, device, norm_min=dataset.norm_min, norm_max=dataset.norm_max,
            base_dose=float(args.extra_noise_dose), base_present=base_present,
            poisson_beta_nll_power=args.poisson_beta_nll_power,
        )
        assert model.config.in_channels == dataset.in_channels_baseline
    elif args.mode == "sinogram":
        sino_vol_shape = (args.crop_h, args.crop_w)
        auto_angles, auto_cols = choose_sino_shape(sino_vol_shape)
        sino_num_angles = args.sino_num_angles if args.sino_num_angles is not None else auto_angles
        sino_det_cols = args.sino_det_cols if args.sino_det_cols is not None else auto_cols
        sino_transform = SinoTransform(sino_vol_shape, num_angles=sino_num_angles, det_cols=sino_det_cols,
                                       angle_max_deg=args.sino_angle_max_deg, device=device)
        print(f"sinogram: vol_shape={sino_vol_shape} -> sino shape=({sino_transform.num_angles},"
              f"{sino_transform.det_cols}), angle_max={sino_transform.angle_max_deg}")
        sino_norm_min, sino_norm_max = fit_sino_norm_range(
            dataset, sino_transform, n_sample=args.sino_norm_sample_frames, seed=args.seed)
        print(f"sino_norm=({sino_norm_min:.3f},{sino_norm_max:.3f}) fit over "
              f"{args.sino_norm_sample_frames} frames")
        model = create_baseline_unet(k=args.k, sample_size=(sino_transform.num_angles, sino_transform.det_cols),
                                     include_mirror=args.include_mirror, neighborhoods=args.neighborhoods,
                                     temporal_raw_pairs=args.temporal_raw_pairs, poisson_head=False)
        loss_fn = SinogramN2VLoss(
            device, sino_transform, norm_min=dataset.norm_min, norm_max=dataset.norm_max,
            sino_norm_min=sino_norm_min, sino_norm_max=sino_norm_max,
            ratio=args.n2v_ratio, window=args.n2v_window,
            conditioning_probability=args.conditioning_probability, loss_type=args.loss_type,
            edge_weight=args.edge_weight,
        )
        assert model.config.in_channels == dataset.in_channels_baseline
    elif args.mode == "noise2clean":
        model = create_baseline_unet(k=args.k, sample_size=sample_size, include_mirror=args.include_mirror,
                                     neighborhoods=args.neighborhoods, temporal_raw_pairs=args.temporal_raw_pairs,
                                     poisson_head=False)
        loss_fn = NoiseToCleanLoss(device, loss_type="mse")
        assert model.config.in_channels == dataset.in_channels_baseline
    elif args.mode == "context_only":
        extra_cond_channels_baseline = extra_cond_channels + (len(args.aux_channel_memmap) if args.aux_channel_memmap else 0)
        model = create_baseline_unet(k=args.k, sample_size=sample_size, include_mirror=args.include_mirror,
                                     neighborhoods=args.neighborhoods, extra_cond_channels=extra_cond_channels_baseline,
                                     temporal_raw_pairs=args.temporal_raw_pairs, poisson_head=False)
        loss_fn = ContextOnlyNoiseToCleanLoss(device, loss_type="mse")
        assert model.config.in_channels == dataset.in_channels_baseline
    else:
        # poisson_head defaults to True but is only wired for n2v (see the guard note
        # above) -- silently disable it for n2n rather than raising, since it was never
        # explicitly requested by an n2n run that just didn't pass --no-poisson_head.
        effective_poisson_head = args.poisson_head and not n2n
        # Give the two-head Poisson belief a running start at the REAL count
        # magnitude -- otherwise the bias sits near its random init (softplus(~0)
        # ~= 0.7) for the whole run, since Adam's step size is roughly bounded by
        # the learning rate regardless of gradient size (drifting a bias by a few
        # hundred units in a few thousand steps is effectively unreachable).
        poisson_mean_only = args.loss_type == "poisson"
        poisson_init_mean = ((dataset.norm_min + dataset.norm_max) / 2.0
                             if (effective_poisson_head or poisson_mean_only) else None)
        # x0 is synthesised via add_poisson_noise(..., dose=extra_noise_dose) when set,
        # so the NB-NLL/posterior-mean must know that SAME thinning fraction (native,
        # non-thinned target => dose=1.0) -- see nb_head.py's dose-aware derivation.
        # (poisson_mean_only's plain Poisson NLL is dose-invariant up to a constant
        # factor, so it doesn't need this -- only poisson_head's NB-NLL does.)
        poisson_dose = float(args.extra_noise_dose) if (effective_poisson_head and args.extra_noise_dose is not None) else 1.0
        extra_cond_channels_baseline = extra_cond_channels + (len(args.aux_channel_memmap) if args.aux_channel_memmap else 0)
        model = create_baseline_unet(k=args.k, sample_size=sample_size, include_mirror=args.include_mirror,
                                     neighborhoods=args.neighborhoods, extra_cond_channels=extra_cond_channels_baseline,
                                     temporal_raw_pairs=args.temporal_raw_pairs, poisson_head=effective_poisson_head,
                                     poisson_init_mean=poisson_init_mean, poisson_init_var=poisson_init_mean)
        sigma_read2 = None
        if args.gaussian_floor:
            assert args.sigma_read2_map, "--gaussian_floor requires --sigma_read2_map"
            sigma_read2 = torch.from_numpy(np.load(args.sigma_read2_map)).float()
        loss_fn = (BaselineN2NLoss(device, loss_type=args.loss_type) if n2n else
                   BaselineN2VLoss(device, ratio=args.n2v_ratio, window=args.n2v_window,
                                   conditioning_probability=args.conditioning_probability, loss_type=args.loss_type,
                                   edge_weight=args.edge_weight, poisson_head=effective_poisson_head,
                                   norm_min=dataset.norm_min, norm_max=dataset.norm_max,
                                   poisson_beta_nll_power=args.poisson_beta_nll_power,
                                   poisson_dose=poisson_dose,
                                   gaussian_floor=args.gaussian_floor, sigma_read2=sigma_read2))
        assert model.config.in_channels == dataset.in_channels_baseline
    print(f"mode={args.mode}  denoise={args.denoise_mode}  model params: {sum(p.numel() for p in model.parameters()):,}")

    if args.save_checkpoint:
        checkpoint_path = args.save_checkpoint
    else:
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = f"checkpoints/tr_denoise_{args.mode}_{args.exp_name}.pt"
    os.makedirs(Path(checkpoint_path).parent, exist_ok=True)

    # For --mode bootstrap, args.k/include_mirror/neighborhoods/temporal_raw_pairs/cond_angle_time
    # describe the DATASET feeding the frozen base model (e.g. k=1), not the shape of THIS
    # checkpoint's own model (always k=0, in_channels=1, no context, no extra cond channels) --
    # persist the latter, or a future load_denoiser/build_model call would reconstruct the wrong
    # (much wider) architecture and fail to load this state_dict.
    if args.mode == "bootstrap":
        cfg_k, cfg_include_mirror = 0, False
        cfg_neighborhoods, cfg_temporal_raw_pairs, cfg_cond_angle_time = "both", False, False
    else:
        cfg_k, cfg_include_mirror = args.k, args.include_mirror
        cfg_neighborhoods = args.neighborhoods
        cfg_temporal_raw_pairs, cfg_cond_angle_time = args.temporal_raw_pairs, args.cond_angle_time

    with open(checkpoint_path.replace(".pt", "_config.json"), "w") as f:
        json.dump({
            "mode": args.mode, "denoise_mode": args.denoise_mode, "k": cfg_k,
            "crop": [args.crop_h, args.crop_w],
            "include_mirror": cfg_include_mirror, "neighborhoods": cfg_neighborhoods,
            "temporal_raw_pairs": cfg_temporal_raw_pairs,
            "in_channels": model.config.in_channels,
            "norm_min": dataset.norm_min, "norm_max": dataset.norm_max,
            "deg_per_frame": args.deg_per_frame, "axis_col": args.axis_col,
            "profile": args.profile,
            "frame_start": args.frame_start, "frame_end": args.frame_end,
            "extra_noise_dose": args.extra_noise_dose,
            "n2v_ratio": args.n2v_ratio, "n2v_window": args.n2v_window,
            "conditioning_probability": args.conditioning_probability,
            "cond_angle_time": cfg_cond_angle_time,
            "loss_type": args.loss_type, "edge_weight": args.edge_weight,
            "poisson_head": effective_poisson_head if args.mode == "baseline" else (args.mode in ("bootstrap", "refine")),
            "poisson_warmup_frac": args.poisson_warmup_frac,
            "poisson_beta_nll_power": args.poisson_beta_nll_power,
            "poisson_init_mean": poisson_init_mean if args.mode == "baseline" else None,
            "poisson_dose": poisson_dose if args.mode == "baseline" else None,
            "p_min": args.p_min, "p_max": args.p_max, "p_bins": args.p_bins,
            "n2n_prediction_type": args.n2n_prediction_type,
            "n2n_consistency_weight": args.n2n_consistency_weight,
            "clean_target_memmap": args.clean_target_memmap,
            "var_target_memmap": args.var_target_memmap,
            "aux_channel_memmap": args.aux_channel_memmap,
            "warped_context_dir": args.warped_context_dir if args.mode in ("baseline", "refine") else None,
            "sigma_min": args.sigma_min, "sigma_max": args.sigma_max,
            "sigma_tn_eff": sigma_tn_eff if args.mode == "ambient_tweedie" else None,
            "anscombe": args.anscombe if args.mode == "ambient_tweedie" else False,
            "anscombe_z_min": dataset.anscombe_z_min if args.anscombe else None,
            "anscombe_z_max": dataset.anscombe_z_max if args.anscombe else None,
            "n_rungs": args.n_rungs, "consistency_weight": args.consistency_weight,
            "ema_decay": args.ema_decay, "consistency_legs": args.consistency_legs,
            "adsm_upper_mult": args.adsm_upper_mult if args.mode == "ambient_tweedie" else None,
            "curriculum": args.curriculum if args.mode == "ambient_tweedie" else False,
            "curriculum_warmup_epochs": args.curriculum_warmup_epochs if args.curriculum else None,
            "curriculum_buffer_size": args.curriculum_buffer_size if args.curriculum else None,
            "faithful": args.faithful if args.mode == "ambient_tweedie" else False,
            "faithful_consistency_warmup_steps": args.faithful_consistency_warmup_steps if args.faithful else None,
            "faithful_consistency_ramp_steps": args.faithful_consistency_ramp_steps if args.faithful else None,
            "mov": args.mov,
            "base_checkpoint": args.base_checkpoint if args.mode in ("bootstrap", "refine") else None,
            "bootstrap_input": args.bootstrap_input if args.mode == "bootstrap" else None,
            "bootstrap_direction": args.bootstrap_direction if args.mode == "bootstrap" else None,
            "sino_num_angles": sino_transform.num_angles if args.mode == "sinogram" else None,
            "sino_det_cols": sino_transform.det_cols if args.mode == "sinogram" else None,
            "sino_angle_max_deg": sino_transform.angle_max_deg if args.mode == "sinogram" else None,
            "sino_norm_min": sino_norm_min if args.mode == "sinogram" else None,
            "sino_norm_max": sino_norm_max if args.mode == "sinogram" else None,
        }, f, indent=2)

    if args.load_checkpoint:
        try:
            ckpt = torch.load(args.load_checkpoint, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded weights from {args.load_checkpoint}")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not load {args.load_checkpoint}: {exc}. Training from scratch.")

    model.to(device)
    exp = PyTorchExperiment(
        args=vars(args), train_dataset=train_ds, test_dataset=test_ds,
        batch_size=args.batch_size, model=model, loss_fn=loss_fn,
        checkpoint_path=checkpoint_path, experiment_name=args.exp_name,
        with_wandb=args.wandb, num_workers=args.num_workers, seed=args.seed,
        save_always=True, mixed_precision=args.mixed_precision,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(exp.train_loader) * args.epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    if args.mode == "baseline" and effective_poisson_head:
        loss_fn.poisson_warmup_steps = int(args.poisson_warmup_frac * total_steps)
        print(f"poisson_head: mu-only warm start for {loss_fn.poisson_warmup_steps}/{total_steps} steps, "
              f"then full NB-NLL (beta_nll_power={args.poisson_beta_nll_power})")
    elif args.mode == "bootstrap":
        loss_fn.poisson_warmup_steps = int(args.poisson_warmup_frac * total_steps)
        print(f"bootstrap poisson_head: mu-only warm start for {loss_fn.poisson_warmup_steps}/{total_steps} steps, "
              f"then full NB-NLL (beta_nll_power={args.poisson_beta_nll_power})")
    exp.train(args.epochs, optimizer, scheduler=lr_scheduler)


if __name__ == "__main__":
    main()
