# tr_diffusion — time-resolved conditional frame denoiser

Self-supervised denoising of individual time-resolved CT **projection frames**
from the `212_Wunderkerze2` continuous-rotation acquisition
(`.../time_resolved/212_Wunderkerze2/212_Wunderkerze2.mov`, 956 526 frames,
528×128, per-frame-normalised HEVC + `.norm.npz` sidecar).

The question this package exists to answer: **can a conditional diffusion model
denoise these frames better than a single-pass regression UNet with the same
capacity and the same conditioning?** The math is shared; only the training
objective and the inference procedure differ.

## Idea

For a central frame `i` we predict a denoised version of it from context that is
physically correlated with it but carries *independent* noise:

- **Rotation-adjacent** frames `i±1 … i±k` — nearly the same view a few frames
  away (rotation ≈ 1.8014 °/frame).
- **Same-angle temporal** frames `i±P … i±kP` — the *identical* viewing geometry
  one or more full turns away (`P ≈ 199.844` frames/turn). `P` is non-integer, so
  these are **linearly interpolated** between bracketing frames.

The central frame is the target `x₀`, but it must never be seen cleanly or the
task is trivial. It is only ever provided through a **Noise2Void blind-spot
corruption** (a small random fraction of pixels replaced by a neighbour value),
and the loss is evaluated **only at those blind-spot pixels** — so the network
must infer them from the (independently-noisy) neighbours, i.e. it denoises
(Noise2Void × Noise2Noise). Training also drops the central channel entirely with
some probability ("with / without central"); those samples get a full-frame loss
(pure conditional-on-neighbours objective).

### Channel contract (single source of truth: `geometry.build_context_layout`)

```
context order:  rot(i-k…i-1), rot(i+1…i+k), tmp(i-k…i-1 turns), tmp(i+1…i+k turns)   -> 4k channels
diffusion input: [ x_t, corrupted_central, <context> ]   in_channels = 2 + 4k
baseline  input: [      corrupted_central, <context> ]   in_channels = 1 + 4k
```

`k` is a parameter (default **3** → 20 diffusion channels, see the k-ablation
below). `--include_mirror` appends the two 180° half-turn taps (`i±P/2`,
flipped about the rotation axis col ≈ 269.85), `+2` channels.

## Value space & the extra-noise regime

Frames are denormalised to a common **count** space via the `.norm.npz` sidecar
(`counts = per_frame_min + decoded/65535·(per_frame_max−per_frame_min)`), cropped
to `(128, 512)` around the rotation axis, then affinely mapped to `[-1, 1]` with a
single `(norm_min, norm_max)` fit saved to the checkpoint config.

- **native** (default): fully self-supervised, `x₀` = measured central frame.
- **extra-noise** (`--extra_noise_dose d`): extra Poisson noise (dose thinning) is
  added independently to every frame; the *original* measured central is kept as
  a lower-noise **pseudo-reference** for PSNR/SSIM across noise levels. Not a true
  GT, but provably less noisy than the input.

## Files

| file | role |
|------|------|
| `geometry.py` | rotation constants, context layout, usable-range math |
| `frames.py` | `.mov`/sidecar decode → counts; ffmpeg + memmap sources |
| `extract_frames.py` | pre-extract a frame range to a fast uint16 memmap |
| `data.py` | `TimeResolvedFrameDataset` (central + context, native/extra-noise) |
| `n2v.py` | blind-spot corruption + mask |
| `noise.py` | extra Poisson noise (dose thinning) |
| `model.py` | diffusers `UNet2DModel` builders (diffusion / baseline) |
| `losses.py` | `DiffusionN2VLoss`, `BaselineN2VLoss` (masked) |
| `pipeline.py` | conditional DDIM inference w/ per-step N2V resampling (phase 2) |
| `train.py` | trainer (`--mode diffusion|baseline`) on `pytorch_base` |

## Usage

Extract the working range once (recommended for training):

```bash
python -m sdate.tr_diffusion.extract_frames \
  --mov .../212_Wunderkerze2/212_Wunderkerze2.mov \
  --out .../212_Wunderkerze2/frames_400k_600k.u16 \
  --frame_start 400000 --frame_end 600000
```

Train the diffusion model and the baseline (same data, same context; `--k 3`
and, for the baseline, the two-head `--poisson_head` are now the defaults --
no need to pass either explicitly):

```bash
python -m sdate.tr_diffusion.train --mode diffusion \
  --mov .../212_Wunderkerze2.mov --memmap .../frames_400k_600k.u16 \
  --batch_size 16 --epochs 100 --exp_name k3 --wandb

python -m sdate.tr_diffusion.train --mode baseline \
  --mov .../212_Wunderkerze2.mov --memmap .../frames_400k_600k.u16 \
  --batch_size 16 --epochs 100 --exp_name k3_poissonhead --wandb
```

Without `--memmap` it decodes straight from the `.mov` via ffmpeg (slower; fine
for small runs / notebooks).

## Inference (phase 2)

`pipeline.denoise_frames(model, central, context, ...)` runs guided DDIM,
resampling the blind-spot corruption on every step; the baseline is a single
`pipeline.denoise_frames_baseline(...)` pass. Wired and shape-checked, but the
project scope so far is training — evaluate/compare here next.

## Status / notes

- Rotation calibration and the axis column come from
  `notebooks/wunderkerze_rotation_calibration.ipynb` (memory
  `project-wunderkerze2-rotation`). Only the *rate* is calibrated, not the
  absolute angle of frame 0.
- Default frame range 400 000–600 000 (rate validated there; constant to <0.05 %).
- `ffmpeg`/`ffprobe` static builds live in `/myhome/bin` (not on default PATH).

## Key experimental findings (dose 0.05, `k=1` unless noted)

**Best inference recipe: single-shot `pred_x0`, not ancestral DDIM.** Noise the
measurement to a moderate timestep (`t≈400–500`, a broad flat plateau — exact
value barely matters), one forward pass, read off `pred_x0`. Full ancestral DDIM
toward `t=0` is *counterproductive*: the corrupted-central conditioning leaks the
noisy input back in, so it reconverges to the measurement rather than denoising.
Short multi-step refinement inside `[400,500]` doesn't beat single-shot either.
See `pipeline.pred_x0_ensemble` (recommended) vs `pipeline.denoise_frames`
(ancestral, inferior) / `pipeline.partial_diffusion` (interval refinement, no
better than single-shot).

**Posterior-mean averaging is the real lever.** Average `B` independent
single-shot draws (fresh diffusion-noise + blind-spot mask per draw) → Monte
Carlo estimate of the MMSE denoiser `E[x0|y]`. Gains saturate fast: the knee is
at **B≈4**, which already both beats the single-pass baseline UNet *and* retains
much more high-frequency detail than the baseline (baseline blurs to buy its
PSNR). B=8 is marginally better and the safe default if compute allows; B≥16 buys
almost nothing further (diminishing returns, variance ∝ 1/B).

**Diffusion vs baseline UNet (same capacity, same conditioning, both dose-matched):**
single diffusion draw *loses* to the baseline on PSNR/SSIM; the baseline's
smoothing wins pointwise metrics but at a real cost in reconstructed sharpness
(see the reconstruction test below — baseline recovers ~51% of GT gradient
energy vs ~81% for single-shot diffusion, ~63–67% for posterior-mean B=4–32).
Posterior-mean B≈4–8 diffusion is the practical sweet spot: PSNR/SSIM ≥ baseline
*and* visibly sharper. A diffusion→baseline cascade (re-denoising the diffusion
output with the baseline) is strictly worse than either alone — it re-smooths
away the sharpness diffusion adds, for no PSNR gain. Don't cascade.

**Reconstruction test (the real end-to-end signal):** even though the
denoiser's *projection-space* edge over the noisy input is modest per frame, the
*downstream FBP reconstruction* gain is large and robust — **+10.6 dB / +0.53
SSIM** at 256² (full 400k–500k sequence, 996 windows) and **+15.9 dB / +0.70
SSIM** at full 512² resolution (noisy floor is essentially unusable, SSIM≈0.10,
once every high-frequency streak is visible at full res). This is the metric
that matters: 5%-dose (20× dose-reduction) acquisitions become reconstructable.
See `reconstruct.py`, `scripts/tr_diffusion_reconstruct_run.py`,
`notebooks/tr_diffusion_reconstruction.ipynb`.

### Context-radius (`k`) ablation — **k=3 is optimal, use it**

Trained diffusion at `k=1` (in_channels=6), `k=2` (10), `k=3` (14), identical
settings otherwise (dose 0.05, ~epoch 23–27 each). Evaluated on the same held-out
128-frame block:

| k | single-shot PSNR / SSIM | posterior-mean B=8 PSNR / SSIM |
|---|---|---|
| 1 | 30.72 / 0.695 | 33.05 / 0.812 |
| 2 | 30.89 / 0.702 | 33.08 / 0.809 |
| **3** | **31.03 / 0.704** | **33.26 / 0.811** |

More context helps **monotonically** — k=1 was actually the *most*-trained
checkpoint of the three yet still the weakest, so the ordering is real, not an
epoch artifact. The gain is modest (~+0.2–0.3 dB k=1→k=3, SSIM ~flat) but free:

**Model size is essentially unaffected by `k`.** Only the first conv layer's
input width changes (`in_channels = 2+4k`); every downstream layer is identical.
Measured: 28,530,241 (k=1) → 28,532,545 (k=2) → 28,534,849 (k=3) parameters —
**+2,304 params per k step (0.008%)**; checkpoint size 342.96 → 342.99 → 343.02
MB. No GPU-memory-for-weights, inference-FLOPs, or capacity penalty.

**The only real cost of higher k is data-loading I/O**: more neighbour frames
read per sample (7 at k=1 → 13 at k=2 → 19 at k=3, since rotation and
interpolated-temporal taps both scale with k), so training and full-sequence
denoising run somewhat slower, and the usable frame range shrinks slightly at
the sequence edges (~1% at k=3, from the larger `±kP` margin in
`usable_frame_range`). No downstream (reconstruction-stage) cost at all — the
denoised output is one frame regardless of k.

**Conclusion: default to k=3.** There is no real drawback to it; k=1/k=2 are
only preferable if data-loading throughput during training/denoising is the
bottleneck. Checkpoints: `checkpoints/tr_denoise_diffusion_k{1,2,3}_dose005.pt`;
ablation eval script: `scripts/tr_diffusion_eval_k.py`.

### Baseline loss correction — **two-head Gamma-Poisson NB-NLL is now the
### standard baseline (`--poisson_head`, default ON), k=1 dose 0.05**

The plain Huber/MAE/MSE baseline scores its point estimate against a
homoscedastic loss, which is the wrong noise model for Poisson-thinned counts
(variance = mean, not constant) -- and independently, the blind-spot
architecture means *any* point-estimate loss (however correct the likelihood)
converges to `E[x | context]` and can't get sharper than that, since the
network never sees the pixel it's predicting. Two follow-up fixes, tested as
separate ablations on `212_Wunderkerze2` (same k=1 dose-0.05 recipe as the
other findings above, 8 epochs from scratch, scored via the same 247-window
flat/dark+destripe FBP reconstruction as everywhere else in this doc):

| variant | PSNR | SSIM | sharpness (1.0 = matches GT) |
|---|---|---|---|
| baseline (Huber) | 26.27 | 0.602 | 0.531 |
| **poisson-mean** (likelihood fix alone, single-channel Poisson NLL) | **26.54** | **0.615** | 0.487 |
| **poisson_head** (2-channel Gamma-Poisson `(mu,var)` NB-NLL + exact posterior-mean combination with the real observation at inference) | 25.58 | 0.590 | **0.921** |

**Fixing the likelihood alone (poisson-mean) gives a small, genuine accuracy
gain (+0.27 dB / +0.013 SSIM) but does NOT fix the blur** -- confirming the
architecture theory above: measured pixelwise correlation between
`poisson_head`'s own `mu` (its context-only belief, before combining with the
observation) and the standalone poisson-mean model's point estimate is
**0.999** -- they are, empirically, the same estimator. Sharpness only comes
back once the posterior step reaches back to the real observed pixel
(`poisson_head`, sharpness 0.92 vs baseline's 0.53), at a modest PSNR/SSIM
cost (imports some residual noise along with the real high-frequency detail).

**Conclusion: `poisson_head` is the standard baseline going forward, default
ON.** It strictly supersedes both the plain Huber baseline and the
single-channel `loss_type="poisson"` ablation, because its own `mu` output IS
that same point estimate (see above) -- train ONE two-head model and get BOTH
behaviours at inference time, no need to train the single-channel variant
separately:

- `poisson_posterior=True` (default) -- sharper, the exact Bayes combination.
- `poisson_posterior=False` -- `mu` alone, the single-head-equivalent point
  estimate (slightly better whole-frame PSNR/SSIM, blurred like any point
  estimate).

See `poisson_posterior` in `pipeline.denoise_frames_baseline` /
`reconstruct.denoise_sequence`/`cascade_sequence`. Pass `--no-poisson_head`
(and `--loss_type poisson`/`mae`/`mse`/`huber`) to train the legacy
single-channel path instead -- kept only for loading/comparing already-trained
checkpoints. Math: `nb_head.py`. Ablation pipelines:
`scripts/tr_diffusion_poissonhead_pipeline.py`,
`scripts/tr_diffusion_poissonmean_pipeline.py`. No retraining of existing
checkpoints was done for this change -- it only affects the CODEBASE DEFAULTS
for future training runs; old checkpoints keep loading exactly as before
(their own saved `poisson_head`/`loss_type` config is always what's used, not
today's CLI default).

### Bootstrap self-distillation (`--mode bootstrap`, 2026-08-09/10)

Idea: instead of the usual multi-frame + N2V-corrupted-central conditioning,
train a SECOND model (same UNet, same two-head Gamma-Poisson NB-NLL) to
predict the noisy measurement `y` from ONLY a frozen, already-trained
`poisson_head` baseline checkpoint's own context-only belief for that frame --
`in_channels=1`, `k=0`, no context at all. A network can't predict independent
Poisson noise from a smoothed input, so minimising the same NLL forces this
second model toward a genuine `E[y | input]` estimate; the question is whether
that's sharper than the base model's own output. `--bootstrap_input` selects
what exactly is fed in: `mean` (the base's deterministic prior mean `mu`) or
`sample` (a FRESH draw from its moment-matched Gamma(mu,var) prior every call,
`nb_head.gamma_sample` -- closer in spirit to Noise2Noise since the input is
never the same value twice, though the input's noise mechanism is Gamma, not a
Poisson split, so it's not a literal N2N pair). The regression target is drawn
from an INDEPENDENT noise realisation (a fresh `add_poisson_noise` draw off the
dataset's pre-thinning `reference` frame, not the draw fed to the frozen
model) -- otherwise the base model's belief would already leak that exact
target through its own corrupted-central input and the loss would trivially
reward copying the leak back out.

Implementation: `losses.BootstrapPoissonLoss`; `train.py --mode bootstrap
--base_checkpoint <baseline .pt> [--bootstrap_input mean|sample]` (geometry/
channel-shape config -- k, crop, neighborhoods, temporal_raw_pairs,
cond_angle_time, axis_col, deg_per_frame, norm_min/max, extra_noise_dose -- is
force-inherited from the base checkpoint so its forward pass stays
in-distribution; only `--profile`/`--mov`/`--memmap`/`--frame_start`/
`--frame_end` are free to differ). Inference: `pipeline.bootstrap_belief` /
`bootstrap_input_from_belief` / `denoise_frames_bootstrap` (shared with
training so the two never drift apart), wired into
`reconstruct.denoise_sequence` (auto-detects `mode=="bootstrap"`, loads the
frozen base checkpoint, builds the dataset from the base's own geometry) --
same cached-memmap output format as every other mode, so it plugs directly
into `run_windows`/the rest of the eval toolkit.

**Training length: 5-8 epochs is enough, not more.** Two ablations trained on
`212_Wunderkerze2` (`tr_denoise_baseline_k1_dose005_poissonhead.pt` as the
frozen base, k=1, dose 0.05, frames 400k-500k, batch=8): `mean` and `sample`
conditioning. Per-epoch train/test loss flatlined from ~epoch 5 onward for
both (train ~-116.21, test ~-116.67, essentially unchanged to the 4th decimal
epoch 5 through 11) -- confirmed this is genuine convergence, NOT the
heteroscedastic-collapse failure mode (var inflating, model ignoring its
input): fed the live checkpoints synthetic low/high-value and spatial-ramp
inputs, `mu` tracked the input value and spatial structure closely (ramp
input/output column-profile correlation r=0.999 for both). Both runs were
stopped once this was confirmed; **default to ~5-8 epochs for this mode**
(the plateau, if anything, arrives EARLIER than the k=1 baseline's own ~8-12
epoch convergence elsewhere in this doc -- a single-channel, context-free
input is a much easier fit than the multi-frame baseline).

**Full result (2026-08-10, `scripts/tr_diffusion_recon_bootstrap.py`, same
247-window flat/dark+destripe FBP convention as the poisson_head table above
-- frame range 400300-450000, det_bin=1):**

| variant | PSNR | SSIM | sharpness (1.0 = matches GT) |
|---|---|---|---|
| poissonhead (base, posterior-mean combined with the real observation) | 25.58 | 0.590 | **0.921** |
| **bootstrap mean** | **26.74** | **0.621** | 0.408 |
| bootstrap sample (single Gamma draw, `num_samples=1`) | 26.64 | 0.615 | 0.434 |
| noisy floor | 4.56 | 0.017 | 14.743 |

**Answering the original question: no, bootstrapping does NOT come out
sharper -- it trades sharpness for accuracy, in the opposite direction from
what `poisson_head`'s posterior combination bought.** Both bootstrap variants
genuinely beat the base model's posterior-mean on PSNR (+1.1-1.2 dB) and SSIM
(+0.025-0.031) -- a real, if modest, accuracy gain. But sharpness collapses to
0.41-0.43, WORSE than even the plain-Huber baseline (0.531) or the single-
channel poisson-mean ablation (0.487) earlier in this doc, and less than half
of poisson_head's 0.921. Mechanistic reason: the bootstrap model's input
(`mu`) IS ALREADY the base model's context-only, pre-observation belief -- the
exact quantity the architecture note above says can never get sharper than
`E[x | context]`. `denoise_frames_bootstrap` reads off the bootstrap model's
own `mu` too (no posterior combination with a real observation at inference,
unlike the base model's `poisson_posterior=True` step) -- so this is a
regression fit ON TOP OF an already-regressed quantity, with the real pixel
never re-entering the pipeline a second time. Two layers of "predict the
context-conditional mean" compounds the blur rather than removing it; PSNR/
SSIM reward that (regression-to-the-mean lowers expected squared error) while
sharpness penalises it -- textbook bias-variance trade, just isolated cleanly
here. `mean` vs `sample` conditioning make almost no difference either way
(consistent with the near-identical training loss curves above) -- injecting
Gamma-sampling noise into the input doesn't change what the model converges
to.

**Follow-up 1 -- posterior-combining the bootstrap model's own `(mu, var)`
with the real observed pixel does NOT help (2026-08-10).** Tested both
`bootstrap_mean_posterior` (`poisson_posterior=True` on top of `mean`
conditioning) and `bootstrap_sample_posterior_b8` (same, on top of `sample`
conditioning + 8-draw averaging, see below) via
`pipeline.denoise_frames_bootstrap(..., poisson_posterior=True)`
(`nb_head.posterior_mean`, same Bayes-update step that gives `poisson_head`
its sharpness). Quick 2-window check (`scripts/tr_diffusion_recon_bootstrap_b8_quick.py`,
frames 400300-400900): `bootstrap_mean_posterior` vs `bootstrap_mean` moved
PSNR/SSIM/sharpness by <0.001/<0.001/<0.001 -- fully negligible. Root cause:
the bootstrap model's own `var` head has effectively collapsed toward zero
(confirmed earlier via a single-frame CPU check feeding synthetic inputs
directly to the live checkpoint -- the "loss-attenuation failure mode"
`nb_head.py`'s own docstring warns about), so `posterior_mean` has almost no
real observation left to blend in. The mechanistic story above already
predicted this; this confirms it rather than changing the conclusion.

**Follow-up 2 -- averaging more Gamma draws for `sample` conditioning gives a
small, real, and quickly-diminishing gain (2026-08-10).** The original "single
Gamma draw" result above (`num_samples=1`) was re-checked against
`num_samples=8` (`pipeline.denoise_frames_bootstrap(..., num_samples=8)`,
i.e. 8 independent fresh Gamma-posterior draws averaged at inference -- an
MMSE ensemble over the sampling mechanism). Same 2-window quick check:

| variant | PSNR | SSIM | sharpness |
|---|---|---|---|
| bootstrap sample, `num_samples=1` | 25.46 | 0.638 | 0.476 |
| bootstrap sample, `num_samples=8` avg | 25.54 | 0.640 | 0.469 |
| bootstrap sample, `num_samples=8` avg + posterior-combine | 25.54 | 0.640 | 0.469 |

+0.08 dB PSNR / +0.002 SSIM for 8x the inference compute, sharpness essentially
unchanged. Consistent with sqrt(N) MMSE noise reduction against a
already-small per-draw variance: most of the benefit of "sample" over "mean"
conditioning already comes from training-time regularisation (fresh noise
every step, not a fixed target), not from inference-time ensembling. Given
the fast-diminishing returns, did NOT scale up to the originally-planned
`num_samples=64` full-sequence run (would have cost ~9h of A100 time for an
expected further gain on the order of 0.1 dB) -- the `num_samples=1` result
already captures the large majority of the achievable benefit.

**Follow-up 3 -- reversed direction (user's N2N-symmetry idea, 2026-08-10/11):
swap which noisy view is input vs. target.** Instead of (base model's belief
-> fresh measurement), train (the REAL measurement -> a fresh Gamma-posterior
draw of the base model's belief). `BootstrapPoissonLoss(direction="reverse")`
(`train.py --bootstrap_direction reverse`, ignores `--bootstrap_input` -- the
input is always the raw measurement): model input = `instance["central"]`
directly (no base-model call needed to build it); target =
`nb_head.gamma_sample` of the frozen base model's `(mu, var)` belief for the
SAME frame (scored at `dose=1.0` regardless of the base's own thinning dose,
since the target is a continuous rate draw, not a further dose-thinned
count -- the mean-value/Bregman property of the Poisson/NB NLL still makes
`mu` converge to `E[target | measurement]` regardless of the exact noise
model assumed for `var`). The key structural consequence: at INFERENCE this
direction needs neither the frozen base model nor multi-frame context at
all -- the trained network stands alone on the raw single-pixel measurement
(`pipeline.denoise_frames_bootstrap_reverse`), testing whether it distilled
the context-rich model's belief into a context-free one-shot regressor.

Trained on the same base checkpoint/frame range/dose as the two variants
above (k=1 base, dose 0.05, batch=8); converged even FASTER than the forward
direction -- test loss flatlined between epoch 1 and 2 already (change
<0.001%, vs. forward's ~5 epochs), consistent with this being an even
simpler function to fit (a direct measurement -> smoothed-target regression,
vs. forward's base-belief -> fresh-measurement regression). Stopped at epoch
3. Full result (`scripts/tr_diffusion_recon_bootstrap_reverse.py`, same
247-window convention, frame 400300-450000):

| variant | PSNR | SSIM | sharpness (1.0 = matches GT) |
|---|---|---|---|
| poissonhead (base) | 25.58 | 0.590 | **0.921** |
| bootstrap mean (forward) | **26.74** | **0.621** | 0.408 |
| bootstrap sample (forward) | 26.64 | 0.615 | 0.434 |
| **bootstrap reverse** | 25.61 | 0.584 | 0.719 |
| noisy floor | 4.56 | 0.017 | 14.743 |

**A genuinely different point on the sharpness/accuracy tradeoff, not a
strict win or loss.** Reverse loses ~1.1dB PSNR / ~0.03 SSIM vs. the forward
bootstrap variants -- it does NOT beat them on accuracy -- but keeps far more
sharpness (0.719 vs. 0.41-0.43), landing much closer to poissonhead's own
0.921 while matching its PSNR almost exactly (25.61 vs 25.58) and beating it
slightly on nothing else. Mechanistic read: forward-bootstrap's input is
ALREADY a regressed quantity (the base's context-only belief), so its own
output is a second layer of "predict the conditional mean" on top of that --
double regression-to-the-mean, hence the collapse to sharpness ~0.41.
Reverse-bootstrap's input is the RAW, unsmoothed measurement instead, so it
never compounds that way -- but it also never sees real multi-frame context
(unlike poissonhead's k=1 neighbours), which is presumably why it still falls
short of poissonhead's sharpness. Inference cost reflects the same structural
difference: 5.4 min to denoise the full 50k-frame range (no base-model
forward pass, no context assembly) vs. the other variants' heavier passes.
Movie `recon_bootstrap_reverse_vs_poissonhead.mov` (panels GT | poissonhead |
bootstrap_mean | bootstrap_sample | bootstrap_reverse | noisy) in
`tr_recon_cache/`. Caveat: `poisson_posterior=True` was used by default here,
which for this direction is a SELF-referential Bayes update (the model's own
`(mu, var)` combined with `central` -- the SAME pixel already used as the
model's input, unlike poissonhead's combination with an independent
observation) -- not yet compared against the `poisson_posterior=False`
(`mu`-alone) case, so it's not established how much of the 0.719 sharpness
comes from that step specifically.

### N2N clean-target diffusion (`DiffusionN2NCleanTargetLoss`, 2026-08-08, in progress)

Motivation: `poisson_head`'s exact Gamma-Poisson posterior-mean reconstruction
is much sharper than any point-estimate baseline (sharpness 0.92 vs 0.53) but
still has residual noise (it imports real high-frequency detail *and* some
noise back from the observation). Question: if a diffusion model is trained
to reproduce that *already-denoised* signal instead of the raw noisy
measurement, can ancestral sampling push further below the noise floor,
instead of reconverging to the noisy input the way it did against a raw-
measurement target (see "N2N ancestral sampler" in project memory --
`n2n_x0c_diff_anc_q100_pm8` only reached 25.85 dB there, still ~6 dB behind
single-shot)?

`DiffusionN2NCleanTargetLoss` (losses.py) keeps the existing N2N recipe
unchanged -- conditioning is still two independent binomial-split views
(fractions `p`/`1-p`) of the SAME raw dose-0.05 measurement, still no N2V --
but both branches now regress toward a fixed **external** clean reference
(`instance["clean_target"]`) instead of toward each other's split half. The
swap-consistency term (`--n2n_consistency_weight`) is unchanged: the two
branches' `x0` estimates still have to agree.

`TimeResolvedFrameDataset(clean_target_memmap=...)` supplies that reference
from a cached `reconstruct.denoise_sequence`-format memmap (float16 counts +
`first_index`/`num_frames`/`crop` sidecar) -- here,
`denoised_212_Wunderkerze2_poissonhead_dose05.f16`. That cache only covers
frames 400201-449799 (the reconstruction-eval half-range, not the checkpoint's
full 400k-500k training range), so the dataset intersects the requested frame
range with the cache's own coverage automatically -- ~49.2k usable centres at
k=1, about half the sample count of the raw-target ablations.

Training job (RunAI, `tr-diff-n2n-cleantarget`, k=1, `--no-temporal_raw_pairs`
to match the existing `x0c` recipe for direct comparability, `prediction_type
=sample`, `consistency_weight=0.2`, same norm range as the other dose-0.05
k=1 checkpoints): `checkpoints/tr_denoise_diffusion_n2n_cleantarget_poissonhead.pt`.
DPS (guiding ancestral sampling toward the actual measurement via the known
Gamma-Poisson likelihood) was discussed as a natural follow-up once plain
ancestral sampling against this cleaner target is evaluated, but is
deliberately deferred -- not yet implemented.

### Noise2clean auxiliary channel + angular-resolution ablation (2026-08-13/15)

**Hypothesis (did not pan out as hoped):** a denoiser trained with REAL ground
truth on a synthetic phantom -- even with a domain gap when run on real data --
might learn sharper edge priors than any self-supervised (N2V) model can, and
feeding its real-data prediction as one extra input channel to the normal
`wunderkerze2` baseline might let that model learn to exploit the prior while
discounting the domain-gap artifacts.

**Setup.** `profiles.REGISTRY["synthetic_wk2geom"]`: a new synthetic profile,
identical to `synthetic_v3` in every respect (phantom scene, crop, dimensions,
axis convention -- `generate_synthetic_phantom.py` writes the CLEAN,
noise-free analytic projection directly; noise is injected later via
`--extra_noise_dose`, same as real data) except `deg_per_frame=1.801402`
(wunderkerze2's own calibrated, non-integer-period rotation rate) instead of
`synthetic_v3`'s exact `deg_per_frame=2.0` (180-frame period). `losses.py`
gained `NoiseToCleanLoss` + `train.py --mode noise2clean`: plain MSE
regression, single forward pass, `poisson_head=False`, target =
`instance["reference"]` (the dataset's pre-thinning clean phantom frame).
`pipeline.denoise_frames_noise2clean` / a matching `reconstruct.denoise_sequence`
branch do inference with NO N2V blind-spot masking (this mode never corrupts
its input at training time either).

Pipeline (`scripts/tr_diffusion_noise2clean_pipeline.py`, all 6 stages
skip-if-exists): render `synthetic_wk2geom` -> train noise2clean on it (k=1,
dose 0.05, 8 epochs) -> cross-domain inference of that frozen checkpoint on
REAL `wunderkerze2` -> cache as `aux_channel_memmap` -> train the real
`--mode baseline --profile wunderkerze2 --k 1 --poisson_head
--aux_channel_memmap <cache>` model -> denoise the eval range -> 247-window
FBP reconstruction vs. the existing (unretrained) `k=1 poisson_head` control.
**Kept `k=1` throughout** (not bumped to 4 as first considered) specifically
so the existing cached `tr_denoise_baseline_k1_dose005_poissonhead.pt` could
serve as a no-retrain control, isolating the aux channel as the one changed
variable.

**Bug caught and fixed mid-run: the cross-domain aux-channel inference must
run on the dose=0.05-thinned real measurement, never the native one.** First
pass called `denoise_sequence(..., dose=None, noise_seed=None, ...)` --
reasoning "run on the real measurement AS MEASURED" -- but that's backwards:
in this project's convention the native/full-dose real frame is NEVER an
actual input available at deployment, it only exists as a pseudo-GT for
offline metric evaluation (`data.py`'s `extra_noise_dose` branch: `reference`
= native frame, eval-only; `central`/`context` = the thinned view, the only
thing any real-data model may assume access to -- same convention the
existing `poisson_head` baseline itself was trained under,
`--extra_noise_dose 0.05`). Visually, feeding the model the native-dose real
frame (which looks close to noise-free, NOT the heavily-thinned regime it was
trained on) vs. the correctly-thinned dose=0.05 frame produced near-identical
output artifacts -- so the severe intensity/contrast distortion in the
cross-domain output is dominated by the phantom-vs-real domain gap itself,
not the dose-regime mismatch this bug introduced -- but the downstream
numbers still had to be redone on the corrected aux cache before trusting
them (moved the invalid first-pass outputs aside as `*.wrongdose_bak`, then
deleted once the corrected run was verified).

**Final result (corrected): the aux channel gives a small genuine
improvement, but NOT the hypothesized sharpness win.**

| variant | PSNR | SSIM | sharpness |
|---|---|---|---|
| `poisson_head` baseline (control, unchanged) | 25.18 dB | 0.536 | 0.926 |
| `poisson_head` + noise2clean aux channel | **25.96 dB** | **0.566** | 0.622 |
| noisy floor | 4.63 dB | 0.019 | 13.31 |

+0.78 dB PSNR / +0.030 SSIM -- real but modest. Sharpness (gradient-energy
ratio vs GT reconstruction, `tr_naf.metrics.masked_sharpness_ratio`) *dropped*
rather than improved, the opposite of the hypothesis -- though the noisy
floor's absurdly high sharpness (13.31) is a reminder this particular metric
conflates real edges with high-frequency noise, so don't over-read the
direction of the "sharpness" number in isolation; the PSNR/SSIM gain is the
more trustworthy signal here.
`checkpoints/tr_denoise_noise2clean_synthwk2geom_k1_dose005.pt` (noise2clean
checkpoint), `.../aux_noise2clean_on_wunderkerze2_dose05.f16` (cached aux
channel), `checkpoints/tr_denoise_baseline_k1_dose005_poissonhead_auxn2c.pt`
(final checkpoint), `tr_recon_cache/recon_summary_noise2clean_aux_vs_poissonhead.json`
+ `recon_noise2clean_aux_vs_poissonhead.mov` (GT | poissonhead |
poissonhead_aux_n2c | noisy panels) -- all under
`/myhome/data/sdate/shared/time_resolved/` /
`/myhome/data/sdate/shared/checkpoints/`.

**Follow-up: does landing back on the same angle every 180 frames actually
matter?** Isolated the angular-resolution variable alone (phantom scene,
crop, dose all held fixed) for BOTH training paradigms this project uses,
scored on each model's own held-out test split (in-domain, real GT available,
raw-count PSNR/SSIM/sharpness):

| model | `synthetic_v3` (deg=2.0, periodic) | `synthetic_wk2geom` (deg=1.801402, real) | gap |
|---|---|---|---|
| noise2clean (supervised, dose 0.05) | 36.96 dB / 0.991 | 32.42 dB / 0.953 | **-4.54 dB / -0.038** |
| baseline N2V (self-supervised, dose 0.025, `poisson_head=False` to match the existing v3 checkpoint's architecture) | 32.18 dB / 0.975 | 30.18 dB / 0.971 | **-2.00 dB / -0.004** |

Non-integer-period rotation costs real recovery accuracy under BOTH training
paradigms, confirming this isn't a noise2clean-specific artifact -- but the
supervised model is ~2x more sensitive to it than the self-supervised one
(N2V's blind-spot training already forces robustness to imperfect local
context/target correlation; noise2clean's clean-GT regression can more
precisely exploit an exact periodic alignment, so it also loses more when
that alignment isn't exact).

Sharpness (gradient-energy ratio vs GT, same metric as above, computed on
projection-domain frames with a full/unmasked region) tells a different story
than PSNR/SSIM: all four (model x geometry) combinations land at essentially
GT-matched sharpness, 0.965-0.981, with no meaningful gap by geometry or model
type:

| model | `synthetic_v3` | `synthetic_wk2geom` |
|---|---|---|
| noise2clean | 0.981 | 0.974 |
| baseline N2V | 0.965 | 0.977 |

So the angular-resolution penalty is NOT extra blur -- both geometries recover
comparable high-frequency content overall, the non-periodic models are just
less *accurate* (residual/misplaced detail) at a matched frequency budget.
Implication for future fixes: a resampling/interpolation approach aimed at
"restoring sharpness" would be attacking the wrong failure mode; something
that improves detail *alignment* is the more relevant direction.
Checkpoints: `checkpoints/tr_denoise_noise2clean_synthv3_k1_dose005.pt`,
`checkpoints/tr_denoise_noise2clean_synthwk2geom_k1_dose005.pt`,
`checkpoints/tr_denoise_baseline_synthetic_v3_final.pt` (pre-existing),
`checkpoints/tr_denoise_baseline_synthwk2geom_dose0025.pt`. Scripts:
`scripts/tr_diffusion_noise2clean_geom_ablation.py`,
`scripts/tr_diffusion_baseline_geom_ablation.py`,
`scripts/tr_diffusion_geom_ablation_sharpness.py`.

### Motion-compensated ("warped") temporal context (2026-08-17/18)

**Follow-up to the angular-resolution-gap finding above.** That ablation
pinned the failure mode as *misplaced detail from unaligned context*, not
blur -- `temporal_raw_pairs=True` hands the model two raw bracketing frames
from adjacent rotations as separate, unaligned channels (the rotation period
is never an integer number of frames) and lets it implicitly learn to
register them. This experiment tests an EXPLICIT fix: dense optical flow
(`cv2.calcOpticalFlowFarneback`, `pyr_scale=0.5 levels=4 winsize=21
iterations=5 poly_n=7 poly_sigma=1.5`) between the CENTRAL frame's already-
denoised ("phase-1", posterior-mean) proxy and each bracket frame's phase-1
proxy, then `cv2.remap` warps the ORIGINAL RAW noisy bracket frame with that
flow. Validated interactively before committing to any training: flow
estimated on the phase-1 denoised proxies gives a smooth, physically-coherent
field and reduces cross-frame residual ~21%; flow estimated directly on raw
noisy (dose=0.05) frames is 2x larger in magnitude, incoherent, and does not
help -- denoise-then-match is necessary. Only the 4 "temporal" context taps
are warped; the 2 "rotation" taps are untouched (genuinely different-angle by
design, not a period-mismatch artifact).

Precompute (`scripts/tr_diffusion_warped_context_precompute.py`): extends the
production checkpoint's own posterior-mean cache to the full training range
(pure registration aid, never a model input, so no target-leak concern), then
computes the 4 per-tap warped caches (`warped_ctx_{tap.name}.f16`, keyed by
`geometry.ContextTap.name` to avoid a positional-index mismatch) over the
frame range common to all taps ([400402, 499598), 99196 frames -- a ~0.8%
shrink from the nominal 100k). `data.py` gained `warped_temporal_memmaps` +
an ADDITIVE `context_warped` output field (a copy of `context` with only the
named taps overridden -- `context` itself always stays raw, since Leg 1's
frozen base model must see exactly what it was trained on).

Two arms, both sharing the same warped-context cache:
- **Leg 2** (`--mode baseline --warped_context_dir ...`): the EXACT production
  recipe (`k=1 poisson_head temporal_raw_pairs extra_noise_dose=0.05`),
  retrained from scratch, with ONLY the context source swapped to warped.
  Isolates context-alignment as the single changed variable vs. the existing
  (unretrained) control checkpoint.
- **Leg 1** (`--mode refine --base_checkpoint ... --warped_context_dir ...`):
  central input replaced by a FRESH Gamma-posterior sample drawn LIVE from the
  frozen control checkpoint's context-only belief (`bootstrap_belief` +
  `bootstrap_input_from_belief(..., "sample")` -- the exact machinery
  `BootstrapPoissonLoss` already validated, always a fresh draw, never a
  static/deterministic proxy -- confirmed with the user this must be the
  Gamma sample, not `mu` alone, to match the earlier validated recipe),
  concatenated with the warped context, trained to predict the SINGLE real
  dose-0.05 measurement we actually have for that frame (`instance["central"]`,
  full-frame NB-NLL, no masking -- see the target-leak fix below for why NOT
  a resampled target, despite that being `BootstrapPoissonLoss`'s own
  convention).

**Both converged fast** (train/test loss plateaued within 5-6 epochs, matching
this project's usual "few epochs is enough" pattern -- see
[[feedback_training_steps_not_epochs]]): Leg 2 flattened around epoch 2-3
(loss ~-116), Leg 1 needed a bit longer (epoch ~4-5, loss ~-116/-115.8, same
floor as Leg 2). Both checkpoints saved via `save_always`; training stopped
manually once the loss curve was clearly flat rather than running the full
epoch budget.

**Operational gotcha caught mid-run (worth remembering): a multiprocessing
worker pool without `torch.set_num_threads(1)` oversubscribes catastrophically
on a CPU-limited container.** The precompute script's per-frame warp step
(one `ds[ci]` dataset-item read + Farneback + remap per call, in 16 parallel
worker processes) measured at ~886ms/item single-threaded and got dramatically
WORSE under 16-way parallelism (~6s/item observed) before adding
`torch.set_num_threads(1)` alongside the existing `cv2.setNumThreads(1)` in
the pool initializer -- after the fix, ~40ms/item (~22x faster), turning a
projected ~42-hour job into ~30 minutes. The SAME bug recurred in the
projection-domain eval script (per-sample `masked_sharpness_ratio` calls in a
tight Python loop) and was fixed the same way. **Any script that does small
per-sample torch ops in a loop -- with or without multiprocessing -- on this
cluster's CPU-limited containers needs an explicit `torch.set_num_threads(1)`
guard, or it silently runs 10-20x slower with zero indication beyond "it's
taking a while."**

**Bug caught before the reconstruction run: `reconstruct.denoise_sequence`'s
generic `baseline` inference branch never read `context_warped`** -- it
always used raw `context`, even for a `--warped_context_dir` checkpoint. Leg
2's training-time fix (`losses.py`'s `BaselineN2VLoss.compute_loss` swapping
to `context_warped` when present) was never mirrored on the inference side, so
running Leg 2 through the shared reconstruction pipeline as-is would have
silently fed it out-of-distribution (unwarped) input. Fixed by swapping to
`context_warped` in the `baseline` branch whenever `warped_context_dir` is set
in the checkpoint's own config -- same one-line pattern as the training fix.

**Target-leak bug caught by the user, fixed, and Leg 1 retrained (2026-08-18):**
the FIRST version of `RefinementLoss` followed `BootstrapPoissonLoss`'s own
convention -- target `y` = a FRESH `add_poisson_noise` draw off
`instance["reference"]` (the higher-SNR native frame), redrawn every training
step, specifically to avoid a DIFFERENT leak (the frozen base model's belief
already having seen ~98% of `instance["central"]` directly through its own
2%-ratio blind-spot corruption). The user caught the problem with THAT choice:
repeatedly drawing independent low-dose realizations from a cleaner reference
is not something a real single-shot deployment ever has access to (one
measurement, period -- no oracle to redraw from); since
`E[add_poisson_noise(reference, dose)] = reference`, scoring against many
fresh redraws over training pushes the model's `mu` toward `reference` itself,
a real information leak, not training-noise. Per the user's explicit
direction, the fix does NOT reintroduce masking to solve the OTHER leak (not
a concern here) -- it simply targets the SINGLE real `instance["central"]`
measurement directly, full-frame NB-NLL, no N2V masking, exactly like the
main baseline's `poisson_head` branch. Training-loss curves for the retrained
model tracked the original (leaky) run almost exactly epoch-for-epoch --
expected, since NB-NLL scored against any single unbiased realization from
the same distribution has similar aggregate magnitude regardless of which
specific realization is used; the leak's real effect only shows up in
held-out evaluation, not the training loss.

**Results (final, target-leak fixed).** Held-out projection-domain scores
(4939 frames, `scripts/tr_diffusion_warped_context_eval.py`; Leg 1 uses a
16-sample posterior-mean):

| arm | PSNR | SSIM | sharpness |
|---|---|---|---|
| noisy input | 17.96 | 0.178 | 6.101 |
| control (unretrained) | 31.73 | 0.794 | 0.777 |
| Leg 2 (warped ctx) | **32.04 (+0.31)** | **0.822 (+0.028)** | 0.552 |
| Leg 1 (refine, warped ctx) | 32.08 (+0.35) | 0.806 (+0.012) | 0.628 |

493-window FBP reconstruction (det_bin=1, same [400402, 499598) range,
`scripts/tr_diffusion_warped_context_recon.py`, flat/dark + destripe
correction -- `dark_map.npy`/`flat_map.npy` from `tr_recon_cache/`,
`destripe_k=31`, same convention as `tr_diffusion_noise2clean_pipeline.py`/
`tr_diffusion_poissonhead_pipeline.py`. A first pass of this script omitted
the correction entirely (copied the OLDER, pre-calibration
`tr_diffusion_recon_ablation.py` convention by mistake) and showed strong ring
artifacts even in GT -- not a detector property, just a missing correction
step; caught when the user asked about it, fixed, and rerun):

| arm | PSNR | SSIM |
|---|---|---|
| noisy | 10.46 | 0.046 |
| control (unretrained) | 29.39 | 0.690 |
| Leg 2 (warped ctx) | **29.80 (+0.41)** | **0.702 (+0.012)** |
| Leg 1 (refine, warped ctx) | 28.77 (-0.62) | 0.667 (-0.023) |

**Verdict: with the target leak fixed, Leg 1 no longer beats control --
it actively LOSES to it in reconstruction space** (-0.62 dB / -0.023 SSIM),
despite still showing a small, genuine projection-domain gain (+0.35 dB).
This is a big swing from the leaky version's reported +0.23 dB reconstruction
win, and it's strong retrospective evidence that the leak was providing real
(if modest) inflation -- consistent with the leaky version's lower
projection-domain sharpness (0.552, now 0.628 after the fix, moving toward
control's 0.777): the pre-fix Leg 1 had partly learned to reproduce a
smoothed, cleaner-reference-like output rather than a genuine single-shot
denoising, which happened to help match a pseudo-GT reference in the metric
but did not translate into better downstream tomographic quality. **Leg 2
(the simple retrain, same architecture/recipe as production, just fed
better-aligned context) is the clear, sole winner of this experiment** --
consistent gains in both domains, no architectural complexity, no leak risk.
Leg 1's added machinery (frozen base model + live resampling) is not
recommended for production use as currently designed; if revisited, the
"context-only belief already leaks 98% of central" issue the ORIGINAL
(leaky-target) design was trying to route around remains unresolved and would
need its own fix (masked loss restricted to the frozen belief's blind-spot
pixels) -- out of scope here per explicit user direction.

Checkpoints: `checkpoints/tr_denoise_baseline_k1_dose005_poissonhead_warpedctx.pt`
(Leg 2), `checkpoints/tr_denoise_refine_warpedctx.pt` (Leg 1, target-leak
fixed; the original leaky-target checkpoint is kept aside as
`tr_denoise_refine_warpedctx.pt.leaky_target_bak` for reference). Caches:
`.../warped_context_dose05/warped_ctx_{tap.name}.f16` (4 taps),
`.../phase1_posteriormean_fullrange_dose05.f16` (registration aid).
`tr_recon_cache/recon_summary_wunderkerze2_warpedctx.json` +
`recon_wunderkerze2_warpedctx.mov` (GT | control | leg2 | leg1 | noisy panels),
`tr_recon_cache/warped_context_eval_summary.json` (projection-domain numbers).

### Joint (non-averaged) k-revolution FBP baseline — the primary baseline to compare against going forward (2026-08-25)

Every classical/learned baseline above compares against either a single noisy
rotation (unusable at dose 0.05) or **SW-FBP**
(`temporal_average_sequence`/`tr_diffusion_recon_swfbp.py`), which *averages*
`2k+1` independent same-angle measurements into fewer, cleaner projections
before FBP — trading noise for temporal blur, but adding no new angular
information (the reconstruction still only ever sees as many effective views
as a single rotation). This new baseline instead keeps **every** projection
from `k` consecutive revolutions at its own exact angle (wrapped mod 360°,
static-volume assumption) and feeds all of them into a single FBP
reconstruction — `k`x the distinct photon measurements of one rotation, no
averaging, no learning, no iterative optimisation
(`sdate.tr_diffusion.reconstruct.reconstruct`, `method="fbp"`, on the raw
k-window's wrapped-angle sinogram). It's also the analytic-FBP control for
the same k-joint sinogram construction used by the Poisson-MLE gradient-
descent/ADMM+TV work (`sdate/tr_diffusion/mle_reconstruct.py`) — same
sinogram, plain FBP instead of iterative MLE.

**k=21, dose 0.05, swept across the ENTIRE wunderkerze2 usable range**
(`scripts/tr_diffusion_jointfbp_k_sweep.py --k 21`; 479 windows, one per
revolution, frames 402149-497749, det_bin=2, real flat/dark calibration,
GT = native full-dose FBP over a matched ~180° window centred at the same
point as each joint window):

| metric | mean over the sweep |
|---|---|
| PSNR | 27.10 dB |
| SSIM | 0.715 |

Per-window PSNR is far from uniform: ~20–22 dB over the more dynamic early
portion of the scene (frames ~402k–442k) vs. ~31–33 dB once the scene settles
(frames ~446k onward) — i.e. very sharp/accurate wherever the sample doesn't
move across the 21-revolution window, markedly worse wherever it does, exactly
the expected sharpness/motion-blur tradeoff of *joining* (not averaging)
measurements across time.

For context against other dose-0.05 baselines measured on this dataset (not a
perfectly matched single-point comparison, but the same ballpark): the
learned N2V+context denoiser (`poisson_head`, see above) scores ~25.8 dB /
0.72 SSIM at a single k=20 sample; single-rotation joint FBP (k=1, i.e. no
joining at all) scores ~23.0 dB / 0.68 at the same sample. This k=21 sweep's
*average* lands close to or above both, using no learned model and no
iterative optimisation at all — a strong, cheap, non-learned floor, and
likely the most important baseline to compare future reconstruction methods
(denoiser-based, Poisson-MLE, or otherwise) against on this dataset.

**Gotcha:** `write_slice_movie`'s `HevcGray10Streamer` invokes the bare
`ffmpeg` command (PATH lookup); RunAI jobs launched via `sdate_launcher.sh`
run under `bash --noprofile --norc`, so `PATH` never picks up `/myhome/bin`
(where this project's ffmpeg actually lives). The job runs its full compute
sweep successfully and only crashes at the final movie-write step
(`RuntimeError: ffmpeg executable not found`) — costly since RunAI's restart
policy re-runs the whole sweep from scratch on each retry. Fix:
`os.environ["PATH"] = f"/myhome/bin:{os.environ.get('PATH','')}"` near the
top of any script that calls `write_slice_movie`/`write_projection_movie`
under this launcher (same fix noted for `write_projection_movie` earlier in
this file's "denoised projection movies" entry).

Outputs: `tr_recon_cache/recon_summary_212_Wunderkerze2_jointfbp_k21.json`
(summary), `recon_results_212_Wunderkerze2_jointfbp_k21.npz` (per-window
PSNR/SSIM), `recon_212_Wunderkerze2_jointfbp_k21.mov` (GT | joint_fbp_k21
movie, one frame per revolution).

### Joint-FBP-context-conditioned baseline denoiser — T=11 same-angle reprojection taps (2026-08-25/26)

*Superseded by the fine-tuned full-scene result below — see "Fine-tuned to the full scene" for the current best result on this dataset. This section covers the initial narrow-range (dynamic-only) proof of concept.*

Takes the k=21 joint-FBP baseline above and feeds it *into* the learned
denoiser as context, instead of using it standalone. For a target frame `i`
at its own exact angle `theta_i`, T=11 different nearby T-revolution-wide
joint-FBP volumes (windows sliding by exactly 1 revolution each step) are
each reprojected back through ASTRA at `theta_i` — giving 11 context values
with **zero angular mismatch** against the target frame, unlike the standard
rotation/temporal taps the baseline normally uses. These 11 reprojections are
cached as extra input channels (`scripts/tr_diffusion_jointfbp_context_cache.py`)
and swapped in for the usual `k=1` context (`train.py --k 0` + `data.py`'s
`aux_channel_memmap`, generalized this session from a single path to a list
of N paths — `in_channels=12`: 1 central frame + 11 taps). Everything else
(loss, `poisson_head`, dose=0.05, norm range) matches the reference
`tr_denoise_baseline_k1_dose005_poissonhead.pt` exactly, so the context
source is the only changed variable.

**Result** (`scripts/tr_diffusion_jointfbpctx_pipeline.py`, frames
[412300,427700), 77 windows, det_bin=1, real flat/dark + destripe(k=31)
correction — same frames scored for both arms):

| arm | PSNR | SSIM |
|---|---|---|
| reference k=1 baseline (`poissonhead_k1`) | 24.46 dB | 0.561 |
| **jointfbpctx_T11 (this)** | **26.81 dB** | **0.685** |
| noisy floor | 2.94 dB | 0.011 |

**+2.35 dB / +0.124 SSIM over the reference baseline** — confirms that
removing angular mismatch from the context lets a learned denoiser recover
much more of the joint-FBP sharpness than the standard taps do. Also lands
within ~0.3 dB of the pure-FBP k=21 floor (27.10 dB/0.715) while being a
*learned* model at T=11 (not 21), which should generalize better to
transient/dynamic content than plain FBP.

**Gotchas hit:** (1) the caching script's window width must be an *exact*
multiple of the per-revolution stride (`win_T = T*stride`, not
`round(T*period_360)`), else the slot-scatter periodically drops to T-1
filled taps instead of T — caught via a cheap offline coverage simulation
before any GPU time was spent. (2) the same ffmpeg-not-on-PATH gotcha noted
above recurred (fix lives in each calling script). (3) `load.py`'s
`build_model` had a stale single-channel assumption for `aux_channel_memmap`
predating this session's list generalization, so eval-time model
reconstruction used the wrong `in_channels` and crashed `load_state_dict` —
fixed to count `len(aux_channel_memmap)` when it's a list. (4) a preemption
without `--load_checkpoint` silently restarts training from scratch and
overwrites the good checkpoint at the same save path — always pass both
`--load_checkpoint` and `--save_checkpoint` at the same path so a future
preemption resumes from the last saved weights (partial resume only:
optimizer/LR-schedule state and the `poisson_head` warmup step counter are
not restored).

Outputs: checkpoint at
`/mydata/sdate/shared/checkpoints/tr_denoise_baseline_jointfbpctx_T11_dose005_poissonhead.pt`;
`tr_recon_cache/recon_jointfbpctx_T11_vs_poissonhead_k1.mov` (GT |
poissonhead_k1 | jointfbpctx_T11 | noisy reconstruction movie),
`tr_recon_cache/proj_jointfbpctx_T11_vs_poissonhead_k1.mov` (projection-domain
movie), `tr_recon_cache/recon_summary_jointfbpctx_T11_vs_poissonhead_k1.json`
(summary).

### Fine-tuned to the full scene (2026-08-26/27) — the best result on this dataset so far

The section above only covered a narrow, still-dynamic 16k-frame slice
(bounded by where the T=11 context taps had been cached) — much smaller than
the k=21 baseline's own ~479-window full-scene sweep. To make a fair
apples-to-apples comparison, the T=11 context taps were re-cached over the
*entire* usable range (`--frame_start 400000 --frame_end 500000`, ~130GB),
and the checkpoint was **fine-tuned** (continued from the narrow-range
checkpoint's weights, not retrained from scratch) over that full range —
preserving what it already learned on the harder dynamic region while
adapting to the easier "settled" region's different distribution.

**Result** (`scripts/tr_diffusion_jointfbpctx_full_pipeline.py`, frames
[402300,497700), 477 windows — matching the k=21 sweep's own
[402149,497749]/479-window coverage almost exactly, det_bin=1, real
flat/dark + destripe(k=31) correction):

| arm | PSNR | SSIM |
|---|---|---|
| reference k=1 baseline (`poissonhead_k1`, full range) | 29.47 dB | 0.693 |
| joint-FBP k=21 (non-learned floor) | 27.10 dB | 0.715 |
| **jointfbpctx_T11_full (this — current best)** | **30.46 dB** | **0.741** |
| noisy floor | 10.52 dB | 0.046 |

**+0.99 dB / +0.048 SSIM over the reference baseline, and +3.36 dB / +0.026
SSIM over the k=21 FBP-only floor** — the first method measured on this
dataset to beat the non-learned k=21 floor, and the best result overall.
The margin over the reference baseline is smaller here than in the
narrow-range-only result above (+0.99dB vs +2.35dB) because this full range
includes the easier settled portion of the scene, where better context
matters less for both arms — the context advantage concentrates in the hard
dynamic region and gets diluted when averaged with the easy static region.

**Operational gotchas hit (this cluster is heavily contended — expect
frequent preemption):** the shared `/mydata/sdate/shared` NFS mount hit
7.1GB free out of 1TB mid-session (`np.memmap`'s `mode="w+"` sparse-allocates
the full logical file size immediately, so a killed job leaves large
near-empty files behind — caught and cleaned up before real damage,
resolved once the user freed space); a preemption during fine-tuning nearly
lost ~3hrs of progress because `--load_checkpoint`/`--save_checkpoint`
pointed at different paths (the exact mistake from the original training
run, repeated) — fixed by pointing both at the same path; with the much
larger full-scene dataset, a single epoch sometimes took longer than this
cluster's preemption cadence (~1.5–3hrs), so several restarts never reached
a save point at all — fixed with `--max_samples 16000` to force epoch
boundaries (and therefore saves) back inside the eviction window; the old
reference-baseline denoised memmap got deleted during the disk cleanup and
had to be cheaply regenerated (~11 min, inference-only). Fine-tuning was
stopped once loss visibly plateaued (~epoch 6 past the warmup transition),
per the project's "judge by loss curve, not epoch count" convention.

Outputs: fine-tuned checkpoint at
`/mydata/sdate/shared/checkpoints/tr_denoise_baseline_jointfbpctx_T11_dose005_poissonhead_full.pt`;
`tr_recon_cache/recon_jointfbpctx_T11_full_vs_poissonhead_k1.mov` (GT |
poissonhead_k1 | jointfbpctx_T11_full | noisy reconstruction movie),
`tr_recon_cache/proj_jointfbpctx_T11_full_vs_poissonhead_k1.mov`
(projection-domain movie),
`tr_recon_cache/recon_summary_jointfbpctx_T11_full_vs_poissonhead_k1.json`
(summary).
