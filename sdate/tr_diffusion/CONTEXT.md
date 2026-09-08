# Time-Resolved Diffusion Denoising

Self-supervised denoising of time-resolved CT projection sequences (no clean ground truth available — the native/real captured frame stands in as reference), plus reconstruction pipelines built on the denoised projections. Covers the existing Noise2Noise/Noise2Void/diffusion baselines and the new Annealed-N2N iterative pipeline.

## Language

**Reference frame**:
The real, natively-captured projection frame for a given angle/time. Stands in as ground truth for all dose-thinning experiments in this codebase, since no true noise-free acquisition exists for real datasets like wunderkerze2.
_Avoid_: ground truth, clean image (when the dataset is real-captured, not a synthetic phantom)

**Dose**:
A fractional Poisson-thinning parameter in (0, 1], not a raw photon count. `dose=0.05` means the simulated measurement carries 5% of the reference frame's photon statistics; mean is preserved, variance is raised by dividing back through the dose.
_Avoid_: photon count, exposure

**Binomial split**:
Drawing a single Poisson realization at a combined dose from the reference frame, then splitting that one realization into two halves via a Binomial(N, p) draw. The two halves are conditionally independent given the shared total count — not two independent measurements of the underlying signal.
_Avoid_: independent draws (when describing this mechanism specifically)

**Annealed-N2N**:
The iterative renoising pipeline (new, this design session): alternates training a Noise2Noise denoiser on the current noisy iterate and "renoising" its output back toward the raw measurement, annealing the mix toward pure denoised output over a fixed number of rounds. Distinct from the existing single-shot N2N baseline (`--mode baseline --denoise_mode n2n`), which has no round loop or renoising step.
_Avoid_: iterative annealed N2N, renoise-anneal (informal names used before this term was settled)

**Round** (k):
One iteration of the Annealed-N2N pipeline: (1) train the round denoiser on the current iterate, targeting `y1`; (2) apply the renoising update to produce the next round's iterate. Indexed 0..K-1.
_Avoid_: epoch, step (both are training-internal units *within* a round, not the round itself)

**y1 / y2**:
The two halves of a Binomial split (combined dose 0.1, p=0.5) of one Poisson draw per projection. `y1` is the fixed Noise2Noise training target for every round of the Annealed-N2N pipeline. `y2` is the raw noisy anchor: it seeds the round-0 iterate and is mixed back in at every renoising step.

**Iterate** (x̂ₖ):
The Annealed-N2N pipeline's evolving noisy estimate of a projection at round k, expressed in Anscombe space. x̂₀ = y2.
_Avoid_: estimate (too generic — use "iterate" when referring specifically to this evolving per-round quantity)

**Round denoiser** (D):
The UNet trained each round via plain MSE Noise2Noise loss (Anscombe space, single-channel regression output) to map the current iterate toward `y1`. Warm-started from the previous round's weights, except round 0 which trains from scratch. Deliberately "round-blind" — not conditioned on the round index or the anneal weight.

**Renoising update**:
The per-round transition x̂ₖ₊₁ = αₖ·y2 + (1−αₖ)·D(x̂ₖ), which mixes the round denoiser's clean output back with raw noise before the next round's training.

**Anneal weight** (αₖ):
The renoising update's mixing coefficient, scheduled from α₀=1 (round-0 iterate is exactly y2) down toward ≈0 by the final round. Two schedules are supported: cosine (default) and linear.

**Final estimate**:
D(x̂_K) — one explicit round-denoiser pass applied to the last iterate. This is the Annealed-N2N pipeline's official output, distinct from the raw trajectory point x̂_K.

**Trajectory**:
The saved sequence x̂₀ … x̂_K for a handful of chosen projections, kept purely for visualizing how the iterate evolves across rounds — not used for evaluation metrics.

**MMSE anchor**:
D(x̂₀) — the Annealed-N2N pipeline's own round-0 output, before any renoising has occurred. Numerically and procedurally identical to a plain single-shot Noise2Noise MMSE estimate, so it doubles as the pipeline's internal baseline without training a separate model.
