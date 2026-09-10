# tr_fbp_codec

Learned lossless codec for time-resolved CT projection streams (Wunderkerze2
first). Reproduces the core idea of Zhang et al., "Fast lossless images
compression for synchrotron radiation facility using deep learning and hybrid
architecture" (2024, DOI 10.1007/s41605-024-00490-9) -- predict frame
`f[n+k]` from `k` preceding raw context frames via a 3D CNN over 32x32
blocks, cross-entropy loss over 4096 (12-bit) pixel-value classes -- and adds
a new input: an FBP-reprojection of the reconstruction built from the frames
immediately preceding the target, at the target's own projection angle.

## Design summary (settled 2026-09-07)

- **Scope**: time-resolved multi-revolution data only for now (Wunderkerze2).
  A limited-angle/single-sweep variant for non-time-resolved tomography is a
  deferred, separate track -- do not assume it shares this package's data
  loading or causality assumptions.
- **Causality / decodability**: this is meant to be a genuinely decodable
  codec, not an oracle study.
  - **Inference**: the FBP prior for target frame `f[n+k]` is built from a
    rolling reconstruction of the ~200 frames immediately *preceding* it
    (one full revolution at Wunderkerze2's rate, ~199.8 frames/360 degrees),
    reprojected to `f[n+k]`'s own angle. Strictly causal: no future frame is
    ever used.
  - **Training**: for throughput, training reuses the *existing* joint-FBP-
    context caches (see DEPENDENCIES.md) built per revolution-aligned chunk,
    not a fresh rolling reconstruction per sample. This means the training
    prior can leak some target-frame information (the chunk reconstruction
    may include frames at or after the target). Accepted as a deliberate
    prototype simplification -- flagged to be stress-tested later (does
    training on a "cheating" prior hurt inference-time performance where the
    prior is strictly causal?). Do not remove this caveat without re-running
    that comparison.
- **Dataset**: prototyping on a small contiguous slice (~1000-3000 frames) of
  **native, full-dose Wunderkerze2 data with no added synthetic noise** --
  explicitly NOT the `dose=0.05` synthetic low-dose regime used elsewhere in
  `tr_diffusion`. See DEPENDENCIES.md for the exact profile/path.
- **Architecture**: 3D ResNet (this package's own design, not a literal port
  of the paper's TCN+Octave-Conv network), per-pixel 4096-class softmax
  (rejected switching to a discretized-mixture-of-logistics head -- see
  chat/PR history: DMoL is well-validated on 8-bit natural images but
  unvalidated on 12-bit CT photon-counting data specifically, adds real
  implementation/numerical-stability complexity, and turned out to be
  unnecessary once the actual memory math was done). **Block size: 64x64**
  (settled 2026-09-07, up from the paper's 32x32 -- the classification
  head's memory cost is trivial even at 64x64, so block size was picked for
  a smaller block-edge bpp tax, not memory pressure; see `config.py:CodecConfig`
  for the exact numbers and the `block_margin` receptive-field fix for the
  fake-boundary issue plain patch training would otherwise introduce). k=3
  context length is a loose rule of thumb from the paper, not a constraint.
- **Loss / entropy coding**: cross-entropy over the 12-bit-quantized ground
  truth. Unlike the paper (which used Huffman-on-residual for FPGA
  portability), we have no FPGA constraint, so entropy coding is arithmetic
  coding driven directly by the predicted per-pixel distribution -- the true
  entropy-limit bpp, not a residual+Huffman approximation.
- **Metric**: compression ratio / bits-per-pixel over the 12-bit-quantized
  ground truth is the only metric that matters (the target is lossless
  reconstruction of the quantized data, not PSNR/SSIM against it). The model
  is trained **per experiment** (self-supervised on that experiment's own
  stream, no cross-dataset generalization claim yet), so report the ratio
  **both** including and excluding the trained model's own weight size --
  track both numbers, do not collapse to one.
- **Baselines**: lossless HEVC at native 12-bit (`gray12le` pixel format fed
  directly, NOT via a `gray16le` + `format=` filter rescale -- that path
  silently corrupts values, see `baselines.py`) and FFV1 (also bit-exact,
  and beat x265-lossless on a pure-noise sanity check -- run both, don't
  assume HEVC wins). The paper's own reported ~1.7-2.0x ratio is a secondary
  reference point, not the primary baseline.
- **Bit-depth quantization check -- DONE, verified on real data**: ran
  `quantization.py:evaluate_quantization_loss` on 200 real frames sampled
  across `profiles.REGISTRY["wunderkerze2"]` frames 400000-402000 (native,
  via `reconstruct.native_window`, not the dose=0.05 synthetic path --
  `DatasetProfile.dose` is only used if a caller explicitly calls
  `noisy_window()`, the memmap itself is native counts). Result: 0%
  clipping, PSNR 69.0dB mean / 68.4dB worst-case, SSIM 0.9999 mean/worst,
  9.8 of 12 bits actually used (local slice max 887.6; the full-dataset
  sidecar max is 1342, still well under 4096). **Verdict: SAFE**, `mode="truncate"`
  in `QuantConfig` is the right default for this dataset. Re-verify before
  pointing the package at a new dataset or a higher-dose acquisition --
  this is dataset/detector-dependent, not a safe universal assumption (see
  `context/compression_data_recon/DATA_REPORT.md` for a worked
  counter-example: some real detectors saturate the full 16-bit range).

## Module status

| Module | Status |
|---|---|
| `quantization.py` | implemented -- int16->int12 mapping + loss evaluation |
| `config.py` | implemented -- experiment/codec config dataclasses |
| `model.py` | implemented -- 3D ResNet predictor |
| `losses.py` | implemented -- CE loss over quantized classes |
| `data.py` | implemented -- training path only (cached tap0, see DEPENDENCIES.md); causal rolling-reconstruction inference path still TODO |
| `coding.py` | implemented -- torchac arithmetic coding, verified bit-exact round-trip against real model output |
| `baselines.py` | implemented -- HEVC-12bit + FFV1, both verified bit-exact (`/myhome/tools/ffmpeg`) |
| `train.py` / `evaluate.py` | implemented, smoke-tested end-to-end (CPU tiny config, then a real RunAI GPU smoke run) |

Full data->model->loss->arithmetic-coding pipeline verified end-to-end
2026-09-07: CE-loss-estimated bpp (11.975) matched the actual arithmetic-
coded bpp (11.975) almost exactly on a real (untrained) sample, and the
encode/decode round-trip was bit-exact. HEVC-12bit/FFV1 baselines on 20 real
frames: 6.36 / 5.71 bpp respectively -- the numbers a trained model needs to
beat. Real training (20000 steps, base_channels=32, n_blocks=6, batch=64)
launched via RunAI as `sdate-fbpcodec-fbp` (use_fbp_prior=1) and
`sdate-fbpcodec-nofbp` (use_fbp_prior=0, paper-reproduction baseline) --
check `runai workspace list -p sdate-luisb | grep fbpcodec` for status, and
`/myhome/data/sdate/shared/time_resolved/tr_fbp_codec/run_{fbp,nofbp}/` for
checkpoints + `eval_result.json` once each job completes.

## First real results (2026-09-08)

Both training runs completed on RunAI (20000 steps, batch_size=8 -- see
"Two bugs found the hard way" below for why batch_size ended up so small;
loss was still descending at the end, not converged, see `history.json` in
each run dir) and evaluated end-to-end via real arithmetic coding on 200
held-out frames:

| | bpp (excl. weights) | bpp (incl. weights) |
|---|---|---|
| **Ours (with FBP prior)** | **6.250** | 7.410 |
| Paper reproduction (no prior) | 6.434 | 7.591 |
| HEVC-12bit baseline | 6.266 | -- |
| FFV1 baseline | **5.602** | -- |

Full results: `run_{fbp,nofbp}/eval_result.json` (per-frame breakdown too).

**Reading of this**: the FBP prior gives a real, consistent improvement
over the paper's own no-prior approach (6.250 vs 6.434 bpp, ~2.9%
relative) and roughly ties the HEVC-12bit baseline -- confirms the core
hypothesis is directionally right. But **FFV1 still wins outright**
(5.602 bpp, beats both model variants even before counting weight
overhead) at this prototype's scale (small data slice, undertrained,
small batch). Not yet a result that justifies the learned approach over a
generic off-the-shelf lossless codec -- more training (batch_size was
capped low by the OOM below, undertraining the model) is the obvious next
lever before concluding anything stronger.

**Important caveat**: `evaluate.py` currently reuses the SAME cached,
non-causal tap0 source (see "Two bugs found the hard way" and
DEPENDENCIES.md) for both training AND this evaluation -- i.e. even the
held-out eval numbers above still benefit from the target-leakage
simplification. The genuinely causal rolling-reconstruction inference path
is still NOT implemented, so these numbers are an optimistic upper bound
on what a real, causally-honest deployed codec would achieve, not the
final answer. Building that causal path and re-measuring the gap is the
single most important piece of remaining work.

## FFV1 on raw frames vs. temporal diff (2026-09-10)

`baselines.py`'s FFV1 encoder was only ever run on raw consecutive frames
(`P_1, P_2, ..., P_n`) -- FFV1 itself is an *intra*-frame codec (spatial
median-predictor only, no temporal referencing), so it was never exploiting
frame-to-frame redundancy at all. Tried the obvious alternative: encode the
temporal-diff stream instead (`P_1, P_2-P_1, P_3-P_2, ...`, biased by
`+4095` to stay unsigned/losslessly invertible -- see
`frames_to_temporal_diff`/`temporal_diff_to_frames`), so FFV1's spatial
predictor now sees a per-pixel diff image instead of the raw frame.

Measured on the holdout range (both a 200-frame slice matching the
original baseline measurement, and the full 4000-frame holdout):

| | bpp (200 frames) | bpp (4000 frames) |
|---|---|---|
| FFV1 on raw frames | 5.892 | 5.900 |
| FFV1 on temporal diff | 6.176 | 6.187 |

**Temporal diffing makes it WORSE**, by ~0.29 bpp, consistently at both
scales (round-trip verified bit-exact both times -- this is a real,
lossless comparison, not an estimate). Counter-intuitive if you're used to
video codecs where temporal prediction helps, but it makes sense here: the
data is shot-noise-dominated, so `P_i` and `P_{i-1}` each carry their own
~independent noise realization on top of a slowly-varying (rotation-driven)
signal. Diffing cancels the correlated, spatially-smooth signal component
that FFV1's own intra-frame predictor already handles well, and leaves
behind noise from *two* frames instead of one (variance roughly doubles) --
harder for a purely spatial predictor to compress, not easier. Diffing
would only help a codec that also has a temporal model to exploit, or on
data where per-pixel signal changes dominate over noise -- neither applies
to this codec's native-noise CT stream.

(Small numeric gap vs. the original recorded `ffv1_bpp=5.602`: that number
predates `DataConfig`'s 2026-09-08 widening to the current holdout range,
so it's a genuinely different frame range, not a methodology
discrepancy -- both measurements above used identical code and are
internally consistent with each other.)

## Two bugs found the hard way (RunAI job history)

1. **Nested-quote corruption in `runai workspace submit --command`.** A
   `python -c "import torchac; print('torchac ok')"` sanity check (single
   quotes nested inside double quotes, inside the outer `bash -c "..."`)
   got silently mangled somewhere in the RunAI CLI's command serialization
   -- the container actually received `python -c "import torchac; print(torchac`
   (truncated, syntax error), even though the exact same pattern worked
   fine locally. Fix: avoid nested quotes entirely in `--command` strings
   for this project's RunAI setup -- `python -c "import torchac"` (no
   inner quotes) is safe. Cost: the first pair of jobs queued ~9.5 hours
   (see point 3) before failing on this in seconds.
2. **CUDA OOM at batch_size=64.** The memory-math table earlier in this
   file (and in the design chat) only accounted for the classification
   head's cost, not the 3D ResNet backbone's own activation memory (which
   scales the same way with batch x depth x channels x spatial extent) --
   real usage was ~452MB/sample against the ~220MB/sample napkin estimate,
   OOMing a 32GB card at batch=64 (28.95GB in use, tried to allocate
   another 4GB). Dropped to batch_size=8 as a conservative fix given the
   queue-time cost of getting this wrong again (see point 3) -- revisit
   with proper memory profiling before assuming 8 is anywhere near optimal.
3. **RunAI queue congestion is severe and was the dominant cost, not
   compute.** Every attempt (including the two above, each requiring a
   fresh submit) queued for hours before a GPU slot opened (cluster showed
   0% free GPU capacity across every A100 node for most of a session) --
   actual training+eval wall-clock, once running, was a small fraction of
   total elapsed time. Budget for this when estimating turnaround.

**Still open / not yet done**:
- **The causal rolling-reconstruction inference path is the top priority**
  (see "Important caveat" above and "Causality / decodability" earlier in
  this file) -- everything so far uses the training-time cache path for
  BOTH train and eval.
- Undertrained: batch_size=8 was a defensive choice, not a considered one.
  Worth a real memory-profiling pass to find a properly-sized batch (or
  add gradient accumulation) and a longer run once the causal path exists.
- No k or block_size ablation yet -- k=5, block_size=64 were reasonable
  defaults, not tuned.
- `evaluate.py`'s edge-padding (replicate-pad by `block_margin` for blocks
  at the true frame boundary) introduces a small train/eval mismatch at
  those edges specifically -- noted in `evaluate.py`'s docstring, not
  expected to be a correctness issue (lossless regardless), just a minor
  efficiency one.

See DEPENDENCIES.md before touching anything that reaches outside this
package.
