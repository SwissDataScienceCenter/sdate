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

`k` is a parameter (default **1** → 6 diffusion channels). `--include_mirror`
appends the two 180° half-turn taps (`i±P/2`, flipped about the rotation axis
col ≈ 269.85), `+2` channels.

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

Train the diffusion model and the baseline (same data, same context):

```bash
python -m sdate.tr_diffusion.train --mode diffusion --k 1 \
  --mov .../212_Wunderkerze2.mov --memmap .../frames_400k_600k.u16 \
  --batch_size 16 --epochs 100 --exp_name k1 --wandb

python -m sdate.tr_diffusion.train --mode baseline --k 1 \
  --mov .../212_Wunderkerze2.mov --memmap .../frames_400k_600k.u16 \
  --batch_size 16 --epochs 100 --exp_name k1 --wandb
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
