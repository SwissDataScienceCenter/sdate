# Cross-package dependencies

`tr_fbp_codec` is meant to stand on its own -- everything specific to this
codec (quantization, model, loss, arithmetic coding, HEVC/FFV1 baselines)
lives inside this package. The only things it reuses from elsewhere in the
repo are CT reconstruction/geometry primitives that already exist and are
non-trivial to reimplement correctly. Every such coupling is listed here;
keep this list exact and up to date -- if you add a new `from sdate.tr_diffusion
import ...` anywhere in this package, add it here too.

## Reused from `sdate.tr_diffusion`

| Symbol | File | Used for |
|---|---|---|
| `MemmapFrameSource` | `frames.py` | Loading native Wunderkerze2 raw projection frames. |
| `PROFILES["wunderkerze2"]`, `DatasetProfile` | `profiles.py` | Geometry constants: `deg_per_frame` (rotation rate), `crop`, `rot_axis_col`, `memmap_path`, `frame_start`/`frame_end`. |
| `native_window_gpu` | `reconstruct.py:666` | Reading a contiguous window of native frames onto GPU. **Gotcha** (see repo memory `feedback_native_window_gpu_contiguous_only`): only contiguous index ranges are safe -- it reads `[idx[0], idx[-1]]` densely regardless of gaps in `idx`. Never pass a non-contiguous index array. |
| `native_window` | `reconstruct.py:642` | CPU per-frame path used in `data.py` (no GPU needed for this dataset's small crops). **Applies `_center_crop`(`profile.crop`, `profile.rot_axis_col`) internally** -- calling `MemmapFrameSource.get()` directly instead (an earlier version of `data.py` did this) returns the *stored* frame at `(profile.height, profile.width)` = `(128, 528)`, NOT the center-cropped `(128, 512)` the tap cache uses -- a real bug caught during smoke-testing (width mismatch between context/target and the prior tap). Always go through `native_window`, never `MemmapFrameSource.get()` directly. |
| `window_length_frames`, `sliding_windows`, `run_windows` | `reconstruct.py:460,465,805` | Turning a frame range into revolution-aligned FBP windows -- basis for both the training-time cache reuse and (adapted, see below) the inference-time rolling reconstruction. |
| `reconstruct` (function) | `reconstruct.py:597` | ASTRA-based FBP reconstruction from a window of projections + angles. |
| `counts_to_attenuation_flatdark`, `attenuation_to_counts_flatdark` | `reconstruct.py:533,549` | Domain round-trip (counts -> attenuation for FBP input, attenuation -> counts for the reprojected prior tap) so the FBP-derived prior tap lives in the same counts-like domain as the raw noisy target. |
| `reproject_at_angles`, `reproject_to_counts` | `phi_context.py:228,253` | Forward-projecting a reconstructed volume at specific target angles and converting back to counts. Despite living in `phi_context.py` (written for phase-gated periodic-motion context taps), these two functions are geometry-agnostic ASTRA calls with no phi-specific logic -- reused directly for our time-domain rolling reconstruction. Do not import anything else from `phi_context.py`; the phi-gating/binning logic there is specific to that project and not applicable here. |

## Reused data artifact (not code -- a cache file on shared storage)

`FBP_TAP_CACHE_*` -- the existing native joint-FBP-context tap cache at
`/myhome/data/sdate/shared/time_resolved/jointfbp_context/212_Wunderkerze2_jointfbpctx_T5_native_tap{0..4}.f16`
(+ matching `.meta.npz` sidecars). Built previously for the T5-native
Gaussian-floor denoiser project (see repo memory
`project-tr-diffusion-t5native-gaussianfloor`), NOT specifically for this
package -- but its metadata confirms it's genuinely native data
(`native_noise=True, dose=1.0`), `T=5` taps, `stride=200` (~1 revolution),
covering absolute frame indices `[412800, 467200)`. This package's
`DataConfig` defaults (`frame_start=413000, frame_end=415000`) were chosen
specifically to fall inside this range so prototyping needs no new
GPU/ASTRA reconstruction job. Per the settled design (README.md), only
**tap0** is used (the slot nearest a causal window, per the tap-cache
script's own "slot 0 = nearest this frame's start" convention) -- taps 1-4
are ignored, not deleted, in case a later experiment wants them. If you
change `DataConfig.frame_start`/`frame_end` outside `[412800, 467200)`,
you must build a new cache first (a real compute job -- see
`scripts/tr_diffusion_jointfbp_context_cache.py`), this is not a
config-only change.

## Reused (as reference pipelines to adapt, not imported directly)

| Script | Relevance |
|---|---|
| `scripts/tr_diffusion_jointfbp_context_cache.py` | Existing training-data caching pipeline: builds per-revolution-chunk FBP reconstructions and reprojects to nearby angles. This is the mechanism `data.py`'s training-time path adapts (per the settled design: training uses these caches for throughput, accepting the resulting target-leakage as a known simplification -- see README.md). |
| `scripts/tr_diffusion_jointfbpctx_t5native_pipeline.py` | Reference pipeline for driving the above cache-building on **native** (not dose-reduced) Wunderkerze2 data specifically -- use this, not the `dose005`-style configs, when locating/building the native data cache. |

## Explicitly NOT reused

| Package | Why not |
|---|---|
| `sdate.compression` (`custom_compression.py`, `h264_utils.py`) | H264/DCT/JPEG-oriented, lossy, no 12-bit support. Unrelated to this codec's lossless-arithmetic-coding design. |
| `sdate.video_compression` (`hevc_grayscale.py`) | HEVC wrapper but hardcoded to 10-bit (`encode_hevc_grayscale_10bit`) via a float-normalized `gray16le`->`format=gray10le` path. Our ground truth is 12-bit and must be fed as `gray12le` directly (no rescale filter) to stay lossless -- see `baselines.py` for the from-scratch implementation and the reasoning. |
| `sdate.stream_hvec.stream_gray10` | Same 10-bit limitation as above; also float-normalized rather than integer-quantized input. |

## Open coupling to revisit later

- If/when the training-time causal-leakage simplification (see README.md) is
  replaced with a genuinely causal rolling-reconstruction training pipeline,
  it will need an efficient incremental-update path through `reconstruct.py`'s
  windowing utilities that does not yet exist -- this is future work, not a
  hidden dependency today.
