# Compression-paper micro-CT dataset — characterization report

Data root: `/myhome/data/sdate/shared/compression_paper` (6 experiments, 6 sub-folders `file_*_extracted`)
Report date: 2026-09-04
Reference notebook: `/myhome/sdate/notebooks/TiffProjectionStream.ipynb`
Scratch analysis scripts: `/tmp/claude-0/-myhome-sdate/b45463be-5f57-4fab-824f-4b75c057339d/scratchpad/{inventory,stats,sinogram,sino_analysis,compressibility,gotchas,poisson_check,summary_plots}.py`
Plots: `/myhome/sdate/context/compression_data_recon/*.png` (paths listed inline below)

---

## 0. Existing loaders in the repo (grep of `/myhome/sdate/sdate/`)

| File | Relevance |
|---|---|
| `sdate/datasets/projection_triplet_dataset.py` | **Closest match.** Contains `load_tomography_params()` (generic regex-based darks/flats/projections auto-detector), `TomographyFolderProcessor` (dark/flat correction + on-disk caching of averages) and `ProjectionTripletDataset`. Verified below that its regex parser happens to extract the correct darks/flats/projections counts for all 6 experiments here — but it does **not** know about trailing flats, and does not extract angular step / exposure / pixel size / energy. |
| `sdate/datasets/tiff_volume_dataset.py` | Generic `TiffVolumeDataset`: loads N consecutive TIFF frames into a 3D volume + sliding-window sub-volumes, optional HEIC dual-channel + residual channel. No darks/flats/log awareness. |
| `sdate/datasets/tiff_dataset.py` | `TIFFDataset`/`tiff_wrapper`: generic 2D-crop dataset over arbitrary TIFFs (e.g. reconstructed slices), unrelated to raw projection sequences. |
| `sdate/datasets/tiff_tomogram_dataset.py` | Tomogram-oriented variant, not log-aware. |
| `sdate/limited_angle_tomo.py` | Time-slice / limited-angle dataset utilities (angular_range_deg, time-slicing of a single big projection sequence) — relevant prior art for slicing a single 180° sweep into pseudo-time chunks. |
| `sdate/streaming/stream_step_prediction.py`, `sdate/training/train_step_prediction.py` | Existing "predict next frame from previous" training pipeline — directly relevant precedent for the t→t+1 angular-prediction compressor. |
| `sdate/stream_hvec/stream_gray10.py` (`HevcGray10Streamer`) | Existing lossless/lossy 10-bit HEVC streaming codec used by the reference notebook as an alternative compression baseline. |

**Verified**: the notebook's/`projection_triplet_dataset.py`'s generic regex fallback (`dark.*?(\d+)`, `proj.*?(\d+)`, etc.) correctly recovers `num_darks/num_flats/num_projections` for all 3 spot-checked experiments here (by coincidence of field layout), e.g. for `file_1`: `{'num_darks': 10, 'num_flats': 100, 'num_projections': 1501}`. It should **not** be trusted for the trailing-flat count or for the other metadata (see §6).

---

## 1. Inventory

All images are single-channel `uint16` TIFF, `(rows=H, cols=W)` per `tifffile`. **Every experiment has exactly `n_darks + n_flats + n_projections + 100` files** — i.e. **100 trailing flat frames after the projections** in all 6 cases (see §6 for whether these are genuine or duplicated).

| exp (folder) | sample / prefix | n_files | size | H×W | dtype | darks | flats(lead) | proj | trailing flats | angle range | Δdeg | exposure | pixel size | energy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| file_1 | JC_62d | 1711 | 18.92 GB | 2160×2560 | uint16 | 10 | 100 | 1501 | 100 | 0→180° | 0.120° | 180 ms | 0.325 µm | 18.001 keV |
| file_2 | PD1511_41_ | 1711 | 18.92 GB | 2160×2560 | uint16 | 10 | 100 | 1501 | 100 | 0→180° | 0.120° | 200 ms | 0.325 µm | 18.000 keV |
| file_3 | drying_2 (log=MI04_02) | 751 | 2.88 GB | 1900×1008 | uint16 | 50 | 100 | 501 | 100 | 0→180° | 0.360° | 1 ms | 1.1 µm | polychromatic (white beam) |
| file_4 | FC_13p5keV_G_LAG100_4ms_70mm_1000p | 1211 | 1.47 GB | 300×2016 | uint16 | 10 | 100 | 1001 | 100 | 0→180° | 0.180° | 4 ms | 2.75 µm | 13.499 keV |
| file_8 | S174482_20x_B1 | 1711 | 18.92 GB | 2160×2560 | uint16 | 10 | 100 | 1501 | 100 | 0→180° | 0.120° | 400 ms | 0.325 µm | 10.000 keV |
| file_10 | Al8Fe_10x_ | 1733 | 15.63 GB | 1762×2560 | uint16 | 32 | 100 | 1501 | 100 | 0→180° | 0.120° | 190 ms | 0.65 µm | (field absent in log) |

Total on-disk: **~76.7 GB** across the 6 experiments.

Acquisition mode: file_1/2/8 = "FAST-TOMO"/"SNAP&STEP-TOMO" on a **PCO.Edge 5.5** camera; file_3/4 = **GigaFRoST** camera (fast, low-exposure, in-situ/dynamic scans). This camera split matters throughout (see §3, §6).

### Raw log verbatim (file_1, `JC_62d.log`)

```
User ID : e11114
FAST-TOMO scan of sample JC_62d started on Sat Nov 21 09:36:47 2015
--------------------Beamline Settings-------------------------
Ring current [mA]           : 400.231
Beam energy [keV]           : 18.001
Monostripe                  : Ru/C
FE-Filter                   : No Filter 100%
OP-Filter 1                 : 100um Al
OP-Filter 2                 : 50um Al
OP-Filter 3                 : No Filter
--------------------Detector Settings-------------------------
Camera                      : PCO.Edge 5.5
Microscope                  : Opt.Peter MB op
Magnification               : 20.0
Scintillator                : LAG:Ce 20 um
Exposure time [ms]          : 180.0
Delay time [ms]             : 0.0
Millisecond shutter [ms]    : not used
X-ROI                       : 1 - 2560
Y-ROI                       : 1 - 2160
Actual pixel size [um]      : 0.325
------------------------Scan Settings-------------------------
Sample folder               : /sls/X02DA/data/e11114/Data10/disk12/JC_62d/
File Prefix                 : JC_62d
Number of projections       : 1501
Number of darks             : 10
Number of flats             : 100
Number of inter-flats       : 0
Flat frequency               : 0
Rot Y min position  [deg]   : -0.0
Rot Y max position  [deg]   : 180.0
Rotation axis position      : Standard
Angular step [deg]          : 0.120
Sample In   [um]            :     0
Sample Out  [um]            :  3000
-----------------------Sample coordinates---------------------
X-coordinate                : 93.00
Y-coordinate                : 4173.24
Z-coordinate                : 32000.00
XX-coordinate               : -1594.27
ZZ-coordinate               : 1798.70
-----------------------Microscope coordinates---------------------
X-coordinate                : -218.02
Y-coordinate                : -136.54
Z-coordinate                : 32.00
--------------------------------------------------------------
TOMOGRAPHIC SCAN STARTED!!!
TOMOGRAPHIC SCAN FINISHED at Sat Nov 21 09:43:00 2015
Original rotation center      : 1283.75
------------------- Projection Information -------------------------------
Original tif projection     : 2560x2160 pixels
Total size of Tiff projection     : 18922291 kb
[... reconstruction-parameters block follows, not relevant to acquisition metadata ...]
```

File naming: darks/flats/projections/trailing-flats are **not separated into subfolders** — they are one flat numeric sequence `<prefix><NNNN>.tif`, sorted lexicographically = acquisition order: `[0:n_darks]` darks, `[n_darks:n_darks+n_flats]` leading flats, `[n_darks+n_flats : n_darks+n_flats+n_proj]` projections, remainder = trailing flats.

---

## 2. Angular sampling

All 6 experiments are a **single 0→180° half-sweep** (no multi-revolution, no 360°). For every experiment, `(n_projections − 1) × Δdeg == 180.0°` exactly:

| exp | n_proj | Δdeg | (n_proj−1)×Δdeg |
|---|---|---|---|
| file_1/2/8/10 | 1501 | 0.120° | 180.00° |
| file_3 | 501 | 0.360° | 180.00° |
| file_4 | 1001 | 0.180° | 180.00° |

**file_4 gotcha resolved this way**: the `_orig.log` file contains two scan-restart entries (1001 then 1000 projections after an apparent abort/restart), and the primary `.log` file's "Number of projections" field (1001) belongs to a metadata block whose sample-name text is inconsistent with the folder (see §6). Using `(n−1)×0.18°` against the stated 180° range unambiguously selects **n=1001** (exact 180.00°) over n=1000 (179.82°, off by 1 step) — and this matches the 1211-file-count budget only when combined with the 100 trailing flats (`10+100+1001+100=1211`, exact).

No experiment here is a multi-revolution time-resolved acquisition like the sdate `tr_diffusion`/Wunderkerze data — this benchmark set is single-sweep (largely static) CT, except file_3 which is a genuinely dynamic (drying) sample scanned during its one 180° sweep (see §4).

---

## 3. Raw data statistics (darks / flats / projections)

Computed from: all dark frames, all 100 leading flats, all 100 trailing flats, and 40 evenly-sampled projection frames per experiment (`stats.py`).

| exp | darks mean (range) | darks std | leading-flat mean (range) | trailing-flat mean (range) | flat decay | proj-sample mean (range) | proj-sample std (range) |
|---|---|---|---|---|---|---|---|
| file_1 | 94.6–94.7 | 4.42–4.45 | 40021.6–40787.9 | 39397.1–39619.5 | **−2.2%** | 19813–20981 | 9121–11318 |
| file_2 | 94.8–94.9 | 4.48–4.50 | 43160.2–44054.1 | 42831.1–43327.2 | **−1.2%** | 26045–26704 | 11111–11414 |
| file_3 | 14.6–75.5* | 7.29–7.77 | 1344.4–1403.2 | **identical to leading (see §6)** | 0% (duplicated) | 1079.4–1123.3 | 294.0–300.6 |
| file_4 | 88.1–88.5 | 7.04–7.11 | 2791.4–2821.4 | **identical to leading (see §6)** | 0% (duplicated) | 1121.5–1189.4 | 398.9–467.2 |
| file_8 | 93.7–93.8 | 5.05–5.13 | 42534.2–42763.2 | 42477.5–42713.6 | **−0.12%** | 41559–41841 | 3643–3997 |
| file_10 | 101.5–110.6* | 4.91–7.62 | 38319.7–39378.4 | 38006.5–38723.6 | **−1.25%** | 9785–10017 | 2611–2792 |

\* first dark frame is an outlier in both cases — see §6.

**Effective bit-depth used** (`log2(global_max+1)` over darks+flats+sampled-projections, `poisson_check.py`/`stats.py`):

| exp | global min | global max | bits used (of 16) |
|---|---|---|---|
| file_1 | 18 | 50027 | 15.61 |
| file_2 | 6 | 53400 | 15.70 |
| file_3 | 0 | 2842 | **11.47** |
| file_4 | 0 | 4095 (= 2¹²−1) | **12.00** |
| file_8 | 9 | 65535 (saturated) | 16.00 |
| file_10 | 36 | 65535 (saturated) | 16.00 |

Plot: `bit_depth_utilization_summary.png`

**Photon-transfer-curve (Poisson) check** on a central 200×200 patch, 40 leading-flat frames, dark-subtracted (`poisson_check.py`). Fit `var = a·signal + b` and report `N_eff = mean²/var` (effective detected-quanta count per pixel — standard PTC gain estimate):

| exp | dark temporal std (read noise, DN) | flat mean (dark-sub, DN) | flat temporal var (DN²) | N_eff (quanta/pixel) |
|---|---|---|---|---|
| file_1 | 3.00 | 42054.7 | 112694 | **15,694** |
| file_2 | 3.04 | 46008.1 | 114988 | **18,408** |
| file_3 | 9.02 | 1671.7 | 276.1 | **10,120** |
| file_4 | 3.01 | 2798.9 | 5402.8 | **1,450** |
| file_8 | 3.10 | 49101.8 | 97164 | **24,814** |
| file_10 | 4.71 | 32138.9 | 109397 | **9,442** |

Plot: `effective_photon_counts_summary.png`

Interpretation: temporal variance across flat frames scales with signal broadly consistently with a shot-noise-dominated detector (not a pure pixel-detector Poisson process — these are indirect scintillator-coupled sCMOS cameras, so "quanta" here means detected-light-quanta equivalent, not X-ray photons 1:1). **file_4 has by far the lowest effective count budget (~1450)** — consistent with its 4 ms exposure and explains its noisier, less-smooth sinograms (§4). Darks are stable (std 4.4–7.6 DN, i.e. read-noise floor only, no obvious thermal drift within a stack) except the file_3/file_10 first-frame outliers (§6).

---

## 4. Sinogram structure (flat/dark-corrected, `-log`)

3 rows each in file_1 (JC_62d), file_3 (drying_2), file_4 (FC_13p5keV…) — `sinogram.py`/`sino_analysis.py`. Plots (sinogram + |consecutive-angle-difference| side by side): `sinogram_file_1_extracted_row{539,1079,1619}.png`, `sinogram_file_3_extracted_row{474,949,1424}.png`, `sinogram_file_4_extracted_row{74,149,224}.png`.

| exp | row (of H) | dynamic range [atten. units] | object occupancy | edge touch (L/R) | rotation-axis offset (px, of W) | MAD(consec)/std |
|---|---|---|---|---|---|---|
| file_1 | 539/2160 | [−0.08, 1.60] | 97.9% | no/no | −47.1 | **0.0306** |
| file_1 | 1079/2160 | [−0.10, 1.65] | 95.2% | no/**yes** | +84.8 | **0.0322** |
| file_1 | 1619/2160 | [−0.06, 1.69] | 100% | **yes/yes** | **+194.8** | **0.0302** |
| file_3 | 474/1900 | [−0.36, 0.45] | 100% | yes/yes | +37.4 | **0.1808** |
| file_3 | 949/1900 | [0.00, 0.65] | 100% | yes/yes | +9.9 | **0.6292** |
| file_3 | 1424/1900 | [0.04, 0.92] | 100% | yes/yes | +16.1 | **0.3283** |
| file_4 | 74/300 | [−0.23, 2.89] | 96.4% | **yes**/no | −2.3 | 0.1161 |
| file_4 | 149/300 | [−0.19, 2.94] | 96.6% | no/no | −0.9 | 0.0971 |
| file_4 | 224/300 | [−0.16, 3.03] | 97.7% | no/no | +0.5 | 0.1408 |

Key findings:

- **Truncation / rotation-axis centering (file_1)**: 2 of 3 tested rows touch a detector edge, and the object's column-center-of-mass drifts up to **194.8 px (7.6% of the 2560 px width)** off the geometric center depending on row. A compressor/reconstructor that assumes a centered, fully-contained object per row would be wrong for a non-trivial fraction of this volume.
- **Angular smoothness is the critical number for the t+1-from-1..t compressor.** It varies **>20×** across experiments/rows: `MAD(consecutive)/std` = 0.030 (file_1, fine 0.12°/1501-proj static tomo scan) → 0.10–0.14 (file_4, low-photon-budget fast scan, N_eff≈1450) → **0.18–0.63 (file_3, drying_2)**. file_3's high ratio reflects a *genuinely time-evolving sample* (drying process changes structure during the 180° sweep) compounded by its coarse 0.36° step and modest photon budget (N_eff≈10,120). Plot: `angular_smoothness_summary.png`.
- **file_3 is truncated at both edges in all 3 tested rows** (small W=1008 detector after ROI/binning, sample fills the full width) — more truncation risk than file_1/file_4.
- **file_4 has the largest raw dynamic range** (up to 3.03 attenuation units) despite the lowest photon budget — consistent with strong phase-contrast edge-enhancement fringes (the log documents a Paganin phase-retrieval step for this sample/setup), which will show up as sharp high-frequency features that are harder to compress spatially but should still be *temporally* predictable across nearby angles.

---

## 5. Compressibility baseline (file_1 JC_62d, row 1080, n_proj=1501, W=2560, raw uint16 — not log-corrected)

`compressibility.py`. This measures the raw-value redundancy that any codec/predictor has to exploit; not the attenuation-domain signal from §4.

| method | size | bits/sample | ratio vs raw |
|---|---|---|---|
| (a) raw uint16 | 7.685 MB | 16.000 (nominal) | 1.00× |
| (b) zlib level-6 on raw bytes | 7.124 MB | 14.831 | 1.079× |
| (b) zstd level-19 on raw bytes | 7.071 MB | 14.722 | 1.087× |
| (c) angular delta (int16, `diff` along projection axis) + zlib | 5.990 MB | 12.480 | 1.283× |
| (c) angular delta + zstd-19 | 5.878 MB | 12.246 | 1.307× |
| (d) zeroth-order entropy, raw values | — | **14.134** | — |
| (d) zeroth-order entropy, angular delta | — | **10.296** | — |

Delta range along the angle axis: `[-2622, 3691]`.

Takeaways:
- Generic byte-oriented lossless compressors barely help on raw 16-bit CT projections (zlib/zstd ≈ **1.08×** only) — most of the entropy is genuine sensor/shot noise in the low bits, not byte-level redundancy.
- A trivial temporal (angular) delta already buys **1.28–1.31×** over raw and drops zeroth-order entropy from **14.13 → 10.30 bits/sample** (a 3.8-bit reduction) — this is a first-order validation of the "predict projection t+1 from 1..t" design premise, but note this is for **file_1**, the *smoothest* case in §4; expect a much smaller entropy reduction for file_3-like data given its 6–20× higher MAD/std.
- The gap between "delta zlib/zstd" (12.2–12.5 bits/sample) and "delta zeroth-order entropy" (10.3 bits/sample) shows headroom: a proper entropy coder matched to the empirical delta distribution (or a smarter, non-linear/learned predictor) should be able to beat naive zlib/zstd-on-delta by roughly another 2 bits/sample even without spatial modeling.

---

## 6. Gotchas

1. **Trailing flats: universal but not uniform in kind.** All 6 experiments have exactly `n_darks + n_flats + n_projections + 100` files, i.e. **100 trailing flat frames after the last projection in every case** (verified by exact file-count arithmetic). But their nature differs by camera:
   - **PCO.Edge scans (file_1, file_2, file_8, file_10): genuine independent re-acquisitions**, showing real beam/detector drift of **−2.2%, −1.2%, −0.12%, −1.25%** mean-intensity decay respectively relative to the leading flats.
   - **GigaFRoST scans (file_3, file_4): the "trailing flats" are byte-for-byte MD5-identical duplicates of the leading-flat stack** (verified: 100/100 hash-matched pairs for both, vs. 0/100 for the other four) — i.e. not a real second measurement, just padding/copying in how this benchmark set was assembled. A loader must not use them for drift correction in these two experiments, and must not assume "trailing flats = leading flats" as a general rule (it's only true here for 2 of 6).
2. **file_4's primary log file describes the wrong scan.** `FC_13p5keV_G_LAG100_4ms_70mm_1000p.log` opens with `"...scan of sample FC_13p5keV_G_LAG100_1ms_70mm_400p started..."` (1 ms exposure, 400-proj name) — text metadata for an unrelated/earlier scan — while the actual acquired data (folder name, `File Prefix`, 4 ms exposure) is only correctly described in the sibling `_orig.log`, which itself contains **two scan-restart entries** (1001 then 1000 projections). Resolved via the angular-range self-consistency check in §2 (n=1001 is correct). Any log parser for this dataset must handle multiple `.log` files per folder and multiple scan blocks within one file, and should cross-validate `(n_proj−1)×Δdeg` against the stated angular range rather than trusting the first match.
3. **Effective bit depth is well below the 16-bit container for 2/6 experiments**: file_3 tops out at 2842 (11.47 bits) and file_4 at exactly 4095 = 2¹²−1 (12-bit ADC ceiling), vs. file_1/2 at ~15.6–15.7 bits and file_8/10 which genuinely hit and saturate the 16-bit ceiling (65535). A one-size-fits-all bit-depth assumption for entropy coding would waste ~4–4.5 bits/sample of header room on file_3/file_4.
4. **Real saturation and dead/stuck pixels** (30-frame leading-flat stack, `gotchas.py`): file_10 has 0.040% of pixels at the 65535 ceiling and **1653/4,510,720 (0.037%) zero-temporal-variance (stuck) pixels**; file_4 has a smaller stuck cluster (19/604,800, 0.0031%). file_1 and file_3 show none of either in the same test. No hot/cold outlier pixels (>8σ) found in any of the four tested experiments.
5. **file_8 sample is nearly transparent**: projection-sample mean (41.6–41.8k) is close to its own flat-field mean (42.5–42.8k) — weak absorption at the lowest beam energy in the set (10 keV), meaning smaller usable dynamic range and worse signal-to-noise-relative-to-signal than the other tomography scans; worth flagging before using it as a "typical" compression test case.
6. **Rotation axis is not centered / object is truncated in a row-dependent way** for file_1 (§4): 2 of 3 rows tested touch a detector edge and the object footprint's column center-of-mass shifts by up to 7.6% of detector width across rows. file_3 is truncated at both edges in every row tested (small 1008 px detector width after ROI/binning).
7. **file_3's first dark frame is an outlier**: mean 14.6 DN vs. 66–75 DN for the remaining 49 frames in the 50-frame dark stack — looks like a camera-not-yet-settled first-frame artifact typical of GigaFRoST's external/auto trigger mode. A robust dark estimate should use a median (or drop frame 0), not a naive mean, at least for file_3/file_4-style acquisitions.
8. **Angular smoothness varies by >20× across experiments** (§4) — any single global model/entropy-coding assumption tuned on file_1-like (smooth, static, fine-angular-step) data will badly under-perform on file_3-like (coarse-step, genuinely dynamic drying sample) or file_4-like (low-photon-budget, phase-contrast) data. This should directly inform whether the compression module needs per-experiment-class tuning or a noise/smoothness-adaptive predictor.
9. **Dataset size is very unevenly distributed**: the 4 PCO.Edge/full-frame scans are 15.6–18.9 GB each; the 2 GigaFRoST scans (smaller cropped ROIs, 300 or ~1000–1900 rows) are only 1.4–2.9 GB. Aggregate compression-ratio numbers across the 6 experiments will be dominated by the 4 large ones unless explicitly weighted or reported per-experiment.

---

## Appendix: file paths

- Raw log dump (all 6, concatenated): available via `cat /myhome/data/sdate/shared/compression_paper/*_extracted/*.log`
- Plots (this dir, `/myhome/sdate/context/compression_data_recon/`):
  - `sinogram_file_1_extracted_row539.png`, `_row1079.png`, `_row1619.png`
  - `sinogram_file_3_extracted_row474.png`, `_row949.png`, `_row1424.png`
  - `sinogram_file_4_extracted_row74.png`, `_row149.png`, `_row224.png`
  - `angular_smoothness_summary.png`
  - `effective_photon_counts_summary.png`
  - `bit_depth_utilization_summary.png`
