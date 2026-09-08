# Residual Computation and Loading Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 1: INITIAL TRAINING                           │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │  TIFF Files  │ (Ground Truth)
    │  (D,H,W)     │
    └──────┬───────┘
           │
           ├──────────────────┐
           │                  │
           v                  v
    ┌──────────────┐    ┌──────────────┐
    │ HEIC Encoder │    │   UNet3D     │
    │  (compress)  │    │   Training   │
    └──────┬───────┘    └──────┬───────┘
           │                   │
           v                   v
    ┌──────────────┐    ┌──────────────┐
    │ HEIC Files   │    │  Checkpoint  │
    │  (lossy)     │    │  (trained)   │
    └──────────────┘    └──────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 2: RESIDUAL COMPUTATION                           │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐         ┌──────────────┐
    │  TIFF Files  │         │ HEIC Files   │
    │  (target)    │         │  (input)     │
    └──────┬───────┘         └──────┬───────┘
           │                        │
           │                        v
           │                 ┌──────────────┐
           │                 │   Trained    │
           │                 │   UNet3D     │
           │                 └──────┬───────┘
           │                        │
           │                        v
           │                 ┌──────────────┐
           │                 │  Predicted   │
           │                 │    TIFF      │
           │                 └──────┬───────┘
           │                        │
           └────────────┬───────────┘
                        │
                        v
                ┌───────────────┐
                │   Residual    │
                │ (Pred - True) │
                └───────┬───────┘
                        │
                        v
           ┌────────────────────────────┐
           │   compute_residuals.py     │
           │                            │
           │  • Loads model checkpoint  │
           │  • Runs inference          │
           │  • Computes differences    │
           │  • Saves to disk           │
           └────────────┬───────────────┘
                        │
                        v
            ┌───────────────────────┐
            │   Residuals on Disk   │
            │                       │
            │ • residuals.npy       │
            │   (num_vols, 64³)     │
            │ • positions.npy       │
            │   (num_vols, 3)       │
            │ • metadata.npz        │
            └───────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                   PHASE 3: ENHANCED DATASET LOADING                         │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  TIFF Files  │  │ HEIC Files   │  │ Residuals    │
    │  (channel 0) │  │ (channel 1)  │  │ (channel 2)  │
    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
           │                 │                 │
           └────────┬────────┴────────┬────────┘
                    │                 │
                    v                 v
           ┌──────────────────────────────────┐
           │     TiffVolumeDataset            │
           │                                  │
           │  use_residuals=True              │
           │  residuals_path='...'            │
           │                                  │
           │  • Loads TIFF frames             │
           │  • Loads HEIC frames             │
           │  • Memory-maps residuals         │
           │  • Validates compatibility       │
           └──────────────┬───────────────────┘
                          │
                          v
                ┌───────────────────┐
                │   3-Channel       │
                │   Sub-volume      │
                │                   │
                │  (3, 64, 64, 64)  │
                │   ↓   ↓    ↓      │
                │ TIFF HEIC  RES    │
                └─────────┬─────────┘
                          │
                          v
                ┌───────────────────┐
                │   DataLoader      │
                │                   │
                │  Batch: (B, 3,    │
                │         64,64,64) │
                └─────────┬─────────┘
                          │
                          v
                ┌───────────────────┐
                │  Training Loop    │
                │                   │
                │  • Error analysis │
                │  • Residual       │
                │    correction     │
                │  • Model refine   │
                └───────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                      DATA STRUCTURE DETAILS                                 │
└─────────────────────────────────────────────────────────────────────────────┘

MEMORY LAYOUT (3-channel mode):

  Sub-volume tensor: (3, 64, 64, 64)
                      ↓
         ┌────────────┼────────────┐
         │            │            │
    Channel 0    Channel 1    Channel 2
      TIFF         HEIC       Residual
   (64,64,64)   (64,64,64)   (64,64,64)
      ↓            ↓            ↓
  Ground Truth   Input      Pred - True


DISK LAYOUT:

  data_path/
  ├── frame_0000.tif      ← Original TIFF files
  ├── frame_0001.tif
  ├── ...
  └── heic/
      ├── frame_0000.heic ← Compressed HEIC files
      ├── frame_0001.heic
      └── ...

  residuals_path/
  ├── data_residuals.npy  ← Memory-mapped (1024, 64, 64, 64)
  ├── data_positions.npy  ← Position indices (1024, 3)
  └── data_metadata.npz   ← Validation metadata


INDEXING:

  dataset[idx] → returns:
    - sub_volume: (3, 64, 64, 64)
    - position:   (3,) [d_start, h_start, w_start]

  Internal lookup:
    volume[d:d+64, :, h:h+64, w:w+64]  → TIFF + HEIC channels
    residuals[idx]                     → corresponding residual


┌─────────────────────────────────────────────────────────────────────────────┐
│                         USAGE PATTERNS                                      │
└─────────────────────────────────────────────────────────────────────────────┘

PATTERN 1: Direct Residual Correction
┌─────┐  ┌─────┐     ┌───────┐     ┌──────┐
│HEIC │  │ RES │ ──> │ Model │ ──> │ Corr │ ──> Corrected = HEIC + Corr
└─────┘  └─────┘     └───────┘     └──────┘

PATTERN 2: Attention Weighting
┌─────┐     ┌─────────┐     ┌──────────┐
│ RES │ ──> │Attention│ ──> │ Weights  │
└─────┘     └─────────┘     └────┬─────┘
┌─────┐                           │
│HEIC │ ─────────────────────────→ × ──> Weighted HEIC
└─────┘

PATTERN 3: Multi-input Processing
┌─────┐ ─────┐
│HEIC │      │
└─────┘      ├──> ┌───────────┐     ┌────────┐
┌─────┐      │    │ Dual-Path │ ──> │  Fused │ ──> Output
│ RES │ ─────┘    │   Model   │     │ Output │
└─────┘           └───────────┘     └────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                    MEMORY EFFICIENCY                                        │
└─────────────────────────────────────────────────────────────────────────────┘

Memory-Mapped Array (mmap_mode='r'):

  ┌─────────────────┐
  │   residuals.npy │  ← On disk (1 GB)
  └────────┬────────┘
           │ mmap
           v
  ┌─────────────────┐
  │  Virtual Memory │  ← Not loaded into RAM
  └────────┬────────┘
           │ On-demand load
           v
  ┌─────────────────┐
  │  Current Batch  │  ← Only batch in RAM (4 MB)
  │  (4, 64³)       │
  └─────────────────┘

Benefits:
  • Low memory footprint
  • Fast random access
  • Multi-process safe
  • Scales to large datasets
```
