"""
gsplat_compress — Video compression via Gaussian splatting delta encoding.

Represents video frames as collections of 2D Gaussians rendered through
gsplat's orthographic rasterisation pipeline.  The first frame (keyframe) is
fitted from scratch; subsequent frames are encoded as codebook-quantised
deltas of the Gaussian parameters.

Modules
-------
camera        Orthographic camera setup
renderer      Differentiable rendering (gsplat wrapper)
initialize    Gaussian parameter initialisation
training      Training loops (keyframe + fine-tune)
compression   Delta compression, K-means codebook, codebook fine-tuning
metrics       Quality (PSNR, SSIM) and compression-ratio accounting
storage       Efficient serialisation (redundancy stripping, FP16)
video         High-level video compression pipeline
"""

# ── Core types ────────────────────────────────────────────────────────────
from gsplat_compress.initialize import GaussianParams, init_gaussians
from gsplat_compress.camera import ortho_camera
from gsplat_compress.renderer import render

# ── Initialisation strategies ─────────────────────────────────────────────
from gsplat_compress.initializations import (
    uniform_2d,
    intensity_2d,
    multiresolution_residual_2d,
    uniform_3d,
)

# ── Training ──────────────────────────────────────────────────────────────
from gsplat_compress.training import (
    TrainConfig,
    FinetuneConfig,
    TrainResult,
    train_keyframe,
    finetune_frame,
)

# ── Compression ───────────────────────────────────────────────────────────
from gsplat_compress.compression import (
    CodebookConfig,
    CompressedFrame,
    compute_deltas,
    build_codebook,
    finetune_codebook,
    encode_delta_frame,
    reconstruct_compressed_frame,
    reconstruct_from_delta,
    split_delta,
    D_FULL,
    D_EFF,
    PARAMS_PER_GAUSSIAN_EFF,
    PARAMS_PER_GAUSSIAN_FULL,
    S_M, S_S, S_Q, S_R, S_O,
)

# ── Metrics ───────────────────────────────────────────────────────────────
from gsplat_compress.metrics import (
    psnr,
    ssim,
    mse,
    keyframe_bytes,
    delta_frame_bytes,
    raw_frame_bytes,
    compression_ratio,
    sequence_storage_summary,
)

# ── Storage ───────────────────────────────────────────────────────────────
from gsplat_compress.storage import save_sequence, load_sequence

# ── Video pipeline ────────────────────────────────────────────────────────
from gsplat_compress.video import (
    VideoCompressor,
    FrameResult,
    decode_sequence,
)

__all__ = [
    # Core types + convenience init
    "GaussianParams",
    "init_gaussians",
    "ortho_camera",
    "render",
    # Initialisation strategies (2-D)
    "uniform_2d",
    "intensity_2d",
    "multiresolution_residual_2d",
    # Initialisation strategies (3-D)
    "uniform_3d",
    # Training
    "TrainConfig",
    "FinetuneConfig",
    "TrainResult",
    "train_keyframe",
    "finetune_frame",
    # Compression
    "CodebookConfig",
    "CompressedFrame",
    "compute_deltas",
    "build_codebook",
    "finetune_codebook",
    "encode_delta_frame",
    "reconstruct_compressed_frame",
    "reconstruct_from_delta",
    "split_delta",
    "D_FULL",
    "D_EFF",
    "PARAMS_PER_GAUSSIAN_EFF",
    "PARAMS_PER_GAUSSIAN_FULL",
    "S_M", "S_S", "S_Q", "S_R", "S_O",
    # Metrics
    "psnr",
    "ssim",
    "mse",
    "keyframe_bytes",
    "delta_frame_bytes",
    "raw_frame_bytes",
    "compression_ratio",
    "sequence_storage_summary",
    # Storage
    "save_sequence",
    "load_sequence",
    # Video pipeline
    "VideoCompressor",
    "FrameResult",
    "decode_sequence",
]
