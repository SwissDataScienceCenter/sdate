"""
CT Projection Compressor.

Compresses a sequence of 2-D tomographic projection frames into a single
binary archive file.

Strategy
--------
1. The first *k* frames (``num_input_projections``, default 3) are stored as
   intra-frame JPEG images.
2. For every subsequent frame *i*, the predictor builds a prediction from
   frames [i-k … i-1].  Residuals = actual – predicted.
3. The residual frame is split into non-overlapping 256×256 patches.
   For each patch:
   - Encode the residual with DCT+Huffman (``ResidualEncoder``).
   - Also encode the raw patch as JPEG at the configured quality.
   - Keep whichever is *smaller* (adaptive fallback).
4. Everything is packed into a single ``.ctc`` binary archive with a
   self-describing header so that the decompressor can reconstruct
   the original frame sequence.
"""

import io
import struct
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, TYPE_CHECKING
from PIL import Image
from tqdm.auto import tqdm

from .predictor import BlockPredictor
from .drift_predictor import DriftPredictor
from .entropy import (
    ResidualEncoder,
    ResidualDecoder,
    MAGIC,
    VERSION,
    FRAME_TYPE_INTRA,
    FRAME_TYPE_PREDICTED,
    PATCH_MODE_RESIDUAL,
    PATCH_MODE_JPEG_FALLBACK,
)

# Type alias for any predictor that exposes predict_frame()
Predictor = Union[BlockPredictor, DriftPredictor]


def _encode_jpeg(patch: np.ndarray, quality: int = 95) -> bytes:
    """Encode a [0,1] float32 patch to JPEG bytes (uint16 stored as 16-bit PNG
    when precision matters, but JPEG for compactness)."""
    # Map to uint8 for JPEG
    arr = np.clip(patch * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _decode_jpeg(blob: bytes) -> np.ndarray:
    """Decode JPEG bytes back to float32 [0,1]."""
    img = Image.open(io.BytesIO(blob)).convert("L")
    return np.array(img, dtype=np.float32) / 255.0


class CTCompressor:
    """
    Compress a list of CT projection frames (2-D float32 arrays in [0,1])
    into a single ``.ctc`` archive file.

    Parameters
    ----------
    predictor : BlockPredictor
        A configured predictor wrapping the trained UNet model.
    patch_size : int
        Size of the non-overlapping patches used for residual coding.
        Must be a multiple of 8. Default 256.
    residual_quality : int
        DCT quantization quality for residuals (1–100). Default 80.
    jpeg_quality : int
        JPEG quality for intra-frames and fallback patches (1–100). Default 95.
    dct_block_size : int
        DCT block size inside the residual encoder. Default 8.
    fallback_threshold : float or None
        If not None, automatically fall back to JPEG when residual MSE
        exceeds this value. Regardless of this setting, the compressor
        always keeps the *smaller* of the two encodings.
    verbose : bool
        Print progress information.
    """

    def __init__(
        self,
        predictor: "Predictor",
        patch_size: int = 256,
        residual_quality: int = 80,
        jpeg_quality: int = 95,
        dct_block_size: int = 8,
        fallback_threshold: Optional[float] = None,
        verbose: bool = True,
    ):
        self.predictor = predictor
        self.patch_size = patch_size
        self.residual_quality = residual_quality
        self.jpeg_quality = jpeg_quality
        self.dct_block_size = dct_block_size
        self.fallback_threshold = fallback_threshold
        self.verbose = verbose
        self.k = predictor.num_input_projections

        self._res_encoder = ResidualEncoder(
            block_size=dct_block_size, quality=residual_quality
        )

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def compress(
        self,
        frames: List[np.ndarray],
        output_path: Union[str, Path],
        center_x_coords: Optional[List[float]] = None,
    ) -> dict:
        """
        Compress *frames* and write the archive to *output_path*.

        Parameters
        ----------
        frames : list of np.ndarray
            Sequence of 2-D float32 arrays (H, W) with values in [0, 1].
            All frames must have the same shape.
        output_path : str or Path
            Destination file (conventionally ``*.ctc``).
        center_x_coords : list of float or None
            Per-frame normalised x-coordinate of the original image
            centre (in [0, 1]).  Required when the predictor was
            created with ``use_conditioning=True``; ignored otherwise.
            If a single float is given it is broadcast to all frames.

        Returns
        -------
        stats : dict
            Compression statistics (total bytes, per-frame breakdown, …).
        """
        output_path = Path(output_path)
        N = len(frames)
        H, W = frames[0].shape
        k = self.k

        assert N > k, f"Need at least {k + 1} frames, got {N}"

        # Resolve center_x conditioning
        use_cond = self.predictor.use_conditioning
        if use_cond:
            if center_x_coords is None:
                raise ValueError(
                    "center_x_coords must be provided when the predictor "
                    "uses conditioning (use_conditioning=True)"
                )
            if isinstance(center_x_coords, (int, float)):
                center_x_coords = [float(center_x_coords)] * N
            assert len(center_x_coords) == N, (
                f"center_x_coords length ({len(center_x_coords)}) != "
                f"number of frames ({N})"
            )

        stats = {
            "num_frames": N,
            "height": H,
            "width": W,
            "frame_bytes": [],
            "fallback_counts": [],
            "residual_counts": [],
        }

        buf = io.BytesIO()

        # ---- File header ---- #
        buf.write(MAGIC)
        buf.write(struct.pack("<B", VERSION))
        buf.write(struct.pack("<I", N))       # number of frames
        buf.write(struct.pack("<HH", H, W))   # frame dims
        buf.write(struct.pack("<B", k))        # num_input_projections
        buf.write(struct.pack("<H", self.patch_size))
        buf.write(struct.pack("<B", self.dct_block_size))
        buf.write(struct.pack("<B", self.residual_quality))
        buf.write(struct.pack("<B", self.jpeg_quality))
        # Conditioning flag (1 byte) + per-frame center_x values (float32)
        buf.write(struct.pack("<B", 1 if use_cond else 0))
        if use_cond:
            for cx in center_x_coords:
                buf.write(struct.pack("<f", cx))

        # ---- Intra-frames (first k) ---- #
        # We store the reconstructed intra-frames so that the predictor
        # and decoder stay in sync (both use the lossy decoded versions).
        reconstructed: List[np.ndarray] = []

        for i in range(min(k, N)):
            jpeg_blob = _encode_jpeg(frames[i], quality=self.jpeg_quality)
            # Write frame header
            buf.write(struct.pack("<B", FRAME_TYPE_INTRA))
            buf.write(struct.pack("<I", len(jpeg_blob)))
            buf.write(jpeg_blob)

            # Decode back so predictor uses the *lossy* version
            reconstructed.append(_decode_jpeg(jpeg_blob))

            stats["frame_bytes"].append(len(jpeg_blob))
            stats["fallback_counts"].append(0)
            stats["residual_counts"].append(0)

            if self.verbose:
                print(f"  Frame {i:4d}  INTRA   {len(jpeg_blob):>8,} B")

        # ---- Predicted frames ---- #
        frame_iter = range(k, N)
        if self.verbose:
            frame_iter = tqdm(frame_iter, desc="Compressing", unit="frame")

        for i in frame_iter:
            prev = reconstructed[-k:]
            cx = center_x_coords[i] if use_cond else None
            predicted = self.predictor.predict_frame(prev, center_x=cx)

            actual = frames[i]
            residual = actual - predicted

            # Pad frame to multiple of patch_size for splitting
            ps = self.patch_size
            pad_h = (ps - H % ps) % ps
            pad_w = (ps - W % ps) % ps
            res_padded = np.pad(residual, ((0, pad_h), (0, pad_w)), mode="constant")
            act_padded = np.pad(actual, ((0, pad_h), (0, pad_w)), mode="constant")
            pH, pW = res_padded.shape

            n_patches_h = pH // ps
            n_patches_w = pW // ps
            n_patches = n_patches_h * n_patches_w

            patch_blobs: List[bytes] = []
            patch_modes: List[int] = []
            n_fallback = 0
            n_residual = 0

            for ph in range(n_patches_h):
                for pw in range(n_patches_w):
                    y0, x0 = ph * ps, pw * ps
                    res_patch = res_padded[y0 : y0 + ps, x0 : x0 + ps]
                    act_patch = act_padded[y0 : y0 + ps, x0 : x0 + ps]

                    # Encode residual
                    # Ensure patch is multiple of dct_block_size
                    res_blob = self._res_encoder.encode(res_patch)

                    # Encode raw patch as JPEG fallback
                    jpeg_blob = _encode_jpeg(act_patch, quality=self.jpeg_quality)

                    # Check if prediction quality is poor
                    mse = float(np.mean(res_patch ** 2))
                    force_fallback = (
                        self.fallback_threshold is not None
                        and mse > self.fallback_threshold
                    )

                    # Keep whichever is smaller (or force JPEG if bad prediction)
                    if force_fallback or len(jpeg_blob) < len(res_blob):
                        patch_blobs.append(jpeg_blob)
                        patch_modes.append(PATCH_MODE_JPEG_FALLBACK)
                        n_fallback += 1
                    else:
                        patch_blobs.append(res_blob)
                        patch_modes.append(PATCH_MODE_RESIDUAL)
                        n_residual += 1

            # Write predicted frame header
            buf.write(struct.pack("<B", FRAME_TYPE_PREDICTED))
            buf.write(struct.pack("<HH", n_patches_h, n_patches_w))
            buf.write(struct.pack("<HH", pad_h, pad_w))

            # Write each patch
            for mode, blob in zip(patch_modes, patch_blobs):
                buf.write(struct.pack("<B", mode))
                buf.write(struct.pack("<I", len(blob)))
                buf.write(blob)

            # Reconstruct this frame for use by future predictions
            recon = self._reconstruct_frame(
                predicted, patch_blobs, patch_modes,
                n_patches_h, n_patches_w, pad_h, pad_w, H, W,
            )
            reconstructed.append(recon)

            frame_bytes = sum(len(b) for b in patch_blobs) + 5 + n_patches * 5
            stats["frame_bytes"].append(frame_bytes)
            stats["fallback_counts"].append(n_fallback)
            stats["residual_counts"].append(n_residual)

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(buf.getvalue())

        total_bytes = buf.tell()
        stats["total_bytes"] = total_bytes
        stats["compression_ratio"] = (N * H * W * 4) / max(total_bytes, 1)

        if self.verbose:
            print(f"\n✅ Compressed {N} frames ({H}×{W}) → {total_bytes:,} bytes")
            print(f"   Compression ratio: {stats['compression_ratio']:.2f}x")
            total_fb = sum(stats["fallback_counts"])
            total_res = sum(stats["residual_counts"])
            print(f"   Residual patches: {total_res}, JPEG fallbacks: {total_fb}")

        return stats

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _reconstruct_frame(
        self,
        predicted: np.ndarray,
        patch_blobs: List[bytes],
        patch_modes: List[int],
        n_patches_h: int,
        n_patches_w: int,
        pad_h: int,
        pad_w: int,
        H: int,
        W: int,
    ) -> np.ndarray:
        """Reconstruct a frame from its prediction + stored patches.

        This mirrors exactly what the decompressor will do, keeping
        encoder and decoder in sync.
        """
        ps = self.patch_size
        pH = n_patches_h * ps
        pW = n_patches_w * ps
        res_decoder = ResidualDecoder(
            block_size=self.dct_block_size, quality=self.residual_quality
        )

        # Pad prediction to match
        pred_padded = np.pad(predicted, ((0, pad_h), (0, pad_w)), mode="constant")

        recon_padded = np.zeros((pH, pW), dtype=np.float32)
        idx = 0
        for ph in range(n_patches_h):
            for pw in range(n_patches_w):
                y0, x0 = ph * ps, pw * ps
                mode = patch_modes[idx]
                blob = patch_blobs[idx]
                idx += 1

                if mode == PATCH_MODE_JPEG_FALLBACK:
                    recon_padded[y0 : y0 + ps, x0 : x0 + ps] = _decode_jpeg(blob)
                else:
                    residual_patch = res_decoder.decode(blob)
                    recon_padded[y0 : y0 + ps, x0 : x0 + ps] = (
                        pred_padded[y0 : y0 + ps, x0 : x0 + ps] + residual_patch
                    )

        return np.clip(recon_padded[:H, :W], 0, 1)
