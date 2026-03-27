"""
CT Projection Decompressor.

Reads a ``.ctc`` archive produced by ``CTCompressor`` and reconstructs
the original sequence of projection frames.
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


def _decode_jpeg(blob: bytes) -> np.ndarray:
    """Decode JPEG bytes back to float32 [0,1]."""
    img = Image.open(io.BytesIO(blob)).convert("L")
    return np.array(img, dtype=np.float32) / 255.0


class CTDecompressor:
    """
    Decompress a ``.ctc`` archive back into a list of 2-D float32 frames.

    Parameters
    ----------
    predictor : BlockPredictor
        The same predictor (with the same model weights) used during
        compression.
    verbose : bool
        Print progress information.
    """

    def __init__(
        self,
        predictor: "Predictor",
        verbose: bool = True,
    ):
        self.predictor = predictor
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def decompress(
        self,
        archive_path: Union[str, Path],
    ) -> List[np.ndarray]:
        """
        Read a ``.ctc`` archive and return the reconstructed frame list.

        Parameters
        ----------
        archive_path : str or Path

        Returns
        -------
        frames : list of np.ndarray
            Each element is a 2-D float32 array (H, W) in [0, 1].
        """
        archive_path = Path(archive_path)

        with open(archive_path, "rb") as fh:
            data = fh.read()

        buf = io.BytesIO(data)

        # ---- Read file header ---- #
        magic = buf.read(4)
        assert magic == MAGIC, f"Invalid archive (magic={magic!r})"
        (version,) = struct.unpack("<B", buf.read(1))
        assert version == VERSION, f"Unsupported version {version}"

        (num_frames,) = struct.unpack("<I", buf.read(4))
        H, W = struct.unpack("<HH", buf.read(4))
        (k,) = struct.unpack("<B", buf.read(1))
        (patch_size,) = struct.unpack("<H", buf.read(2))
        (dct_block_size,) = struct.unpack("<B", buf.read(1))
        (residual_quality,) = struct.unpack("<B", buf.read(1))
        (jpeg_quality,) = struct.unpack("<B", buf.read(1))

        # Conditioning flag + per-frame center_x values
        (use_cond_flag,) = struct.unpack("<B", buf.read(1))
        use_cond = bool(use_cond_flag)
        center_x_coords: Optional[List[float]] = None
        if use_cond:
            center_x_coords = [
                struct.unpack("<f", buf.read(4))[0]
                for _ in range(num_frames)
            ]

        if self.verbose:
            print(f"📂 Archive: {archive_path.name}")
            print(f"   Frames: {num_frames}, Size: {H}×{W}, k={k}")
            print(f"   Patch: {patch_size}, DCT block: {dct_block_size}")
            print(f"   Residual quality: {residual_quality}, JPEG quality: {jpeg_quality}")
            print(f"   Conditioning: {'ON' if use_cond else 'OFF'}")

        res_decoder = ResidualDecoder(
            block_size=dct_block_size, quality=residual_quality
        )

        reconstructed: List[np.ndarray] = []

        frame_iter = range(num_frames)
        if self.verbose:
            frame_iter = tqdm(frame_iter, desc="Decompressing", unit="frame")

        for frame_i in frame_iter:
            (frame_type,) = struct.unpack("<B", buf.read(1))

            if frame_type == FRAME_TYPE_INTRA:
                # ---- Intra frame (JPEG) ---- #
                (blob_len,) = struct.unpack("<I", buf.read(4))
                blob = buf.read(blob_len)
                frame = _decode_jpeg(blob)
                # Resize to (H, W) in case JPEG rounding changed dims
                if frame.shape != (H, W):
                    frame = np.array(
                        Image.fromarray(
                            (frame * 255).astype(np.uint8)
                        ).resize((W, H), Image.BILINEAR),
                        dtype=np.float32,
                    ) / 255.0
                reconstructed.append(frame)

            elif frame_type == FRAME_TYPE_PREDICTED:
                # ---- Predicted frame ---- #
                n_patches_h, n_patches_w = struct.unpack("<HH", buf.read(4))
                pad_h, pad_w = struct.unpack("<HH", buf.read(4))

                # Build prediction from previous k frames
                prev = reconstructed[-k:]
                cx = center_x_coords[frame_i] if use_cond else None
                predicted = self.predictor.predict_frame(prev, center_x=cx)

                # Pad prediction to match
                pred_padded = np.pad(
                    predicted, ((0, pad_h), (0, pad_w)), mode="constant"
                )

                ps = patch_size
                pH = n_patches_h * ps
                pW = n_patches_w * ps
                recon_padded = np.zeros((pH, pW), dtype=np.float32)

                for ph in range(n_patches_h):
                    for pw in range(n_patches_w):
                        y0, x0 = ph * ps, pw * ps
                        (mode,) = struct.unpack("<B", buf.read(1))
                        (blob_len,) = struct.unpack("<I", buf.read(4))
                        blob = buf.read(blob_len)

                        if mode == PATCH_MODE_JPEG_FALLBACK:
                            patch = _decode_jpeg(blob)
                            # Resize if needed
                            if patch.shape != (ps, ps):
                                patch = np.array(
                                    Image.fromarray(
                                        (patch * 255).astype(np.uint8)
                                    ).resize((ps, ps), Image.BILINEAR),
                                    dtype=np.float32,
                                ) / 255.0
                            recon_padded[y0 : y0 + ps, x0 : x0 + ps] = patch
                        elif mode == PATCH_MODE_RESIDUAL:
                            residual_patch = res_decoder.decode(blob)
                            recon_padded[y0 : y0 + ps, x0 : x0 + ps] = (
                                pred_padded[y0 : y0 + ps, x0 : x0 + ps]
                                + residual_patch
                            )
                        else:
                            raise ValueError(f"Unknown patch mode {mode}")

                frame = np.clip(recon_padded[:H, :W], 0, 1)
                reconstructed.append(frame)

            else:
                raise ValueError(f"Unknown frame type {frame_type}")

        if self.verbose:
            print(f"✅ Decompressed {len(reconstructed)} frames ({H}×{W})")

        return reconstructed
