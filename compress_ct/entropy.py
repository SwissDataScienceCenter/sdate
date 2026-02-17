"""
Entropy coding utilities for residual compression.

Provides DCT + quantization + Huffman coding for residual frames (float32),
plus helpers for reading / writing the binary archive format.
"""

import io
import struct
import numpy as np
from collections import Counter
from typing import Dict, Tuple, Optional

try:
    from torch_dct import dct_2d, idct_2d
    import torch
    _HAS_TORCH_DCT = True
except ImportError:
    _HAS_TORCH_DCT = False

try:
    import huffman as _huffman
    _HAS_HUFFMAN = True
except ImportError:
    _HAS_HUFFMAN = False


# -----------------------------------------------------------------------
# Binary format magic numbers / identifiers
# -----------------------------------------------------------------------
MAGIC = b"CTCM"                # CT Compression Magic
VERSION = 1
FRAME_TYPE_INTRA = 0           # JPEG intra-frame
FRAME_TYPE_PREDICTED = 1       # model prediction + residual
PATCH_MODE_RESIDUAL = 0        # DCT-coded residual
PATCH_MODE_JPEG_FALLBACK = 1   # JPEG fallback for bad prediction


# -----------------------------------------------------------------------
# Residual encoder / decoder using DCT + Quantization + Huffman
# -----------------------------------------------------------------------

class ResidualEncoder: 
    """
    Compress a floating-point residual patch into a compact byte stream.

    Pipeline: DCT → Adaptive Quantize → Zig-zag scan → Huffman.

    Parameters
    ----------
    block_size : int
        DCT block size (must be a multiple of 8). Default 8.
    quality : int
        Quality parameter (1–100). Higher = less quantization. Default 80.
    adaptive : bool
        If True, use adaptive quantization scaled to data. Default True.
    """

    def __init__(self, block_size: int = 8, quality: int = 80, adaptive: bool = True):
        if not _HAS_TORCH_DCT:
            raise ImportError("torch_dct is required for residual compression")
        if not _HAS_HUFFMAN:
            raise ImportError("huffman package is required for entropy coding")

        self.block_size = block_size
        self.quality = quality
        self.adaptive = adaptive
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Standard JPEG luminance quantization pattern (8×8) - relative importance only
        self._base_pattern = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ], dtype=np.float32)

        if block_size != 8:
            if block_size % 8 != 0:
                raise ValueError("block_size must be a multiple of 8")
            self._base_pattern = np.repeat(
                np.repeat(self._base_pattern, block_size // 8, axis=0), 
                block_size // 8, axis=1
            )

        # If not adaptive, compute static quantization matrix now
        if not adaptive:
            scale = (5000 / quality) if quality < 50 else (200 - 2 * quality)
            self._q_matrix = torch.from_numpy((self._base_pattern * scale / 100).clip(1)).float().to(self.device)
        else:
            self._q_matrix = None  # Will be computed adaptively per encode
    
    def _compute_adaptive_quantization_matrix(self, dct_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive quantization matrix scaled to actual DCT coefficient magnitudes.
        
        Parameters
        ----------
        dct_coeffs : torch.Tensor
            Shape (N, 1, block_size, block_size) where N is number of blocks
        
        Returns
        -------
        q_matrix : torch.Tensor
            Adaptive quantization matrix of shape (1, 1, block_size, block_size)
        """
        # Compute per-frequency statistics across all blocks
        freq_std = dct_coeffs.std(dim=0, keepdim=True)  # (1, 1, bs, bs)
        
        # Normalize base pattern and convert to torch
        base_pattern_normalized = torch.from_numpy(
            self._base_pattern / self._base_pattern.max()
        ).float().to(self.device)
        
        # Scale by frequency statistics
        adaptive_scale = freq_std + 1e-6
        
        # Apply quality-based scaling
        if self.quality < 50:
            quality_scale = 5000 / max(self.quality, 1)
        else:
            quality_scale = 200 - 2 * self.quality
        
        # Combine: base pattern × frequency scale × quality scale
        q_matrix = base_pattern_normalized * adaptive_scale * (quality_scale / 100)
        q_matrix = torch.clamp(q_matrix, min=1.0)
        
        return q_matrix

    # ------------------------------------------------------------------ #
    #  Encode
    # ------------------------------------------------------------------ #

    def encode(self, residual: np.ndarray) -> bytes:
        """
        Encode a 2-D float32 residual patch into a compact byte blob.

        Parameters
        ----------
        residual : np.ndarray   (H, W), float32

        Returns
        -------
        blob : bytes
        """
        H, W = residual.shape
        bs = self.block_size
        assert H % bs == 0 and W % bs == 0, (
            f"Patch dims ({H},{W}) must be multiples of block_size={bs}"
        )

        nb_h, nb_w = H // bs, W // bs
        res_t = torch.from_numpy(residual).float().to(self.device)

        # Reshape into (nb_h*nb_w, 1, bs, bs) for batch DCT
        blocks = (
            res_t
            .reshape(nb_h, bs, nb_w, bs)
            .permute(0, 2, 1, 3)
            .reshape(-1, 1, bs, bs)
        )
        dct_c = dct_2d(blocks)
        
        # Compute adaptive or static quantization matrix
        if self.adaptive:
            q_matrix = self._compute_adaptive_quantization_matrix(dct_c)
        else:
            q_matrix = self._q_matrix
        
        quant = torch.round(dct_c / q_matrix).to(torch.int32)
        flat = quant.cpu().numpy().flatten().tolist()

        # Huffman coding
        freqs = Counter(flat).items()
        if len(list(freqs)) < 2:
            freqs = list(Counter(flat).items()) + [("__dummy__", 1)]
        else:
            freqs = list(Counter(flat).items())
        codebook = _huffman.codebook(freqs)
        codebook.pop("__dummy__", None)

        bitstring = "".join(codebook[v] for v in flat)

        # Pack with quantization matrix for decoder
        return self._pack(H, W, codebook, bitstring, q_matrix.cpu().numpy())

    # ------------------------------------------------------------------ #
    #  Binary packing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pack(H: int, W: int, codebook: dict, bitstring: str, q_matrix: np.ndarray) -> bytes:
        """Serialize Huffman-coded residual to bytes, including quantization matrix."""
        buf = io.BytesIO()

        # Header: H, W (uint16)
        buf.write(struct.pack("<HH", H, W))
        
        # Quantization matrix (flattened float32 array)
        bs = q_matrix.shape[-1]  # block size
        q_flat = q_matrix.flatten().astype(np.float32)
        buf.write(struct.pack("<I", len(q_flat)))  # number of elements
        buf.write(q_flat.tobytes())

        # Codebook size (uint32)
        buf.write(struct.pack("<I", len(codebook)))

        for symbol, code in codebook.items():
            buf.write(struct.pack("<i", int(symbol)))  # int32 symbol
            code_bytes = code.encode("ascii")
            buf.write(struct.pack("<B", len(code_bytes)))
            buf.write(code_bytes)

        # Bit-packed encoded data
        n_bits = len(bitstring)
        buf.write(struct.pack("<I", n_bits))
        padding = (8 - n_bits % 8) % 8
        bitstring += "0" * padding
        byte_arr = bytearray(int(bitstring[i : i + 8], 2) for i in range(0, len(bitstring), 8))
        buf.write(byte_arr)

        return buf.getvalue()


class ResidualDecoder:
    """
    Decode a byte blob produced by ``ResidualEncoder`` back to a float32
    residual patch.

    Parameters
    ----------
    block_size : int
    quality : int
        Must match the encoder settings (only used for backwards compatibility).
    adaptive : bool
        If True, read quantization matrix from encoded data. Default True.
    """

    def __init__(self, block_size: int = 8, quality: int = 80, adaptive: bool = True):
        if not _HAS_TORCH_DCT:
            raise ImportError("torch_dct is required for residual decompression")

        self.block_size = block_size
        self.quality = quality
        self.adaptive = adaptive
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # For backwards compatibility with non-adaptive encoding
        if not adaptive:
            q8 = np.array([
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99],
            ], dtype=np.float32)

            if block_size != 8:
                if block_size % 8 != 0:
                    raise ValueError("block_size must be a multiple of 8")
                q8 = np.repeat(np.repeat(q8, block_size // 8, axis=0), block_size // 8, axis=1)

            scale = (5000 / quality) if quality < 50 else (200 - 2 * quality)
            self._q_matrix = torch.from_numpy((q8 * scale / 100).clip(1)).float().to(self.device)
        else:
            self._q_matrix = None  # Will be read from encoded data

    def decode(self, blob: bytes) -> np.ndarray:
        """
        Decode a byte blob back to a 2-D float32 residual.

        Returns
        -------
        residual : np.ndarray  (H, W)
        """
        buf = io.BytesIO(blob)

        H, W = struct.unpack("<HH", buf.read(4))
        bs = self.block_size
        nb_h, nb_w = H // bs, W // bs
        total_elements = nb_h * nb_w * bs * bs

        # Read quantization matrix if adaptive
        if self.adaptive:
            (n_q_elements,) = struct.unpack("<I", buf.read(4))
            q_bytes = buf.read(n_q_elements * 4)  # float32 = 4 bytes each
            q_flat = np.frombuffer(q_bytes, dtype=np.float32)
            q_matrix = torch.from_numpy(q_flat.reshape(1, 1, bs, bs)).float().to(self.device)
        else:
            q_matrix = self._q_matrix

        # Codebook
        (n_entries,) = struct.unpack("<I", buf.read(4))
        codebook: Dict[str, int] = {}
        for _ in range(n_entries):
            (symbol,) = struct.unpack("<i", buf.read(4))
            (code_len,) = struct.unpack("<B", buf.read(1))
            code = buf.read(code_len).decode("ascii")
            codebook[code] = symbol

        # Bits
        (n_bits,) = struct.unpack("<I", buf.read(4))
        raw = buf.read()
        bitstring = "".join(f"{b:08b}" for b in raw)[:n_bits]

        # Huffman decode
        decoded = []
        cur = ""
        for bit in bitstring:
            cur += bit
            if cur in codebook:
                decoded.append(codebook[cur])
                cur = ""
                if len(decoded) == total_elements:
                    break

        quant = (
            torch.tensor(decoded, dtype=torch.float32)
            .reshape(-1, 1, bs, bs)
            .to(self.device)
        )

        dequant = quant * q_matrix
        blocks = idct_2d(dequant)

        # Reassemble
        residual = (
            blocks
            .reshape(nb_h, nb_w, 1, bs, bs)
            .permute(0, 2, 3, 1, 4)          # (nb_h, 1, bs, nb_w, bs)
            .reshape(H, W)
        )
        return residual.cpu().numpy()
