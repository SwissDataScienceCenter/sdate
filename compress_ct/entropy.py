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

    Pipeline: DCT → Quantize → Zig-zag scan → Huffman.

    Parameters
    ----------
    block_size : int
        DCT block size (must be a multiple of 8). Default 8.
    quality : int
        Quality parameter (1–100). Higher = less quantization. Default 80.
    """

    def __init__(self, block_size: int = 8, quality: int = 80):
        if not _HAS_TORCH_DCT:
            raise ImportError("torch_dct is required for residual compression")
        if not _HAS_HUFFMAN:
            raise ImportError("huffman package is required for entropy coding")

        self.block_size = block_size
        self.quality = quality
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Standard JPEG luminance quantization matrix (8×8), scaled to block_size
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
        quant = torch.round(dct_c / self._q_matrix).to(torch.int16)
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

        return self._pack(H, W, codebook, bitstring)

    # ------------------------------------------------------------------ #
    #  Binary packing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pack(H: int, W: int, codebook: dict, bitstring: str) -> bytes:
        """Serialize Huffman-coded residual to bytes."""
        buf = io.BytesIO()

        # Header: H, W (uint16)
        buf.write(struct.pack("<HH", H, W))

        # Codebook size (uint16)
        buf.write(struct.pack("<H", len(codebook)))

        for symbol, code in codebook.items():
            buf.write(struct.pack("<h", int(symbol)))  # int16 symbol
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
        Must match the encoder settings.
    """

    def __init__(self, block_size: int = 8, quality: int = 80):
        if not _HAS_TORCH_DCT:
            raise ImportError("torch_dct is required for residual decompression")

        self.block_size = block_size
        self.quality = quality
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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

        # Codebook
        (n_entries,) = struct.unpack("<H", buf.read(2))
        codebook: Dict[str, int] = {}
        for _ in range(n_entries):
            (symbol,) = struct.unpack("<h", buf.read(2))
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

        dequant = quant * self._q_matrix
        blocks = idct_2d(dequant)

        # Reassemble
        residual = (
            blocks
            .reshape(nb_h, nb_w, 1, bs, bs)
            .permute(0, 2, 3, 1, 4)          # (nb_h, 1, bs, nb_w, bs)
            .reshape(H, W)
        )
        return residual.cpu().numpy()
