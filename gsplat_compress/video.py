"""
High-level video compression pipeline using Gaussian-splatting delta encoding.

Orchestrates the full encode/decode workflow:
1. Fit keyframe (frame 0) from scratch
2. For each subsequent frame:
   a. Fine-tune from the *reconstructed* previous frame
   b. Compute deltas → K-means → codebook fine-tune
   c. Reconstruct from base + quantised delta  (this becomes the starting
      point for the next frame)
3. Track quality (PSNR, SSIM) and compression ratios throughout
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from gsplat_compress.camera import ortho_camera
from gsplat_compress.compression import (
    CodebookConfig,
    CompressedFrame,
    encode_delta_frame,
    reconstruct_compressed_frame,
)
from gsplat_compress.initialize import GaussianParams, init_gaussians
from gsplat_compress.metrics import (
    compression_ratio,
    delta_frame_bytes,
    keyframe_bytes,
    psnr,
    raw_frame_bytes,
    sequence_storage_summary,
    ssim,
)
from gsplat_compress.renderer import render
from gsplat_compress.storage import save_sequence
from gsplat_compress.training import (
    FinetuneConfig,
    TrainConfig,
    TrainResult,
    finetune_frame,
    train_keyframe,
)


# ═══════════════════════════════════════════════════════════════════════════
# Per-frame result
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FrameResult:
    """Quality and storage metrics for a single encoded frame."""

    frame_idx: int
    frame_type: str            # "keyframe" or "delta"
    psnr: float
    ssim: float
    compressed_bytes: int
    raw_bytes: int
    compression_ratio: float
    elapsed: float
    rendered_np: np.ndarray | None = None  # optional: keep for visualisation


# ═══════════════════════════════════════════════════════════════════════════
# Video compression pipeline
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VideoCompressor:
    """Stateful video compressor using Gaussian-splatting delta encoding.

    Usage
    -----
    >>> vc = VideoCompressor(device=torch.device("cuda"))
    >>> results = vc.compress_sequence(frames, ...)
    """

    device: torch.device
    use_fp16: bool = True

    # Set after encode_keyframe
    keyframe_params: GaussianParams | None = None
    current_params: GaussianParams | None = None  # reconstructed params of latest frame
    compressed_frames: list[CompressedFrame] = field(default_factory=list)
    frame_results: list[FrameResult] = field(default_factory=list)

    # Camera (set once)
    K: torch.Tensor | None = None
    viewmat: torch.Tensor | None = None

    def _setup_camera(self):
        if self.K is None:
            self.K, self.viewmat = ortho_camera(self.device)

    def _to_gt(self, image_gray: torch.Tensor) -> torch.Tensor:
        """Convert [H, W] grayscale to [H, W, 3] for gsplat."""
        return image_gray.unsqueeze(-1).repeat(1, 1, 3).to(self.device)

    def _render_np(self, params: GaussianParams, W: int, H: int) -> np.ndarray:
        """Render and return grayscale [H, W] numpy array."""
        with torch.no_grad():
            rendered, _, _ = render(
                params.means, params.log_scales, params.quats,
                params.rgbs, params.opacities,
                self.viewmat, self.K, W, H,
            )
        return np.clip(rendered[:, :, 0].cpu().numpy(), 0, 1)

    # ── Keyframe ──────────────────────────────────────────────────────

    def encode_keyframe(
        self,
        image: torch.Tensor,
        n_points: int = 50_000,
        init_mode: str = "intensity",
        train_cfg: TrainConfig | None = None,
        verbose: bool = True,
    ) -> FrameResult:
        """Fit the first frame from scratch.

        Parameters
        ----------
        image : Tensor [H, W]
            Grayscale image normalised to [0, 1].
        n_points : int
            Number of initial Gaussians.
        init_mode : str
            ``'uniform'`` or ``'intensity'``.
        train_cfg : TrainConfig
            Training hyperparameters.
        verbose : bool

        Returns
        -------
        FrameResult
        """
        self._setup_camera()
        H, W = image.shape
        gt = self._to_gt(image)

        if train_cfg is None:
            train_cfg = TrainConfig()

        # Initialise and train
        params = init_gaussians(
            image, n_points, self.device, mode=init_mode,
        )
        result = train_keyframe(
            params, gt, self.viewmat, self.K, W, H, train_cfg, verbose,
        )

        self.keyframe_params = result.params.detach_clone()
        self.current_params = result.params.detach_clone()

        # Metrics
        rendered = self._render_np(result.params, W, H)
        target_np = image.cpu().numpy()
        kf_bytes = keyframe_bytes(result.params.n_gaussians, self.use_fp16)
        raw_bytes = raw_frame_bytes(H, W)
        cr = raw_bytes / kf_bytes if kf_bytes > 0 else float("inf")

        fr = FrameResult(
            frame_idx=0,
            frame_type="keyframe",
            psnr=psnr(rendered, target_np),
            ssim=ssim(rendered, target_np),
            compressed_bytes=kf_bytes,
            raw_bytes=raw_bytes,
            compression_ratio=cr,
            elapsed=result.elapsed,
            rendered_np=rendered,
        )
        self.frame_results.append(fr)
        return fr

    # ── Delta frame ───────────────────────────────────────────────────

    def encode_delta_frame(
        self,
        image: torch.Tensor,
        ft_cfg: FinetuneConfig | None = None,
        cb_cfg: CodebookConfig | None = None,
        verbose: bool = True,
    ) -> FrameResult:
        """Encode a frame as delta from the current reconstructed parameters.

        Important: Fine-tuning starts from ``self.current_params``, which is
        the *reconstructed* (quantised) version of the previous frame — not
        the unquantised fine-tune.  This ensures that the decoder can
        reproduce the chain exactly.

        Parameters
        ----------
        image : Tensor [H, W]
            Grayscale target normalised to [0, 1].
        ft_cfg : FinetuneConfig
            Fine-tuning config.
        cb_cfg : CodebookConfig
            Codebook compression config.
        verbose : bool

        Returns
        -------
        FrameResult
        """
        assert self.current_params is not None, "Encode a keyframe first."
        self._setup_camera()
        H, W = image.shape
        gt = self._to_gt(image)

        if ft_cfg is None:
            ft_cfg = FinetuneConfig()
        if cb_cfg is None:
            cb_cfg = CodebookConfig()

        frame_idx = len(self.frame_results)
        t0 = time.time()

        # 1. Fine-tune from current (reconstructed) params
        if verbose:
            print(f"\n{'='*60}")
            print(f"Frame {frame_idx}: fine-tuning from reconstructed previous frame")
        ft_result = finetune_frame(
            self.current_params, gt, self.viewmat, self.K, W, H,
            ft_cfg, verbose,
        )

        # 2. Delta-encode: base = current_params, target = fine-tuned
        if verbose:
            print(f"Frame {frame_idx}: delta compression (K={cb_cfg.n_clusters})")
        compressed, reconstructed = encode_delta_frame(
            self.current_params, ft_result.params,
            gt, self.viewmat, self.K, W, H,
            cb_cfg, verbose,
        )

        # 3. Update state: the reconstructed params become the new base
        self.current_params = reconstructed
        self.compressed_frames.append(compressed)

        # 4. Metrics
        rendered = self._render_np(reconstructed, W, H)
        target_np = image.cpu().numpy()
        d_bytes = compressed.storage_bytes(self.use_fp16)["total_bytes"]
        raw = raw_frame_bytes(H, W)
        cr = raw / d_bytes if d_bytes > 0 else float("inf")
        elapsed = time.time() - t0

        fr = FrameResult(
            frame_idx=frame_idx,
            frame_type="delta",
            psnr=psnr(rendered, target_np),
            ssim=ssim(rendered, target_np),
            compressed_bytes=d_bytes,
            raw_bytes=raw,
            compression_ratio=cr,
            elapsed=elapsed,
            rendered_np=rendered,
        )
        self.frame_results.append(fr)
        return fr

    # ── Convenience: encode a full sequence ───────────────────────────

    def compress_sequence(
        self,
        frames: list[torch.Tensor],
        n_points: int = 50_000,
        init_mode: str = "intensity",
        train_cfg: TrainConfig | None = None,
        ft_cfg: FinetuneConfig | None = None,
        cb_cfg: CodebookConfig | None = None,
        verbose: bool = True,
    ) -> list[FrameResult]:
        """Compress a list of frames (first = keyframe, rest = delta).

        Parameters
        ----------
        frames : list[Tensor [H, W]]
            Sequence of grayscale images normalised to [0, 1].

        Returns
        -------
        list[FrameResult] — one per frame.
        """
        assert len(frames) >= 1, "Need at least one frame."

        # Keyframe
        if verbose:
            print(f"Compressing sequence: {len(frames)} frames")
            print(f"{'='*60}")
        self.encode_keyframe(
            frames[0], n_points=n_points, init_mode=init_mode,
            train_cfg=train_cfg, verbose=verbose,
        )

        # Delta frames
        for i in range(1, len(frames)):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Encoding frame {i}/{len(frames)-1} as delta")
            self.encode_delta_frame(
                frames[i], ft_cfg=ft_cfg, cb_cfg=cb_cfg, verbose=verbose,
            )

        return self.frame_results

    # ── Summary ───────────────────────────────────────────────────────

    def summary(self, H: int, W: int) -> dict:
        """Full compression summary."""
        if self.keyframe_params is None:
            return {}
        return sequence_storage_summary(
            self.keyframe_params.n_gaussians,
            self.compressed_frames,
            H, W,
            self.use_fp16,
        )

    def save(self, path: str | Path, metadata: dict | None = None) -> Path:
        """Save the compressed sequence to disk."""
        assert self.keyframe_params is not None
        return save_sequence(
            path, self.keyframe_params, self.compressed_frames,
            self.use_fp16, metadata,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Decoder
# ═══════════════════════════════════════════════════════════════════════════

def decode_sequence(
    keyframe_params: GaussianParams,
    compressed_frames: list[CompressedFrame],
    viewmat: torch.Tensor,
    K: torch.Tensor,
    W: int,
    H: int,
) -> list[np.ndarray]:
    """Decode a compressed sequence into rendered images.

    Returns a list of numpy arrays [H, W] in [0, 1], one per frame
    (including the keyframe).
    """
    renders = []

    # Keyframe
    with torch.no_grad():
        r, _, _ = render(
            keyframe_params.means, keyframe_params.log_scales,
            keyframe_params.quats, keyframe_params.rgbs,
            keyframe_params.opacities,
            viewmat, K, W, H,
        )
    renders.append(np.clip(r[:, :, 0].cpu().numpy(), 0, 1))

    # Delta frames
    current = keyframe_params
    for cf in compressed_frames:
        current = reconstruct_compressed_frame(current, cf)
        with torch.no_grad():
            r, _, _ = render(
                current.means, current.log_scales, current.quats,
                current.rgbs, current.opacities,
                viewmat, K, W, H,
            )
        renders.append(np.clip(r[:, :, 0].cpu().numpy(), 0, 1))

    return renders
