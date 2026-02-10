"""
Block-level predictor that uses a trained UNet2DModel to predict the next
projection frame from k previous frames, operating on 256×256 patches.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path


class BlockPredictor:
    """
    Splits frames into overlapping 256×256 blocks, feeds each block through
    a trained UNet model (k input channels → 1 output channel), and
    reassembles the output into a full-resolution predicted frame.

    Overlap is handled with linear blending so that seams are invisible.

    Parameters
    ----------
    model : torch.nn.Module
        A UNet2DModel (or compatible) that accepts (B, k, 256, 256) and
        returns (B, 1, 256, 256).
    block_size : int
        Spatial size of each patch (default 256).
    num_input_projections : int
        Number of previous frames used as input channels (k).
    device : torch.device or str
        Device for inference.
    overlap : int
        Number of pixels of overlap between adjacent patches.
        Must be even. Default 32.
    use_conditioning : bool
        If True, pass the normalised center-x coordinate of each patch
        as the ``timestep`` argument to the UNet (matching the
        ``Noise2NoiseWithConditioningLoss`` training mode).  When False
        (default) zeros are passed, matching the base ``Noise2NoiseLoss``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        block_size: int = 256,
        num_input_projections: int = 3,
        device: Optional[torch.device] = None,
        overlap: int = 32,
        use_conditioning: bool = False,
    ):
        self.model = model
        self.block_size = block_size
        self.num_input_projections = num_input_projections
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.overlap = overlap
        self.use_conditioning = use_conditioning

        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_frame(
        self,
        previous_frames: List[np.ndarray],
        batch_size: int = 16,
        center_x: Optional[float] = None,
    ) -> np.ndarray:
        """
        Predict the next frame given *k* previous frames.

        Parameters
        ----------
        previous_frames : list of np.ndarray
            List of k 2-D arrays (H, W), each in [0, 1].
        batch_size : int
            Maximum number of patches to process at once.
        center_x : float or None
            Normalised x-coordinate of the original image centre inside
            the (possibly cropped / padded) frame, in [0, 1].  Required
            when ``use_conditioning=True``; ignored otherwise.

        Returns
        -------
        predicted : np.ndarray
            2-D array of shape (H, W) with predicted pixel values.
        """
        k = self.num_input_projections
        assert len(previous_frames) == k, (
            f"Expected {k} previous frames, got {len(previous_frames)}"
        )
        if self.use_conditioning and center_x is None:
            raise ValueError(
                "center_x must be provided when use_conditioning=True"
            )

        H, W = previous_frames[0].shape
        bs = self.block_size
        step = bs - self.overlap  # stride between patch origins

        # Pad so that all blocks fit exactly
        pad_h = (step - (H % step)) % step + self.overlap
        pad_w = (step - (W % step)) % step + self.overlap
        padded_H = H + pad_h
        padded_W = W + pad_w

        # Pad each frame
        padded_frames = []
        for frame in previous_frames:
            padded = np.pad(frame, ((0, pad_h), (0, pad_w)), mode="reflect")
            padded_frames.append(padded)

        # Collect patch coordinates
        positions: List[Tuple[int, int]] = []
        for y in range(0, padded_H - bs + 1, step):
            for x in range(0, padded_W - bs + 1, step):
                positions.append((y, x))

        # Pre-compute per-patch conditioning values.
        # During training the model receives center_coords[:, 0] — the
        # normalised x-position of the *original* image centre within
        # the crop.  For each patch we compute the analogous quantity:
        # where the original-image centre falls relative to the patch.
        if self.use_conditioning:
            # center_x is in [0,1] relative to the (unpadded) frame of
            # width W.  Convert to pixel coordinate in the *padded* frame.
            cx_pixel = center_x * W  # pixel position in unpadded frame
            patch_cond = np.zeros(len(positions), dtype=np.float32)
            for idx, (_y, x) in enumerate(positions):
                # Normalised position of the original centre within this
                # patch (same convention as the dataset).
                patch_cond[idx] = (cx_pixel - x) / bs
        else:
            patch_cond = None

        # Extract patches — shape (N, k, bs, bs)
        patches = np.zeros((len(positions), k, bs, bs), dtype=np.float32)
        for idx, (y, x) in enumerate(positions):
            for ch, pf in enumerate(padded_frames):
                patches[idx, ch] = pf[y : y + bs, x : x + bs]

        # Run model in mini-batches
        pred_patches = np.zeros((len(positions), 1, bs, bs), dtype=np.float32)
        for start in range(0, len(positions), batch_size):
            end = min(start + batch_size, len(positions))
            inp = torch.from_numpy(patches[start:end]).to(self.device)

            if self.use_conditioning:
                # Pass the per-patch centre-x as the "timestep" argument
                # (float tensor, same as Noise2NoiseWithConditioningLoss).
                cond = torch.from_numpy(patch_cond[start:end]).to(self.device)
            else:
                cond = torch.zeros(inp.shape[0], dtype=torch.long, device=self.device)

            out = self.model(inp, cond, return_dict=False)[0]
            pred_patches[start:end] = out.cpu().numpy()

        # Reassemble with linear blending
        predicted = self._reassemble(pred_patches, positions, padded_H, padded_W)

        # Remove padding
        predicted = predicted[:H, :W]
        return predicted

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_blend_weights(self, bs: int) -> np.ndarray:
        """Create a 2-D raised-cosine blending window of size (bs, bs)."""
        ramp = np.ones(bs, dtype=np.float32)
        ov = self.overlap
        if ov > 0:
            # cosine ramp for the overlap region
            t = np.linspace(0, np.pi / 2, ov, dtype=np.float32)
            ramp[:ov] = np.sin(t) ** 2
            ramp[-ov:] = np.cos(t) ** 2
        return ramp[:, None] * ramp[None, :]

    def _reassemble(
        self,
        pred_patches: np.ndarray,
        positions: List[Tuple[int, int]],
        H: int,
        W: int,
    ) -> np.ndarray:
        """Stitch patches back into a full frame using blending weights."""
        bs = self.block_size
        weight_map = np.zeros((H, W), dtype=np.float32)
        accum = np.zeros((H, W), dtype=np.float32)
        blend = self._make_blend_weights(bs)

        for idx, (y, x) in enumerate(positions):
            patch = pred_patches[idx, 0]
            accum[y : y + bs, x : x + bs] += patch * blend
            weight_map[y : y + bs, x : x + bs] += blend

        # Avoid division by zero
        weight_map = np.maximum(weight_map, 1e-8)
        return accum / weight_map

    # ------------------------------------------------------------------
    # Model loading helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        block_size: int = 256,
        num_input_projections: int = 3,
        device: Optional[torch.device] = None,
        overlap: int = 32,
        model_kwargs: Optional[dict] = None,
        use_conditioning: bool = False,
    ) -> "BlockPredictor":
        """
        Convenience constructor that loads a checkpoint and builds the
        predictor in one call.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to a ``.pt`` checkpoint saved during training.
        block_size : int
            Patch size to use for prediction.
        num_input_projections : int
            k — number of previous frames.
        device : torch.device, optional
        overlap : int
        model_kwargs : dict, optional
            Extra kwargs forwarded to ``UNet2DModel``.
        use_conditioning : bool
            Pass centre-x conditioning to the model.

        Returns
        -------
        BlockPredictor
        """
        from diffusers import UNet2DModel

        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        default_kwargs = dict(
            sample_size=block_size,
            in_channels=num_input_projections,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(64, 64, 128, 128, 256, 256),
            down_block_types=(
                "DownBlock2D", "DownBlock2D", "DownBlock2D",
                "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
                "UpBlock2D", "UpBlock2D", "UpBlock2D",
            ),
        )
        if model_kwargs:
            default_kwargs.update(model_kwargs)

        model = UNet2DModel(**default_kwargs)

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(dev)
        model.eval()

        return cls(
            model=model,
            block_size=block_size,
            num_input_projections=num_input_projections,
            device=dev,
            overlap=overlap,
            use_conditioning=use_conditioning,
        )
