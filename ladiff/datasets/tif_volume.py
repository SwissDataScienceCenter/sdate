"""TifVolumeSliceDataset — 2D slice dataset from multi-slice TIFF volumes.

Each TIFF file is a stack of 2D slices with shape (num_slices, H, W).
The dataset concatenates slices across all specified files and returns
individual float32 tensors, optionally resized and normalised.

Usage
-----
Single file::

    ds = TifVolumeSliceDataset(
        data_path='/data/212_Wunderkerze2_rotate_04001.tif',
    )

Multiple consecutive files (5 total from the start file)::

    ds = TifVolumeSliceDataset(
        data_path='/data/212_Wunderkerze2_rotate_04001.tif',
        file_range=5,
        resize=256,
        normalize_range=None,   # auto min/max
    )
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset


def _resolve_file_list(start_file: Path, file_range: Optional[int]) -> List[Path]:
    """Return the list of tif files to load.

    Parameters
    ----------
    start_file:
        Path to the first tif file (must match the ``prefix + 5-digit number``
        naming convention).
    file_range:
        Total number of files to include (starting from *start_file*).
        ``None`` means only *start_file*.

    Returns
    -------
    list of Path
        Sorted list of resolved file paths.
    """
    if file_range is None:
        return [start_file]

    name = start_file.name
    match = re.match(r"^(.+?)(\d{5})\.tif$", name, re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"Filename '{name}' does not match the expected pattern "
            "'<prefix><5-digit-number>.tif'"
        )

    prefix = match.group(1)
    start_num = int(match.group(2))
    parent = start_file.parent

    pattern = re.compile(
        r"^" + re.escape(prefix) + r"(\d{5})\.tif$", re.IGNORECASE
    )
    candidates: List[Tuple[int, Path]] = []
    for entry in parent.iterdir():
        m = pattern.match(entry.name)
        if m:
            num = int(m.group(1))
            if num >= start_num:
                candidates.append((num, entry))

    candidates.sort(key=lambda x: x[0])
    selected = [f for _, f in candidates[:file_range]]

    if not selected:
        raise FileNotFoundError(
            f"No tif files found in '{parent}' matching prefix '{prefix}' "
            f"starting from number {start_num:05d}."
        )
    return selected


class TifVolumeSliceDataset(Dataset):
    """PyTorch dataset of 2D slices extracted from time-resolved 4D TIFF volumes.

    Each TIFF file stores one time-step as a 3D volume of shape
    ``(num_slices, H, W)``.  The dataset linearises all slices across the
    requested files so that ``dataset[i]`` returns the *i*-th 2D slice as a
    float32 tensor of shape ``(H, W)`` (or ``(resize, resize)`` when
    *resize* is given).

    Parameters
    ----------
    data_path:
        Path to the **starting** ``.tif`` file.  The filename must end in a
        5-digit integer suffix, e.g. ``prefix_04001.tif``.
    file_range:
        How many consecutive tif files (sorted by their numeric suffix) to
        include, *counting from and including data_path*.  ``None`` or ``1``
        loads just the single file.
    resize:
        If given, resize every slice to a square of this size using bilinear
        interpolation.  ``None`` returns the native resolution.
    normalize_range:
        ``(vmin, vmax)`` pair used for linear normalisation to ``[0, 1]``.
        When ``None``, the global min and max are computed from all slices in
        the selected files and used for normalisation.
    augment:
        When ``True``, apply random data augmentation at each ``__getitem__``
        call: random scaling (0.9–1.2), random rotation (±180°), random
        horizontal and vertical flips, and random translations (±20% of the
        image size).
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        file_range: Optional[int] = None,
        resize: Optional[int] = None,
        normalize_range: Optional[Tuple[float, float]] = None,
        augment: bool = True,
    ) -> None:
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"File not found: {data_path}")

        self.files: List[Path] = _resolve_file_list(data_path, file_range)
        self.resize = resize
        self.augment = augment

        # Count slices per file using tifffile metadata (no full read yet).
        import tifffile  # deferred import so the module is optional at import time

        self._tifffile = tifffile
        self._slices_per_file: List[int] = []
        for f in self.files:
            with tifffile.TiffFile(str(f)) as tif:
                self._slices_per_file.append(len(tif.pages))

        self._cumulative: np.ndarray = np.concatenate(
            [[0], np.cumsum(self._slices_per_file)]
        ).astype(np.int32)

        # Per-file float32 array cache.
        self._cache: Dict[int, np.ndarray] = {}

        # Normalisation.
        if normalize_range is None:
            gmin: float = float("inf")
            gmax: float = float("-inf")
            for i, f in enumerate(self.files):
                arr = self._read_file(i)
                gmin = min(gmin, float(arr.min()))
                gmax = max(gmax, float(arr.max()))
            self.norm_min = gmin
            self.norm_max = gmax
        else:
            self.norm_min = float(normalize_range[0])
            self.norm_max = float(normalize_range[1])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _augment(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply random geometric augmentation to a ``(H, W)`` tensor."""
        h, w = tensor.shape[-2], tensor.shape[-1]

        # Random scale in [0.9, 1.2].
        scale = float(torch.empty(1).uniform_(0.9, 1.2))

        # Random rotation in [-180, 180] degrees.
        angle = float(torch.empty(1).uniform_(-180.0, 180.0))

        # Random translation: ±20% of image dimensions, in pixels.
        translate = [
            float(torch.empty(1).uniform_(-0.0 * w, 0.0 * w)),
            float(torch.empty(1).uniform_(-0.0 * h, 0.0 * h)),
        ]

        # Combined affine transform (scale + rotation + translation).
        # TF.affine requires a leading channel dim: (1, H, W).
        tensor = TF.affine(
            tensor.unsqueeze(0),
            angle=angle,
            translate=translate,
            scale=scale,
            shear=0.0,
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        ).squeeze(0)

        # Random horizontal flip.
        if torch.rand(1).item() > 0.5:
            tensor = TF.hflip(tensor.unsqueeze(0)).squeeze(0)

        # Random vertical flip.
        if torch.rand(1).item() > 0.5:
            tensor = TF.vflip(tensor.unsqueeze(0)).squeeze(0)

        return tensor

    def _read_file(self, file_idx: int) -> np.ndarray:
        """Load (and cache) file *file_idx* as a float32 numpy array."""
        if file_idx not in self._cache:
            arr = self._tifffile.imread(str(self.files[file_idx]))
            self._cache[file_idx] = arr.astype(np.float32)
        return self._cache[file_idx]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        # Locate file and intra-file slice index.
        file_idx = int(np.searchsorted(self._cumulative[1:], idx, side="right"))
        slice_idx = idx - int(self._cumulative[file_idx])

        arr = self._read_file(file_idx)
        img = arr[slice_idx]  # (H, W), float32

        # Normalise to [0, 1].
        denom = self.norm_max - self.norm_min
        if denom > 0.0:
            img = (img - self.norm_min) / denom
        else:
            img = img - self.norm_min

        tensor = torch.from_numpy(img)  # (H, W)

        if self.resize is not None:
            tensor = F.interpolate(
                tensor.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
                size=(self.resize, self.resize),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)  # (resize, resize)

        if self.augment:
            tensor = self._augment(tensor)

        return tensor

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def num_files(self) -> int:
        """Number of tif files loaded."""
        return len(self.files)

    @property
    def slices_per_file(self) -> List[int]:
        """Number of 2D slices contributed by each file."""
        return list(self._slices_per_file)

    @property
    def image_size(self) -> Tuple[int, int]:
        """Returned image size ``(H, W)``."""
        if self.resize is not None:
            return (self.resize, self.resize)
        arr = self._read_file(0)
        return (arr.shape[1], arr.shape[2])

    def get_volume(self, file_idx: int = 0) -> torch.Tensor:
        """Return all slices from a single file as a ``(N, H, W)`` tensor.

        Slices are normalised using the same ``norm_min``/``norm_max`` as
        ``__getitem__``, and resized if *resize* was set.

        Parameters
        ----------
        file_idx:
            Index into ``self.files`` (0-based).
        """
        n = self._slices_per_file[file_idx]
        start = int(self._cumulative[file_idx])
        return torch.stack([self[start + i] for i in range(n)])

    def __repr__(self) -> str:
        size_str = (
            f"{self.resize}×{self.resize}" if self.resize else f"native {self.image_size}"
        )
        return (
            f"TifVolumeSliceDataset("
            f"files={self.num_files}, "
            f"total_slices={len(self)}, "
            f"slices_per_file={self.slices_per_file}, "
            f"image_size={size_str}, "
            f"norm=[{self.norm_min:.4g}, {self.norm_max:.4g}])"
        )
