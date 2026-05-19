"""ladiff.datasets.em_io — helpers for loading, visualising, and saving
electron-microscopy volumes (.mrc / .rec).

Axis convention
---------------
Raw EM tomograms (e.g. produced by IMOD/AreTomo) are stored as (Z, Y, X) arrays
where the **sample tilts around the Y axis**.  This makes the missing wedge
visible in the XZ Fourier-space slice (⊥Y, axis=1).

The rest of the ladiff pipeline (Fourier-cone removal, training datasets) uses
the convention ``TILT_AXIS=0``, where the tilt axis is **Z (axis 0)** and the
missing wedge is visible in the XY Fourier-space slice (⊥Z, axis=0).

``permute_em_to_tilt_axis0`` converts between the two conventions by swapping
the Y and Z axes:

    (Z, Y, X)  →  np.transpose(vol, (1, 0, 2))  →  (Y, Z, X)

After this permutation the missing wedge is visible in the XY slice of the new
volume, matching TILT_AXIS=0.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_mrc_volume(path: Path | str) -> np.ndarray:
    """Load a ``.mrc`` or ``.rec`` file and return a float32 array ``(Z, Y, X)``.

    Uses ``mrcfile`` with ``permissive=True`` so files with minor header
    inconsistencies (common in cryo-ET workflows) are accepted.
    """
    import mrcfile  # optional dependency

    with mrcfile.open(str(path), mode='r', permissive=True) as mrc:
        vol = mrc.data.copy().astype(np.float32)
    return vol


def collect_em_files(
    root: Path | str,
    max_per_folder: int = 4,
) -> Dict[Path, List[Path]]:
    """Walk *root* and collect ``.mrc`` / ``.rec`` files grouped by subfolder.

    Returns a ``dict`` mapping each subfolder ``Path`` to a sorted list of up
    to *max_per_folder* volume paths.  Subfolders are returned in sorted order.
    """
    root = Path(root)
    groups: Dict[Path, List[Path]] = defaultdict(list)
    for f in sorted(root.rglob('*')):
        if f.is_file() and f.suffix.lower() in ('.mrc', '.rec'):
            groups[f.parent].append(f)
    return {
        folder: files[:max_per_folder]
        for folder, files in sorted(groups.items())
    }


# ---------------------------------------------------------------------------
# Axis permutation
# ---------------------------------------------------------------------------

def permute_em_to_tilt_axis0(vol: np.ndarray) -> np.ndarray:
    """Permute a raw EM volume so the missing-wedge convention matches TILT_AXIS=0.

    Raw EM volumes are ``(Z, Y, X)`` with the tilt axis along **Y** (axis 1),
    meaning the missing wedge is visible in the kX-kZ Fourier slice (⊥Y).

    After ``np.transpose(vol, (1, 0, 2))`` the shape becomes ``(Y, Z, X)``
    (renaming the new leading axis as the new Z).  The missing wedge is now in
    the kX-kY plane (⊥ new Z = axis 0), which matches the ``TILT_AXIS=0``
    convention used throughout ladiff.

    Parameters
    ----------
    vol:
        Float32 array of shape ``(Z, Y, X)``.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(old_Y, old_Z, old_X)`` with the same data
        but re-ordered axes.
    """
    return np.transpose(vol, (1, 0, 2))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def center_crop_cubic(vol: np.ndarray) -> np.ndarray:
    """Center-crop *vol* to a cube whose side equals ``min(vol.shape)``.

    Parameters
    ----------
    vol:
        Array of shape ``(D0, D1, D2)``.

    Returns
    -------
    np.ndarray
        Cubic sub-volume of shape ``(s, s, s)`` where ``s = min(vol.shape)``.
    """
    s = min(vol.shape)
    crops = tuple(
        slice((dim - s) // 2, (dim - s) // 2 + s)
        for dim in vol.shape
    )
    return vol[crops]


def rescale_to_min_size(vol: np.ndarray, min_size: int = 256) -> np.ndarray:
    """Rescale *vol* to ``(min_size,) * 3`` if any dimension is smaller than *min_size*.

    Uses trilinear (order=1) interpolation via ``scipy.ndimage.zoom``.
    No-op when all dimensions are already >= *min_size*.

    Parameters
    ----------
    vol:
        Float32 array of shape ``(D0, D1, D2)``.
    min_size:
        Minimum voxel count per dimension.  Default 256.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(min_size, min_size, min_size)``
        when rescaling is triggered, otherwise *vol* unchanged.
    """
    if all(d >= min_size for d in vol.shape):
        return vol
    from scipy.ndimage import zoom as _zoom
    zoom_factors = tuple(min_size / d for d in vol.shape)
    return _zoom(vol.astype(np.float32), zoom_factors, order=1)


def tile_into_cubes(
    vol: np.ndarray,
    tile_size: int,
) -> List[Tuple[int, int, int, np.ndarray]]:
    """Partition *vol* into disjoint ``tile_size³`` cubes.

    Voxels that do not fit into a complete tile along any axis are discarded
    (trailing partial tiles are dropped).

    Parameters
    ----------
    vol:
        Float32 array of shape ``(D0, D1, D2)``.
    tile_size:
        Side length (in voxels) of each cubic tile.

    Returns
    -------
    list of ``(i, j, k, tile)``
        ``i``, ``j``, ``k`` are 0-based tile indices along the three axes.
        ``tile`` is a float32 array of shape
        ``(tile_size, tile_size, tile_size)``.
        Returns an empty list if any dimension is smaller than ``tile_size``.
    """
    s = tile_size
    n0, n1, n2 = vol.shape[0] // s, vol.shape[1] // s, vol.shape[2] // s
    tiles: List[Tuple[int, int, int, np.ndarray]] = []
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                tile = vol[i * s:(i + 1) * s,
                           j * s:(j + 1) * s,
                           k * s:(k + 1) * s]
                tiles.append((i, j, k, tile))
    return tiles


# ---------------------------------------------------------------------------
# Normalisation & saving
# ---------------------------------------------------------------------------

def normalize_percentile(
    vol: np.ndarray,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> Tuple[np.ndarray, float, float]:
    """Clip and rescale *vol* to ``[0, 1]`` using percentile-based bounds.

    Parameters
    ----------
    vol:
        Input array (any shape, float32 recommended).
    p_low, p_high:
        Lower and upper percentiles used as the mapping range.

    Returns
    -------
    norm_vol : np.ndarray
        Float32 array clipped and linearly rescaled to ``[0, 1]``.
    norm_min : float
        The *p_low*-th percentile value (maps to 0).
    norm_max : float
        The *p_high*-th percentile value (maps to 1).
    """
    norm_min = float(np.percentile(vol, p_low))
    norm_max = float(np.percentile(vol, p_high))
    denom = norm_max - norm_min if norm_max != norm_min else 1.0
    norm_vol = np.clip((vol.astype(np.float32) - norm_min) / denom, 0.0, 1.0)
    return norm_vol, norm_min, norm_max


def save_em_volume_npy(
    vol: np.ndarray,
    out_path: Path | str,
    p_low: float = 2.0,
    p_high: float = 98.0,
    permute: bool = True,
    make_cubic: bool = False,
    min_size: int = 256,
) -> List[Tuple[Path, Path]]:
    """Normalise and save an EM volume to ``.npy`` with ``_norm.json`` sidecars.

    Steps
    -----
    1. Optionally permute axes with :func:`permute_em_to_tilt_axis0` so that the
       missing wedge aligns with ``TILT_AXIS=0``.
    2a. **Tiling mode** (``make_cubic=True``): normalise the full volume using
        ``p_low``/``p_high`` percentiles, then partition it into disjoint
        ``min_size³`` cubes with :func:`tile_into_cubes`.  Each tile is saved as
        ``<stem>_tile_<i>_<j>_<k>.npy`` with a companion ``_norm.json``.
        Trailing voxels that do not fill a complete tile are discarded.
    2b. **Single-volume mode** (``make_cubic=False``): if any dimension is smaller
        than *min_size*, rescale to ``(min_size, min_size, min_size)`` via
        trilinear interpolation (:func:`rescale_to_min_size`), then normalise
        and save to *out_path*.

    Parameters
    ----------
    vol:
        Float32 EM volume ``(Z, Y, X)``.
    out_path:
        Destination ``.npy`` path (used as the base name for tile files).
    p_low, p_high:
        Percentiles for normalisation (default 2 and 98).
    permute:
        If ``True`` (default), apply :func:`permute_em_to_tilt_axis0` before
        processing.
    make_cubic:
        If ``True``, tile the volume into disjoint ``min_size³`` cubes and save
        each tile separately.  If ``False``, save a single volume (rescaling to
        ``min_size³`` if necessary).
    min_size:
        In tiling mode (``make_cubic=True``): target side length for upscaling
        each tile.  Tiles are cut at ``min(vol.shape)`` voxels and then rescaled
        to ``(min_size, min_size, min_size)`` when the natural tile size is
        smaller than *min_size*.  In single-volume mode (``make_cubic=False``):
        minimum voxels per dimension; the whole volume is rescaled if any
        dimension falls below this threshold.  Default 256.  Set to 0 to
        disable rescaling in both modes.

    Returns
    -------
    list of ``(npy_path, norm_path)`` pairs
        One entry per saved tile (tiling mode) or a single-element list
        (single-volume mode).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if permute:
        vol = permute_em_to_tilt_axis0(vol)

    if make_cubic:
        # Normalise the full volume so all tiles share the same intensity scale.
        norm_vol, _, _ = normalize_percentile(vol, p_low, p_high)

        # Tile using the natural cube size (= minimum volume dimension).
        tile_size = min(norm_vol.shape)
        tiles = tile_into_cubes(norm_vol, tile_size=tile_size)
        if not tiles:
            raise ValueError(
                f"Volume shape {vol.shape} is too small to produce any "
                f"{tile_size}³ tiles."
            )

        sidecar = {'norm_min': 0.0, 'norm_max': 1.0}
        results: List[Tuple[Path, Path]] = []
        for i, j, k, tile in tiles:
            if min_size > 0:
                tile = rescale_to_min_size(tile, min_size=min_size)
            tile_path = out_path.parent / f"{out_path.stem}_tile_{i}_{j}_{k}.npy"
            norm_path = out_path.parent / f"{out_path.stem}_tile_{i}_{j}_{k}_norm.json"
            np.save(str(tile_path), tile)
            with open(norm_path, 'w') as f:
                json.dump(sidecar, f, indent=2)
            results.append((tile_path, norm_path))
        return results

    # ── single-volume mode ────────────────────────────────────────────────────
    if min_size > 0:
        vol = rescale_to_min_size(vol, min_size=min_size)

    norm_vol, _, _ = normalize_percentile(vol, p_low, p_high)
    np.save(str(out_path), norm_vol)

    norm_path = out_path.parent / (out_path.stem + '_norm.json')
    with open(norm_path, 'w') as f:
        json.dump({'norm_min': 0.0, 'norm_max': 1.0}, f, indent=2)

    return [(out_path, norm_path)]
