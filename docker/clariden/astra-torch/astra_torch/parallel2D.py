"""Astra 2D parallel beam utilities.

This module provides GPU-friendly PyTorch functions for 2D parallel beam reconstruction
using ASTRA toolbox.

Public functions
----------------
fbp_reconstruction_masked(...):
    Perform FBP (Filtered Back Projection) reconstruction on (possibly masked) subset of views for 2D parallel beam.

sirt_reconstruction_masked(...):
    Perform SIRT (Simultaneous Iterative Reconstruction Technique) on (possibly masked) subset of views for 2D parallel beam.

gd_reconstruction_masked(...):
    Gradient descent reconstruction using differentiable 2D parallel beam projector.

build_parallel2d_projector(...):
    Factory that returns a torch.autograd.Function for 2D parallel beam projection.

Notes
-----
* Requires the ASTRA toolbox compiled with CUDA support.
* All tensors are float32 and (by default) CUDA if available.
* Projection tensor layout is (V, C) following the convention (views, detector_cols).
* This is 2D parallel beam geometry where all rays are parallel within each projection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Optional, Dict, Any
import numpy as np
import torch
from tqdm import tqdm

try:
    import astra  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError("astra toolbox is required for astra_torch.parallel2D module") from e

try:
    import cupy as cp  # type: ignore
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


@dataclass
class Parallel2DAcquisition:
    """2D parallel beam acquisition parameters."""
    proj_cols: int
    angles_deg: np.ndarray  # rotation angles in degrees
    vol_shape: Tuple[int, int]  # (height, width)


# ---------------------------------------------------------------------------
# Geometry setup helpers
# ---------------------------------------------------------------------------

def _make_volume_geom(vol_shape: Tuple[int, int], voxel_size_mm: float = 1.0):
    """Create ASTRA 2D volume geometry with proper scaling and centering.
    
    Parameters
    ----------
    vol_shape : (ny, nx)
        Volume shape in PyTorch convention (height, width)
    voxel_size_mm : float
        Voxel size in mm
        
    Returns
    -------
    vol_geom : dict
        ASTRA volume geometry
    """
    ny, nx = vol_shape  # PyTorch convention: (height, width)
    
    # ASTRA expects (nx, ny) convention
    vol_geom = astra.create_vol_geom(ny, nx)
    
    # Set the volume to be centered around the origin
    # ASTRA default is [-nx/2, nx/2] x [-ny/2, ny/2] which is what we want
    vol_geom['option'] = {}
    vol_geom['option']['WindowMinX'] = -nx * voxel_size_mm / 2.0
    vol_geom['option']['WindowMaxX'] = nx * voxel_size_mm / 2.0
    vol_geom['option']['WindowMinY'] = -ny * voxel_size_mm / 2.0
    vol_geom['option']['WindowMaxY'] = ny * voxel_size_mm / 2.0
    
    return vol_geom


def _create_parallel2d_geometry(
    vol_shape: Tuple[int, int],  # (ny, nx)
    det_cols: int,
    angles_deg: np.ndarray,
    voxel_size_mm: float = 1.0,
    det_spacing_mm: float = 1.0,
) -> Tuple[dict, dict]:
    """Create ASTRA geometries for 2D parallel beam.
    
    Parameters
    ----------
    vol_shape : (ny, nx)
        Volume dimensions (height, width)
    det_cols : int
        Number of detector columns
    angles_deg : array
        Rotation angles in degrees (0-360)
    voxel_size_mm : float
        Voxel size in mm
    det_spacing_mm : float
        Detector pixel spacing in mm
        
    Returns
    -------
    vol_geom : dict
        ASTRA volume geometry
    proj_geom : dict
        ASTRA projection geometry
        
    Notes
    -----
    This function implements 2D parallel beam geometry where all rays are parallel.
    """
    vol_geom = _make_volume_geom(vol_shape, voxel_size_mm)
    
    n_angles = len(angles_deg)
    
    # Convert angles to radians
    angles_rad = np.deg2rad(angles_deg)
    
    # Create parallel beam geometry
    proj_geom = astra.create_proj_geom(
        'parallel', 
        det_spacing_mm, 
        det_cols, 
        angles_rad
    )
    
    return vol_geom, proj_geom


# ---------------------------------------------------------------------------
# Filtering utilities for FBP
# ---------------------------------------------------------------------------

def _apply_ramp_filter(
    sino_vc,
    use_gpu: bool = False,
    filter_type: str = 'ram-lak',
    det_spacing_mm: float = 1.0,
):
    """Apply ramp filter to sinogram for FBP reconstruction.
    
    This function applies the ramp filter in the frequency domain to all projections
    simultaneously for maximum GPU efficiency. The filter can be windowed to reduce
    noise and artifacts.
    
    Parameters
    ----------
    sino_vc : array-like (V, C)
        Sinogram in ASTRA format (views, cols)
    use_gpu : bool
        Whether to use GPU (CuPy) or CPU (NumPy)
    filter_type : str, optional
        Filter type: 'ram-lak' (default), 'shepp-logan', 'cosine', 'hamming', 'hann'
        - 'ram-lak': Pure ramp filter (no windowing)
        - 'shepp-logan': Multiplies ramp by sinc function
        - 'cosine': Multiplies ramp by cosine
        - 'hamming': Hamming window on ramp
        - 'hann': Hann window on ramp
        
    Returns
    -------
    filtered_sino : array-like (V, C)
        Filtered sinogram
        
    Notes
    -----
    The ramp filter is the derivative operator in Fourier space, which compensates
    for the 1/r weighting in backprojection. Windowing reduces high-frequency noise
    at the cost of slightly reduced resolution.
    """
    xp = cp if use_gpu else np

    n_views, det_cols = sino_vc.shape

    if det_spacing_mm <= 0:
        raise ValueError("det_spacing_mm must be positive for ramp filtering")

    # Pad to next power-of-two (following TomoPy convention) to avoid wrap-around.
    padded_cols = int(2 ** math.ceil(math.log2(max(64, 2 * det_cols))))
    pad_cols = padded_cols - det_cols

    # Remove mean along detector columns to suppress low-frequency bias
    sino_zero_mean = sino_vc - xp.mean(sino_vc, axis=1, keepdims=True)

    if pad_cols > 0:
        pad_config = ((0, 0), (0, pad_cols))
        sino_padded = xp.pad(sino_zero_mean, pad_config, mode='edge')
    else:
        sino_padded = sino_zero_mean

    # Work in Fourier domain along detector axis using rFFT for efficiency.
    sino_fft = xp.fft.rfft(sino_padded, axis=1)

    freq = xp.fft.rfftfreq(padded_cols, d=det_spacing_mm)
    freq_abs = xp.abs(freq)
    freq_abs[0] = 0.0

    freq_max = float(freq_abs.max()) if freq_abs.size > 0 else 1.0
    if freq_max == 0:
        freq_max = 1.0
    freq_norm = freq_abs / freq_max

    window = xp.ones_like(freq_abs)
    ftype = filter_type.lower()

    if ftype == 'ram-lak':
        pass
    elif ftype == 'shepp-logan':
        window[1:] = xp.sinc(freq_norm[1:])
    elif ftype == 'cosine':
        window = xp.cos((xp.pi / 2.0) * freq_norm)
    elif ftype == 'hamming':
        window = 0.54 + 0.46 * xp.cos(xp.pi * freq_norm)
    elif ftype == 'hann':
        window = 0.5 * (1.0 + xp.cos(xp.pi * freq_norm))
    else:
        raise ValueError(
            f"Unknown filter type: {filter_type}. Choose from: 'ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann'"
        )

    ramp = 2.0 * freq_abs
    filter_kernel = ramp * window

    filter_kernel = filter_kernel.reshape(1, -1)
    filtered_fft = sino_fft * filter_kernel

    filtered_sino = xp.fft.irfft(filtered_fft, n=padded_cols, axis=1)

    if pad_cols > 0:
        filtered_sino = filtered_sino[..., :det_cols]

    # Scale by detector sampling step to approximate continuous integral
    filtered_sino = filtered_sino.astype(xp.float32) * det_spacing_mm

    return filtered_sino


# ---------------------------------------------------------------------------
# 2D Parallel beam reconstruction with optional masking  
# ---------------------------------------------------------------------------

def fbp_reconstruction_masked(
    projs_vc: torch.Tensor,
    angles_deg: np.ndarray,
    mask: Optional[Sequence[Any]] = None,
    voxel_per_mm: int = 1,
    voxel_size_mm: float = -1.0,
    vol_shape: Optional[Tuple[int, int]] = None,
    det_spacing_mm: Optional[float] = 1.0,
    filter_type: str = 'hann',
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Run 2D parallel beam reconstruction (FBP) on a subset of views.

    Parameters
    ----------
    projs_vc : (V, C) tensor 
        2D parallel beam projections (CPU or CUDA)
    angles_deg : (V,) numpy array
        Rotation angles in degrees (0-360)
    mask : optional boolean sequence length V
        If None, all views are used
    voxel_per_mm : int
        Resolution parameter
    voxel_size_mm : float
        Voxel size in mm (if -1, computed from voxel_per_mm)
    vol_shape : optional explicit (ny, nx)
        Volume dimensions
    det_spacing_mm : float
        Detector pixel spacing in mm
    filter_type : str, optional
        Ramp filter type: 'ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann' (default)
        'hann' provides good balance between noise and resolution
    device : torch.device, optional
        Target device

    Returns
    -------
    vol : (ny, nx) float32 tensor on device

    Notes
    -----
    * On CUDA, the reconstruction buffer is backed by a torch tensor. CuPy is only
      used for intermediate arrays and its memory pools are flushed at the end of
      the routine so repeated reconstructions do not accumulate GPU allocations.
    """
    if device is None:
        device = projs_vc.device if isinstance(projs_vc, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if voxel_size_mm <= 0:
        voxel_size_mm = 1.0 / voxel_per_mm

    if mask is not None:
        # Support torch.Tensor or numpy / sequence masks (boolean or index)
        if isinstance(mask, torch.Tensor):
            if mask.dtype == torch.bool:
                if mask.numel() != len(angles_deg):
                    raise ValueError("boolean mask length must match number of views")
                mask_cpu = mask.detach().cpu().numpy()
                sel_angles = angles_deg[mask_cpu]
                sel_projs = projs_vc[mask]
            else:
                idx = mask.detach().cpu().numpy()
                sel_angles = angles_deg[idx]
                sel_projs = projs_vc[idx]
        else:
            mask_arr = np.asarray(mask)
            if mask_arr.dtype == bool:
                if mask_arr.shape[0] != len(angles_deg):
                    raise ValueError("boolean mask length must match number of views")
                sel_angles = angles_deg[mask_arr]
                sel_projs = projs_vc[mask_arr]
            else:
                sel_angles = angles_deg[mask_arr]
                sel_projs = projs_vc[mask_arr]
    else:
        sel_angles = angles_deg
        sel_projs = projs_vc

    # Convert to ASTRA format (V, C)
    # For GPU path, use CuPy to properly interface PyTorch CUDA tensors with ASTRA
    if sel_projs.is_cuda:
        if not CUPY_AVAILABLE:
            raise RuntimeError(
                "CuPy is required for GPU-accelerated reconstruction. "
                "Install it with: pip install cupy-cuda11x (or cupy-cuda12x for CUDA 12)"
            )
        # GPU path using CuPy - keeps data on GPU throughout
        # Get pointer to PyTorch tensor's CUDA memory
        ptr = sel_projs.data_ptr()
        shape = sel_projs.shape  # (V, C)
        
        # Create CuPy array from PyTorch tensor's memory (zero-copy view)
        sel_projs_cp = cp.ndarray(
            shape=shape,
            dtype=cp.float32,
            memptr=cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(ptr, sel_projs.numel() * 4, sel_projs), 0
            )
        )
        
        sino_vc = cp.ascontiguousarray(sel_projs_cp)
        use_gpu = True
    else:
        # CPU path - convert to numpy
        sel_projs_np = sel_projs.detach().cpu().numpy()  # (V, C)
        sino_vc = sel_projs_np.copy()
        use_gpu = False
    
    n_views, det_cols = sino_vc.shape

    if vol_shape is None:
        # Default: square volume with size matching detector
        vol_shape = (det_cols, det_cols)
    
    # Allocate reconstruction buffer
    if use_gpu:
        # Hold reconstruction directly in a torch tensor so we can return it without
        # additional copies. ASTRA will write into this memory via a CuPy view.
        vol_t = torch.zeros(vol_shape, dtype=torch.float32, device=device)
        vol_rec = cp.ndarray(
            shape=vol_shape,
            dtype=cp.float32,
            memptr=cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(vol_t.data_ptr(), vol_t.numel() * 4, vol_t),
                0,
            ),
        )
    else:
        vol_rec = np.zeros(vol_shape, dtype=np.float32)

    # Create 2D parallel beam geometry
    vol_geom, proj_geom = _create_parallel2d_geometry(
        vol_shape=vol_shape,
        det_cols=det_cols,
        angles_deg=sel_angles,
        voxel_size_mm=voxel_size_mm,
        det_spacing_mm=det_spacing_mm,
    )

    # Apply ramp filter for FBP (Filtered Backprojection)
    sino_vc_filtered = _apply_ramp_filter(
        sino_vc,
        use_gpu=use_gpu,
        filter_type=filter_type,
        det_spacing_mm=det_spacing_mm if det_spacing_mm is not None else 1.0,
    )

    vol_id = astra.data2d.link('-vol', vol_geom, vol_rec)
    proj_id = astra.data2d.link('-sino', proj_geom, sino_vc_filtered)

    # Use BP_CUDA for backprojection after filtering
    # This completes the FBP (Filtered Backprojection) reconstruction
    cfg = astra.astra_dict('BP_CUDA')
    cfg['ProjectionDataId'] = proj_id
    cfg['ReconstructionDataId'] = vol_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, 1)  # Single FBP iteration
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(proj_id)
    astra.data2d.delete(vol_id)

    # Normalize by number of projections (matching analytical FBP scaling)
    norm_factor = math.pi / (2.0 * len(sel_angles)) if len(sel_angles) > 0 else 1.0
    if use_gpu:
        vol_t.mul_(norm_factor)
    else:
        vol_rec *= norm_factor

    if use_gpu:
        # Release intermediate CuPy allocations back to the driver; only the returned
        # torch tensor should retain GPU memory after this function completes.
        del sino_vc_filtered
        del sino_vc
        del sel_projs_cp
        del vol_rec
        cp.cuda.runtime.deviceSynchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        return vol_t

    # CPU path: convert numpy array back to torch tensor
    vol_t = torch.from_numpy(vol_rec).to(torch.float32).to(device)
    return vol_t


def sirt_reconstruction_masked(
    projs_vc: torch.Tensor,
    angles_deg: np.ndarray,
    mask: Optional[Sequence[Any]] = None,
    voxel_per_mm: int = 1,
    voxel_size_mm: float = -1.0,
    vol_shape: Optional[Tuple[int, int]] = None,
    det_spacing_mm: Optional[float] = 1.0,
    num_iterations: int = 100,
    min_constraint: Optional[float] = 0.0,
    max_constraint: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Run SIRT (Simultaneous Iterative Reconstruction Technique) on a subset of 2D parallel beam views.
    
    SIRT is an iterative reconstruction algorithm that minimizes the difference between
    measured and forward-projected data. It's more robust to noise and artifacts than FBP
    but requires more computation time.

    Parameters
    ----------
    projs_vc : (V, C) tensor 
        2D parallel beam projections (CPU or CUDA)
    angles_deg : (V,) numpy array
        Rotation angles in degrees (0-360)
    mask : optional boolean sequence length V
        If None, all views are used
    voxel_per_mm : int
        Resolution parameter
    voxel_size_mm : float
        Voxel size in mm (if -1, computed from voxel_per_mm)
    vol_shape : optional explicit (ny, nx)
        Volume dimensions
    det_spacing_mm : float
        Detector pixel spacing in mm
    num_iterations : int
        Number of SIRT iterations (default: 100)
        More iterations = better quality but slower
    min_constraint : float, optional
        Minimum value constraint (default: 0.0 for non-negative reconstruction)
        Set to None to disable
    max_constraint : float, optional
        Maximum value constraint (default: None for no upper limit)
    device : torch.device, optional
        Target device

    Returns
    -------
    vol : (ny, nx) float32 tensor on device
    
    Notes
    -----
    SIRT is particularly useful for:
    - Noisy data
    - Limited angle tomography
    - Sparse view reconstruction
    - When you need non-negativity constraints
    
    FBP is typically faster but SIRT can provide better quality for challenging cases.
    """
    if device is None:
        device = projs_vc.device if isinstance(projs_vc, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if voxel_size_mm <= 0:
        voxel_size_mm = 1.0 / voxel_per_mm

    if mask is not None:
        # Support torch.Tensor or numpy / sequence masks (boolean or index)
        if isinstance(mask, torch.Tensor):
            if mask.dtype == torch.bool:
                if mask.numel() != len(angles_deg):
                    raise ValueError("boolean mask length must match number of views")
                mask_cpu = mask.detach().cpu().numpy()
                sel_angles = angles_deg[mask_cpu]
                sel_projs = projs_vc[mask]
            else:
                idx = mask.detach().cpu().numpy()
                sel_angles = angles_deg[idx]
                sel_projs = projs_vc[idx]
        else:
            mask_arr = np.asarray(mask)
            if mask_arr.dtype == bool:
                if mask_arr.shape[0] != len(angles_deg):
                    raise ValueError("boolean mask length must match number of views")
                sel_angles = angles_deg[mask_arr]
                sel_projs = projs_vc[mask_arr]
            else:
                sel_angles = angles_deg[mask_arr]
                sel_projs = projs_vc[mask_arr]
    else:
        sel_angles = angles_deg
        sel_projs = projs_vc

    # Convert to ASTRA format (V, C)
    # For GPU path, use CuPy to keep data on GPU; for CPU path, use NumPy
    if sel_projs.is_cuda:
        if not CUPY_AVAILABLE:
            raise RuntimeError(
                "CuPy is required for GPU-accelerated reconstruction. "
                "Install it with: pip install cupy-cuda11x (or cupy-cuda12x for CUDA 12)"
            )
        # GPU path using CuPy - keeps data on GPU throughout
        # Get pointer to PyTorch tensor's CUDA memory
        ptr = sel_projs.data_ptr()
        shape = sel_projs.shape  # (V, C)
        
        # Create CuPy array from PyTorch tensor's memory (zero-copy view)
        sel_projs_cp = cp.ndarray(
            shape=shape,
            dtype=cp.float32,
            memptr=cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(ptr, sel_projs.numel() * 4, sel_projs),
                0
            )
        )
        
        # Make contiguous
        sino_vc = cp.ascontiguousarray(sel_projs_cp)
    else:
        # CPU path - convert to numpy
        sel_projs_np = sel_projs.detach().cpu().numpy()  # (V, C)
        sino_vc = sel_projs_np.copy()  # (V, C)
    
    n_views, det_cols = sino_vc.shape

    if vol_shape is None:
        # Default: square volume with size matching detector
        vol_shape = (det_cols, det_cols)

    # Create 2D parallel beam geometry
    vol_geom, proj_geom = _create_parallel2d_geometry(
        vol_shape=vol_shape,
        det_cols=det_cols,
        angles_deg=sel_angles,
        voxel_size_mm=voxel_size_mm,
        det_spacing_mm=det_spacing_mm,
    )

    # Note: SIRT_CUDA requires using astra.data2d.create() instead of link()
    # due to internal memory management requirements. We convert CuPy/NumPy to 
    # NumPy for ASTRA to manage, which copies to GPU internally.
    if isinstance(sino_vc, cp.ndarray):
        sino_vc_np = cp.asnumpy(sino_vc)
    else:
        sino_vc_np = sino_vc

    # Create ASTRA data objects (ASTRA manages GPU memory internally)
    vol_id = astra.data2d.create('-vol', vol_geom)
    proj_id = astra.data2d.create('-sino', proj_geom, sino_vc_np)

    # Create SIRT_CUDA algorithm
    cfg = astra.astra_dict('SIRT_CUDA')
    cfg['ProjectionDataId'] = proj_id
    cfg['ReconstructionDataId'] = vol_id
    
    # Add constraints if specified
    if min_constraint is not None or max_constraint is not None:
        cfg['option'] = {}
        if min_constraint is not None:
            cfg['option']['MinConstraint'] = float(min_constraint)
        if max_constraint is not None:
            cfg['option']['MaxConstraint'] = float(max_constraint)
    
    alg_id = astra.algorithm.create(cfg)
    
    # Run SIRT iterations
    astra.algorithm.run(alg_id, num_iterations)
    
    # Get result from ASTRA
    vol_rec = astra.data2d.get(vol_id)
    
    # Cleanup ASTRA objects
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(proj_id)
    astra.data2d.delete(vol_id)

    # Convert result to PyTorch tensor on target device
    vol_t = torch.from_numpy(vol_rec).to(torch.float32).to(device)
    
    return vol_t


# ---------------------------------------------------------------------------
# Gradient descent reconstruction
# ---------------------------------------------------------------------------

def gd_reconstruction_masked(
    projs_vc: torch.Tensor,
    angles_deg: np.ndarray,
    mask: Optional[Sequence[Any]] = None,
    voxel_per_mm: int = 1,
    voxel_size_mm: float = -1.0,
    vol_shape: Optional[Tuple[int, int]] = None,
    det_spacing_mm: float = 1.0,
    device: Optional[torch.device] = None,
    # Optimization hyper-parameters
    max_epochs: int | Sequence[int] = 30,
    batch_size: int | Sequence[int] = 20,
    lr: float | Sequence[float] = 1e-3,
    clamp_min: float = 0.0,
    # Loss function
    loss_type: str = "l2",
    # Optimizer settings
    optimizer_type: str = "adam",
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    # Initialization
    vol_init: Optional[torch.Tensor] = None,
    # Control verbosity
    verbose: bool = False,
    # Restart optimizer
    restart_optimizer: bool = True,  # Restart optimizer for each segment
) -> torch.Tensor:
    """Gradient-descent 2D parallel beam reconstruction using differentiable projector.

    This function performs iterative reconstruction by minimizing the mean squared error
    between measured projections and forward projections of the volume using mini-batches.

    Parameters
    ----------
    projs_vc : (V, C) torch.Tensor
        2D parallel beam projections (full set of views) on any device
    angles_deg : (V,) np.ndarray
        Rotation angles in degrees
    mask : sequence or boolean np.ndarray, optional
        Subset of views to use. If None, all views are used
    voxel_per_mm : int
        Resolution parameter
    vol_shape : (ny, nx), optional
        Explicit volume shape
    det_spacing_mm : float
        Detector pixel spacing
    device : torch.device, optional
        Target device for reconstruction
    max_epochs : int or Sequence[int]
        Number of optimization epochs (can be staged)
    batch_size : int or Sequence[int]
        Number of views per mini-batch
    lr : float or Sequence[float]
        Learning rate(s) for optimizer
    clamp_min : float
        Minimum value to clamp volume after each step
    loss_type : str
        Type of loss function to use. Options:
        - "l2": Mean squared error (assumes Gaussian noise)
        - "poisson": Poisson negative log-likelihood (assumes Poisson noise)
        Default is "l2"
    optimizer_type : str
        Type of optimizer ("adam" or "sgd")
    momentum : float
        Momentum for SGD optimizer
    weight_decay : float
        Weight decay (L2 regularization) parameter
    vol_init : torch.Tensor, optional
        Initial volume (if None, uses zero initialization)
    verbose : bool
        Show progress bars
    restart_optimizer : bool
        If True, restart optimizer for each segment (useful for staged learning rates)

    Returns
    -------
    vol : (ny, nx) torch.Tensor
        Final reconstructed volume on the specified device
    """
    if device is None:
        device = projs_vc.device if isinstance(projs_vc, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Validate loss_type
    if loss_type not in ["l2", "poisson"]:
        raise ValueError(f"loss_type must be 'l2' or 'poisson', got '{loss_type}'")

    # Select masked subset of projections & angles
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            if mask.dtype == torch.bool:
                if mask.numel() != len(angles_deg):
                    raise ValueError("boolean mask length must match number of views")
                mask_cpu = mask.detach().cpu().numpy()
                sel_angles = angles_deg[mask_cpu]
                sel_projs = projs_vc[mask]
            else:
                idx = mask.detach().cpu().numpy()
                sel_angles = angles_deg[idx]
                sel_projs = projs_vc[idx]
        else:
            mask_arr = np.asarray(mask)
            if mask_arr.dtype == bool:
                if mask_arr.shape[0] != len(angles_deg):
                    raise ValueError("boolean mask length must match number of views")
                sel_angles = angles_deg[mask_arr]
                sel_projs = projs_vc[mask_arr]
            else:
                sel_angles = angles_deg[mask_arr]
                sel_projs = projs_vc[mask_arr]
    else:
        sel_angles = angles_deg
        sel_projs = projs_vc

    V_sel = len(sel_angles)
    det_cols = sel_projs.shape[1]

    if vol_shape is None:
        # Default: square volume with size matching detector
        vol_shape = (det_cols, det_cols)
    
    if voxel_size_mm < 0:
        voxel_size_mm = 1.0 / voxel_per_mm

    # Initialization
    if vol_init is not None:
        if vol_init.ndim == 4:  # (1, 1, ny, nx)
            vol0 = vol_init[0, 0]
        elif vol_init.ndim == 3:  # (1, ny, nx)
            vol0 = vol_init[0]
        elif vol_init.ndim == 2:
            vol0 = vol_init
        else:
            raise ValueError("Unsupported vol_init shape")
        # If vol_shape not supplied, infer from vol_init
        if vol_shape is None:
            vol_shape = tuple(vol0.shape)
        if tuple(vol0.shape) != vol_shape:
            raise ValueError(f"vol_init shape {tuple(vol0.shape)} does not match target vol_shape {vol_shape}")
        recon = vol0.clone().to(device)
    else:
        # Default: zero initialization
        recon = torch.zeros(vol_shape, dtype=torch.float32, device=device)

    recon = recon.unsqueeze(0).unsqueeze(0)  # (1, 1, ny, nx)
    recon.requires_grad_(True)

    # Measurements tensor shaped (1, V, C) for convenience
    meas_full = sel_projs.to(device).unsqueeze(0)

    # Build learning rate / epoch schedule
    if isinstance(batch_size, int):
        batch_list = [int(batch_size)]
    else:
        batch_list = [int(b) for b in batch_size]
        if len(batch_list) == 0:
            raise ValueError("batch_size sequence must be non-empty")
        
    if isinstance(max_epochs, int):
        epochs_list = [int(max_epochs)]
    else:
        epochs_list = [int(e) for e in max_epochs]
        if len(epochs_list) == 0:
            raise ValueError("max_epochs sequence must be non-empty")
    if isinstance(lr, (float, int)):
        lr_list = [float(lr)] * len(epochs_list)
    else:
        lr_list = [float(x) for x in lr]
        if len(lr_list) == 0:
            raise ValueError("lr sequence must be non-empty")
        if len(lr_list) != len(epochs_list):
            raise ValueError("lr and max_epochs sequences must have same length")

    total_epochs = sum(epochs_list)

    # Outer progress bar
    if verbose:
        outer_pbar = tqdm(total=total_epochs, leave=True, disable=not verbose)
    epoch_global = 0

    if verbose:
        print(lr_list, epochs_list)
    optimizer = None

    for seg_idx, (lr_i, seg_epochs, batch_size) in enumerate(zip(lr_list, epochs_list, batch_list)):
        # Create optimizer
        if optimizer is None or restart_optimizer:
            if optimizer_type.lower() == "adam":
                optimizer = torch.optim.Adam([recon], lr=lr_i, weight_decay=weight_decay)
            elif optimizer_type.lower() == "sgd":
                optimizer = torch.optim.SGD([recon], lr=lr_i, momentum=momentum, weight_decay=weight_decay)
            else:
                raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Supported: 'adam', 'sgd'")
        
        for local_epoch in range(seg_epochs):
            perm = torch.randperm(V_sel, device=device)
            iters = math.ceil(V_sel / batch_size)
            
            epoch_loss = 0.0
            for it in range(iters):
                sel = perm[it * batch_size:(it + 1) * batch_size]
                angles_batch = sel_angles[sel.cpu().numpy()]
                proj_layer = build_parallel2d_projector(
                    vol_shape=vol_shape,
                    det_cols=det_cols,
                    angles_deg=angles_batch,
                    voxel_size_mm=voxel_size_mm,
                    det_spacing_mm=det_spacing_mm,
                    device=device
                )
                meas_batch = meas_full[:, sel]  # (1, k, C)
                optimizer.zero_grad(set_to_none=True)
                pred = proj_layer(recon)  # (1, k, C)
                
                # Compute loss based on loss_type
                if loss_type == "l2":
                    loss = torch.mean((pred - meas_batch) ** 2)
                elif loss_type == "poisson":
                    # Poisson negative log-likelihood: pred - meas * log(pred)
                    # Add small epsilon to avoid log(0)
                    eps = 1e-8
                    loss = torch.mean(pred - meas_batch * torch.log(pred + eps))
                
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    if clamp_min is not None:
                        recon.clamp_(min=clamp_min)
                epoch_loss += loss.item() * sel.numel()
                if verbose:
                    denom = (it * batch_size + sel.numel())
                    inner_postfix = {
                        'loss': f"{loss.item():.3e}",
                        'avg': f"{epoch_loss/denom:.3e}",
                        'k': sel.numel(),
                        'lr': f"{lr_i:.1e}",
                        'opt': optimizer_type.upper()
                    }
                    outer_pbar.set_postfix(inner_postfix)  # type: ignore[attr-defined]
            epoch_global += 1
            if verbose:
                outer_pbar.update(1)

    if verbose:
        outer_pbar.close()

    final_vol = recon.detach()[0, 0]
    return final_vol


# ---------------------------------------------------------------------------
# Differentiable 2D parallel beam projector
# ---------------------------------------------------------------------------

class _AstraParallel2DOp:
    """ASTRA 2D parallel beam forward/adjoint operator."""
    
    def __init__(self, vol_geom, proj_geom, vol_shape, det_cols, n_views):
        self.vol_geom = vol_geom
        self.proj_geom = proj_geom
        self.vol_shape = vol_shape
        self.det_cols = det_cols
        self.n_views = n_views
    
    def forward(self, vol_t, out_sino_t=None):
        """Forward projection: volume -> sinogram."""
        if out_sino_t is None:
            out_sino_t = torch.empty((self.n_views, self.det_cols), device=vol_t.device, dtype=torch.float32)
        
        # Convert PyTorch tensors to NumPy for ASTRA
        # ASTRA will handle GPU memory internally when using CUDA algorithms
        if vol_t.is_cuda:
            if not CUPY_AVAILABLE:
                raise RuntimeError(
                    "CuPy is required for GPU-accelerated projector. "
                    "Install it with: pip install cupy-cuda11x (or cupy-cuda12x for CUDA 12)"
                )
            # Create CuPy views of PyTorch tensors (zero-copy)
            vol_cp = cp.ndarray(
                shape=vol_t.shape,
                dtype=cp.float32,
                memptr=cp.cuda.MemoryPointer(
                    cp.cuda.UnownedMemory(vol_t.data_ptr(), vol_t.numel() * 4, vol_t), 0
                )
            )
            # Convert CuPy to NumPy for ASTRA (ASTRA manages GPU internally)
            vol_np = cp.asnumpy(vol_cp)
            
            # Create ASTRA data objects (ASTRA manages GPU memory)
            vol_id = astra.data2d.create('-vol', self.vol_geom, vol_np)
            sino_id = astra.data2d.create('-sino', self.proj_geom)
        else:
            # CPU path: convert to numpy
            vol_np = vol_t.detach().cpu().numpy()
            vol_id = astra.data2d.create('-vol', self.vol_geom, vol_np)
            sino_id = astra.data2d.create('-sino', self.proj_geom)
        
        cfg = astra.astra_dict('FP_CUDA' if vol_t.is_cuda else 'FP')
        cfg['VolumeDataId'] = vol_id
        cfg['ProjectionDataId'] = sino_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 1)
        
        # Get result from ASTRA
        sino_result = astra.data2d.get(sino_id)
        
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(sino_id)
        astra.data2d.delete(vol_id)
        
        # Copy result back to output tensor
        out_sino_t.copy_(torch.from_numpy(sino_result).to(out_sino_t.device))
        
        return out_sino_t

    def adjoint(self, sino_t, out_vol_t=None):
        """Adjoint/backprojection: sinogram -> volume."""
        if out_vol_t is None:
            out_vol_t = torch.zeros(self.vol_shape, device=sino_t.device, dtype=torch.float32)
        
        # Convert PyTorch tensors to NumPy for ASTRA
        # ASTRA will handle GPU memory internally when using CUDA algorithms
        if sino_t.is_cuda:
            if not CUPY_AVAILABLE:
                raise RuntimeError(
                    "CuPy is required for GPU-accelerated projector. "
                    "Install it with: pip install cupy-cuda11x (or cupy-cuda12x for CUDA 12)"
                )
            # Create CuPy view of PyTorch tensor (zero-copy)
            sino_cp = cp.ndarray(
                shape=sino_t.shape,
                dtype=cp.float32,
                memptr=cp.cuda.MemoryPointer(
                    cp.cuda.UnownedMemory(sino_t.data_ptr(), sino_t.numel() * 4, sino_t), 0
                )
            )
            # Convert CuPy to NumPy for ASTRA (ASTRA manages GPU internally)
            sino_np = cp.asnumpy(sino_cp)
            
            # Create ASTRA data objects (ASTRA manages GPU memory)
            vol_id = astra.data2d.create('-vol', self.vol_geom)
            sino_id = astra.data2d.create('-sino', self.proj_geom, sino_np)
        else:
            # CPU path: convert to numpy
            sino_np = sino_t.detach().cpu().numpy()
            vol_id = astra.data2d.create('-vol', self.vol_geom)
            sino_id = astra.data2d.create('-sino', self.proj_geom, sino_np)
        
        cfg = astra.astra_dict('BP_CUDA' if sino_t.is_cuda else 'BP')
        cfg['ReconstructionDataId'] = vol_id
        cfg['ProjectionDataId'] = sino_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 1)
        
        # Get result from ASTRA
        vol_result = astra.data2d.get(vol_id)
        
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(sino_id)
        astra.data2d.delete(vol_id)
        
        # Copy result back to output tensor
        out_vol_t.copy_(torch.from_numpy(vol_result).to(out_vol_t.device))
        
        return out_vol_t


def build_parallel2d_projector(
    vol_shape: Tuple[int, int],  # (ny, nx)
    det_cols: int,
    angles_deg: np.ndarray,
    voxel_size_mm: float = 1.0,
    det_spacing_mm: float = 1.0,
    device: Optional[torch.device] = None,
):
    """Return a differentiable 2D parallel beam projection layer.

    Returns a function handle that acts like: y = projector(x) where
    x is (B, 1, ny, nx) and y is (B, V, C). Autograd is supported.
    
    Parameters
    ----------
    vol_shape : (ny, nx)
        Volume dimensions
    det_cols : int
        Number of detector columns
    angles_deg : array
        Rotation angles in degrees
    voxel_size_mm : float
        Voxel size in mm
    det_spacing_mm : float
        Detector pixel spacing in mm
    device : torch.device
        Compute device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vol_geom, proj_geom = _create_parallel2d_geometry(
        vol_shape=vol_shape,
        det_cols=det_cols,
        angles_deg=angles_deg,
        voxel_size_mm=voxel_size_mm,
        det_spacing_mm=det_spacing_mm,
    )
    
    op = _AstraParallel2DOp(vol_geom, proj_geom, vol_shape, det_cols, len(angles_deg))

    class Parallel2DProjectorFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor):
            if x.ndim != 4 or x.shape[1] != 1:
                raise ValueError('Input must have shape (B, 1, ny, nx)')
            B = x.shape[0]
            y_out = torch.empty((B, op.n_views, det_cols), device=x.device, dtype=torch.float32)
            for i in range(B):
                vol_t = x[i, 0]
                sino_vc = op.forward(vol_t)  # (V, C)
                y_out[i] = sino_vc
            ctx.op = op
            return y_out

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            op_local = ctx.op
            B = grad_out.shape[0]
            g_vols = []
            for i in range(B):
                g_y = grad_out[i]  # (V, C)
                g_vol = torch.zeros(op_local.vol_shape, device=g_y.device, dtype=torch.float32)
                op_local.adjoint(g_y, g_vol)
                g_vols.append(g_vol)
            g_stack = torch.stack(g_vols, dim=0).unsqueeze(1)  # (B, 1, ny, nx)
            return g_stack

    def layer(x: torch.Tensor) -> torch.Tensor:
        return Parallel2DProjectorFn.apply(x)

    return layer
