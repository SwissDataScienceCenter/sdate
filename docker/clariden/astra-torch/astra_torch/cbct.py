"""Astra / Walnut dataset utilities.

This module factors logic from the CBCT notebook into reusable GPU-friendly
PyTorch functions without relying on notebook globals.

Public functions

fdk_reconstruction_masked(...):
    Perform FDK reconstruction on (possibly masked) subset of views.

project_conebeam_masked(...):
    Forward project a volume for a subset of views producing cone-beam
    projections with autograd support (PyTorch tensors in / out) using ASTRA CUDA.

build_conebeam_projector(...):
    Lower-level factory that returns a torch.autograd.Function apply handle
    for a given subset of projection geometry vectors.

Notes
-----
* Requires the ASTRA toolbox compiled with CUDA support.
* All tensors are float32 and (by default) CUDA if available.
* Projection tensor layout exposed to the user is (V, R, C) following the
  notebook convention (views, detector_rows, detector_cols).
"""
from __future__ import annotations

import locale
import os

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Optional, Dict, Any
import os
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading

try:
    import astra  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError("astra toolbox is required for chip.astra.walnut module") from e


@dataclass
class CBCTAcquisition:
    proj_rows: int
    proj_cols: int
    vecs: np.ndarray  # shape (V, 12)
    voxel_per_mm: int
    vol_shape: Tuple[int, int, int]
    voxel_size_mm: float



# ---------------------------------------------------------------------------
# FDK reconstruction with optional masking
# ---------------------------------------------------------------------------

def _make_volume_geom(vol_shape: Tuple[int, int, int], voxel_size_mm: float):
    vol_geom = astra.create_vol_geom(vol_shape)
    # Scale window by voxel size
    for key in [
        'WindowMinX','WindowMaxX','WindowMinY','WindowMaxY','WindowMinZ','WindowMaxZ'
    ]:
        vol_geom['option'][key] = vol_geom['option'][key] * voxel_size_mm
    return vol_geom


def fdk_reconstruction_masked(
    projs_vrc: torch.Tensor,
    vecs: np.ndarray,
    voxel_per_mm: int,
    voxel_size_mm: float,
    mask: Optional[Sequence[Any]] = None,
    vol_shape: Optional[Tuple[int, int, int]] = None,
    short_scan: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Run FDK on a subset of views indicated by mask.

    Parameters
    ----------
    projs_vrc : (V,R,C) tensor (CPU or CUDA)
    vecs : (V,12) numpy array
    mask : optional boolean sequence length V; if None all used.
    voxel_per_mm : resolution parameter (50*voxel_per_mm+1 per dim if vol_shape None)
    vol_shape : optional explicit (nx,ny,nz)
    short_scan : set ASTRA ShortScan option

    Returns
    -------
    vol : (nx,ny,nz) float32 CPU numpy wrapped as torch.Tensor on device.
    """
    if device is None:
        device = projs_vrc.device if isinstance(projs_vrc, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if vecs is a torch tensor, convert to numpy
    if isinstance(vecs, torch.Tensor):
        vecs = vecs.detach().cpu().numpy()

    if mask is not None:
        # Support torch.Tensor or numpy / sequence masks (boolean or index)
        if isinstance(mask, torch.Tensor):
            if mask.dtype == torch.bool:
                if mask.numel() != vecs.shape[0]:
                    raise ValueError("boolean mask length must match number of views")
                mask_cpu = mask.detach().cpu().numpy()
                sel_vecs = vecs[mask_cpu]
                sel_projs = projs_vrc[mask]
            else:
                idx = mask.detach().cpu().numpy()
                sel_vecs = vecs[idx]
                sel_projs = projs_vrc[idx]
        else:
            mask_arr = np.asarray(mask)
            if mask_arr.dtype == bool:
                if mask_arr.shape[0] != vecs.shape[0]:
                    raise ValueError("boolean mask length must match number of views")
                sel_vecs = vecs[mask_arr]
                sel_projs = projs_vrc[mask_arr]
            else:
                sel_vecs = vecs[mask_arr]
                sel_projs = projs_vrc[mask_arr]
    else:
        sel_vecs = vecs
        sel_projs = projs_vrc

    # Convert to ASTRA format (R,V,C) - PyTorch CUDA tensors work directly with ASTRA!
    # PyTorch CUDA tensors have __cuda_array_interface__, so ASTRA can access GPU memory directly
    if sel_projs.is_cuda:
        # Keep on GPU - ASTRA will access directly via __cuda_array_interface__
        sino_rvc = sel_projs.permute(1, 0, 2).contiguous().detach()  # (V,R,C) -> (R,V,C)
    else:
        # CPU path - convert to numpy
        sel_projs_np = sel_projs.detach().cpu().numpy()  # (V,R,C)
        sino_rvc = np.transpose(sel_projs_np, (1, 0, 2)).copy()
    
    det_rows, n_views, det_cols = sino_rvc.shape

    if vol_shape is None:
        dim = int(50 * voxel_per_mm + 1)
        vol_shape = (dim, dim, dim)
    if voxel_size_mm is None:
        voxel_size_mm = 1.0 / voxel_per_mm

    vol_rec = np.zeros(vol_shape, dtype=np.float32)

    vol_geom = _make_volume_geom(vol_shape, voxel_size_mm)
    proj_geom = astra.create_proj_geom('cone_vec', det_rows, det_cols, sel_vecs)

    vol_id = astra.data3d.link('-vol', vol_geom, vol_rec)
    proj_id = astra.data3d.link('-sino', proj_geom, sino_rvc)

    cfg = astra.astra_dict('FDK_CUDA')
    cfg['ProjectionDataId'] = proj_id
    cfg['ReconstructionDataId'] = vol_id
    cfg['option'] = {'ShortScan': bool(short_scan)}
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, 1)
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(vol_id)

    vol_t = torch.from_numpy(vol_rec).to(torch.float32).to(device)
    return vol_t


def gd_reconstruction_masked(
    projs_vrc: torch.Tensor,
    vecs: np.ndarray,
    voxel_per_mm: int,
    voxel_size_mm: float,
    mask: Optional[Sequence[Any]] = None,
    vol_shape: Optional[Tuple[int, int, int]] = None,
    device: Optional[torch.device] = None,
    # Optimization hyper-parameters
    max_epochs: int | Sequence[int] = 30,
    batch_size: int = 10,
    lr: float | Sequence[float] = 1e-3,
    clamp_min: float = 0.0,
    # Optimizer settings
    optimizer_type: str = "adam",
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    # Initialization (if vol_init is None we default to ZERO initialization)
    vol_init: Optional[torch.Tensor] = None,
    # Control verbosity / progress bars
    verbose: bool = False,
) -> torch.Tensor:
    """Gradient-descent style masked reconstruction using differentiable projector.

    This function mirrors the logic prototyped in the CBCT notebook: starting from
    an initialization (optionally an FDK reconstruction on the masked subset) and
    iteratively minimizing the mean squared error between measured masked
    projections and forward projections of the volume using mini-batches of views.

    Parameters
    ----------
    projs_vrc : (V,R,C) torch.Tensor
        Log-corrected cone-beam projections (full set of views) on any device.
    vecs : (V,12) np.ndarray
        Geometry vectors (Walnut cone-beam format for ASTRA 'cone_vec').
    mask : sequence or boolean np.ndarray, optional
        Subset of views to use. If None, all views are used.
    voxel_per_mm : int
        Resolution parameter; if vol_shape not provided we set dim = 50*voxel_per_mm+1.
    vol_shape : (nx,ny,nz), optional
        Explicit volume shape. If None it's derived from voxel_per_mm like FDK helper.
    device : torch.device, optional
        Target device for reconstruction (defaults to projections' device or CUDA).
    max_epochs : int or Sequence[int]
        If int, total number of optimization epochs with a single learning rate.
        If a sequence, interpreted as a list of epoch counts for a staged schedule
        (one optimizer / learning rate segment per entry). When a sequence is
        provided, lr can be either a single float (broadcast to all segments) or
        a sequence of the same length (pairwise zipped like (lr_i, epochs_i)).
    batch_size : int
        Number of views per mini-batch (last batch may be smaller).
    lr : float or Sequence[float]
        Learning rate(s) for optimizer. If a sequence is provided along with
        a sequence for max_epochs, they are zipped to form (lr_i, epochs_i)
        schedule segments. If only one of (lr, max_epochs) is a sequence, the
        scalar counterpart is broadcast to match its length.
    clamp_min : float
        Minimum value to clamp volume after each optimization step (non-negativity).
    optimizer_type : str
        Type of optimizer to use. Supported values: "adam", "sgd".
    momentum : float
        Momentum parameter for SGD optimizer (ignored for Adam). Default: 0.9.
    weight_decay : float
        Weight decay (L2 regularization) parameter for optimizer. Default: 0.0.
    vol_init : torch.Tensor, optional
        If provided, used as initialization. Accepts shapes (nx,ny,nz), (1,nx,ny,nz) or (1,1,nx,ny,nz).
        If None, the volume is initialized with zeros (non-negative). To use an FDK initialization,
        call fdk_reconstruction_masked(...) first and pass the result as vol_init.
    verbose : bool
        If True, show tqdm progress bars (epoch + inner batches). Outer epoch bar only otherwise.

    Returns
    -------
    vol : (nx,ny,nz) torch.Tensor
        Final reconstructed volume on the specified device.
    """
    import math, time

    if device is None:
        device = projs_vrc.device if isinstance(projs_vrc, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(vecs, torch.Tensor):
        vecs = vecs.detach().cpu().numpy()

    # Select masked subset of projections & geometry (reuse logic from FDK function)
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            if mask.dtype == torch.bool:
                if mask.numel() != vecs.shape[0]:
                    raise ValueError("boolean mask length must match number of views")
                mask_cpu = mask.detach().cpu().numpy()
                sel_vecs = vecs[mask_cpu]
                sel_projs = projs_vrc[mask]
            else:
                idx = mask.detach().cpu().numpy()
                sel_vecs = vecs[idx]
                sel_projs = projs_vrc[idx]
        else:
            mask_arr = np.asarray(mask)
            if mask_arr.dtype == bool:
                if mask_arr.shape[0] != vecs.shape[0]:
                    raise ValueError("boolean mask length must match number of views")
                sel_vecs = vecs[mask_arr]
                sel_projs = projs_vrc[mask_arr]
            else:
                sel_vecs = vecs[mask_arr]
                sel_projs = projs_vrc[mask_arr]
    else:
        sel_vecs = vecs
        sel_projs = projs_vrc

    V_sel = sel_vecs.shape[0]
    det_rows, det_cols = sel_projs.shape[1], sel_projs.shape[2]
    
    if voxel_size_mm is None:
        voxel_size_mm = 1.0 / voxel_per_mm

    # Initialization
    if vol_init is not None:
        if vol_init.ndim == 5:
            vol0 = vol_init[0, 0]
        elif vol_init.ndim == 4:  # (1,nx,ny,nz)
            vol0 = vol_init[0]
        elif vol_init.ndim == 3:
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
        if vol_shape is None:
            raise ValueError("vol_shape must be provided if vol_init is None")
        recon = torch.zeros(vol_shape, dtype=torch.float32, device=device)

    recon = recon.unsqueeze(0).unsqueeze(0)  # (1,1,nx,ny,nz)
    recon.requires_grad_(True)

    # Measurements tensor shaped (1,V,R,C) for convenience
    meas_full = sel_projs.to(device).unsqueeze(0)

    # ------------------------------------------------------------------
    # Build learning rate / epoch schedule
    # ------------------------------------------------------------------
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
            # Allow broadcast if one of them length 1 (already handled earlier) -> raise otherwise
            raise ValueError("lr and max_epochs sequences must have same length")

    total_epochs = sum(epochs_list)

    # Outer progress bar accumulative over schedule
    outer_iter = range(total_epochs)
    if verbose:
        outer_pbar = tqdm(total=total_epochs, leave=True, disable=not verbose)
    epoch_global = 0
    if verbose:
        print(lr_list, epochs_list)
    for seg_idx, (lr_i, seg_epochs) in enumerate(zip(lr_list, epochs_list)):
        # Create optimizer based on type
        if optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam([recon], lr=lr_i, weight_decay=weight_decay)
        elif optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD([recon], lr=lr_i, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Supported: 'adam', 'sgd'")
        
        for local_epoch in range(seg_epochs):
            perm = torch.randperm(V_sel, device=device)
            iters = math.ceil(V_sel / batch_size)
            inner_iter = range(iters)
    
            epoch_loss = 0.0
            for it in inner_iter:
                sel = perm[it * batch_size:(it + 1) * batch_size]
                vecs_batch = sel_vecs[sel.cpu().numpy()]
                proj_layer = build_conebeam_projector(vol_shape, det_rows, det_cols, vecs_batch, voxel_size_mm, device=device)
                meas_batch = meas_full[:, sel]  # (1,k,R,C)
                optimizer.zero_grad(set_to_none=True)
                pred = proj_layer(recon)  # (1,k,R,C)
                loss = torch.mean((pred - meas_batch) ** 2)
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
# Differentiable projector (forward + adjoint) for masked views
# ---------------------------------------------------------------------------



class _AstraConeOp:
    def __init__(self, vol_geom, proj_geom, vol_shape, det_rows, det_cols, n_views):
        self.vol_geom = vol_geom
        self.proj_geom = proj_geom
        self.vol_shape = vol_shape
        self.det_rows = det_rows
        self.det_cols = det_cols
        self.n_views = n_views
    
    def forward(self, vol_t, out_sino_t=None):
        if out_sino_t is None:
            out_sino_t = torch.empty((self.det_rows, self.n_views, self.det_cols), device=vol_t.device, dtype=torch.float32)
        vol_id  = astra.data3d.link('-vol',  self.vol_geom, vol_t.detach())
        sino_id = astra.data3d.link('-sino', self.proj_geom, out_sino_t.detach())
        cfg = astra.astra_dict('FP3D_CUDA')
        cfg['VolumeDataId'] = vol_id
        cfg['ProjectionDataId'] = sino_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 1)
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(sino_id)
        astra.data3d.delete(vol_id)
        return out_sino_t

    def adjoint(self, sino_t, out_vol_t=None):
        if out_vol_t is None:
            out_vol_t = torch.zeros(self.vol_shape, device=sino_t.device, dtype=torch.float32)
        vol_id  = astra.data3d.link('-vol',  self.vol_geom, out_vol_t.detach())
        sino_id = astra.data3d.link('-sino', self.proj_geom, sino_t.detach())
        cfg = astra.astra_dict('BP3D_CUDA')
        cfg['ReconstructionDataId'] = vol_id
        cfg['ProjectionDataId']     = sino_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 1)
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(sino_id)
        astra.data3d.delete(vol_id)
        return out_vol_t
    


def build_conebeam_projector(
    vol_shape: Tuple[int, int, int],
    det_rows: int,
    det_cols: int,
    vecs_subset: np.ndarray,
    voxel_size_mm: float,
    device: Optional[torch.device] = None,
):
    """Return a differentiable projection layer (callable) for given subset views.

    Returns a function handle that acts like: y = projector(x) where
    x is (B,1,nx,ny,nz) and y is (B, V, R, C). Autograd is supported.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vol_geom = _make_volume_geom(vol_shape, voxel_size_mm)
    proj_geom = astra.create_proj_geom('cone_vec', det_rows, det_cols, vecs_subset)
    op = _AstraConeOp(vol_geom, proj_geom, vol_shape, det_rows, det_cols, len(vecs_subset))

    class ConeBeamProjectorFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor):
            if x.ndim != 5 or x.shape[1] != 1:
                raise ValueError('Input must have shape (B,1,nx,ny,nz)')
            B = x.shape[0]
            y_out = torch.empty((B, op.n_views, det_rows, det_cols), device=x.device, dtype=torch.float32)
            for i in range(B):
                vol_t = x[i, 0]
                sino_rvc = op.forward(vol_t)  # (R,V,C)
                y_out[i] = sino_rvc.permute(1, 0, 2).contiguous()
            ctx.op = op
            return y_out

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            op_local = ctx.op
            B = grad_out.shape[0]
            g_vols = []
            for i in range(B):
                g_y = grad_out[i]  # (V,R,C)
                # reorder to ASTRA (R,V,C)
                g_sino_rvc = g_y.permute(1, 0, 2).contiguous()
                g_vol = torch.zeros(op_local.vol_shape, device=g_y.device, dtype=torch.float32)
                op_local.adjoint(g_sino_rvc, g_vol)
                g_vols.append(g_vol)
            g_stack = torch.stack(g_vols, dim=0).unsqueeze(1)  # (B,1,nx,ny,nz)
            return g_stack

    def layer(x: torch.Tensor) -> torch.Tensor:
        return ConeBeamProjectorFn.apply(x)

    return layer


def project_conebeam_masked(
    volume: torch.Tensor,
    vecs: np.ndarray,
    mask: Optional[Sequence[Any]],
    voxel_size_mm: float,
    det_rows: int,
    det_cols: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Project a volume for selected views returning (V,R,C) tensor with autograd.

    volume: (1,1,nx,ny,nz) or (nx,ny,nz) or (B,1,nx,ny,nz)
    mask: optional boolean or index sequence selecting subset of vecs.
    Supports batching if volume has shape (B,1,nx,ny,nz).
    """
    if device is None:
        device = volume.device if isinstance(volume, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Normalize volume to (B,1,nx,ny,nz)
    if volume.ndim == 5:
        if volume.shape[1] != 1:
            raise ValueError('Channel dimension must be 1')
        vols = volume
    elif volume.ndim == 3:
        vols = volume.unsqueeze(0).unsqueeze(0)
    elif volume.ndim == 4:
        # assume (B,nx,ny,nz)
        vols = volume.unsqueeze(1)
    else:
        raise ValueError('Unsupported volume dimensionality')

    if mask is not None:
        if isinstance(mask, torch.Tensor):
            if mask.dtype == torch.bool:
                if mask.numel() != vecs.shape[0]:
                    raise ValueError('boolean mask length mismatch with vecs')
                vecs_subset = vecs[mask.detach().cpu().numpy()]
            else:
                vecs_subset = vecs[mask.detach().cpu().numpy()]
        else:
            mask_arr = np.asarray(mask)
            if mask_arr.dtype == bool:
                if mask_arr.shape[0] != vecs.shape[0]:
                    raise ValueError('boolean mask length mismatch with vecs')
                vecs_subset = vecs[mask_arr]
            else:
                vecs_subset = vecs[mask_arr]
    else:
        vecs_subset = vecs

    vol_shape = tuple(vols.shape[2:])
    projector = build_conebeam_projector(vol_shape, det_rows, det_cols, vecs_subset, voxel_size_mm, device=device)
    projs = projector(vols)  # (B,V,R,C)
    if volume.ndim == 3:
        return projs[0]
    return projs
