"""
Utility functions for INCT.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def psnr(mse: float, max_val: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Args:
        mse: Mean Squared Error
        max_val: Maximum pixel value (1.0 for normalized images)
        
    Returns:
        PSNR in dB
    """
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Mean Squared Error.
    
    Args:
        pred: Predicted tensor
        target: Target tensor
        
    Returns:
        MSE value
    """
    return ((pred - target) ** 2).mean().item()


def normalize_coords(
    coords: torch.Tensor,
    shape: Tuple[int, ...],
) -> torch.Tensor:
    """
    Normalize integer coordinates to [0, 1] range.
    
    Args:
        coords: Integer coordinates of shape (..., n_dims)
        shape: Shape of the volume (D1, D2, ..., Dn)
        
    Returns:
        Normalized coordinates in [0, 1]
    """
    shape_tensor = torch.tensor(shape, dtype=torch.float32, device=coords.device)
    return coords.float() / (shape_tensor - 1).clamp(min=1)


def denormalize_coords(
    coords: torch.Tensor,
    shape: Tuple[int, ...],
) -> torch.Tensor:
    """
    Convert normalized coordinates back to integer indices.
    
    Args:
        coords: Normalized coordinates in [0, 1]
        shape: Shape of the volume
        
    Returns:
        Integer coordinates
    """
    shape_tensor = torch.tensor(shape, dtype=torch.float32, device=coords.device)
    return (coords * (shape_tensor - 1)).round().long()


def create_coord_grid(
    shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Create a coordinate grid for a volume.
    
    Args:
        shape: Shape of the volume (D1, D2, ..., Dn)
        device: Device to create tensor on
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        Coordinate tensor of shape (*shape, n_dims)
    """
    grids = []
    for i, s in enumerate(shape):
        if normalize:
            grid = torch.linspace(0, 1, s, device=device)
        else:
            grid = torch.arange(s, device=device, dtype=torch.float32)
        grids.append(grid)
    
    meshes = torch.meshgrid(*grids, indexing='ij')
    return torch.stack(meshes, dim=-1)


def compute_compression_ratio(
    original_shape: Tuple[int, ...],
    model_size_bytes: int,
    dtype_bits: int = 16,
) -> float:
    """
    Compute the compression ratio.
    
    Args:
        original_shape: Shape of the original volume
        model_size_bytes: Size of the compressed model in bytes
        dtype_bits: Bits per value in original data
        
    Returns:
        Compression ratio (original_size / compressed_size)
    """
    original_size = np.prod(original_shape) * (dtype_bits / 8)
    return original_size / model_size_bytes


def get_model_size_bytes(model: torch.nn.Module) -> int:
    """
    Get the size of a model in bytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Size in bytes
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return param_size + buffer_size


def print_model_summary(model: torch.nn.Module, input_shape: Optional[Tuple[int, ...]] = None):
    """
    Print a summary of the model.
    
    Args:
        model: PyTorch model
        input_shape: Optional input shape for testing
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 50)
    print(f"Model Summary")
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {get_model_size_bytes(model) / 1024:.2f} KB")
    print("=" * 50)


class EarlyStopping:
    """
    Early stopping handler.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current score (loss or metric)
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class MetricTracker:
    """
    Track and compute running averages of metrics.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked values."""
        self.values = {}
        self.counts = {}
    
    def update(self, metrics: dict, n: int = 1):
        """
        Update metrics with new values.
        
        Args:
            metrics: Dictionary of metric names to values
            n: Number of samples
        """
        for key, value in metrics.items():
            if key not in self.values:
                self.values[key] = 0.0
                self.counts[key] = 0
            self.values[key] += value * n
            self.counts[key] += n
    
    def get_average(self, key: str) -> float:
        """Get the running average for a metric."""
        if key not in self.values or self.counts[key] == 0:
            return 0.0
        return self.values[key] / self.counts[key]
    
    def get_averages(self) -> dict:
        """Get all running averages."""
        return {key: self.get_average(key) for key in self.values}
