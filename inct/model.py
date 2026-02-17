"""
Neural Network Models for Instant NGP.

Implements the small MLP that decodes hash-encoded features
into output values (e.g., intensity for tomographic data).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .hash_encoding import MultiResolutionHashEncoding


class TinyMLP(nn.Module):
    """
    Small Multi-Layer Perceptron as used in Instant NGP.
    
    Uses ReLU activations and optionally a sigmoid output.
    The network is kept small (typically 2-3 hidden layers with 64 neurons)
    to maintain fast inference while the hash encoding provides capacity.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: List[int] = [64, 64],
        activation: str = "relu",
        output_activation: Optional[str] = None,
    ):
        """
        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (1 for scalar intensity)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'gelu', 'silu')
            output_activation: Output activation (None, 'sigmoid', 'tanh')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "silu":
                layers.append(nn.SiLU(inplace=True))
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Output activation
        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation == "tanh":
            layers.append(nn.Tanh())
        elif output_activation is not None:
            raise ValueError(f"Unknown output activation: {output_activation}")
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        return self.network(x)


class InstantNGPModel(nn.Module):
    """
    Complete Instant NGP model for learning volumetric data.
    
    Combines multi-resolution hash encoding with a small MLP decoder
    to learn a continuous function from coordinates to values.
    
    For tomographic data:
    - 3D version: learns a single 3D volume (height, width, projection_index)
    - 4D version: could include additional dimensions
    """
    
    def __init__(
        self,
        n_dims: int = 3,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        base_resolution: int = 16,
        max_resolution: int = 512,
        table_size: int = 2**19,
        hidden_dims: List[int] = [64, 64],
        output_dim: int = 1,
        activation: str = "relu",
        output_activation: Optional[str] = "sigmoid",
        include_input: bool = True,
    ):
        """
        Args:
            n_dims: Number of input coordinate dimensions
            n_levels: Number of hash encoding resolution levels
            n_features_per_level: Features per level
            base_resolution: Coarsest resolution
            max_resolution: Finest resolution
            table_size: Hash table size per level
            hidden_dims: MLP hidden layer dimensions
            output_dim: Output dimension (1 for grayscale)
            activation: MLP activation function
            output_activation: Output activation (sigmoid for [0,1] outputs)
            include_input: Whether to include raw coords in encoding
        """
        super().__init__()
        
        self.n_dims = n_dims
        
        # Hash encoding
        self.encoding = MultiResolutionHashEncoding(
            n_dims=n_dims,
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            base_resolution=base_resolution,
            max_resolution=max_resolution,
            table_size=table_size,
            include_input=include_input,
        )
        
        # MLP decoder
        self.mlp = TinyMLP(
            input_dim=self.encoding.output_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            output_activation=output_activation,
        )
        
        # Store config for saving/loading
        self.config = {
            'n_dims': n_dims,
            'n_levels': n_levels,
            'n_features_per_level': n_features_per_level,
            'base_resolution': base_resolution,
            'max_resolution': max_resolution,
            'table_size': table_size,
            'hidden_dims': hidden_dims,
            'output_dim': output_dim,
            'activation': activation,
            'output_activation': output_activation,
            'include_input': include_input,
        }
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: coordinates -> hash encoding -> MLP -> output.
        
        Args:
            coords: Normalized coordinates in [0, 1], shape (..., n_dims)
            
        Returns:
            Output values of shape (..., output_dim)
        """
        # Get hash encoding features
        features = self.encoding(coords)
        
        # Decode with MLP
        output = self.mlp(features)
        
        return output
    
    def predict_volume(
        self,
        shape: Tuple[int, ...],
        batch_size: int = 65536,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Predict values for an entire volume.
        
        Args:
            shape: Shape of the output volume (e.g., (H, W, D) for 3D)
            batch_size: Number of coordinates to process at once
            device: Device to use for computation
            
        Returns:
            Tensor of shape (*shape, output_dim)
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Create coordinate grid
        grids = [torch.linspace(0, 1, s, device=device) for s in shape]
        coords = torch.stack(torch.meshgrid(*grids, indexing='ij'), dim=-1)
        coords = coords.reshape(-1, self.n_dims)
        
        # Process in batches
        outputs = []
        with torch.no_grad():
            for i in range(0, coords.shape[0], batch_size):
                batch_coords = coords[i:i+batch_size]
                batch_output = self.forward(batch_coords)
                outputs.append(batch_output)
        
        # Reshape to volume
        output = torch.cat(outputs, dim=0)
        output = output.reshape(*shape, -1)
        
        # Squeeze if single channel
        if output.shape[-1] == 1:
            output = output.squeeze(-1)
        
        return output
    
    def get_model_info(self) -> str:
        """Return string with model information."""
        n_params = sum(p.numel() for p in self.parameters())
        n_hash_params = self.encoding.n_params
        n_mlp_params = n_params - n_hash_params
        
        info = [
            f"InstantNGPModel:",
            f"  Input dims: {self.n_dims}",
            f"  Total parameters: {n_params:,}",
            f"  Hash table params: {n_hash_params:,} ({100*n_hash_params/n_params:.1f}%)",
            f"  MLP params: {n_mlp_params:,} ({100*n_mlp_params/n_params:.1f}%)",
            f"  Encoding: {self.encoding.n_levels} levels, {self.encoding.n_features_per_level} features/level",
            f"  {self.encoding.get_resolution_info()}",
        ]
        return "\n".join(info)
    
    @classmethod
    def from_config(cls, config: dict) -> 'InstantNGPModel':
        """Create model from config dictionary."""
        return cls(**config)
    
    def save(self, path: str):
        """Save model to file."""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict(),
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'InstantNGPModel':
        """Load model from file."""
        checkpoint = torch.load(path, map_location=device)
        model = cls.from_config(checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model
