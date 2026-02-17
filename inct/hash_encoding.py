"""
Multi-Resolution Hash Encoding for Instant NGP.

Implements the hash encoding scheme from:
"Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"

The key idea is to use multiple levels of resolution, each with its own
hash table. For each input coordinate, we:
1. Scale to each resolution level
2. Find the surrounding voxel vertices
3. Hash the vertex coordinates to get indices into the hash table
4. Trilinearly interpolate the feature vectors

For 4D data (projections), we extend to quadrilinear interpolation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List


def hash_function(coords: torch.Tensor, table_size: int) -> torch.Tensor:
    """
    Spatial hash function for arbitrary dimensional integer coordinates.
    
    Uses XOR of coordinate-specific hashes with large prime numbers.
    Based on the hash function from Instant NGP paper.
    
    Args:
        coords: Integer coordinates of shape (..., D) where D is dimensionality
        table_size: Size of hash table (T in the paper)
        
    Returns:
        Hash indices of shape (...,) in range [0, table_size)
    """
    # Large prime numbers for hashing (one per dimension, up to 4D)
    primes = torch.tensor([1, 2654435761, 805459861, 3674653429], 
                          dtype=torch.int64, device=coords.device)
    
    D = coords.shape[-1]
    primes = primes[:D]
    
    # XOR hash: h = (x1 * p1) XOR (x2 * p2) XOR ... XOR (xD * pD)
    coords_long = coords.long()
    result = torch.zeros(coords.shape[:-1], dtype=torch.int64, device=coords.device)
    
    for d in range(D):
        result = result ^ (coords_long[..., d] * primes[d])
    
    # Take modulo to get index in table
    return (result % table_size).long()


class HashEncoding(nn.Module):
    """
    Single-level hash encoding.
    
    Maps D-dimensional coordinates to F-dimensional feature vectors
    using a hash table and multilinear interpolation.
    """
    
    def __init__(
        self,
        n_dims: int = 3,
        n_features: int = 2,
        resolution: int = 16,
        table_size: int = 2**14,
    ):
        """
        Args:
            n_dims: Number of input dimensions (3 for 3D, 4 for 4D)
            n_features: Number of features per hash entry (F in paper)
            resolution: Grid resolution for this level (N in paper)
            table_size: Size of hash table (T in paper)
        """
        super().__init__()
        
        self.n_dims = n_dims
        self.n_features = n_features
        self.resolution = resolution
        self.table_size = table_size
        
        # Hash table: learnable feature vectors
        # Initialized with small random values as in the paper
        self.hash_table = nn.Parameter(
            torch.randn(table_size, n_features) * 1e-4
        )
        
        # Precompute vertex offsets for the hypercube corners
        # For 3D: 8 corners, for 4D: 16 corners
        n_vertices = 2 ** n_dims
        offsets = torch.zeros(n_vertices, n_dims, dtype=torch.long)
        for i in range(n_vertices):
            for d in range(n_dims):
                offsets[i, d] = (i >> d) & 1
        self.register_buffer('vertex_offsets', offsets)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute hash encoding for input coordinates.
        
        Args:
            coords: Normalized coordinates in [0, 1], shape (..., n_dims)
            
        Returns:
            Feature vectors of shape (..., n_features)
        """
        # Scale coordinates to grid resolution
        scaled = coords * (self.resolution - 1)
        
        # Get floor coordinates (lower corner of the cell)
        floor_coords = torch.floor(scaled).long()
        
        # Get interpolation weights
        weights = scaled - floor_coords.float()
        
        # Clamp floor coords to valid range
        floor_coords = torch.clamp(floor_coords, 0, self.resolution - 2)
        
        # Shape: (..., n_vertices, n_dims)
        batch_shape = coords.shape[:-1]
        n_vertices = 2 ** self.n_dims
        
        # Compute all vertex coordinates
        # Expand floor_coords: (..., 1, n_dims) + (n_vertices, n_dims) -> (..., n_vertices, n_dims)
        all_vertices = floor_coords.unsqueeze(-2) + self.vertex_offsets
        
        # Hash all vertices to get table indices
        # Shape: (..., n_vertices)
        hash_indices = hash_function(all_vertices, self.table_size)
        
        # Look up features from hash table
        # Shape: (..., n_vertices, n_features)
        features = self.hash_table[hash_indices]
        
        # Compute multilinear interpolation weights for each vertex
        # For vertex i, weight = prod_d (w_d if bit d is 1, else 1-w_d)
        vertex_weights = torch.ones(*batch_shape, n_vertices, device=coords.device)
        
        for d in range(self.n_dims):
            # bit_d tells us whether to use w or (1-w) for this dimension
            bit_d = self.vertex_offsets[:, d]  # Shape: (n_vertices,)
            w_d = weights[..., d:d+1]  # Shape: (..., 1)
            
            # Expand: (..., n_vertices)
            weight_d = torch.where(
                bit_d.unsqueeze(0).expand(*batch_shape, -1) == 1,
                w_d.expand(*batch_shape, n_vertices),
                (1 - w_d).expand(*batch_shape, n_vertices)
            )
            vertex_weights = vertex_weights * weight_d
        
        # Weighted sum of features
        # vertex_weights: (..., n_vertices, 1)
        # features: (..., n_vertices, n_features)
        vertex_weights = vertex_weights.unsqueeze(-1)
        output = (vertex_weights * features).sum(dim=-2)
        
        return output


class MultiResolutionHashEncoding(nn.Module):
    """
    Multi-resolution hash encoding as described in Instant NGP.
    
    Uses L levels of resolution, each with its own hash table.
    The resolutions follow a geometric progression from N_min to N_max.
    """
    
    def __init__(
        self,
        n_dims: int = 3,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        base_resolution: int = 16,
        max_resolution: int = 512,
        table_size: int = 2**14,
        include_input: bool = True,
    ):
        """
        Args:
            n_dims: Number of input dimensions (3 for 3D, 4 for 4D)
            n_levels: Number of resolution levels (L in paper)
            n_features_per_level: Features per hash entry per level (F in paper)
            base_resolution: Coarsest resolution (N_min in paper)
            max_resolution: Finest resolution (N_max in paper)
            table_size: Size of each hash table (T in paper)
            include_input: Whether to include original coordinates in output
        """
        super().__init__()
        
        self.n_dims = n_dims
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.table_size = table_size
        self.include_input = include_input
        
        # Compute resolution for each level using geometric progression
        # N_l = floor(N_min * b^l) where b = exp(ln(N_max/N_min) / (L-1))
        if n_levels > 1:
            b = np.exp(np.log(max_resolution / base_resolution) / (n_levels - 1))
        else:
            b = 1.0
        
        self.resolutions = []
        self.encodings = nn.ModuleList()
        
        for level in range(n_levels):
            resolution = int(np.floor(base_resolution * (b ** level)))
            self.resolutions.append(resolution)
            
            self.encodings.append(HashEncoding(
                n_dims=n_dims,
                n_features=n_features_per_level,
                resolution=resolution,
                table_size=table_size,
            ))
        
        # Output dimension
        self.output_dim = n_levels * n_features_per_level
        if include_input:
            self.output_dim += n_dims
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-resolution hash encoding.
        
        Args:
            coords: Normalized coordinates in [0, 1], shape (..., n_dims)
            
        Returns:
            Concatenated feature vectors of shape (..., output_dim)
        """
        features = []
        
        # Optionally include raw coordinates
        if self.include_input:
            features.append(coords)
        
        # Get features from each level
        for encoding in self.encodings:
            level_features = encoding(coords)
            features.append(level_features)
        
        # Concatenate all features
        return torch.cat(features, dim=-1)
    
    def get_resolution_info(self) -> str:
        """Return string describing resolution levels."""
        return f"Resolutions: {self.resolutions}"
    
    @property
    def n_params(self) -> int:
        """Total number of parameters in hash tables."""
        return sum(enc.hash_table.numel() for enc in self.encodings)
