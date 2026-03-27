"""
Neural Attenuation Field (NAF) for Tomographic Reconstruction.

Implements the NAF approach from:
"NAF: Neural Attenuation Fields for Sparse-View CBCT Reconstruction"
https://arxiv.org/abs/2209.14540

This module learns a continuous 3D attenuation field and uses
differentiable ray tracing (Beer-Lambert law) to compute projections.
The forward model is:
    projection(r) = integral_0^L mu(x(t)) dt

where mu is the attenuation coefficient at point x along ray r,
and L is the ray length through the volume.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from .hash_encoding import MultiResolutionHashEncoding
from .model import TinyMLP


@dataclass
class ParallelBeamGeometry:
    """
    Parallel beam geometry parameters for 2D tomography.
    
    For a detector with n_detector pixels at angle theta:
    - Rays are perpendicular to the detector
    - Ray direction: (cos(theta), sin(theta))
    - Detector spans from -det_width/2 to det_width/2
    """
    n_angles: int
    n_detector: int
    det_spacing: float = 1.0  # Detector pixel spacing
    volume_size: Tuple[int, int] = (256, 256)  # (H, W) of the volume
    angle_range: float = 180.0  # Total angle range in degrees
    start_angle: float = 0.0  # Starting angle in degrees
    
    def get_angles_rad(self) -> torch.Tensor:
        """Get projection angles in radians."""
        angles_deg = torch.linspace(
            self.start_angle,
            self.start_angle + self.angle_range * (1 - 1/self.n_angles),
            self.n_angles
        )
        return angles_deg * np.pi / 180.0
    
    def get_angles_deg(self) -> torch.Tensor:
        """Get projection angles in degrees."""
        return torch.linspace(
            self.start_angle,
            self.start_angle + self.angle_range * (1 - 1/self.n_angles),
            self.n_angles
        )


class DifferentiableRayTracer(nn.Module):
    """
    Differentiable ray tracer for parallel beam tomography.
    
    Uses sampling-based integration along rays to compute line integrals
    of the attenuation field. This approach is fully differentiable and
    works well with neural fields.
    
    Coordinate convention (matching scipy.ndimage.rotate):
    - Image coordinates: (row, col) where row=0 is top, col=0 is left
    - Angle 0°: rays are vertical (along column direction), detector spans rows
    - Angle 90°: rays are horizontal (along row direction), detector spans columns
    - Rotation is counter-clockwise
    """
    
    def __init__(
        self,
        volume_size: Tuple[int, int] = (256, 256),
        n_samples_per_ray: int = 256,
        sample_strategy: str = 'uniform',  # 'uniform', 'stratified', 'importance'
    ):
        """
        Args:
            volume_size: (H, W) size of the reconstruction volume
            n_samples_per_ray: Number of sample points along each ray
            sample_strategy: How to sample points along rays
        """
        super().__init__()
        self.volume_size = volume_size
        self.n_samples = n_samples_per_ray
        self.sample_strategy = sample_strategy
        
    def _generate_rays_parallel(
        self,
        angles_rad: torch.Tensor,
        n_detector: int,
        det_spacing: float = 1.0,
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate ray origins and directions for parallel beam geometry.
        
        Uses tomography convention where:
        - angle=0: vertical rays (top to bottom), detector spans horizontally (columns)
        - angle=90: horizontal rays (left to right), detector spans vertically (rows)
        
        Args:
            angles_rad: Projection angles in radians, shape (n_angles,)
            n_detector: Number of detector pixels
            det_spacing: Detector pixel spacing (relative to volume size)
            device: Device to create tensors on
            
        Returns:
            origins: Ray origins, shape (n_angles, n_detector, 2)
            directions: Ray directions, shape (n_angles, n_detector, 2)
        """
        if device is None:
            device = angles_rad.device
            
        n_angles = len(angles_rad)
        H, W = self.volume_size
        
        # Detector positions in normalized coordinates [-0.5, 0.5]
        det_positions = (torch.arange(n_detector, device=device, dtype=torch.float32) - (n_detector - 1) / 2) 
        det_positions = det_positions / n_detector  # Normalize to ~[-0.5, 0.5]
        
        # For tomography convention:
        # At angle=0: rays go downward (positive row direction), detector spans columns
        # Ray direction: (sin(theta), cos(theta)) in (col, row) = (x, y) format
        #   theta=0: direction = (0, 1) - vertical down
        #   theta=90: direction = (1, 0) - horizontal right
        
        cos_theta = torch.cos(angles_rad)  # (n_angles,)
        sin_theta = torch.sin(angles_rad)  # (n_angles,)
        
        # Ray direction in (x, y) coordinates where x=col, y=row (row increases downward)
        # At theta=0: rays point in +y (downward), so direction = (sin(0), cos(0)) = (0, 1)
        directions = torch.stack([sin_theta, cos_theta], dim=-1)  # (n_angles, 2)
        directions = directions.unsqueeze(1).expand(-1, n_detector, -1)  # (n_angles, n_detector, 2)
        
        # Detector normal (perpendicular to ray): (cos(theta), -sin(theta))
        # At theta=0: detector spans x direction (columns), normal = (1, 0)
        det_normal = torch.stack([cos_theta, -sin_theta], dim=-1)  # (n_angles, 2)
        
        # Origins: place rays to start before the volume
        # Center is at (0.5, 0.5),  we offset along detector normal then back along ray direction
        
        # Offset along detector normal based on detector position
        # det_positions: (n_detector,) -> (1, n_detector, 1)
        # det_normal: (n_angles, 2) -> (n_angles, 1, 2)
        offset_along_normal = det_positions.view(1, -1, 1) * det_normal.unsqueeze(1)  # (n_angles, n_detector, 2)
        
        # Start far enough back (1.5 units along negative ray direction from center)
        start_offset = -1.5 * directions  # Start well outside [0,1]^2
        
        # Origin = center + detector offset + start offset
        origins = torch.full((n_angles, n_detector, 2), 0.5, device=device)
        origins = origins + offset_along_normal + start_offset
        
        return origins, directions
    
    def _sample_along_rays(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        n_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Sample points along rays.
        
        Args:
            origins: Ray origins, shape (..., 2)
            directions: Ray directions (normalized), shape (..., 2)
            n_samples: Number of samples per ray
            
        Returns:
            sample_points: Shape (..., n_samples, 2) in [0,1] coordinates
            t_vals: Shape (..., n_samples) - parameter values along ray
            step_size: Physical step size for integration
        """
        device = origins.device
        batch_shape = origins.shape[:-1]
        
        # Rays need to travel ~3 units to fully traverse [0,1]^2 from outside
        # (starting at -1.5, ending at +1.5 relative to center 0.5)
        t_max = 3.0
        
        if self.sample_strategy == 'uniform':
            t_vals = torch.linspace(0, t_max, n_samples, device=device)
            t_vals = t_vals.expand(*batch_shape, -1)
        elif self.sample_strategy == 'stratified':
            # Stratified sampling for better coverage
            bin_size = t_max / n_samples
            t_vals = torch.linspace(0, t_max - bin_size, n_samples, device=device)
            rand = torch.rand(*batch_shape, n_samples, device=device) * bin_size
            t_vals = t_vals.expand(*batch_shape, -1) + rand
        else:
            t_vals = torch.linspace(0, t_max, n_samples, device=device)
            t_vals = t_vals.expand(*batch_shape, -1)
        
        # Compute sample positions: origin + t * direction
        sample_points = origins.unsqueeze(-2) + t_vals.unsqueeze(-1) * directions.unsqueeze(-2)
        
        # Step size in normalized coordinates
        # The actual path length through [0,1]^2 is ~sqrt(2) ≈ 1.41 for diagonal
        # We scale by the volume size to get proper physical units
        step_size = t_max / n_samples
        
        return sample_points, t_vals, step_size
    
    def forward(
        self,
        attenuation_field: nn.Module,
        geometry: ParallelBeamGeometry,
        batch_angles_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute projections by ray tracing through the attenuation field.
        
        Args:
            attenuation_field: Neural network that takes (x, y) coords and returns attenuation
            geometry: Parallel beam geometry parameters
            batch_angles_idx: Optional indices of angles to compute (for batching)
            
        Returns:
            projections: Shape (n_angles, n_detector) - the computed sinogram
        """
        device = next(attenuation_field.parameters()).device
        
        angles_rad = geometry.get_angles_rad().to(device)
        
        if batch_angles_idx is not None:
            angles_rad = angles_rad[batch_angles_idx]
        
        # Generate rays
        origins, directions = self._generate_rays_parallel(
            angles_rad, 
            geometry.n_detector,
            geometry.det_spacing,
            device
        )
        
        # Sample points along rays
        sample_points, t_vals, step_size = self._sample_along_rays(
            origins, directions, self.n_samples
        )
        
        # Reshape for batch processing: (n_angles * n_detector * n_samples, 2)
        n_angles, n_detector, n_samples, _ = sample_points.shape
        points_flat = sample_points.reshape(-1, 2)
        
        # Query attenuation field (only for points inside volume [0,1]^2)
        # Create mask for valid points
        valid_mask = ((points_flat >= 0) & (points_flat <= 1)).all(dim=-1)
        
        # Initialize attenuation values
        attenuations = torch.zeros(points_flat.shape[0], device=device)
        
        # Query only valid points
        if valid_mask.any():
            valid_points = points_flat[valid_mask]
            valid_attenuations = attenuation_field(valid_points).squeeze(-1)
            attenuations[valid_mask] = valid_attenuations
        
        # Reshape back
        attenuations = attenuations.reshape(n_angles, n_detector, n_samples)
        
        # Integrate along rays (numerical integration using midpoint rule)
        # projection = integral(mu(x) dx) ≈ sum(mu(x_i) * step_size)
        projections = attenuations.sum(dim=-1) * step_size
        
        # Flip both axes to match scipy's rotate + sum convention.
        # The scipy.ndimage.rotate followed by sum(axis=0) uses a different
        # detector ordering and angle direction than standard parallel beam
        # tomography. This flip corrects for that difference, ensuring
        # sinograms generated by scipy can be used directly for training.
        # Note: This is a coordinate convention issue, not a bug.
        projections = projections.flip(dims=[0, 1])
        
        return projections


class NeuralAttenuationField(nn.Module):
    """
    Neural Attenuation Field for 2D CT reconstruction.
    
    Combines a multi-resolution hash encoding with an MLP to represent
    the attenuation coefficient at any continuous 2D position.
    Uses differentiable ray tracing to compute projections matching
    the measured sinogram data.
    
    Coordinate System:
        The field uses normalized [0, 1] coordinates where:
        - x corresponds to columns (left=0, right=1)
        - y corresponds to rows (top=0, bottom=1)
        
    Training Notes:
        NAF typically needs 500-2000 epochs to converge well, depending
        on image complexity and network size. Use at least 90-180 angles
        in the sinogram for good reconstruction quality.
        
    Example:
        >>> geometry = ParallelBeamGeometry(n_angles=180, n_detector=256)
        >>> model = NeuralAttenuationField(n_dims=2, n_levels=12).cuda()
        >>> trainer = NAFTrainer(model, geometry, learning_rate=1e-3)
        >>> trainer.train(sinogram_tensor, num_epochs=1000)
        >>> volume = model.predict_volume((256, 256))
    """
    
    def __init__(
        self,
        n_dims: int = 2,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        base_resolution: int = 16,
        max_resolution: int = 512,
        table_size: int = 2**19,
        hidden_dims: List[int] = [64, 64],
        activation: str = 'relu',
        output_activation: Optional[str] = 'softplus',  # Attenuation must be >= 0
        include_input: bool = True,
        n_samples_per_ray: int = 256,
    ):
        """
        Args:
            n_dims: Number of spatial dimensions (2 for 2D CT)
            n_levels: Number of hash encoding levels
            n_features_per_level: Features per level
            base_resolution: Coarsest resolution
            max_resolution: Finest resolution  
            table_size: Hash table size per level
            hidden_dims: MLP hidden layer dimensions
            activation: MLP activation function
            output_activation: Output activation ('softplus' ensures positive values)
            include_input: Whether to include raw coords in encoding
            n_samples_per_ray: Number of samples along each ray for integration
        """
        super().__init__()
        
        self.n_dims = n_dims
        
        # Hash encoding for position
        self.encoding = MultiResolutionHashEncoding(
            n_dims=n_dims,
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            base_resolution=base_resolution,
            max_resolution=max_resolution,
            table_size=table_size,
            include_input=include_input,
        )
        
        # MLP decoder (output is attenuation coefficient)
        # Using softplus activation to ensure non-negative output
        self.mlp = TinyMLP(
            input_dim=self.encoding.output_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=activation,
            output_activation=None,  # We apply softplus separately for stability
        )
        
        self.output_activation = output_activation
        
        # Ray tracer
        self.ray_tracer = DifferentiableRayTracer(
            n_samples_per_ray=n_samples_per_ray,
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
            'activation': activation,
            'output_activation': output_activation,
            'include_input': include_input,
            'n_samples_per_ray': n_samples_per_ray,
        }
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Evaluate attenuation at given coordinates.
        
        Args:
            coords: Coordinates in [0, 1], shape (..., n_dims)
            
        Returns:
            Attenuation values, shape (..., 1)
        """
        features = self.encoding(coords)
        output = self.mlp(features)
        
        # Apply output activation (softplus for non-negative attenuation)
        if self.output_activation == 'softplus':
            output = F.softplus(output)
        elif self.output_activation == 'relu':
            output = F.relu(output)
        elif self.output_activation == 'sigmoid':
            output = torch.sigmoid(output)
        elif self.output_activation == 'exp':
            output = torch.exp(output.clamp(-10, 10))  # Clamp for stability
            
        return output
    
    def compute_projections(
        self,
        geometry: ParallelBeamGeometry,
        batch_angles_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute projections (sinogram) using ray tracing.
        
        Args:
            geometry: Parallel beam geometry parameters
            batch_angles_idx: Optional indices for batched computation
            
        Returns:
            projections: Shape (n_angles, n_detector)
        """
        # Update ray tracer volume size
        self.ray_tracer.volume_size = geometry.volume_size
        
        return self.ray_tracer(self, geometry, batch_angles_idx)
    
    def predict_volume(
        self,
        shape: Tuple,
        batch_size: int = 65536,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Predict the attenuation volume.

        Args:
            shape: (H, W) for 2D, or (D, H, W) for 3D
            batch_size: Points processed per forward pass
            device: Target device (defaults to model device)

        Returns:
            2D tensor (H, W) or 3D tensor (D, H, W)
        """
        if device is None:
            device = next(self.parameters()).device

        if len(shape) == 2:
            H, W = shape
            y_coords = torch.linspace(0, 1, H, device=device)
            x_coords = torch.linspace(0, 1, W, device=device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        elif len(shape) == 3:
            D, H, W = shape
            z_coords = torch.linspace(0, 1, D, device=device)
            y_coords = torch.linspace(0, 1, H, device=device)
            x_coords = torch.linspace(0, 1, W, device=device)
            zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
            coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        else:
            raise ValueError(f"shape must be 2D (H,W) or 3D (D,H,W), got {shape}")

        outputs = []
        with torch.no_grad():
            for i in range(0, coords.shape[0], batch_size):
                batch_coords = coords[i : i + batch_size]
                outputs.append(self.forward(batch_coords))

        output = torch.cat(outputs, dim=0)
        return output.reshape(*shape)

    def get_model_info(self) -> str:
        """Return string with model information."""
        n_params = sum(p.numel() for p in self.parameters())
        n_hash_params = self.encoding.n_params
        n_mlp_params = n_params - n_hash_params
        
        info = [
            f"NeuralAttenuationField:",
            f"  Input dims: {self.n_dims}",
            f"  Total parameters: {n_params:,}",
            f"  Hash table params: {n_hash_params:,} ({100*n_hash_params/n_params:.1f}%)",
            f"  MLP params: {n_mlp_params:,} ({100*n_mlp_params/n_params:.1f}%)",
            f"  Encoding: {self.encoding.n_levels} levels, {self.encoding.n_features_per_level} features/level",
            f"  Ray samples: {self.ray_tracer.n_samples}",
            f"  {self.encoding.get_resolution_info()}",
        ]
        return "\n".join(info)
    
    @classmethod
    def from_config(cls, config: dict) -> 'NeuralAttenuationField':
        """Create model from config dictionary."""
        return cls(**config)
    
    def save(self, path: str):
        """Save model to file."""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict(),
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'NeuralAttenuationField':
        """Load model from file."""
        checkpoint = torch.load(path, map_location=device)
        model = cls.from_config(checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model


class NAFTrainer:
    """
    Trainer for Neural Attenuation Field.
    
    Optimizes the NAF to match measured projection data (sinogram).
    """
    
    def __init__(
        self,
        model: NeuralAttenuationField,
        geometry: ParallelBeamGeometry,
        learning_rate: float = 1e-3,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: The NAF model to train
            geometry: Parallel beam geometry
            learning_rate: Learning rate for optimizer
            device: Device to train on
        """
        self.model = model
        self.geometry = geometry
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = None
        
        self.loss_history = []
    
    def setup_scheduler(self, num_epochs: int, warmup_epochs: int = 5):
        """Setup learning rate scheduler with warmup."""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        
        warmup_scheduler = LinearLR(
            self.optimizer, 
            start_factor=0.01, 
            total_iters=warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=num_epochs - warmup_epochs
        )
        self.scheduler = SequentialLR(
            self.optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
    
    def train_step(
        self,
        target_projections: torch.Tensor,
        batch_angle_indices: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Single training step.
        
        Args:
            target_projections: Target sinogram, shape (n_angles, n_detector) or (batch_angles, n_detector)
            batch_angle_indices: If provided, only compute projections for these angles
            
        Returns:
            loss: The computed loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        target = target_projections.to(self.device)
        
        # Compute predicted projections
        pred_projections = self.model.compute_projections(
            self.geometry,
            batch_angles_idx=batch_angle_indices
        )
        
        # Match dimensions if batching angles
        if batch_angle_indices is not None:
            target = target[batch_angle_indices]
        
        # MSE loss
        loss = F.mse_loss(pred_projections, target)
        
        # Backprop
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(
        self,
        target_projections: torch.Tensor,
        num_epochs: int = 500,
        batch_angles: Optional[int] = None,
        verbose: bool = True,
        log_interval: int = 10,
    ) -> Dict:
        """
        Train the NAF model.
        
        Args:
            target_projections: Target sinogram, shape (n_angles, n_detector)
            num_epochs: Number of training epochs. NAF typically needs 500-2000 epochs
                       to converge well, depending on image complexity and network size.
            batch_angles: If set, randomly sample this many angles per step
            verbose: Print progress
            log_interval: Steps between log messages
            
        Returns:
            Dictionary with training history
        
        Note:
            For best results, use at least 90-180 angles in the sinogram and train
            for 500+ epochs. Smaller networks may need longer training.
        """
        from tqdm import tqdm
        
        target = target_projections.to(self.device)
        self.setup_scheduler(num_epochs)
        
        n_angles = target.shape[0]
        
        pbar = tqdm(range(num_epochs), disable=not verbose, desc="Training NAF")
        
        for epoch in pbar:
            # Sample random angles if batching
            if batch_angles is not None and batch_angles < n_angles:
                batch_idx = torch.randperm(n_angles)[:batch_angles].to(self.device)
            else:
                batch_idx = None
            
            # Training step
            loss = self.train_step(target, batch_idx)
            self.loss_history.append(loss)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Logging
            if verbose and (epoch + 1) % log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({'loss': f'{loss:.6f}', 'lr': f'{lr:.2e}'})
        
        return {
            'loss_history': self.loss_history,
            'final_loss': self.loss_history[-1] if self.loss_history else None,
        }
    
    def evaluate(
        self,
        target_projections: torch.Tensor,
    ) -> Dict:
        """
        Evaluate the model on target projections.
        
        Args:
            target_projections: Target sinogram
            
        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        target = target_projections.to(self.device)
        
        with torch.no_grad():
            pred_projections = self.model.compute_projections(self.geometry)
            mse = F.mse_loss(pred_projections, target).item()
            
            # PSNR
            max_val = target.max().item() - target.min().item()
            psnr = 10 * np.log10((max_val ** 2) / mse) if mse > 0 else float('inf')
        
        return {
            'mse': mse,
            'psnr': psnr,
        }


@dataclass
class ParallelBeamGeometry3D:
    """
    3D parallel beam geometry for volumetric CT reconstruction.

    The scanner rotates around the Z axis.  Each 2D projection image
    (indexed by angle) has:
      - U axis (columns): perpendicular to the ray direction in XY
      - V axis (rows):    along the Z / height axis

    Volume coordinates are normalized to [0, 1]^3 = (x, y, z).
    """
    n_angles: int
    n_detector_u: int   # detector width  (horizontal, XY-plane)
    n_detector_v: int   # detector height (vertical, Z-axis)
    volume_size: Tuple[int, int, int]   # (D, H, W) = (Z, Y, X) voxels
    angle_range: float = 180.0          # total angular span in degrees
    start_angle: float = 0.0

    def get_angles_rad(self) -> torch.Tensor:
        angles_deg = torch.linspace(
            self.start_angle,
            self.start_angle + self.angle_range * (1 - 1 / self.n_angles),
            self.n_angles,
        )
        return angles_deg * np.pi / 180.0

    def get_angles_deg(self) -> torch.Tensor:
        return torch.linspace(
            self.start_angle,
            self.start_angle + self.angle_range * (1 - 1 / self.n_angles),
            self.n_angles,
        )


class DifferentiableRayTracer3D(nn.Module):
    """
    Differentiable 3D ray tracer for parallel beam tomography.

    The scanner rotates around the Z axis.  At angle θ:
      - Ray direction : (sin θ, cos θ, 0)  — same XY convention as the 2D tracer
      - Detector U axis: (cos θ, −sin θ, 0) — horizontal, perpendicular to ray
      - Detector V axis: (0, 0, 1)          — vertical, along Z

    Each ray at (angle θ, detector column u, detector row v) travels at a
    constant z = v / (n_detector_v − 1), integrating attenuation in XY only.

    The output projection array is (n_angles, n_detector_v, n_detector_u) and
    is flipped on the angle and U axes to match the scipy.ndimage.rotate /
    column-sum convention used by the 2D counterpart.
    """

    def __init__(
        self,
        volume_size: Tuple[int, int, int] = (256, 256, 256),
        n_samples_per_ray: int = 256,
    ):
        super().__init__()
        self.volume_size = volume_size
        self.n_samples = n_samples_per_ray

    def _generate_rays_3d(
        self,
        angles_rad: torch.Tensor,     # (n_angles,)
        row_indices: torch.Tensor,    # (n_rows,) - which detector V rows
        n_detector_u: int,
        n_detector_v: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build ray origins and directions.

        Returns:
            origins:    (n_angles, n_rows, n_detector_u, 3)
            directions: (n_angles, n_rows, n_detector_u, 3)
        """
        n_angles = len(angles_rad)
        n_rows = len(row_indices)

        sin_t = torch.sin(angles_rad)   # (n_angles,)
        cos_t = torch.cos(angles_rad)   # (n_angles,)

        # Ray direction in (x, y, z) — no z component for rotation around Z
        # shape → (n_angles, 1, 1, 3) for broadcasting
        ray_dir = torch.stack([sin_t, cos_t, torch.zeros_like(sin_t)], dim=-1)
        ray_dir = ray_dir.view(n_angles, 1, 1, 3)
        directions = ray_dir.expand(n_angles, n_rows, n_detector_u, 3)

        # Detector U axis: (cos θ, -sin θ, 0)  — (n_angles, 1, 1, 3)
        det_u_vec = torch.stack([cos_t, -sin_t, torch.zeros_like(cos_t)], dim=-1)
        det_u_vec = det_u_vec.view(n_angles, 1, 1, 3)

        # U detector positions: (n_detector_u,) in ~[−0.5, 0.5]
        u_pos = (torch.arange(n_detector_u, device=device, dtype=torch.float32)
                 - (n_detector_u - 1) / 2.0) / n_detector_u
        u_pos = u_pos.view(1, 1, n_detector_u, 1)          # (1,1,n_u,1)

        # U offset in world space: (n_angles, 1, n_detector_u, 3)
        offset_u = u_pos * det_u_vec

        # V (Z) positions: row_indices map directly to normalized z in [0, 1]
        v_z = row_indices.float() / (n_detector_v - 1)     # (n_rows,)
        v_z = v_z.view(1, n_rows, 1)                        # (1, n_rows, 1)

        # Build origins: center (0.5, 0.5, z) + U-offset − 1.5 * direction
        origins = torch.zeros(n_angles, n_rows, n_detector_u, 3, device=device)
        origins[..., 0] = 0.5
        origins[..., 1] = 0.5
        origins[..., 2] = v_z          # broadcast to (n_angles, n_rows, n_u)
        origins = origins + offset_u - 1.5 * ray_dir

        return origins, directions

    def forward(
        self,
        attenuation_field: nn.Module,
        geometry: ParallelBeamGeometry3D,
        batch_angle_indices: Optional[torch.Tensor] = None,
        batch_row_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute 2D projections through the 3D field.

        Args:
            attenuation_field: 3D NAF field (n_dims=3), takes coords (..., 3)
            geometry:          3D geometry
            batch_angle_indices: subset of angle indices (for training batches)
            batch_row_indices:   subset of detector-row indices (for training batches)

        Returns:
            projections: (n_angles_batch, n_rows_batch, n_detector_u)
        """
        device = next(attenuation_field.parameters()).device

        angles_rad = geometry.get_angles_rad().to(device)
        if batch_angle_indices is not None:
            angles_rad = angles_rad[batch_angle_indices]

        if batch_row_indices is not None:
            row_indices = batch_row_indices.to(device)
        else:
            row_indices = torch.arange(geometry.n_detector_v, device=device)

        origins, directions = self._generate_rays_3d(
            angles_rad, row_indices,
            geometry.n_detector_u, geometry.n_detector_v, device
        )
        # shapes: (n_a, n_r, n_u, 3)

        t_max = 3.0
        t_vals = torch.linspace(0, t_max, self.n_samples, device=device)

        # sample_points: (n_a, n_r, n_u, n_samples, 3)
        sample_points = (origins.unsqueeze(-2) +
                         t_vals.view(1, 1, 1, -1, 1) * directions.unsqueeze(-2))

        step_size = t_max / self.n_samples

        n_a, n_r, n_u, n_s, _ = sample_points.shape
        points_flat = sample_points.reshape(-1, 3)

        valid_mask = ((points_flat >= 0) & (points_flat <= 1)).all(dim=-1)
        attenuations = torch.zeros(points_flat.shape[0], device=device)

        if valid_mask.any():
            valid_atten = attenuation_field(points_flat[valid_mask]).squeeze(-1)
            attenuations[valid_mask] = valid_atten

        attenuations = attenuations.reshape(n_a, n_r, n_u, n_s)
        projections = attenuations.sum(dim=-1) * step_size   # (n_a, n_r, n_u)

        # Flip angle and U-detector axes to match scipy convention (same as 2D).
        # The Z / V axis is NOT flipped — z maps directly to z.
        projections = projections.flip(dims=[0, 2])

        return projections


class NAFTrainer3D:
    """
    Trainer for a 3D Neural Attenuation Field.

    Trains by sampling random batches of (angle, detector-row) pairs per step
    to keep memory usage constant regardless of volume size.

    Args:
        model:          NeuralAttenuationField with n_dims=3
        geometry:       ParallelBeamGeometry3D
        learning_rate:  Adam learning rate (default 1e-3)
        device:         torch device
        batch_angles:   angles to sample per step (default 8)
        batch_rows:     detector rows to sample per step (default 16)
    """

    def __init__(
        self,
        model: NeuralAttenuationField,
        geometry: ParallelBeamGeometry3D,
        learning_rate: float = 1e-3,
        device: Optional[torch.device] = None,
        batch_angles: int = 8,
        batch_rows: int = 16,
    ):
        self.model = model
        self.geometry = geometry
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_angles = batch_angles
        self.batch_rows = batch_rows

        self.model = self.model.to(self.device)

        # Replace the 2D ray tracer with the 3D one
        self.ray_tracer = DifferentiableRayTracer3D(
            volume_size=geometry.volume_size,
            n_samples_per_ray=model.ray_tracer.n_samples,
        )

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler: Optional[object] = None
        self.loss_history: List[float] = []

    def setup_scheduler(self, num_epochs: int, warmup_epochs: int = 5):
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs - warmup_epochs)
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

    def train_step(self, target_projections: torch.Tensor) -> float:
        """
        Single training step with random batch of angles and rows.

        Args:
            target_projections: shape (n_angles, n_detector_v, n_detector_u)

        Returns:
            scalar loss value
        """
        self.model.train()
        self.optimizer.zero_grad()

        n_angles, n_det_v, n_det_u = target_projections.shape
        target = target_projections.to(self.device)

        # Sample random angles and rows
        angle_idx = torch.randperm(n_angles, device=self.device)[:self.batch_angles]
        row_idx   = torch.randperm(n_det_v,  device=self.device)[:self.batch_rows]

        # Compute predicted projections for this batch
        pred = self.ray_tracer(
            self.model, self.geometry,
            batch_angle_indices=angle_idx,
            batch_row_indices=row_idx,
        )   # (batch_angles, batch_rows, n_det_u)

        # Gather matching target values
        tgt = target[angle_idx][:, row_idx, :]   # (batch_angles, batch_rows, n_det_u)

        loss = F.mse_loss(pred, tgt)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(
        self,
        target_projections: torch.Tensor,
        num_epochs: int = 500,
        verbose: bool = True,
        log_interval: int = 10,
    ) -> Dict:
        """
        Train the 3D NAF model.

        Args:
            target_projections: shape (n_angles, n_detector_v, n_detector_u)
            num_epochs:         training epochs (500–2000 recommended)
            verbose:            show tqdm progress bar
            log_interval:       epochs between log updates

        Returns:
            dict with 'loss_history' and 'final_loss'
        """
        from tqdm import tqdm

        self.setup_scheduler(num_epochs)
        pbar = tqdm(range(num_epochs), disable=not verbose, desc='Training 3D NAF')

        for epoch in pbar:
            loss = self.train_step(target_projections)
            self.loss_history.append(loss)

            if self.scheduler is not None:
                self.scheduler.step()

            if verbose and (epoch + 1) % log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({'loss': f'{loss:.6f}', 'lr': f'{lr:.2e}'})

        return {
            'loss_history': self.loss_history,
            'final_loss': self.loss_history[-1] if self.loss_history else None,
        }

    def evaluate(self, target_projections: torch.Tensor) -> Dict:
        """
        Evaluate sinogram reconstruction quality.

        Args:
            target_projections: shape (n_angles, n_det_v, n_det_u)

        Returns:
            dict with 'mse' and 'psnr'
        """
        self.model.eval()
        target = target_projections.to(self.device)

        # Evaluate in chunks over rows to avoid OOM
        n_angles, n_det_v, n_det_u = target.shape
        all_pred = []

        row_chunk = 32
        with torch.no_grad():
            for angle_start in range(0, n_angles, 8):
                angle_batch = torch.arange(
                    angle_start, min(angle_start + 8, n_angles), device=self.device
                )
                row_preds = []
                for row_start in range(0, n_det_v, row_chunk):
                    row_batch = torch.arange(
                        row_start, min(row_start + row_chunk, n_det_v), device=self.device
                    )
                    pred = self.ray_tracer(
                        self.model, self.geometry,
                        batch_angle_indices=angle_batch,
                        batch_row_indices=row_batch,
                    )
                    row_preds.append(pred.cpu())
                all_pred.append(torch.cat(row_preds, dim=1))

        pred_all = torch.cat(all_pred, dim=0)   # (n_angles, n_det_v, n_det_u)
        mse_val = F.mse_loss(pred_all, target.cpu()).item()
        max_val = float(target.max() - target.min())
        psnr_val = 10 * np.log10(max_val ** 2 / mse_val) if mse_val > 0 else float('inf')

        return {'mse': mse_val, 'psnr': psnr_val}


def create_sinogram_dataset(
    projections: torch.Tensor,
    n_projections: int,
    angle_range: float = 180.0,
) -> Tuple[torch.Tensor, ParallelBeamGeometry]:
    """
    Create a sinogram dataset from stacked projection images.
    
    This assumes projections are evenly spaced within the angle_range.
    
    Args:
        projections: Stack of projections, shape (H, W, n_projections) or (n_projections, H, W)
        n_projections: Number of projections to use
        angle_range: Total angle range in degrees (default 180)
        
    Returns:
        sinogram: Sinogram tensor, shape (n_angles, n_detector)
        geometry: Parallel beam geometry object
    """
    # Handle different input formats
    if projections.ndim == 3:
        if projections.shape[-1] <= projections.shape[0]:
            # Shape is (H, W, n_proj) - extract middle row as sinogram
            H, W, N = projections.shape
            N = min(N, n_projections)
            # For 2D reconstruction from projections, we typically use one row
            # The sinogram is (angles, detector_positions)
            mid_row = H // 2
            sinogram = projections[mid_row, :, :N].T  # (N, W) = (angles, detector)
            volume_size = (W, W)  # Assume square volume for reconstruction
            n_detector = W
        else:
            # Shape is (n_proj, H, W)
            N, H, W = projections.shape
            N = min(N, n_projections)
            mid_row = H // 2
            sinogram = projections[:N, mid_row, :]  # (N, W) = (angles, detector)
            volume_size = (W, W)
            n_detector = W
    else:
        raise ValueError(f"Expected 3D tensor, got shape {projections.shape}")
    
    geometry = ParallelBeamGeometry(
        n_angles=sinogram.shape[0],
        n_detector=n_detector,
        det_spacing=1.0,
        volume_size=volume_size,
        angle_range=angle_range,
        start_angle=0.0,
    )
    
    return sinogram, geometry
