#!/usr/bin/env python3
"""
Laminography Reconstruction Example
===================================

This example demonstrates laminography reconstruction using filtered backprojection.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from astra_torch import lamino_fbp_reconstruction_masked


def create_layered_phantom(shape=(64, 64, 64)):
    """Create a layered phantom suitable for laminography.
    
    Laminography is particularly useful for thin, layered samples,
    so we create a phantom with multiple layers and features.
    
    Args:
        shape: Volume dimensions
        
    Returns:
        phantom: 3D numpy array
    """
    phantom = np.zeros(shape)
    
    # Add several layers with different structures
    z_center = shape[2] // 2
    layer_thickness = shape[2] // 8
    
    # Bottom layer - grid pattern
    z_start = z_center - 2 * layer_thickness
    z_end = z_center - layer_thickness
    for z in range(max(0, z_start), min(shape[2], z_end)):
        # Grid pattern
        phantom[::8, :, z] = 0.6
        phantom[:, ::8, z] = 0.6
    
    # Middle layer - circular features  
    x, y, z = np.mgrid[:shape[0], :shape[1], :shape[2]]
    z_start = z_center - layer_thickness//2
    z_end = z_center + layer_thickness//2
    
    layer_mask = (z >= z_start) & (z <= z_end)
    
    # Add circular features in middle layer
    centers = [(shape[0]//3, shape[1]//3), 
               (2*shape[0]//3, shape[1]//3),
               (shape[0]//2, 2*shape[1]//3)]
    
    for cx, cy in centers:
        circle = ((x - cx)**2 + (y - cy)**2) <= (shape[0]//10)**2
        phantom[circle & layer_mask] = 0.8
    
    # Top layer - line features
    z_start = z_center + layer_thickness
    z_end = z_center + 2 * layer_thickness
    for z in range(max(0, z_start), min(shape[2], z_end)):
        # Diagonal lines
        for i in range(0, shape[0], 6):
            if i + 2 < shape[0]:
                phantom[i:i+2, :, z] = 0.4
        for j in range(0, shape[1], 6):
            if j + 2 < shape[1]:
                phantom[:, j:j+2, z] = 0.4
    
    return phantom


def simulate_laminography_projections(phantom, num_angles):
    """Simulate laminography projections.
    
    Args:
        phantom: 3D volume
        num_angles: Number of projection angles
        
    Returns:
        projections: Simulated projection data
    """
    detector_shape = (256, 256)
    
    # Simple projection simulation
    projections = np.zeros((num_angles,) + detector_shape)
    
    for i in range(num_angles):
        # Simple line integral approximation
        # In practice, this would use proper ray tracing
        projection_sum = np.sum(phantom)
        
        # Add some angular variation and noise
        angle_factor = 1 + 0.2 * np.sin(2 * np.pi * i / num_angles)
        base_intensity = projection_sum * angle_factor / (num_angles * 10)
        
        # Generate Poisson noise
        projections[i] = np.random.poisson(
            base_intensity * 1000, 
            size=detector_shape
        ).astype(np.float32) / 1000.0
        
        # Add some detector-specific patterns
        u, v = np.meshgrid(np.linspace(-1, 1, detector_shape[1]),
                          np.linspace(-1, 1, detector_shape[0]))
        projections[i] *= (1 + 0.1 * np.sin(5 * u) * np.cos(3 * v))
    
    return projections


def main():
    """Main laminography example."""
    print("Laminography Reconstruction Example")
    print("=" * 40)
    
    # Check device
    # Note: For ASTRA compatibility, CPU mode is more stable
    # CUDA direct linking may cause memory access issues with some ASTRA versions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create laminography geometry
    print("Creating laminography geometry...")
    num_angles = 360
    tilt_angle = np.pi/6  # 30 degrees (lamino angle)
    angles_deg = np.linspace(0, 180, num_angles, endpoint=False)
    print(f"Created geometry with {num_angles} angles, lamino tilt = {np.degrees(tilt_angle):.1f}°")
    
    # Create layered phantom
    print("Creating layered phantom...")
    phantom = create_layered_phantom((64, 64, 64))
    print(f"Phantom shape: {phantom.shape}")
    
    # Simulate projections
    print("Simulating laminography projections...")
    projections = simulate_laminography_projections(phantom, num_angles)
    print(f"Projection data shape: {projections.shape}")
    
    # Convert to PyTorch tensors (V, R, C) format
    projections_torch = torch.from_numpy(projections).float()  # Shape: (180, 256, 256)
    projections_torch = projections_torch.to(device)
    
    # Perform FBP reconstruction
    print("Performing laminography FBP reconstruction...")
    try:
        volume = lamino_fbp_reconstruction_masked(
            projections_torch,
            angles_deg,
            lamino_angle_deg=np.degrees(tilt_angle),
            voxel_per_mm=2.0,
            vol_shape=(64, 64, 64),
        )
        print(f"Reconstruction successful! Volume shape: {volume.shape}")
        
        # Convert to numpy for visualization
        volume_np = volume.squeeze().cpu().numpy()
        
        # Visualize results
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        # Show different Z slices of original phantom
        z_slices = [16, 28, 36, 48]
        for i, z_idx in enumerate(z_slices):
            axes[0, i].imshow(phantom[:, :, z_idx], cmap='gray')
            axes[0, i].set_title(f'Original Z={z_idx}')
            axes[0, i].axis('off')
        
        # Show same Z slices of reconstruction
        for i, z_idx in enumerate(z_slices):
            axes[1, i].imshow(volume_np[:, :, z_idx], cmap='gray')
            axes[1, i].set_title(f'Reconstructed Z={z_idx}')
            axes[1, i].axis('off')
        
        # Show some projections
        proj_indices = [0, num_angles//4, num_angles//2, 3*num_angles//4]
        for i, proj_idx in enumerate(proj_indices):
            axes[2, i].imshow(projections[proj_idx], cmap='gray')
            axes[2, i].set_title(f'Projection {proj_idx}')
            axes[2, i].axis('off')
        
        plt.suptitle('Laminography Reconstruction Results', fontsize=14)
        plt.tight_layout()
        plt.savefig('laminography_example.png', dpi=150, bbox_inches='tight')
        print("Results saved to 'laminography_example.png'")
        plt.show()
        
        # Calculate and display some metrics
        print("\nReconstruction Quality Metrics:")
        print(f"Original phantom range: [{phantom.min():.3f}, {phantom.max():.3f}]")
        print(f"Reconstructed range: [{volume_np.min():.3f}, {volume_np.max():.3f}]")
        
        # Focus on central region for comparison
        center_slice = slice(16, 48)
        phantom_center = phantom[center_slice, center_slice, center_slice]
        volume_center = volume_np[center_slice, center_slice, center_slice]
        
        mse = np.mean((phantom_center - volume_center)**2)
        print(f"MSE (center region): {mse:.6f}")
        
    except Exception as e:
        print(f"Reconstruction failed: {e}")
        print("This might be due to missing ASTRA toolbox installation.")
        print("Please install ASTRA: conda install -c astra-toolbox astra-toolbox")


if __name__ == "__main__":
    main()
