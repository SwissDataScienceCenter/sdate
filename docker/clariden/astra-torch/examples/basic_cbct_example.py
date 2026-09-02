#!/usr/bin/env python3
"""
Basic CBCT Reconstruction Example
=================================

This example demonstrates basic cone-beam CT reconstruction using the FDK algorithm.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from astra_torch import fdk_reconstruction_masked


def create_circular_cbct_geometry(num_angles=180, source_distance=1000, detector_distance=500):
    """Create a circular CBCT acquisition geometry.
    
    Args:
        num_angles: Number of projection angles
        source_distance: Distance from origin to source
        detector_distance: Distance from origin to detector
        
    Returns:
        vectors: ASTRA geometry vectors (num_angles, 12)
    """
    angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    
    # Source positions
    source_positions = np.column_stack([
        source_distance * np.cos(angles),
        source_distance * np.sin(angles), 
        np.zeros(num_angles)
    ])
    
    # Detector positions
    detector_positions = np.column_stack([
        -detector_distance * np.cos(angles),
        -detector_distance * np.sin(angles),
        np.zeros(num_angles)
    ])
    
    # Detector u vectors (horizontal)
    u_vectors = np.column_stack([
        -np.sin(angles),
        np.cos(angles),
        np.zeros(num_angles)
    ])
    
    # Detector v vectors (vertical)
    v_vectors = np.tile([0, 0, 1], (num_angles, 1))
    
    # Combine into ASTRA vectors
    vectors = np.column_stack([
        source_positions,                          # Source position
        detector_positions - source_positions,     # Ray direction  
        u_vectors,                                 # Detector u
        v_vectors                                  # Detector v
    ])
    
    return vectors


def create_phantom(shape=(64, 64, 64)):
    """Create a simple 3D phantom for testing.
    
    Args:
        shape: Volume dimensions
        
    Returns:
        phantom: 3D numpy array
    """
    phantom = np.zeros(shape)
    
    # Add a sphere in the center
    x, y, z = np.mgrid[:shape[0], :shape[1], :shape[2]]
    center = np.array(shape) // 2
    
    # Central sphere
    sphere1 = ((x - center[0])**2 + 
               (y - center[1])**2 + 
               (z - center[2])**2) <= (shape[0]//6)**2
    phantom[sphere1] = 1.0
    
    # Smaller offset sphere
    offset = shape[0] // 4
    sphere2 = ((x - center[0] + offset)**2 + 
               (y - center[1])**2 + 
               (z - center[2])**2) <= (shape[0]//12)**2
    phantom[sphere2] = 0.5
    
    return phantom


def simulate_projections(phantom, vectors):
    """Simulate CBCT projections from a phantom.
    
    This is a simplified simulation - in practice you would use
    the forward projection operator from astra_torch.
    
    Args:
        phantom: 3D volume
        vectors: ASTRA geometry vectors
        
    Returns:
        projections: Simulated projection data
    """
    # For this example, create synthetic noisy projections
    num_angles = vectors.shape[0]
    detector_shape = (256, 256)  # Assuming square detector
    
    # Simple simulation based on phantom statistics
    mean_projection = np.sum(phantom) / (num_angles * 10)
    
    projections = np.random.poisson(
        mean_projection * 1000, 
        size=(num_angles,) + detector_shape
    ).astype(np.float32) / 1000.0
    
    # Add some structure
    for i in range(num_angles):
        angle = i * 2 * np.pi / num_angles
        projections[i] *= (1 + 0.3 * np.sin(angle))
    
    return projections


def main():
    """Main reconstruction example."""
    print("CBCT Reconstruction Example")
    print("=" * 40)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create geometry
    print("Creating acquisition geometry...")
    num_angles = 180
    vectors = create_circular_cbct_geometry(num_angles)
    print(f"Created geometry with {num_angles} angles")
    
    # Create phantom
    print("Creating phantom...")
    phantom = create_phantom((64, 64, 64))
    print(f"Phantom shape: {phantom.shape}")
    
    # Simulate projections
    print("Simulating projections...")
    projections = simulate_projections(phantom, vectors)
    print(f"Projection data shape: {projections.shape}")
    
    # Convert to PyTorch tensors with proper batch dimensions
    projections_torch = torch.from_numpy(projections).float()
    projections_torch = projections_torch.unsqueeze(0).unsqueeze(0)  # (batch, channel, angles, h, w)
    projections_torch = projections_torch.to(device)
    
    print(f"PyTorch projections shape: {projections_torch.shape}")
    
    # Perform FDK reconstruction
    print("Performing FDK reconstruction...")
    try:
        volume = fdk_reconstruction_masked(
            projections_torch,
            vectors,
            voxel_per_mm=2.0,
            volume_shape=(64, 64, 64)
        )
        print(f"Reconstruction successful! Volume shape: {volume.shape}")
        
        # Convert back to numpy for visualization
        volume_np = volume.squeeze().cpu().numpy()
        
        # Display results
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # Original phantom slices
        mid_slice = phantom.shape[2] // 2
        axes[0, 0].imshow(phantom[:, :, mid_slice], cmap='gray')
        axes[0, 0].set_title('Original Phantom (Z slice)')
        axes[0, 1].imshow(phantom[:, mid_slice, :], cmap='gray') 
        axes[0, 1].set_title('Original Phantom (Y slice)')
        axes[0, 2].imshow(phantom[mid_slice, :, :], cmap='gray')
        axes[0, 2].set_title('Original Phantom (X slice)')
        
        # Reconstructed volume slices
        axes[1, 0].imshow(volume_np[:, :, mid_slice], cmap='gray')
        axes[1, 0].set_title('Reconstructed (Z slice)')
        axes[1, 1].imshow(volume_np[:, mid_slice, :], cmap='gray')
        axes[1, 1].set_title('Reconstructed (Y slice)') 
        axes[1, 2].imshow(volume_np[mid_slice, :, :], cmap='gray')
        axes[1, 2].set_title('Reconstructed (X slice)')
        
        for ax in axes.flat:
            ax.axis('off')
            
        plt.tight_layout()
        plt.savefig('cbct_reconstruction_example.png', dpi=150, bbox_inches='tight')
        print("Results saved to 'cbct_reconstruction_example.png'")
        plt.show()
        
    except Exception as e:
        print(f"Reconstruction failed: {e}")
        print("This might be due to missing ASTRA toolbox installation.")
        print("Please install ASTRA: conda install -c astra-toolbox astra-toolbox")


if __name__ == "__main__":
    main()
