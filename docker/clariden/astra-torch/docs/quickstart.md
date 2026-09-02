# Quick Start

This guide will get you up and running with ASTRA-Torch for tomographic reconstruction.

## Basic CBCT Reconstruction

```python
import torch
import numpy as np
from astra_torch import fdk_reconstruction_masked, CBCTAcquisition

# Create acquisition geometry
num_angles = 180
angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)

# Define source and detector positions (circular orbit)
source_positions = np.column_stack([
    1000 * np.cos(angles),  # X positions  
    1000 * np.sin(angles),  # Y positions
    np.zeros(num_angles)    # Z positions
])

detector_positions = np.column_stack([
    -500 * np.cos(angles),  # X positions
    -500 * np.sin(angles),  # Y positions  
    np.zeros(num_angles)    # Z positions
])

# Detector orientation vectors
u_vectors = np.column_stack([
    -np.sin(angles),        # X components
    np.cos(angles),         # Y components
    np.zeros(num_angles)    # Z components
])

v_vectors = np.tile([0, 0, 1], (num_angles, 1))  # Vertical detector

# Combine into ASTRA geometry vectors
vectors = np.column_stack([
    source_positions,
    detector_positions - source_positions,  # Ray directions
    u_vectors,
    v_vectors
])

# Load your projection data (shape: batch_size, 1, num_angles, det_height, det_width)
projections = torch.randn(1, 1, 180, 256, 256)

# Perform FDK reconstruction  
volume = fdk_reconstruction_masked(
    projections, 
    vectors,
    voxel_per_mm=2.0,
    volume_shape=(128, 128, 128)
)

print(f"Reconstructed volume shape: {volume.shape}")
```

## Laminography Reconstruction

```python
from astra_torch import lamino_fbp_reconstruction_masked

# Create laminography geometry with tilted rotation axis
tilt_angle = np.pi/6  # 30 degrees
vectors = []

for angle in angles:
    # Ray direction (parallel beam)
    ray_dir = np.array([
        np.cos(angle) * np.cos(tilt_angle),
        np.sin(angle) * np.cos(tilt_angle), 
        np.sin(tilt_angle)
    ])
    
    # Detector u vector
    u_vec = np.array([-np.sin(angle), np.cos(angle), 0])
    
    # Detector v vector
    v_vec = np.cross(ray_dir, u_vec)
    v_vec = v_vec / np.linalg.norm(v_vec)
    
    vectors.append(np.concatenate([ray_dir, u_vec, v_vec]))

vectors = np.array(vectors)

# Load laminography projection data
projections = torch.randn(1, 1, 180, 256, 256)

# Perform FBP reconstruction
volume = lamino_fbp_reconstruction_masked(
    projections,
    vectors, 
    voxel_per_mm=2.0,
    volume_shape=(128, 128, 128)
)
```

## Gradient Descent Reconstruction

For optimization-based reconstruction:

```python  
from astra_torch import cbct_gd_reconstruction_masked

# Perform gradient descent reconstruction with custom loss
volume = cbct_gd_reconstruction_masked(
    projections,
    vectors,
    voxel_per_mm=2.0,
    volume_shape=(128, 128, 128),
    num_iterations=100,
    learning_rate=0.01,
    loss_function='mse',  # or custom loss function
    regularization_weight=0.001
)
```

## Working with Real Data

### Walnut Dataset Format

```python
import h5py

# Load Walnut dataset
with h5py.File('walnut_data.h5', 'r') as f:
    projections = torch.from_numpy(f['projections'][:]).float()
    vectors = f['vectors'][:]

# Reconstruction
volume = fdk_reconstruction_masked(
    projections.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
    vectors,
    voxel_per_mm=10.0
)
```

### Custom Data Loading

```python
def load_projections(file_path):
    """Load projections from your custom format."""
    # Your data loading logic here
    projections = ...  # Shape: (num_angles, height, width)
    
    # Convert to PyTorch tensor with proper dimensions
    projections = torch.from_numpy(projections).float()
    projections = projections.unsqueeze(0).unsqueeze(0)  # Add batch and channel
    
    return projections

projections = load_projections('my_data.tiff')
volume = fdk_reconstruction_masked(projections, vectors)
```

## GPU Acceleration

ASTRA-Torch automatically uses GPU acceleration when available:

```python
# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move data to GPU
projections = projections.to(device)

# Reconstruction will automatically use GPU
volume = fdk_reconstruction_masked(projections, vectors)
```

## Next Steps

- See the [API Reference](api/index.md) for detailed function documentation
- Check out more [Examples](examples/index.md) for advanced use cases
- Learn about [Contributing](contributing.md) to the project
