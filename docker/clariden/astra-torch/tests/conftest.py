"""Test configuration and fixtures."""

import pytest
import numpy as np
import torch


@pytest.fixture
def device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def simple_phantom():
    """Create a simple 3D phantom for testing."""
    phantom = np.zeros((64, 64, 64))
    # Add a sphere in the center
    x, y, z = np.mgrid[:64, :64, :64]
    center = np.array([32, 32, 32])
    sphere = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= 15**2
    phantom[sphere] = 1.0
    return phantom


@pytest.fixture
def simple_projections():
    """Create simple projection data for testing."""
    # Simple 2D projection data
    projs = np.random.rand(180, 256, 256)
    return projs


@pytest.fixture
def cbct_geometry():
    """Create CBCT geometry vectors for testing."""
    num_angles = 180
    angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    
    # Simple circular orbit
    source_positions = np.column_stack([
        1000 * np.cos(angles),
        1000 * np.sin(angles),
        np.zeros(num_angles)
    ])
    
    detector_positions = np.column_stack([
        -500 * np.cos(angles),
        -500 * np.sin(angles),
        np.zeros(num_angles)
    ])
    
    # Detector orientation vectors
    u_vectors = np.column_stack([
        -np.sin(angles),
        np.cos(angles),
        np.zeros(num_angles)
    ])
    
    v_vectors = np.tile([0, 0, 1], (num_angles, 1))
    
    # Combine into geometry vectors
    vectors = np.column_stack([
        source_positions,
        detector_positions - source_positions,
        u_vectors,
        v_vectors
    ])
    
    return vectors


@pytest.fixture
def lamino_geometry():
    """Create laminography geometry for testing."""
    num_angles = 180
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)
    
    # Laminography with tilted rotation axis
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
    
    return np.array(vectors)
