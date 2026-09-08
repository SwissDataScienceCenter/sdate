"""
Quick Reference: Using Residuals with TiffVolumeDataset

This is a quick reference for using the residual functionality. 
For detailed documentation, see scripts/README_residuals.md
"""

# ============================================================================
# STEP 1: Compute Residuals
# ============================================================================

# After training your model, compute residuals:
"""
python scripts/compute_residuals.py \
    --data_path=/myhome/data/sdate/shared/compression_paper/file_3_extracted \
    --checkpoint_path=outputs/heic_to_tiff/checkpoint-final \
    --output_path=outputs/residuals/file_3_residuals.npy \
    --volume_size=64 \
    --stride=64 \
    --num_frames=100 \
    --heic_quality=85 \
    --batch_size=4
"""

# This creates:
# - outputs/residuals/file_3_residuals.npy (memory-mapped residuals)
# - outputs/residuals/file_3_positions.npy (position indices)
# - outputs/residuals/file_3_metadata.npz (metadata)


# ============================================================================
# STEP 2: Load Dataset with Residuals
# ============================================================================

from sdate.datasets import TiffVolumeDataset

# Standard 2-channel dataset (TIFF + HEIC)
dataset_2ch = TiffVolumeDataset(
    data_path='/path/to/data',
    volume_size=64,
    stride=64,
    num_frames=100,
    use_heic_compression=True,
    dual_channel=True,
    normalize=True,
)

# 3-channel dataset (TIFF + HEIC + Residuals)
dataset_3ch = TiffVolumeDataset(
    data_path='/path/to/data',
    volume_size=64,
    stride=64,
    num_frames=100,
    use_heic_compression=True,
    dual_channel=True,
    use_residuals=True,  # ← ADD THIS
    residuals_path='outputs/residuals/file_3_residuals.npy',  # ← AND THIS
    normalize=True,
)


# ============================================================================
# STEP 3: Access Data
# ============================================================================

# Get a sample
sub_volume, position = dataset_3ch[0]

# sub_volume shape: (3, 64, 64, 64)
# Channel 0: TIFF (ground truth)
# Channel 1: HEIC (compressed input)
# Channel 2: Residuals (predicted - target)

tiff = sub_volume[0]      # (64, 64, 64)
heic = sub_volume[1]      # (64, 64, 64)
residual = sub_volume[2]  # (64, 64, 64)


# ============================================================================
# STEP 4: Use with DataLoader
# ============================================================================

from torch.utils.data import DataLoader

loader = DataLoader(
    dataset_3ch,
    batch_size=4,
    shuffle=True,
    num_workers=4,
)

for batch_volumes, batch_positions in loader:
    # batch_volumes shape: (4, 3, 64, 64, 64)
    
    tiff_batch = batch_volumes[:, 0:1]      # (4, 1, 64, 64, 64)
    heic_batch = batch_volumes[:, 1:2]      # (4, 1, 64, 64, 64)
    residual_batch = batch_volumes[:, 2:3]  # (4, 1, 64, 64, 64)
    
    # Your training code here
    # loss = criterion(model(heic_batch, residual_batch), tiff_batch)


# ============================================================================
# STEP 5: Example Training with Residuals
# ============================================================================

import torch
import torch.nn as nn

# Example: Train a residual correction model
class ResidualCorrector(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 2 channels (HEIC + Residuals)
        # Output: 1 channel (corrected TIFF)
        self.conv1 = nn.Conv3d(2, 32, 3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv3d(64, 1, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, heic, residual):
        x = torch.cat([heic, residual], dim=1)  # (B, 2, D, H, W)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Training loop
model = ResidualCorrector()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

for epoch in range(10):
    for batch_volumes, batch_positions in loader:
        tiff = batch_volumes[:, 0:1]      # Ground truth
        heic = batch_volumes[:, 1:2]      # Compressed
        residual = batch_volumes[:, 2:3]  # Model residuals
        
        # Predict corrected TIFF
        predicted = model(heic, residual)
        
        # Compute loss
        loss = criterion(predicted, tiff)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# ============================================================================
# Common Patterns
# ============================================================================

# Pattern 1: Use residuals as additional context
# concatenate HEIC + residual as input to model
input_channels = torch.cat([heic, residual], dim=1)  # (B, 2, D, H, W)
output = model(input_channels)

# Pattern 2: Learn residual correction
# predict correction to residual, then add to HEIC
correction = model(heic, residual)
corrected = heic + correction

# Pattern 3: Attention over residuals
# use residuals to weight different regions
attention = attention_model(residual)
weighted_heic = heic * attention
output = decoder(weighted_heic)

# Pattern 4: Multi-scale residuals
# use residuals at different spatial scales
residual_pyramid = create_pyramid(residual)
output = multi_scale_model(heic, residual_pyramid)


# ============================================================================
# Troubleshooting
# ============================================================================

# Error: "residuals_path must be provided"
# Solution: Add residuals_path parameter
dataset = TiffVolumeDataset(..., use_residuals=True, residuals_path='path/to/residuals.npy')

# Error: "use_residuals=True requires dual_channel=True"  
# Solution: Enable dual channel mode
dataset = TiffVolumeDataset(..., dual_channel=True, use_residuals=True, ...)

# Error: "Residuals volume_size does not match"
# Solution: Recompute residuals with matching volume_size
# python scripts/compute_residuals.py --volume_size=64 ...

# Error: "Number of residual sub-volumes does not match"
# Solution: Use same stride and data when computing residuals
# python scripts/compute_residuals.py --stride=64 ...


# ============================================================================
# Performance Tips
# ============================================================================

# 1. Residuals are memory-mapped: efficient for large datasets
# 2. Use batch_size in compute_residuals.py to fit GPU memory
# 3. num_workers in DataLoader works with memory-mapped residuals
# 4. Residuals are loaded on-demand, not all at once
# 5. Use ProcessPoolExecutor for parallel residual computation


# ============================================================================
# Validation
# ============================================================================

# Test your setup:
"""
python scripts/test_residuals.py
"""

# Check residual statistics:
import numpy as np
residuals = np.load('outputs/residuals/file_3_residuals.npy', mmap_mode='r')
print(f"Shape: {residuals.shape}")
print(f"Mean: {residuals.mean():.6f}")
print(f"Std: {residuals.std():.6f}")
print(f"Min: {residuals.min():.6f}")
print(f"Max: {residuals.max():.6f}")
