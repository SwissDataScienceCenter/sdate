"""
2D Parallel Beam Reconstruction Example
========================================

This example demonstrates 2D parallel beam tomographic reconstruction using
the astra_torch library. We create a simple phantom, generate synthetic
projections, and reconstruct using FBP, SIRT, and gradient descent methods.

Requirements:
- ASTRA toolbox with CUDA support
- PyTorch with CUDA support
- CuPy (for GPU acceleration)
- matplotlib (for visualization)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from astra_torch.parallel2D import (
    build_parallel2d_projector,
    fbp_reconstruction_masked,
    sirt_reconstruction_masked,
    gd_reconstruction_masked,
)


def create_shepp_logan_phantom(size=256):
    """Create a simplified Shepp-Logan phantom for 2D reconstruction.
    
    Parameters
    ----------
    size : int
        Size of the phantom (size x size)
    
    Returns
    -------
    phantom : torch.Tensor
        2D phantom on GPU
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phantom = torch.zeros((size, size), dtype=torch.float32, device=device)
    
    # Create coordinate grids
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, size, device=device),
        torch.linspace(-1, 1, size, device=device),
        indexing='ij'
    )
    
    # Large outer ellipse
    ellipse1 = ((x/0.69)**2 + (y/0.92)**2 < 1).float()
    phantom += ellipse1 * 2.0
    
    # Inner ellipse (dark)
    ellipse2 = (((x-0.0)/0.6624)**2 + ((y-0.0)/0.874)**2 < 1).float()
    phantom -= ellipse2 * 1.8
    
    # Small bright ellipse (top left)
    ellipse3 = (((x+0.22)/0.11)**2 + ((y+0.25)/0.31)**2 < 1).float()
    phantom += ellipse3 * 1.0
    
    # Small bright ellipse (top right)
    ellipse4 = (((x-0.22)/0.16)**2 + ((y+0.25)/0.41)**2 < 1).float()
    phantom += ellipse4 * 1.0
    
    # Small circle (center)
    circle = ((x-0.0)**2 + (y-0.35)**2 < 0.05**2).float()
    phantom += circle * 1.5
    
    return phantom


def generate_projections(phantom, num_angles=180):
    """Generate parallel beam projections from a phantom.
    
    Parameters
    ----------
    phantom : torch.Tensor
        2D phantom image
    num_angles : int
        Number of projection angles
    
    Returns
    -------
    projs : torch.Tensor
        Projections with shape (num_angles, det_cols)
    angles_deg : np.ndarray
        Projection angles in degrees
    """
    device = phantom.device
    size = phantom.shape[0]
    
    # Create angles from 0 to 180 degrees
    angles_deg = np.linspace(0, 180, num_angles, endpoint=False)
    
    # Detector size (slightly larger than object)
    det_cols = int(size * 1.5)
    
    # Build projector
    projector = build_parallel2d_projector(
        vol_shape=(size, size),
        det_cols=det_cols,
        angles_deg=angles_deg,
        voxel_size_mm=1.0,
        det_spacing_mm=1.0,
        device=device
    )
    
    # Generate projections
    phantom_batch = phantom.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    projs = projector(phantom_batch)  # (1, V, C)
    projs_vc = projs[0]  # (V, C)
    
    return projs_vc, angles_deg, det_cols


def add_noise(projs, noise_level=0.05):
    """Add Gaussian noise to projections.
    
    Parameters
    ----------
    projs : torch.Tensor
        Clean projections
    noise_level : float
        Standard deviation of noise relative to signal
    
    Returns
    -------
    noisy_projs : torch.Tensor
        Projections with added noise
    """
    signal_std = projs.std()
    noise = torch.randn_like(projs) * (signal_std * noise_level)
    return projs + noise


def main():
    """Run the 2D parallel beam reconstruction example."""
    print("2D Parallel Beam Reconstruction Example")
    print("=" * 50)
    
    # Check for GPU
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. This example requires GPU support.")
        return
    
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # Parameters
    phantom_size = 256
    num_angles = 180
    noise_level = 0.02  # 2% noise
    
    print(f"\nPhantom size: {phantom_size} x {phantom_size}")
    print(f"Number of angles: {num_angles}")
    print(f"Noise level: {noise_level*100}%")
    
    # Step 1: Create phantom
    print("\n[1/5] Creating Shepp-Logan phantom...")
    phantom = create_shepp_logan_phantom(size=phantom_size)
    
    # Step 2: Generate projections
    print("[2/5] Generating projections...")
    projs_clean, angles_deg, det_cols = generate_projections(phantom, num_angles)
    
    # Add noise
    projs = add_noise(projs_clean, noise_level)
    
    print(f"    Projection shape: {projs.shape}")
    print(f"    Detector columns: {det_cols}")
    
    # Step 3: FBP Reconstruction
    print("[3/5] Running FBP reconstruction...")
    recon_fbp = fbp_reconstruction_masked(
        projs_vc=projs,
        angles_deg=angles_deg,
        vol_shape=(phantom_size, phantom_size),
        det_spacing_mm=1.0,
        filter_type='hann',  # Hann filter for noise suppression
        device=device,
    )
    
    # Step 4: SIRT Reconstruction
    print("[4/5] Running SIRT reconstruction (100 iterations)...")
    recon_sirt = sirt_reconstruction_masked(
        projs_vc=projs,
        angles_deg=angles_deg,
        vol_shape=(phantom_size, phantom_size),
        det_spacing_mm=1.0,
        num_iterations=100,
        min_constraint=0.0,  # Non-negativity constraint
        device=device,
    )
    
    # Step 5: Gradient Descent Reconstruction
    print("[5/5] Running gradient descent reconstruction...")
    recon_gd = gd_reconstruction_masked(
        projs_vc=projs,
        angles_deg=angles_deg,
        vol_shape=(phantom_size, phantom_size),
        det_spacing_mm=1.0,
        max_epochs=20,
        batch_size=30,
        lr=1e-2,
        clamp_min=0.0,
        optimizer_type="adam",
        device=device,
        verbose=True,
    )
    
    # Visualization
    print("\n[6/6] Creating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Convert to CPU for plotting
    phantom_cpu = phantom.cpu().numpy()
    recon_fbp_cpu = recon_fbp.cpu().numpy()
    recon_sirt_cpu = recon_sirt.cpu().numpy()
    recon_gd_cpu = recon_gd.cpu().numpy()
    projs_cpu = projs.cpu().numpy()
    
    # Plot phantom
    im0 = axes[0, 0].imshow(phantom_cpu, cmap='gray')
    axes[0, 0].set_title('Original Phantom')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # Plot sinogram
    im1 = axes[0, 1].imshow(projs_cpu, cmap='gray', aspect='auto')
    axes[0, 1].set_title(f'Sinogram ({num_angles} views)')
    axes[0, 1].set_xlabel('Detector position')
    axes[0, 1].set_ylabel('Angle index')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Plot profile comparison
    mid = phantom_size // 2
    axes[0, 2].plot(phantom_cpu[mid, :], 'k-', label='Phantom', linewidth=2)
    axes[0, 2].plot(recon_fbp_cpu[mid, :], 'r--', label='FBP', alpha=0.7)
    axes[0, 2].plot(recon_sirt_cpu[mid, :], 'g--', label='SIRT', alpha=0.7)
    axes[0, 2].plot(recon_gd_cpu[mid, :], 'b--', label='GD', alpha=0.7)
    axes[0, 2].set_title('Horizontal Profile (center)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot FBP reconstruction
    im2 = axes[1, 0].imshow(recon_fbp_cpu, cmap='gray')
    axes[1, 0].set_title('FBP Reconstruction')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    
    # Plot SIRT reconstruction
    im3 = axes[1, 1].imshow(recon_sirt_cpu, cmap='gray')
    axes[1, 1].set_title('SIRT Reconstruction')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
    
    # Plot GD reconstruction
    im4 = axes[1, 2].imshow(recon_gd_cpu, cmap='gray')
    axes[1, 2].set_title('Gradient Descent Reconstruction')
    axes[1, 2].axis('off')
    plt.colorbar(im4, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'parallel2d_reconstruction_example.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    # Compute and display error metrics
    print("\nReconstruction Quality Metrics:")
    print("-" * 50)
    
    def compute_metrics(recon, phantom):
        """Compute RMSE and PSNR."""
        mse = torch.mean((recon - phantom) ** 2).item()
        rmse = np.sqrt(mse)
        psnr = 10 * np.log10(phantom.max().item()**2 / mse) if mse > 0 else float('inf')
        return rmse, psnr
    
    rmse_fbp, psnr_fbp = compute_metrics(recon_fbp, phantom)
    rmse_sirt, psnr_sirt = compute_metrics(recon_sirt, phantom)
    rmse_gd, psnr_gd = compute_metrics(recon_gd, phantom)
    
    print(f"FBP:  RMSE = {rmse_fbp:.4f}, PSNR = {psnr_fbp:.2f} dB")
    print(f"SIRT: RMSE = {rmse_sirt:.4f}, PSNR = {psnr_sirt:.2f} dB")
    print(f"GD:   RMSE = {rmse_gd:.4f}, PSNR = {psnr_gd:.2f} dB")
    
    print("\nExample completed successfully!")
    print(f"You can view the results in '{output_file}'")
    
    # Show plot
    plt.show()


if __name__ == '__main__':
    main()
