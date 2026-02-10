"""
Noise2Noise Loss for Tomographic Projection Denoising.

This module implements the loss function for training a denoising network
using the Noise2Noise paradigm on tomographic projections.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

try:
    from pytorch_base.base_loss import BaseLoss
except ImportError:
    # Fallback if pytorch_base is not available
    class BaseLoss:
        def __init__(self, stats_names):
            self.stats_names = stats_names
        
        def compute_loss(self, instance, model):
            raise NotImplementedError


class Noise2NoiseLoss(BaseLoss):
    """
    Loss function for Noise2Noise training on tomographic projections.
    
    The model takes two noisy observations (P_{i-1}, P_{i+1}) and predicts
    a denoised version of P_i. We use MSE loss between the prediction and P_i.
    
    Optionally includes perceptual loss and gradient-based losses.
    """
    
    def __init__(
        self,
        device: torch.device,
        use_l1: bool = False,
        use_gradient_loss: bool = False,
        gradient_weight: float = 0.1,
        use_ssim_loss: bool = False,
        ssim_weight: float = 0.1
    ):
        """
        Initialize the loss function.
        
        Args:
            device: Torch device for computation
            use_l1: If True, use L1 loss instead of MSE
            use_gradient_loss: If True, add gradient consistency loss
            gradient_weight: Weight for gradient loss
            use_ssim_loss: If True, add SSIM loss
            ssim_weight: Weight for SSIM loss
        """
        stats_names = ["loss", "mse_loss"]
        if use_gradient_loss:
            stats_names.append("gradient_loss")
        if use_ssim_loss:
            stats_names.append("ssim_loss")
        
        super().__init__(stats_names)
        
        self.device = device
        self.use_l1 = use_l1
        self.use_gradient_loss = use_gradient_loss
        self.gradient_weight = gradient_weight
        self.use_ssim_loss = use_ssim_loss
        self.ssim_weight = ssim_weight
        
        # Loss functions
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def _compute_gradient_loss(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient consistency loss."""
        # Sobel filters for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        # Compute gradients
        pred_grad_x = nn.functional.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = nn.functional.conv2d(pred, sobel_y, padding=1)
        target_grad_x = nn.functional.conv2d(target, sobel_x, padding=1)
        target_grad_y = nn.functional.conv2d(target, sobel_y, padding=1)
        
        # MSE on gradients
        grad_loss = self.mse(pred_grad_x, target_grad_x) + self.mse(pred_grad_y, target_grad_y)
        return grad_loss
    
    def _compute_ssim(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        window_size: int = 11,
        sigma: float = 1.5
    ) -> torch.Tensor:
        """Compute SSIM loss (1 - SSIM for minimization)."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Create Gaussian window
        coords = torch.arange(window_size, dtype=torch.float32, device=self.device)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(0) * g.unsqueeze(1)
        window = window.unsqueeze(0).unsqueeze(0)
        
        # Compute means
        mu_pred = nn.functional.conv2d(pred, window, padding=window_size // 2)
        mu_target = nn.functional.conv2d(target, window, padding=window_size // 2)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        # Compute variances
        sigma_pred_sq = nn.functional.conv2d(pred ** 2, window, padding=window_size // 2) - mu_pred_sq
        sigma_target_sq = nn.functional.conv2d(target ** 2, window, padding=window_size // 2) - mu_target_sq
        sigma_pred_target = nn.functional.conv2d(pred * target, window, padding=window_size // 2) - mu_pred_target
        
        # SSIM formula
        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        
        return 1 - ssim_map.mean()
    
    def compute_loss(
        self, 
        instance: Dict[str, torch.Tensor], 
        model: nn.Module
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the Noise2Noise loss.
        
        Args:
            instance: Dictionary with 'input', 'target', 'center_coords' keys
            model: The denoising model (UNet2DModel)
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Get data
        input_tensor = instance['input'].to(self.device)  # (B, k, H, W) where k = num_input_projections
        target = instance['target'].to(self.device)  # (B, 1, H, W)
        center_coords = instance['center_coords'].to(self.device)  # (B, 2)
        
        # Forward pass through model
        # The UNet2DModel expects timestep argument - we use 0 for non-diffusion models
        # or we can ignore it if not using diffusion
        model.zero_grad()
        
        # Check if model expects timestep (diffusers UNet)
        try:
            # For diffusers UNet2DModel, we need a dummy timestep
            # We use 0 as we're not doing diffusion
            batch_size = input_tensor.shape[0]
            timesteps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            pred = model(input_tensor, timesteps, return_dict=False)[0]
        except TypeError:
            # For regular UNet without timestep
            pred = model(input_tensor)
        
        # Compute main loss
        if self.use_l1:
            mse_loss = self.l1(pred, target)
        else:
            mse_loss = self.mse(pred, target)
        
        total_loss = mse_loss
        loss_dict = {"loss": total_loss, "mse_loss": mse_loss}
        
        # Optional gradient loss
        if self.use_gradient_loss:
            grad_loss = self._compute_gradient_loss(pred, target)
            total_loss = total_loss + self.gradient_weight * grad_loss
            loss_dict["gradient_loss"] = grad_loss
            loss_dict["loss"] = total_loss
        
        # Optional SSIM loss
        if self.use_ssim_loss:
            ssim_loss = self._compute_ssim(pred, target)
            total_loss = total_loss + self.ssim_weight * ssim_loss
            loss_dict["ssim_loss"] = ssim_loss
            loss_dict["loss"] = total_loss
        
        return total_loss, loss_dict


class Noise2NoiseWithConditioningLoss(Noise2NoiseLoss):
    """
    Extended Noise2Noise loss that passes center coordinates as conditioning to the model.
    
    This version embeds the center coordinates and concatenates them with the input
    or uses class conditioning if the model supports it.
    """
    
    def __init__(
        self,
        device: torch.device,
        conditioning_method: str = "concat_embedding",
        embedding_dim: int = 256,
        **kwargs
    ):
        """
        Initialize the conditioned loss function.
        
        Args:
            device: Torch device
            conditioning_method: One of "concat_embedding", "class_embedding", "none"
            embedding_dim: Dimension of coordinate embedding
            **kwargs: Additional arguments for parent class
        """
        super().__init__(device=device, **kwargs)
        
        self.conditioning_method = conditioning_method
        self.embedding_dim = embedding_dim
        
        # Coordinate embedding network (simple MLP)
        if conditioning_method == "concat_embedding":
            self.coord_embedding = nn.Sequential(
                nn.Linear(2, embedding_dim),
                nn.SiLU(),
                nn.Linear(embedding_dim, embedding_dim)
            ).to(device)
    
    def compute_loss(
        self, 
        instance: Dict[str, torch.Tensor], 
        model: nn.Module
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the conditioned Noise2Noise loss.
        """
        input_tensor = instance['input'].to(self.device)
        target = instance['target'].to(self.device)
        center_coords = instance['center_coords'].to(self.device)
        
        model.zero_grad()
        
        conditioning = center_coords[:, 0].to(self.device)
        
        pred = model(input_tensor, conditioning, return_dict=False)[0]
            
        
        # Compute losses (same as parent)
        if self.use_l1:
            mse_loss = self.l1(pred, target)
        else:
            mse_loss = self.mse(pred, target)
        
        total_loss = mse_loss
        loss_dict = {"loss": total_loss, "mse_loss": mse_loss}
        
        if self.use_gradient_loss:
            grad_loss = self._compute_gradient_loss(pred, target)
            total_loss = total_loss + self.gradient_weight * grad_loss
            loss_dict["gradient_loss"] = grad_loss
            loss_dict["loss"] = total_loss
        
        if self.use_ssim_loss:
            ssim_loss = self._compute_ssim(pred, target)
            total_loss = total_loss + self.ssim_weight * ssim_loss
            loss_dict["ssim_loss"] = ssim_loss
            loss_dict["loss"] = total_loss
        
        return total_loss, loss_dict


# Simple loss function for use without pytorch_base
def noise2noise_mse_loss(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Simple MSE loss for Noise2Noise training.
    
    Args:
        model: UNet2DModel
        input_tensor: Input with shape (B, 2, H, W)
        target: Target with shape (B, 1, H, W)
        device: Torch device
        
    Returns:
        MSE loss
    """
    input_tensor = input_tensor.to(device)
    target = target.to(device)
    
    batch_size = input_tensor.shape[0]
    timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    pred = model(input_tensor, timesteps, return_dict=False)[0]
    
    return nn.functional.mse_loss(pred, target)
