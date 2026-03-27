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
    a denoised version of P_i.
    
    Implements the combined loss function:
        L = λ₁||I - Î||₁ + λ_g||∇I - ∇Î||₁ + λ_r·ProxyRate(r)
    
    where:
        - I and Î are the target and predicted frames respectively
        - ||·||₁ denotes L1 norm
        - ∇ represents spatial gradients (computed via Sobel filters)
        - r = I - Î is the residual
        - ProxyRate(r) = Σ log(ε + |r|) is a proxy for compression rate
    
    The three terms encourage:
        1. L1 loss: Direct pixel-wise accuracy
        2. Gradient loss: Preservation of edges and fine details
        3. Proxy rate: Implicit compression efficiency via residual sparsity
    """
    
    def __init__(
        self,
        device: torch.device,
        use_l1: bool = True,
        lambda_l1: float = 1.0,
        use_gradient_loss: bool = True,
        lambda_gradient: float = 0.1,
        use_proxy_rate: bool = True,
        lambda_rate: float = 0.01,
        proxy_rate_epsilon: float = 1e-6,
        use_ssim_loss: bool = False,
        ssim_weight: float = 0.1
    ):
        """
        Initialize the loss function.
        
        Args:
            device: Torch device for computation
            use_l1: If True, use L1 loss (default: True)
            lambda_l1: Weight for L1 loss
            use_gradient_loss: If True, add gradient consistency loss (default: True)
            lambda_gradient: Weight for gradient loss
            use_proxy_rate: If True, add proxy rate term (default: True)
            lambda_rate: Weight for proxy rate loss
            proxy_rate_epsilon: Epsilon for log stability in proxy rate
            use_ssim_loss: If True, add SSIM loss
            ssim_weight: Weight for SSIM loss
        """
        stats_names = ["loss", "l1_loss"]
        if use_gradient_loss:
            stats_names.append("gradient_loss")
        if use_proxy_rate:
            stats_names.append("proxy_rate_loss")
        if use_ssim_loss:
            stats_names.append("ssim_loss")
        
        super().__init__(stats_names)
        
        self.device = device
        self.use_l1 = use_l1
        self.lambda_l1 = lambda_l1
        self.use_gradient_loss = use_gradient_loss
        self.lambda_gradient = lambda_gradient
        self.use_proxy_rate = use_proxy_rate
        self.lambda_rate = lambda_rate
        self.proxy_rate_epsilon = proxy_rate_epsilon
        self.use_ssim_loss = use_ssim_loss
        self.ssim_weight = ssim_weight
        
        # Loss functions
        self.l1 = nn.L1Loss(reduction='mean')
    
    def _compute_gradient_loss(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient consistency loss using L1 norm.
        Computes ||∇I - ∇Î||_1 where ∇ is the spatial gradient.
        """
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
        
        # L1 loss on gradients
        grad_loss = self.l1(pred_grad_x, target_grad_x) + self.l1(pred_grad_y, target_grad_y)
        return grad_loss
    
    def _compute_proxy_rate(
        self,
        residual: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute proxy rate loss: ProxyRate(r) = Σ log(ε + |r|)
        where r is the residual (target - predicted).
        
        Args:
            residual: The residual tensor (I - Î)
            
        Returns:
            Scalar tensor representing the proxy rate
        """
        # Compute log(epsilon + |r|) and sum over all elements
        proxy_rate = torch.log(self.proxy_rate_epsilon + torch.abs(residual)).sum()
        # Normalize by the number of elements for stability
        proxy_rate = proxy_rate / residual.numel()
        return proxy_rate
    
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
        Compute the combined loss:
        L = λ₁||I - Î||₁ + λ_g||∇I - ∇Î||₁ + λ_r·ProxyRate(r)
        
        where:
        - I is the target frame
        - Î is the predicted frame
        - r = I - Î is the residual
        - ProxyRate(r) = Σ log(ε + |r|)
        
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
        model.zero_grad()
        
        # Check if model expects timestep (diffusers UNet)
        try:
            conditioning = center_coords[:, 0].long()  # Use the first coordinate as timestep to condition on the x position
            pred = model(input_tensor, conditioning, return_dict=False)[0]
        except TypeError:
            # For regular UNet without timestep
            pred = model(input_tensor)
        
        # Compute residual for proxy rate term
        residual = target - pred
        
        # Initialize total loss
        total_loss = torch.tensor(0.0, device=self.device)
        loss_dict = {}
        
        # 1. L1 loss: λ₁||I - Î||₁
        if self.use_l1:
            l1_loss = self.l1(pred, target)
            total_loss = total_loss + self.lambda_l1 * l1_loss
            loss_dict["l1_loss"] = l1_loss
        
        # 2. Gradient loss: λ_g||∇I - ∇Î||₁
        if self.use_gradient_loss:
            grad_loss = self._compute_gradient_loss(pred, target)
            total_loss = total_loss + self.lambda_gradient * grad_loss
            loss_dict["gradient_loss"] = grad_loss
        
        # 3. Proxy rate loss: λ_r·ProxyRate(r)
        if self.use_proxy_rate:
            proxy_rate_loss = self._compute_proxy_rate(residual)
            total_loss = total_loss + self.lambda_rate * proxy_rate_loss
            loss_dict["proxy_rate_loss"] = proxy_rate_loss
        
        # 4. Optional SSIM loss (legacy support)
        if self.use_ssim_loss:
            ssim_loss = self._compute_ssim(pred, target)
            total_loss = total_loss + self.ssim_weight * ssim_loss
            loss_dict["ssim_loss"] = ssim_loss
        
        loss_dict["loss"] = total_loss
        
        return total_loss, loss_dict
