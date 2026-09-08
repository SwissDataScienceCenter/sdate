"""
Example demonstrating the new combined loss function.

The loss function implements:
    L = λ₁||I - Î||₁ + λ_g||∇I - ∇Î||₁ + λ_r·ProxyRate(r)

where ProxyRate(r) = Σ log(ε + |r|) and r = I - Î (residual)
"""

import torch
from sdate.losses.noise2noise_loss import Noise2NoiseLoss

def main():
    """Demonstrate the new loss function."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create loss function with default parameters
    # L1 loss, gradient loss, and proxy rate are enabled by default
    loss_fn = Noise2NoiseLoss(
        device=device,
        use_l1=True,
        lambda_l1=1.0,              # Weight for L1 loss
        use_gradient_loss=True,
        lambda_gradient=0.1,         # Weight for gradient loss
        use_proxy_rate=True,
        lambda_rate=0.01,            # Weight for proxy rate term
        proxy_rate_epsilon=1e-6      # Epsilon for numerical stability
    )
    
    print("Loss Function Configuration:")
    print(f"  L1 Loss: Enabled (λ₁ = {loss_fn.lambda_l1})")
    print(f"  Gradient Loss: Enabled (λ_g = {loss_fn.lambda_gradient})")
    print(f"  Proxy Rate: Enabled (λ_r = {loss_fn.lambda_rate})")
    print(f"  Proxy Rate Epsilon: {loss_fn.proxy_rate_epsilon}")
    print(f"\nLoss components tracked: {loss_fn.stats_names}")
    
    # Example: Create dummy data
    batch_size = 4
    height, width = 256, 256
    num_input_projections = 2
    
    # Simulate input data
    input_tensor = torch.randn(batch_size, num_input_projections, height, width, device=device)
    target = torch.randn(batch_size, 1, height, width, device=device)
    center_coords = torch.tensor([[128, 128]] * batch_size, device=device)
    
    instance = {
        'input': input_tensor,
        'target': target,
        'center_coords': center_coords
    }
    
    # Create a dummy model (simple convolution for demonstration)
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(num_input_projections, 1, kernel_size=3, padding=1)
        
        def forward(self, x):
            return self.conv(x)
    
    model = DummyModel().to(device)
    
    # Compute loss
    total_loss, loss_dict = loss_fn.compute_loss(instance, model)
    
    print("\n" + "="*50)
    print("Loss Computation Results:")
    print("="*50)
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.6f}")
    
    print("\nLoss breakdown:")
    if 'l1_loss' in loss_dict:
        weighted_l1 = loss_fn.lambda_l1 * loss_dict['l1_loss'].item()
        print(f"  Weighted L1:         {weighted_l1:.6f} (λ₁={loss_fn.lambda_l1})")
    
    if 'gradient_loss' in loss_dict:
        weighted_grad = loss_fn.lambda_gradient * loss_dict['gradient_loss'].item()
        print(f"  Weighted Gradient:   {weighted_grad:.6f} (λ_g={loss_fn.lambda_gradient})")
    
    if 'proxy_rate_loss' in loss_dict:
        weighted_rate = loss_fn.lambda_rate * loss_dict['proxy_rate_loss'].item()
        print(f"  Weighted ProxyRate:  {weighted_rate:.6f} (λ_r={loss_fn.lambda_rate})")
    
    print(f"\n  Total Loss:          {total_loss.item():.6f}")
    
    # Example with different lambda values
    print("\n" + "="*50)
    print("Alternative Configuration:")
    print("="*50)
    
    loss_fn_alt = Noise2NoiseLoss(
        device=device,
        lambda_l1=2.0,           # Emphasize reconstruction accuracy
        lambda_gradient=0.5,     # Strongly encourage edge preservation
        lambda_rate=0.001,       # Light regularization for compression
    )
    
    total_loss_alt, loss_dict_alt = loss_fn_alt.compute_loss(instance, model)
    
    print(f"Configuration: λ₁={loss_fn_alt.lambda_l1}, "
          f"λ_g={loss_fn_alt.lambda_gradient}, "
          f"λ_r={loss_fn_alt.lambda_rate}")
    print(f"Total Loss: {total_loss_alt.item():.6f}")
    
    # Example: Disable proxy rate
    print("\n" + "="*50)
    print("Without Proxy Rate Term:")
    print("="*50)
    
    loss_fn_no_rate = Noise2NoiseLoss(
        device=device,
        use_proxy_rate=False,
        lambda_l1=1.0,
        lambda_gradient=0.1
    )
    
    total_loss_no_rate, loss_dict_no_rate = loss_fn_no_rate.compute_loss(instance, model)
    print(f"Components: {loss_fn_no_rate.stats_names}")
    print(f"Total Loss: {total_loss_no_rate.item():.6f}")


if __name__ == '__main__':
    main()
