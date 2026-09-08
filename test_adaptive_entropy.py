#!/usr/bin/env python3
"""
Test adaptive quantization in ResidualEncoder/Decoder
"""

import numpy as np
import torch
from compress_ct.entropy import ResidualEncoder, ResidualDecoder

def test_adaptive_quantization():
    """Test that adaptive quantization works with large-range residuals."""
    
    # Create a residual with large magnitude (like CT data)
    np.random.seed(42)
    H, W = 256, 256
    residual = np.random.randn(H, W).astype(np.float32) * 10000  # Large range
    
    print("Testing Adaptive Quantization")
    print("=" * 60)
    print(f"Residual shape: {residual.shape}")
    print(f"Residual range: [{residual.min():.2f}, {residual.max():.2f}]")
    print(f"Residual std: {residual.std():.2f}")
    
    # Test with adaptive quantization
    print("\n1. Testing ADAPTIVE quantization:")
    encoder_adaptive = ResidualEncoder(block_size=8, quality=50, adaptive=True)
    decoder_adaptive = ResidualDecoder(block_size=8, quality=50, adaptive=True)
    
    encoded_adaptive = encoder_adaptive.encode(residual)
    decoded_adaptive = decoder_adaptive.decode(encoded_adaptive)
    
    error_adaptive = np.abs(residual - decoded_adaptive).mean()
    relative_error_adaptive = error_adaptive / (np.abs(residual).mean() + 1e-6)
    
    print(f"   Encoded size: {len(encoded_adaptive)} bytes")
    print(f"   Compression ratio: {residual.nbytes / len(encoded_adaptive):.2f}x")
    print(f"   Mean absolute error: {error_adaptive:.4f}")
    print(f"   Relative error: {relative_error_adaptive * 100:.2f}%")
    
    # Test with static (old) quantization for comparison
    print("\n2. Testing STATIC (old) quantization:")
    encoder_static = ResidualEncoder(block_size=8, quality=50, adaptive=False)
    decoder_static = ResidualDecoder(block_size=8, quality=50, adaptive=False)
    
    encoded_static = encoder_static.encode(residual)
    decoded_static = decoder_static.decode(encoded_static)
    
    error_static = np.abs(residual - decoded_static).mean()
    relative_error_static = error_static / (np.abs(residual).mean() + 1e-6)
    
    print(f"   Encoded size: {len(encoded_static)} bytes")
    print(f"   Compression ratio: {residual.nbytes / len(encoded_static):.2f}x")
    print(f"   Mean absolute error: {error_static:.4f}")
    print(f"   Relative error: {relative_error_static * 100:.2f}%")
    
    print("\n" + "=" * 60)
    print("Comparison:")
    print(f"   Adaptive vs Static error ratio: {error_adaptive / error_static:.2f}x")
    print(f"   Size difference: {len(encoded_adaptive) - len(encoded_static):+d} bytes")
    
    if error_adaptive < error_static * 0.8:
        print("\n✅ Adaptive quantization provides BETTER quality!")
    elif error_adaptive > error_static * 1.2:
        print("\n⚠️  Adaptive quantization provides WORSE quality (unexpected)")
    else:
        print("\n✅ Both methods provide similar quality")
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    test_adaptive_quantization()
