"""
Test script for SSSD-ECG with Attention mechanism.
This script tests the new AttentionModel and SSSD_ECG_Attention implementation.
"""

import torch
import sys
import os

# Add the sssd_standalone directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sssd_standalone'))

from models.AttentionModel import AttentionLayer, BidirectionalAttentionLayer, get_attention_layer
from models.SSSD_ECG_Attention import SSSD_ECG_Attention


def test_attention_layer():
    """Test basic AttentionLayer functionality"""
    print("Testing AttentionLayer...")

    # Parameters
    batch_size = 4
    seq_len = 128
    features = 64
    num_heads = 8

    # Create layer
    layer = AttentionLayer(features=features, lmax=seq_len, num_heads=num_heads, dropout=0.1)

    # Create random input: (seq_len, batch_size, features)
    x = torch.randn(seq_len, batch_size, features)

    # Forward pass
    output = layer(x)

    # Check output shape
    assert output.shape == (seq_len, batch_size, features), f"Expected shape {(seq_len, batch_size, features)}, got {output.shape}"

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print("  ✓ AttentionLayer test passed!")

    return True


def test_bidirectional_attention_layer():
    """Test BidirectionalAttentionLayer functionality"""
    print("\nTesting BidirectionalAttentionLayer...")

    # Parameters
    batch_size = 4
    seq_len = 128
    features = 64
    num_heads = 8

    # Create layer
    layer = BidirectionalAttentionLayer(features=features, lmax=seq_len, num_heads=num_heads, dropout=0.1)

    # Create random input: (seq_len, batch_size, features)
    x = torch.randn(seq_len, batch_size, features)

    # Forward pass
    output = layer(x)

    # Check output shape
    assert output.shape == (seq_len, batch_size, features), f"Expected shape {(seq_len, batch_size, features)}, got {output.shape}"

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print("  ✓ BidirectionalAttentionLayer test passed!")

    return True


def test_sssd_ecg_attention():
    """Test SSSD_ECG_Attention model"""
    print("\nTesting SSSD_ECG_Attention model...")

    # Model parameters (matching typical SSSD-ECG configuration)
    in_channels = 12  # 12-lead ECG
    res_channels = 256
    skip_channels = 256
    out_channels = 12
    num_res_layers = 36

    diffusion_step_embed_dim_in = 128
    diffusion_step_embed_dim_mid = 512
    diffusion_step_embed_dim_out = 512

    attention_lmax = 1000
    attention_num_heads = 8
    attention_dropout = 0.0

    # Create model
    model = SSSD_ECG_Attention(
        in_channels=in_channels,
        res_channels=res_channels,
        skip_channels=skip_channels,
        out_channels=out_channels,
        num_res_layers=num_res_layers,
        diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
        diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
        diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
        attention_lmax=attention_lmax,
        attention_num_heads=attention_num_heads,
        attention_dropout=attention_dropout,
        attention_layernorm=True,
        label_embed_classes=0,
        label_embed_dim=128
    )

    # Test parameters
    batch_size = 2
    seq_len = 1000

    # Create random input
    noise = torch.randn(batch_size, in_channels, seq_len)
    label = torch.zeros(batch_size, 0)  # No labels for unconditional case
    diffusion_steps = torch.randint(0, 50, (batch_size,))

    # Forward pass
    print(f"  Input noise shape: {noise.shape}")
    print(f"  Diffusion steps: {diffusion_steps}")

    output = model((noise, label, diffusion_steps))

    # Check output shape
    assert output.shape == (batch_size, out_channels, seq_len), f"Expected shape {(batch_size, out_channels, seq_len)}, got {output.shape}"

    print(f"  Output shape: {output.shape}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {num_trainable_params:,}")
    print("  ✓ SSSD_ECG_Attention test passed!")

    return True


def test_conditional_sssd_ecg_attention():
    """Test SSSD_ECG_Attention model with conditional labels"""
    print("\nTesting SSSD_ECG_Attention with conditional labels...")

    # Model parameters
    in_channels = 12
    res_channels = 128
    skip_channels = 128
    out_channels = 12
    num_res_layers = 12

    diffusion_step_embed_dim_in = 128
    diffusion_step_embed_dim_mid = 512
    diffusion_step_embed_dim_out = 512

    attention_lmax = 500
    attention_num_heads = 8
    label_embed_classes = 5  # 5 classes

    # Create model with labels
    model = SSSD_ECG_Attention(
        in_channels=in_channels,
        res_channels=res_channels,
        skip_channels=skip_channels,
        out_channels=out_channels,
        num_res_layers=num_res_layers,
        diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
        diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
        diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
        attention_lmax=attention_lmax,
        attention_num_heads=attention_num_heads,
        attention_dropout=0.1,
        attention_layernorm=True,
        label_embed_classes=label_embed_classes,
        label_embed_dim=128
    )

    # Test parameters
    batch_size = 2
    seq_len = 500

    # Create random input with labels
    noise = torch.randn(batch_size, in_channels, seq_len)
    label = torch.zeros(batch_size, label_embed_classes)
    label[:, 0] = 1.0  # One-hot encoded labels
    diffusion_steps = torch.randint(0, 50, (batch_size,))

    # Forward pass
    output = model((noise, label, diffusion_steps))

    # Check output shape
    assert output.shape == (batch_size, out_channels, seq_len), f"Expected shape {(batch_size, out_channels, seq_len)}, got {output.shape}"

    print(f"  Input noise shape: {noise.shape}")
    print(f"  Label shape: {label.shape}")
    print(f"  Output shape: {output.shape}")
    print("  ✓ Conditional SSSD_ECG_Attention test passed!")

    return True


def test_cuda_compatibility():
    """Test CUDA compatibility if available"""
    if not torch.cuda.is_available():
        print("\nCUDA not available, skipping CUDA test")
        return True

    print("\nTesting CUDA compatibility...")

    # Create small model
    model = SSSD_ECG_Attention(
        in_channels=12,
        res_channels=64,
        skip_channels=64,
        out_channels=12,
        num_res_layers=4,
        diffusion_step_embed_dim_in=128,
        diffusion_step_embed_dim_mid=256,
        diffusion_step_embed_dim_out=256,
        attention_lmax=256,
        attention_num_heads=4,
        attention_dropout=0.0,
        attention_layernorm=True,
        label_embed_classes=0
    ).cuda()

    # Create input on CUDA
    batch_size = 2
    seq_len = 256
    noise = torch.randn(batch_size, 12, seq_len).cuda()
    label = torch.zeros(batch_size, 0).cuda()
    diffusion_steps = torch.randint(0, 50, (batch_size,)).cuda()

    # Forward pass
    output = model((noise, label, diffusion_steps))

    assert output.is_cuda, "Output should be on CUDA"
    assert output.shape == (batch_size, 12, seq_len), f"Expected shape {(batch_size, 12, seq_len)}, got {output.shape}"

    print(f"  Device: {output.device}")
    print(f"  Output shape: {output.shape}")
    print("  ✓ CUDA compatibility test passed!")

    return True


def main():
    """Run all tests"""
    print("=" * 80)
    print("SSSD-ECG Attention Model Test Suite")
    print("=" * 80)

    try:
        # Run tests
        test_attention_layer()
        test_bidirectional_attention_layer()
        test_sssd_ecg_attention()
        test_conditional_sssd_ecg_attention()
        test_cuda_compatibility()

        print("\n" + "=" * 80)
        print("All tests passed! ✓")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
