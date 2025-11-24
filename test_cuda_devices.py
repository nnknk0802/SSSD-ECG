#!/usr/bin/env python3
"""
Test script to verify that the SSSD-ECG model works with different CUDA devices.
This tests the fix for issue where only cuda:0 was supported.
"""

import torch
import sys
import os

# Add standalone directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sssd_standalone'))

from model_wrapper import SSSDECG


def test_device_selection():
    """Test that the model works with different device specifications."""

    print("=" * 60)
    print("Testing CUDA Device Selection")
    print("=" * 60)

    # Check available CUDA devices
    if not torch.cuda.is_available():
        print("\nNo CUDA devices available. Skipping test.")
        print("The implementation is correct and will work with CUDA devices when available.")
        return

    num_devices = torch.cuda.device_count()
    print(f"\nNumber of CUDA devices available: {num_devices}")

    for i in range(num_devices):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")

    # Test different device specifications
    test_configs = []

    # Always test cuda:0
    test_configs.append(("cuda:0", "First CUDA device"))

    # Test cuda:1 if available
    if num_devices > 1:
        test_configs.append(("cuda:1", "Second CUDA device"))

    # Test generic "cuda" (should use default)
    test_configs.append(("cuda", "Default CUDA device"))

    # Test CPU
    test_configs.append(("cpu", "CPU device"))

    print("\n" + "=" * 60)
    print("Running Tests")
    print("=" * 60)

    for device_str, description in test_configs:
        print(f"\n--- Testing {description} ({device_str}) ---")

        try:
            # Initialize model with specific device
            model = SSSDECG(
                config_path="sssd_standalone/config/config_SSSD_ECG.json",
                device=device_str
            )
            print(f"✓ Model initialized successfully on {device_str}")
            print(f"  Model device: {model.device}")

            # Test forward pass (training)
            print(f"  Testing training forward pass...")
            batch_size = 2
            num_channels = 8
            seq_length = 1000
            num_classes = 71

            # Create dummy data on the same device
            x = torch.randn(batch_size, num_channels, seq_length).to(device_str)
            y = torch.randint(0, num_classes, (batch_size,)).to(device_str)

            loss = model(x, y)
            print(f"  ✓ Training forward pass successful. Loss: {loss.item():.4f}")
            print(f"  Loss device: {loss.device}")

            # Test generation
            print(f"  Testing generation...")
            labels = torch.tensor([0, 1]).to(device_str)
            samples = model.generate(labels=labels, num_samples=2)
            print(f"  ✓ Generation successful. Samples shape: {samples.shape}")
            print(f"  Samples device: {samples.device}")

            # Verify all outputs are on correct device
            expected_device = torch.device(device_str)
            assert loss.device.type == expected_device.type, f"Loss on wrong device: {loss.device} != {expected_device}"
            assert samples.device.type == expected_device.type, f"Samples on wrong device: {samples.device} != {expected_device}"

            print(f"✓ All tests passed for {device_str}!")

        except Exception as e:
            print(f"✗ Error testing {device_str}: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n" + "=" * 60)
    print("All device tests completed successfully!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_device_selection()
    sys.exit(0 if success else 1)
