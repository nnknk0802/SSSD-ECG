#!/usr/bin/env python3
"""
Simple script to check available CUDA devices.
Run this before training to see which devices you can use.
"""

import sys
import os

# Add standalone directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sssd_standalone'))

try:
    from model_wrapper import SSSDECG

    print("=" * 60)
    print("SSSD-ECG Device Check")
    print("=" * 60)
    print()

    # Print available devices
    SSSDECG.print_available_devices()

    print()
    print("=" * 60)
    print("Usage Examples:")
    print("=" * 60)

    devices = SSSDECG.list_available_devices()

    print()
    print("To use a specific device, initialize the model like this:")
    print()

    for device in devices:
        print(f'  model = SSSDECG(config_path="config.json", device="{device}")')

    print()
    print("Or let the model auto-select (CUDA if available, else CPU):")
    print('  model = SSSDECG(config_path="config.json")')
    print()

except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Make sure PyTorch is installed:")
    print("  pip install torch")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
