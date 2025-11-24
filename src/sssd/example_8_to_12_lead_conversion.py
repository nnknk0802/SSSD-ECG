#!/usr/bin/env python3
"""
Example usage of 8-lead to 12-lead ECG conversion.

This script demonstrates how to use the convert_8_to_12_lead module
both as a library and how to create test data.
"""

import numpy as np
from convert_8_to_12_lead import convert_8_to_12_lead


def create_sample_8lead_data(num_samples=10, signal_length=1000):
    """
    Create sample 8-lead ECG data for testing.

    Args:
        num_samples (int): Number of ECG samples to generate
        signal_length (int): Length of each ECG signal (default: 1000 for 10s at 100Hz)

    Returns:
        np.ndarray: Sample 8-lead ECG data with shape (num_samples, 8, signal_length)
    """
    # Create synthetic ECG-like signals
    # In practice, you would load real ECG data here
    data_8lead = np.random.randn(num_samples, 8, signal_length).astype(np.float32)

    # Add some ECG-like features (simple sine waves for demonstration)
    time = np.arange(signal_length) / 100.0  # Assuming 100 Hz sampling rate
    for i in range(num_samples):
        for lead_idx in range(8):
            # Add a slow oscillation (heart rate ~60 bpm = 1 Hz)
            data_8lead[i, lead_idx, :] += np.sin(2 * np.pi * 1.0 * time) * 0.5

    return data_8lead


def validate_conversion(data_8lead, data_12lead):
    """
    Validate that the conversion was performed correctly.

    Args:
        data_8lead (np.ndarray): Original 8-lead data
        data_12lead (np.ndarray): Converted 12-lead data

    Returns:
        bool: True if validation passes
    """
    # Handle single sample vs batch
    if data_8lead.ndim == 2:
        data_8lead = data_8lead[np.newaxis, :]
    if data_12lead.ndim == 2:
        data_12lead = data_12lead[np.newaxis, :]

    # Check shapes
    num_samples, _, signal_length = data_8lead.shape
    assert data_12lead.shape == (num_samples, 12, signal_length), \
        f"Invalid output shape: {data_12lead.shape}"

    # Verify that original leads are preserved
    # Lead I should be the same
    assert np.allclose(data_8lead[:, 0, :], data_12lead[:, 0, :]), \
        "Lead I not preserved"

    # Lead II should be the same
    assert np.allclose(data_8lead[:, 1, :], data_12lead[:, 1, :]), \
        "Lead II not preserved"

    # V1-V6 should be the same (indices 2-7 in 8-lead, indices 6-11 in 12-lead)
    for i in range(6):
        assert np.allclose(data_8lead[:, 2+i, :], data_12lead[:, 6+i, :]), \
            f"Lead V{i+1} not preserved"

    # Verify derived leads
    lead_I = data_12lead[:, 0, :]
    lead_II = data_12lead[:, 1, :]
    lead_III = data_12lead[:, 2, :]
    lead_aVR = data_12lead[:, 3, :]
    lead_aVL = data_12lead[:, 4, :]
    lead_aVF = data_12lead[:, 5, :]

    # Check derivation formulas
    assert np.allclose(lead_III, lead_II - lead_I, atol=1e-5), \
        "Lead III derivation incorrect"

    assert np.allclose(lead_aVR, -(lead_I + lead_II) / 2, atol=1e-5), \
        "Lead aVR derivation incorrect"

    assert np.allclose(lead_aVL, lead_I - lead_II / 2, atol=1e-5), \
        "Lead aVL derivation incorrect"

    assert np.allclose(lead_aVF, lead_II - lead_I / 2, atol=1e-5), \
        "Lead aVF derivation incorrect"

    print("✓ All validation checks passed!")
    return True


def example_single_sample():
    """Example: Convert a single ECG sample."""
    print("\n=== Example 1: Single Sample Conversion ===")

    # Create a single 8-lead ECG sample
    data_8lead = create_sample_8lead_data(num_samples=1, signal_length=1000)
    data_8lead = data_8lead[0]  # Remove batch dimension for single sample

    print(f"Input shape: {data_8lead.shape}")  # (8, 1000)

    # Convert to 12-lead
    data_12lead = convert_8_to_12_lead(data_8lead)

    print(f"Output shape: {data_12lead.shape}")  # (12, 1000)

    # Validate
    validate_conversion(data_8lead, data_12lead)


def example_batch_conversion():
    """Example: Convert a batch of ECG samples."""
    print("\n=== Example 2: Batch Conversion ===")

    # Create a batch of 8-lead ECG samples
    num_samples = 100
    signal_length = 1000
    data_8lead = create_sample_8lead_data(num_samples, signal_length)

    print(f"Input shape: {data_8lead.shape}")  # (100, 8, 1000)

    # Convert to 12-lead
    data_12lead = convert_8_to_12_lead(data_8lead)

    print(f"Output shape: {data_12lead.shape}")  # (100, 12, 1000)

    # Validate
    validate_conversion(data_8lead, data_12lead)


def example_save_and_load():
    """Example: Save and load converted data."""
    print("\n=== Example 3: Save and Load ===")

    # Create sample data
    data_8lead = create_sample_8lead_data(num_samples=50, signal_length=1000)

    print(f"Created 8-lead data with shape: {data_8lead.shape}")

    # Save 8-lead data
    np.save('sample_8lead_data.npy', data_8lead)
    print("Saved 8-lead data to: sample_8lead_data.npy")

    # Convert to 12-lead
    data_12lead = convert_8_to_12_lead(data_8lead)

    # Save 12-lead data
    np.save('sample_12lead_data.npy', data_12lead)
    print("Saved 12-lead data to: sample_12lead_data.npy")

    # Load and verify
    loaded_12lead = np.load('sample_12lead_data.npy')
    assert np.allclose(loaded_12lead, data_12lead), "Loaded data doesn't match"

    print("✓ Data saved and loaded successfully!")

    # Print file sizes
    import os
    size_8lead = os.path.getsize('sample_8lead_data.npy') / 1024 / 1024
    size_12lead = os.path.getsize('sample_12lead_data.npy') / 1024 / 1024
    print(f"File size - 8-lead: {size_8lead:.2f} MB")
    print(f"File size - 12-lead: {size_12lead:.2f} MB")


def example_ptbxl_format():
    """Example: Convert data in PTB-XL format."""
    print("\n=== Example 4: PTB-XL Format Conversion ===")

    # PTB-XL typically has shape (num_samples, signal_length, num_leads)
    # We need to transpose to (num_samples, num_leads, signal_length)

    # Simulate PTB-XL format data
    num_samples = 20
    signal_length = 1000
    num_leads = 8

    # PTB-XL format: (num_samples, signal_length, num_leads)
    data_ptbxl_format = np.random.randn(num_samples, signal_length, num_leads).astype(np.float32)
    print(f"PTB-XL format shape: {data_ptbxl_format.shape}")

    # Transpose to our expected format: (num_samples, num_leads, signal_length)
    data_8lead = np.transpose(data_ptbxl_format, (0, 2, 1))
    print(f"Transposed shape: {data_8lead.shape}")

    # Convert to 12-lead
    data_12lead = convert_8_to_12_lead(data_8lead)
    print(f"12-lead shape: {data_12lead.shape}")

    # Optionally transpose back to PTB-XL format
    data_12lead_ptbxl_format = np.transpose(data_12lead, (0, 2, 1))
    print(f"12-lead PTB-XL format shape: {data_12lead_ptbxl_format.shape}")

    validate_conversion(data_8lead, data_12lead)


def main():
    """Run all examples."""
    print("=" * 60)
    print("8-Lead to 12-Lead ECG Conversion Examples")
    print("=" * 60)

    # Run examples
    example_single_sample()
    example_batch_conversion()
    example_save_and_load()
    example_ptbxl_format()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
