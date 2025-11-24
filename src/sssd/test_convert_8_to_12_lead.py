#!/usr/bin/env python3
"""
Unit tests for 8-lead to 12-lead ECG conversion.

Run with: python -m pytest test_convert_8_to_12_lead.py
Or: python test_convert_8_to_12_lead.py
"""

import sys
import numpy as np

# Import the conversion function
from convert_8_to_12_lead import convert_8_to_12_lead


def test_single_sample_conversion():
    """Test conversion of a single ECG sample."""
    # Create a single 8-lead sample
    signal_length = 1000
    data_8lead = np.random.randn(8, signal_length).astype(np.float32)

    # Convert to 12-lead
    data_12lead = convert_8_to_12_lead(data_8lead)

    # Check output shape
    assert data_12lead.shape == (12, signal_length), \
        f"Expected shape (12, {signal_length}), got {data_12lead.shape}"

    print("✓ test_single_sample_conversion passed")


def test_batch_conversion():
    """Test conversion of a batch of ECG samples."""
    num_samples = 50
    signal_length = 1000
    data_8lead = np.random.randn(num_samples, 8, signal_length).astype(np.float32)

    # Convert to 12-lead
    data_12lead = convert_8_to_12_lead(data_8lead)

    # Check output shape
    assert data_12lead.shape == (num_samples, 12, signal_length), \
        f"Expected shape ({num_samples}, 12, {signal_length}), got {data_12lead.shape}"

    print("✓ test_batch_conversion passed")


def test_lead_preservation():
    """Test that original leads are preserved in the output."""
    num_samples = 10
    signal_length = 500
    data_8lead = np.random.randn(num_samples, 8, signal_length).astype(np.float32)

    data_12lead = convert_8_to_12_lead(data_8lead)

    # Check that Lead I is preserved
    assert np.allclose(data_8lead[:, 0, :], data_12lead[:, 0, :]), \
        "Lead I not preserved"

    # Check that Lead II is preserved
    assert np.allclose(data_8lead[:, 1, :], data_12lead[:, 1, :]), \
        "Lead II not preserved"

    # Check that V1-V6 are preserved (indices 2-7 in 8-lead, 6-11 in 12-lead)
    for i in range(6):
        assert np.allclose(data_8lead[:, 2+i, :], data_12lead[:, 6+i, :]), \
            f"Lead V{i+1} not preserved"

    print("✓ test_lead_preservation passed")


def test_derived_leads():
    """Test that derived leads are calculated correctly."""
    num_samples = 10
    signal_length = 500
    data_8lead = np.random.randn(num_samples, 8, signal_length).astype(np.float32)

    data_12lead = convert_8_to_12_lead(data_8lead)

    # Extract leads
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

    print("✓ test_derived_leads passed")


def test_invalid_input_shape():
    """Test that invalid input shapes raise errors."""
    # Test with wrong number of leads
    try:
        data_wrong = np.random.randn(10, 12, 1000)  # 12 leads instead of 8
        convert_8_to_12_lead(data_wrong)
        assert False, "Should have raised ValueError for wrong number of leads"
    except ValueError as e:
        assert "Expected 8 leads" in str(e)
        print("✓ test_invalid_input_shape passed")


def test_different_signal_lengths():
    """Test conversion with different signal lengths."""
    for signal_length in [100, 500, 1000, 2500, 5000]:
        data_8lead = np.random.randn(5, 8, signal_length).astype(np.float32)
        data_12lead = convert_8_to_12_lead(data_8lead)

        assert data_12lead.shape == (5, 12, signal_length), \
            f"Failed for signal length {signal_length}"

    print("✓ test_different_signal_lengths passed")


def test_data_type_preservation():
    """Test that data type is preserved."""
    # Test float32
    data_8lead_f32 = np.random.randn(10, 8, 1000).astype(np.float32)
    data_12lead_f32 = convert_8_to_12_lead(data_8lead_f32)
    assert data_12lead_f32.dtype == np.float32, "float32 not preserved"

    # Test float64
    data_8lead_f64 = np.random.randn(10, 8, 1000).astype(np.float64)
    data_12lead_f64 = convert_8_to_12_lead(data_8lead_f64)
    assert data_12lead_f64.dtype == np.float64, "float64 not preserved"

    print("✓ test_data_type_preservation passed")


def run_all_tests():
    """Run all test functions."""
    print("\n" + "=" * 60)
    print("Running 8-lead to 12-lead Conversion Tests")
    print("=" * 60 + "\n")

    tests = [
        test_single_sample_conversion,
        test_batch_conversion,
        test_lead_preservation,
        test_derived_leads,
        test_invalid_input_shape,
        test_different_signal_lengths,
        test_data_type_preservation,
    ]

    failed_tests = []
    for test_func in tests:
        try:
            test_func()
        except AssertionError as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            failed_tests.append(test_func.__name__)
        except Exception as e:
            print(f"✗ {test_func.__name__} error: {e}")
            failed_tests.append(test_func.__name__)

    print("\n" + "=" * 60)
    if failed_tests:
        print(f"FAILED: {len(failed_tests)} test(s) failed")
        for test_name in failed_tests:
            print(f"  - {test_name}")
        print("=" * 60)
        return 1
    else:
        print("SUCCESS: All tests passed!")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
