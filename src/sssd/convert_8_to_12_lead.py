#!/usr/bin/env python3
"""
8-Lead to 12-Lead ECG Conversion Script

This standalone script converts 8-lead ECG data (I, II, V1-V6) to 12-lead ECG data
by deriving the missing leads (III, aVR, aVL, aVF) using standard ECG formulas.

Standard 12-lead ECG order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6

Derivation formulas:
- Lead III = Lead II - Lead I
- aVR = -(Lead I + Lead II) / 2
- aVL = Lead I - Lead II / 2
- aVF = Lead II - Lead I / 2

Input format:
    8-lead ECG data in shape (num_samples, 8, signal_length)
    where the 8 leads are ordered as: I, II, V1, V2, V3, V4, V5, V6

Output format:
    12-lead ECG data in shape (num_samples, 12, signal_length)
    where the 12 leads are ordered as: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
"""

import argparse
import numpy as np
import os
from pathlib import Path


def convert_8_to_12_lead(data_8lead):
    """
    Convert 8-lead ECG data to 12-lead ECG data.

    Args:
        data_8lead (np.ndarray): 8-lead ECG data with shape (num_samples, 8, signal_length)
                                 or (8, signal_length) for single sample
                                 Leads ordered as: I, II, V1, V2, V3, V4, V5, V6

    Returns:
        np.ndarray: 12-lead ECG data with shape (num_samples, 12, signal_length)
                    or (12, signal_length) for single sample
                    Leads ordered as: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    """
    # Handle single sample vs batch
    single_sample = False
    if data_8lead.ndim == 2:
        single_sample = True
        data_8lead = data_8lead[np.newaxis, :]  # Add batch dimension

    num_samples, num_leads, signal_length = data_8lead.shape

    if num_leads != 8:
        raise ValueError(f"Expected 8 leads, got {num_leads}")

    # Extract the limb leads (I, II) and chest leads (V1-V6)
    lead_I = data_8lead[:, 0, :]   # Index 0
    lead_II = data_8lead[:, 1, :]  # Index 1
    lead_V1 = data_8lead[:, 2, :]  # Index 2
    lead_V2 = data_8lead[:, 3, :]  # Index 3
    lead_V3 = data_8lead[:, 4, :]  # Index 4
    lead_V4 = data_8lead[:, 5, :]  # Index 5
    lead_V5 = data_8lead[:, 6, :]  # Index 6
    lead_V6 = data_8lead[:, 7, :]  # Index 7

    # Derive the missing limb leads using Einthoven's triangle
    lead_III = lead_II - lead_I
    lead_aVR = -(lead_I + lead_II) / 2
    lead_aVL = lead_I - lead_II / 2
    lead_aVF = lead_II - lead_I / 2

    # Construct 12-lead ECG in standard order
    # I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    data_12lead = np.stack([
        lead_I,    # Index 0
        lead_II,   # Index 1
        lead_III,  # Index 2
        lead_aVR,  # Index 3
        lead_aVL,  # Index 4
        lead_aVF,  # Index 5
        lead_V1,   # Index 6
        lead_V2,   # Index 7
        lead_V3,   # Index 8
        lead_V4,   # Index 9
        lead_V5,   # Index 10
        lead_V6    # Index 11
    ], axis=1)

    # Remove batch dimension if input was single sample
    if single_sample:
        data_12lead = data_12lead[0]

    return data_12lead


def load_8lead_data(input_path):
    """
    Load 8-lead ECG data from file.

    Args:
        input_path (str): Path to input .npy file

    Returns:
        np.ndarray: 8-lead ECG data
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = np.load(input_path)
    print(f"Loaded data shape: {data.shape}")

    return data


def save_12lead_data(data_12lead, output_path):
    """
    Save 12-lead ECG data to file.

    Args:
        data_12lead (np.ndarray): 12-lead ECG data
        output_path (str): Path to output .npy file
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    np.save(output_path, data_12lead)
    print(f"Saved 12-lead data to: {output_path}")
    print(f"Output shape: {data_12lead.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert 8-lead ECG data to 12-lead ECG data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single file
  python convert_8_to_12_lead.py --input data_8lead.npy --output data_12lead.npy

  # With custom input/output paths
  python convert_8_to_12_lead.py -i /path/to/8lead.npy -o /path/to/12lead.npy

Expected input format:
  - Shape: (num_samples, 8, signal_length) or (8, signal_length)
  - Lead order: I, II, V1, V2, V3, V4, V5, V6

Output format:
  - Shape: (num_samples, 12, signal_length) or (12, signal_length)
  - Lead order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
        """
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to input 8-lead ECG data (.npy file)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Path to output 12-lead ECG data (.npy file)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed information'
    )

    args = parser.parse_args()

    try:
        # Load 8-lead data
        if args.verbose:
            print(f"Loading 8-lead data from: {args.input}")
        data_8lead = load_8lead_data(args.input)

        # Validate input shape
        if data_8lead.ndim not in [2, 3]:
            raise ValueError(
                f"Invalid input shape: {data_8lead.shape}. "
                f"Expected (num_samples, 8, signal_length) or (8, signal_length)"
            )

        if data_8lead.ndim == 2 and data_8lead.shape[0] != 8:
            raise ValueError(
                f"Invalid number of leads: {data_8lead.shape[0]}. Expected 8 leads"
            )
        elif data_8lead.ndim == 3 and data_8lead.shape[1] != 8:
            raise ValueError(
                f"Invalid number of leads: {data_8lead.shape[1]}. Expected 8 leads"
            )

        # Convert to 12-lead
        if args.verbose:
            print("Converting to 12-lead ECG...")
        data_12lead = convert_8_to_12_lead(data_8lead)

        # Save result
        save_12lead_data(data_12lead, args.output)

        if args.verbose:
            print("\nConversion completed successfully!")
            print(f"Input shape:  {data_8lead.shape}")
            print(f"Output shape: {data_12lead.shape}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
