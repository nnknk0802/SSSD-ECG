"""
Example usage of PTB-XL DataLoader

This script demonstrates how to use the PTB-XL dataloader module.
"""

import sys
from pathlib import Path

# Add parent directory to path to import ptbxl_dataloader
sys.path.insert(0, str(Path(__file__).parent))

from ptbxl_dataloader import (
    get_ptbxl_dataloaders,
    create_ptbxl_dataloaders,
    prepare_ptbxl_data
)


def example_full_pipeline():
    """
    Example: Full pipeline from raw data to dataloaders
    """
    print("=" * 80)
    print("Example 1: Full Pipeline (Prepare Data + Create DataLoaders)")
    print("=" * 80)

    # Configuration
    data_folder = "./data_folder_ptb_xl/"
    target_folder = "./processed_ptb_xl_fs100"
    target_fs = 100  # 100 Hz or 500 Hz
    batch_size = 8
    num_workers = 4

    # Create dataloaders (this will also prepare data if recreate_data=True)
    train_loader, val_loader, test_loader, lbl_itos, mean, std = get_ptbxl_dataloaders(
        data_folder_ptb_xl=data_folder,
        target_folder_ptb_xl=target_folder,
        target_fs=target_fs,
        batch_size=batch_size,
        num_workers=num_workers,
        recreate_data=True,  # Set to False if data is already prepared
        min_cnt=0,  # Minimum count for label filtering
        ptb_xl_label="label_diag"  # Use diagnostic labels
    )

    # Print information
    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    print(f"Number of classes: {len(lbl_itos)}")
    print(f"Classes: {lbl_itos[:10]}...")  # Print first 10 classes
    print(f"\nNormalization stats:")
    print(f"  Mean shape: {mean.shape}")
    print(f"  Std shape: {std.shape}")

    # Test loading a batch
    print("\n" + "-" * 80)
    print("Testing batch loading...")
    print("-" * 80)
    for batch_idx, (data, labels) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Data shape: {data.shape}")  # [batch_size, channels, timesteps]
        print(f"  Labels shape: {labels.shape}")  # [batch_size, num_classes]
        print(f"  Data dtype: {data.dtype}")
        print(f"  Labels dtype: {labels.dtype}")
        print(f"  Data range: [{data.min():.4f}, {data.max():.4f}]")
        print(f"  Positive labels in batch: {labels.sum(dim=1).mean():.2f}")
        if batch_idx == 2:  # Load 3 batches
            break


def example_existing_data():
    """
    Example: Create dataloaders from already prepared data
    """
    print("\n\n")
    print("=" * 80)
    print("Example 2: Create DataLoaders from Existing Prepared Data")
    print("=" * 80)

    # Configuration
    target_folder = "./processed_ptb_xl_fs100"
    batch_size = 16
    num_workers = 4

    # Create dataloaders from already prepared data
    train_loader, val_loader, test_loader, lbl_itos = create_ptbxl_dataloaders(
        target_folder_ptb_xl=target_folder,
        input_size=1000,  # 10 seconds at 100 Hz
        batch_size=batch_size,
        num_workers=num_workers,
        chunkify_train=False,  # Don't split training samples into chunks
        chunkify_valtest=False,  # Don't split val/test samples into chunks
        ptb_xl_label="label_diag"
    )

    # Print information
    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    print(f"Number of classes: {len(lbl_itos)}")


def example_prepare_data_only():
    """
    Example: Only prepare data without creating dataloaders
    """
    print("\n\n")
    print("=" * 80)
    print("Example 3: Prepare Data Only (No DataLoaders)")
    print("=" * 80)

    # Configuration
    data_folder = "./data_folder_ptb_xl/"
    target_folder = "./processed_ptb_xl_fs100"

    # Only prepare data
    df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl = prepare_ptbxl_data(
        data_folder_ptb_xl=data_folder,
        target_folder_ptb_xl=target_folder,
        target_fs=100,
        min_cnt=0,
        recreate_data=True,
        delete_npys=True  # Delete intermediate .npy files to save space
    )

    # Print information
    print(f"\nDataFrame shape: {df_ptb_xl.shape}")
    print(f"Number of label types: {len(lbl_itos_ptb_xl)}")
    print(f"Label types: {list(lbl_itos_ptb_xl.keys())}")
    print(f"Mean shape: {mean_ptb_xl.shape}")
    print(f"Std shape: {std_ptb_xl.shape}")


def example_with_custom_parameters():
    """
    Example: Custom parameters for specific use cases
    """
    print("\n\n")
    print("=" * 80)
    print("Example 4: Custom Parameters")
    print("=" * 80)

    # Configuration with custom parameters
    target_folder = "./processed_ptb_xl_fs100"

    # Use different label types
    for label_type in ["label_diag", "label_form", "label_rhythm"]:
        print(f"\n--- Using {label_type} ---")

        train_loader, val_loader, test_loader, lbl_itos = create_ptbxl_dataloaders(
            target_folder_ptb_xl=target_folder,
            input_size=1000,
            batch_size=8,
            num_workers=2,
            chunkify_train=False,
            chunkify_valtest=False,
            ptb_xl_label=label_type
        )

        print(f"Number of {label_type} classes: {len(lbl_itos)}")
        print(f"Classes: {lbl_itos[:5]}...")  # Print first 5 classes


def main():
    """
    Run examples based on user input
    """
    import argparse

    parser = argparse.ArgumentParser(description="PTB-XL DataLoader Examples")
    parser.add_argument(
        "--example",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Which example to run (1-4)"
    )

    args = parser.parse_args()

    examples = {
        1: example_full_pipeline,
        2: example_existing_data,
        3: example_prepare_data_only,
        4: example_with_custom_parameters
    }

    try:
        examples[args.example]()
        print("\n" + "=" * 80)
        print("Example completed successfully!")
        print("=" * 80)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure you have the PTB-XL dataset downloaded and the paths are correct.")
        print("You can download the dataset from: https://physionet.org/content/ptb-xl/")
    except Exception as e:
        print(f"\nError running example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
