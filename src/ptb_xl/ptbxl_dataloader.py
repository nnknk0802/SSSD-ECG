"""
PTB-XL DataLoader Module

This module provides functionality to create PyTorch dataloaders for the PTB-XL dataset.
Converted from ecg_data_preprocessing.ipynb notebook.
"""

import sys
from pathlib import Path

# Handle both relative and absolute imports
try:
    # Try relative import first (when used as a module)
    from .clinical_ts.timeseries_utils import (
        load_dataset,
        reformat_as_memmap,
        TimeseriesDatasetCrops,
        ToTensor
    )
    from .clinical_ts.ecg_utils import prepare_data_ptb_xl, channel_stoi_default
except ImportError:
    # Fall back to absolute import (when run as a script)
    # Add the clinical_ts directory to sys.path
    clinical_ts_path = Path(__file__).parent / 'clinical_ts'
    if str(clinical_ts_path) not in sys.path:
        sys.path.insert(0, str(clinical_ts_path.parent))

    from clinical_ts.timeseries_utils import (
        load_dataset,
        reformat_as_memmap,
        TimeseriesDatasetCrops,
        ToTensor
    )
    from clinical_ts.ecg_utils import prepare_data_ptb_xl, channel_stoi_default

import numpy as np
from torch.utils.data import DataLoader


def multihot_encode(x, num_classes):
    """
    Convert list of class indices to multi-hot encoded vector.

    Args:
        x: List of class indices
        num_classes: Total number of classes

    Returns:
        Multi-hot encoded numpy array of shape (num_classes,)
    """
    res = np.zeros(num_classes, dtype=np.float32)
    for y in x:
        res[y] = 1
    return res


def prepare_ptbxl_data(
    data_folder_ptb_xl,
    target_folder_ptb_xl,
    target_fs=100,
    min_cnt=0,
    channels=12,
    channel_stoi=None,
    recreate_data=True,
    delete_npys=True
):
    """
    Prepare PTB-XL data for training.

    Args:
        data_folder_ptb_xl: Path to raw PTB-XL data folder
        target_folder_ptb_xl: Path to output folder for processed data
        target_fs: Target sampling rate (100 Hz or 500 Hz)
        min_cnt: Minimum count for label filtering
        channels: Number of channels
        channel_stoi: Channel string-to-index mapping
        recreate_data: Whether to recreate data from scratch
        delete_npys: Whether to delete intermediate .npy files after creating memmap

    Returns:
        df_ptb_xl: DataFrame with processed data
        lbl_itos: Label index-to-string mapping
        mean_ptb_xl: Mean values for normalization
        std_ptb_xl: Standard deviation values for normalization
    """
    data_folder_ptb_xl = Path(data_folder_ptb_xl)
    target_folder_ptb_xl = Path(target_folder_ptb_xl)

    if channel_stoi is None:
        channel_stoi = channel_stoi_default

    # Prepare the dataset
    df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl = prepare_data_ptb_xl(
        data_folder_ptb_xl,
        min_cnt=min_cnt,
        target_fs=target_fs,
        channels=channels,
        channel_stoi=channel_stoi,
        target_folder=target_folder_ptb_xl,
        recreate_data=recreate_data
    )

    # Reformat everything as memmap for efficiency
    if recreate_data:
        reformat_as_memmap(
            df_ptb_xl,
            target_folder_ptb_xl / "memmap.npy",
            data_folder=target_folder_ptb_xl,
            delete_npys=delete_npys
        )

    return df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl


def create_ptbxl_dataloaders(
    target_folder_ptb_xl,
    input_size=1000,
    batch_size=8,
    num_workers=4,
    chunkify_train=False,
    chunkify_valtest=False,
    ptb_xl_label="label_diag",
    ds_mean=None,
    ds_std=None
):
    """
    Create PyTorch DataLoaders for PTB-XL dataset.

    Args:
        target_folder_ptb_xl: Path to processed PTB-XL data folder
        input_size: Sample length (default: 1000 for 10 seconds at 100 Hz)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        chunkify_train: Whether to split training samples into chunks
        chunkify_valtest: Whether to split validation/test samples into chunks
        ptb_xl_label: Label type to use (e.g., "label_diag", "label_all")
        ds_mean: Dataset mean for normalization (optional)
        ds_std: Dataset standard deviation for normalization (optional)

    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        lbl_itos: Label index-to-string mapping
    """
    target_folder_ptb_xl = Path(target_folder_ptb_xl)

    # Load dataset
    df_mapped, lbl_itos, mean, std = load_dataset(target_folder_ptb_xl)

    # Default normalization values from the notebook
    # These are computed from the entire PTB-XL dataset at 100 Hz
    if ds_mean is None:
        ds_mean = np.array([
            -0.00184586, -0.00130277,  0.00017031, -0.00091313, -0.00148835,
            -0.00174687, -0.00077071, -0.00207407,  0.00054329,  0.00155546,
            -0.00114379, -0.00035649
        ])

    if ds_std is None:
        ds_std = np.array([
            0.16401004, 0.1647168,  0.23374124, 0.33767231, 0.33362807,
            0.30583013, 0.2731171,  0.27554379, 0.17128962, 0.14030828,
            0.14606956, 0.14656108
        ])

    # Extract label information
    lbl_itos_label = np.array(lbl_itos[ptb_xl_label])

    # Create multi-hot encoded labels
    df_mapped["label"] = df_mapped[ptb_xl_label + "_filtered_numeric"].apply(
        lambda x: multihot_encode(x, len(lbl_itos_label))
    )

    # Define chunk parameters
    chunk_length_train = input_size if chunkify_train else 0
    stride_train = input_size

    chunk_length_valtest = input_size if chunkify_valtest else 0
    stride_valtest = input_size

    # Create transforms
    tfms_ptb_xl = ToTensor()

    # Split data into train/val/test based on strat_fold
    max_fold_id = df_mapped.strat_fold.max()
    df_train = df_mapped[df_mapped.strat_fold < max_fold_id - 1]
    df_val = df_mapped[df_mapped.strat_fold == max_fold_id - 1]
    df_test = df_mapped[df_mapped.strat_fold == max_fold_id]

    # Create datasets
    ds_train = TimeseriesDatasetCrops(
        df_train,
        input_size,
        num_classes=len(lbl_itos_label),
        data_folder=target_folder_ptb_xl,
        chunk_length=chunk_length_train,
        min_chunk_length=input_size,
        stride=stride_train,
        transforms=tfms_ptb_xl,
        annotation=False,
        col_lbl="label",
        memmap_filename=target_folder_ptb_xl / "memmap.npy"
    )

    ds_val = TimeseriesDatasetCrops(
        df_val,
        input_size,
        num_classes=len(lbl_itos_label),
        data_folder=target_folder_ptb_xl,
        chunk_length=chunk_length_valtest,
        min_chunk_length=input_size,
        stride=stride_valtest,
        transforms=tfms_ptb_xl,
        annotation=False,
        col_lbl="label",
        memmap_filename=target_folder_ptb_xl / "memmap.npy"
    )

    ds_test = TimeseriesDatasetCrops(
        df_test,
        input_size,
        num_classes=len(lbl_itos_label),
        data_folder=target_folder_ptb_xl,
        chunk_length=chunk_length_valtest,
        min_chunk_length=input_size,
        stride=stride_valtest,
        transforms=tfms_ptb_xl,
        annotation=False,
        col_lbl="label",
        memmap_filename=target_folder_ptb_xl / "memmap.npy"
    )

    # Create dataloaders
    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True
    )

    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, lbl_itos_label


def get_ptbxl_dataloaders(
    data_folder_ptb_xl,
    target_folder_ptb_xl,
    target_fs=100,
    input_size=1000,
    batch_size=8,
    num_workers=4,
    min_cnt=0,
    recreate_data=False,
    delete_npys=True,
    chunkify_train=False,
    chunkify_valtest=False,
    ptb_xl_label="label_diag"
):
    """
    Complete pipeline to prepare data and create dataloaders for PTB-XL.

    Args:
        data_folder_ptb_xl: Path to raw PTB-XL data folder
        target_folder_ptb_xl: Path to output folder for processed data
        target_fs: Target sampling rate (100 Hz or 500 Hz)
        input_size: Sample length (default: 1000 for 10 seconds at 100 Hz)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        min_cnt: Minimum count for label filtering
        recreate_data: Whether to recreate data from scratch
        delete_npys: Whether to delete intermediate .npy files after creating memmap
        chunkify_train: Whether to split training samples into chunks
        chunkify_valtest: Whether to split validation/test samples into chunks
        ptb_xl_label: Label type to use (e.g., "label_diag", "label_all")

    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        lbl_itos: Label index-to-string mapping
        mean: Mean values for normalization
        std: Standard deviation values for normalization
    """
    # Prepare data if needed
    if recreate_data:
        df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl = prepare_ptbxl_data(
            data_folder_ptb_xl=data_folder_ptb_xl,
            target_folder_ptb_xl=target_folder_ptb_xl,
            target_fs=target_fs,
            min_cnt=min_cnt,
            recreate_data=recreate_data,
            delete_npys=delete_npys
        )

    # Create dataloaders
    train_loader, val_loader, test_loader, lbl_itos = create_ptbxl_dataloaders(
        target_folder_ptb_xl=target_folder_ptb_xl,
        input_size=input_size,
        batch_size=batch_size,
        num_workers=num_workers,
        chunkify_train=chunkify_train,
        chunkify_valtest=chunkify_valtest,
        ptb_xl_label=ptb_xl_label
    )

    # Load normalization stats
    _, _, mean, std = load_dataset(Path(target_folder_ptb_xl))

    return train_loader, val_loader, test_loader, lbl_itos, mean, std


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="PTB-XL DataLoader")
    parser.add_argument(
        "--data_folder",
        type=str,
        default="./data_folder_ptb_xl/",
        help="Path to raw PTB-XL data folder"
    )
    parser.add_argument(
        "--target_folder",
        type=str,
        default="./processed_ptb_xl_fs100",
        help="Path to output folder for processed data"
    )
    parser.add_argument(
        "--target_fs",
        type=int,
        default=100,
        help="Target sampling rate (100 or 500 Hz)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for dataloaders"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading"
    )
    parser.add_argument(
        "--recreate_data",
        action="store_true",
        help="Recreate data from scratch"
    )

    args = parser.parse_args()

    # Create dataloaders
    train_loader, val_loader, test_loader, lbl_itos, mean, std = get_ptbxl_dataloaders(
        data_folder_ptb_xl=args.data_folder,
        target_folder_ptb_xl=args.target_folder,
        target_fs=args.target_fs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        recreate_data=args.recreate_data
    )

    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    print(f"Number of classes: {len(lbl_itos)}")
    print(f"Mean shape: {mean.shape}")
    print(f"Std shape: {std.shape}")

    # Test loading a batch
    for batch_idx, (data, labels) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Data shape: {data.shape}")
        print(f"  Labels shape: {labels.shape}")
        if batch_idx == 0:
            break
