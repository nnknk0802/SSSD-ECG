"""
PTB-XL DataLoader Module (Version 2 - Workaround)

This version fixes the issue where label columns are lost after reformat_as_memmap.
"""

import sys
from pathlib import Path

# Handle both relative and absolute imports
try:
    from .clinical_ts.timeseries_utils import (
        load_dataset,
        reformat_as_memmap,
        TimeseriesDatasetCrops,
        ToTensor
    )
    from .clinical_ts.ecg_utils import prepare_data_ptb_xl, channel_stoi_default
except ImportError:
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
    """
    res = np.zeros(num_classes, dtype=np.float32)
    for y in x:
        res[y] = 1
    return res


def get_ptbxl_dataloaders_v2(
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
    Complete pipeline with workaround for label column issue.

    This version ensures that label columns are preserved through the memmap process.
    """
    data_folder_ptb_xl = Path(data_folder_ptb_xl)
    target_folder_ptb_xl = Path(target_folder_ptb_xl)

    # Step 1: Prepare data
    print("Step 1: Preparing PTB-XL data...")
    df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl = prepare_data_ptb_xl(
        data_folder_ptb_xl,
        min_cnt=min_cnt,
        target_fs=target_fs,
        channels=12,
        channel_stoi=channel_stoi_default,
        target_folder=target_folder_ptb_xl,
        recreate_data=recreate_data
    )
    print(f"  DataFrame shape: {df_ptb_xl.shape}")
    print(f"  Columns: {len(df_ptb_xl.columns)}")

    # Check if label columns exist
    label_col_key = ptb_xl_label + "_filtered_numeric"
    if label_col_key not in df_ptb_xl.columns:
        raise ValueError(
            f"Column '{label_col_key}' not found in DataFrame. "
            f"Available columns: {list(df_ptb_xl.columns)}"
        )

    # Step 2: Reformat as memmap
    if recreate_data:
        print("\nStep 2: Reformatting as memmap...")
        df_mapped = reformat_as_memmap(
            df_ptb_xl,
            target_folder_ptb_xl / "memmap.npy",
            data_folder=target_folder_ptb_xl,
            delete_npys=delete_npys
        )
        print(f"  Mapped DataFrame shape: {df_mapped.shape}")
        print(f"  Mapped DataFrame columns: {len(df_mapped.columns)}")

        # Verify that label columns are still there
        if label_col_key not in df_mapped.columns:
            raise ValueError(
                f"Label column '{label_col_key}' was lost during memmap reformatting! "
                f"This is a bug in reformat_as_memmap()."
            )
    else:
        print("\nStep 2: Loading existing memmap data...")
        df_mapped, _, _, _ = load_dataset(target_folder_ptb_xl)

    # Step 3: Create multi-hot encoded labels directly from df_mapped
    print("\nStep 3: Creating multi-hot encoded labels...")
    lbl_itos_label = np.array(lbl_itos_ptb_xl[ptb_xl_label])

    df_mapped["label"] = df_mapped[label_col_key].apply(
        lambda x: multihot_encode(x, len(lbl_itos_label))
    )
    print(f"  Number of classes: {len(lbl_itos_label)}")

    # Step 4: Create datasets and dataloaders
    print("\nStep 4: Creating datasets and dataloaders...")

    # Define chunk parameters
    chunk_length_train = input_size if chunkify_train else 0
    stride_train = input_size
    chunk_length_valtest = input_size if chunkify_valtest else 0
    stride_valtest = input_size

    # Create transforms
    tfms_ptb_xl = ToTensor()

    # Split data
    max_fold_id = df_mapped.strat_fold.max()
    df_train = df_mapped[df_mapped.strat_fold < max_fold_id - 1]
    df_val = df_mapped[df_mapped.strat_fold == max_fold_id - 1]
    df_test = df_mapped[df_mapped.strat_fold == max_fold_id]

    print(f"  Train samples: {len(df_train)}")
    print(f"  Val samples: {len(df_val)}")
    print(f"  Test samples: {len(df_test)}")

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

    print("\n✓ Dataloaders created successfully!")

    return train_loader, val_loader, test_loader, lbl_itos_label, mean_ptb_xl, std_ptb_xl


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PTB-XL DataLoader V2")
    parser.add_argument("--data_folder", type=str, default="./data_folder_ptb_xl/")
    parser.add_argument("--target_folder", type=str, default="./processed_ptb_xl_fs100")
    parser.add_argument("--target_fs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--recreate_data", action="store_true")

    args = parser.parse_args()

    train_loader, val_loader, test_loader, lbl_itos, mean, std = get_ptbxl_dataloaders_v2(
        data_folder_ptb_xl=args.data_folder,
        target_folder_ptb_xl=args.target_folder,
        target_fs=args.target_fs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        recreate_data=args.recreate_data
    )

    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")

    # Test loading a batch
    for batch_idx, (data, labels) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Data shape: {data.shape}")
        print(f"  Labels shape: {labels.shape}")
        if batch_idx == 0:
            break
