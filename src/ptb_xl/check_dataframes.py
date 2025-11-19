"""
Check saved DataFrames for PTB-XL data
"""

import sys
from pathlib import Path
import pickle
import numpy as np

# Set your target folder here
target_folder = "./processed_ptb_xl_fs100"  # Replace with your actual path

print("=" * 80)
print("PTB-XL DataFrame Checker")
print("=" * 80)

target_path = Path(target_folder)

if not target_path.exists():
    print(f"\nERROR: Target folder does not exist: {target_folder}")
    sys.exit(1)

print(f"\nTarget folder: {target_folder}")
print(f"\nFiles in target folder:")
for f in sorted(target_path.glob("*")):
    if f.is_file():
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name:40s} {size_mb:10.2f} MB")

# Check for pickle files
print("\n" + "=" * 80)
print("Checking DataFrame files:")
print("=" * 80)

df_files = {
    "df.pkl": "Original DataFrame (from prepare_data_ptb_xl)",
    "df_memmap.pkl": "Memmap DataFrame (from reformat_as_memmap)"
}

for filename, description in df_files.items():
    filepath = target_path / filename
    print(f"\n{filename} - {description}")
    print("-" * 80)

    if not filepath.exists():
        print(f"  ❌ File does not exist")
        continue

    try:
        df = pickle.load(open(filepath, "rb"))
        print(f"  ✓ File loaded successfully")
        print(f"  Shape: {df.shape}")
        print(f"  Number of columns: {len(df.columns)}")
        print(f"\n  Columns:")
        for col in df.columns:
            print(f"    - {col}")

        # Check for label columns
        print(f"\n  Label-related columns:")
        label_cols = [col for col in df.columns if 'label' in col.lower()]
        if label_cols:
            for col in label_cols:
                print(f"    ✓ {col}")
        else:
            print(f"    ❌ No label columns found!")

        # Check specifically for the problematic column
        if 'label_diag_filtered_numeric' in df.columns:
            print(f"\n  ✓ 'label_diag_filtered_numeric' column EXISTS")
            print(f"    Sample value: {df['label_diag_filtered_numeric'].iloc[0]}")
        else:
            print(f"\n  ❌ 'label_diag_filtered_numeric' column MISSING")

    except Exception as e:
        print(f"  ❌ Error loading file: {e}")
        import traceback
        traceback.print_exc()

# Check lbl_itos files
print("\n" + "=" * 80)
print("Checking label mapping files:")
print("=" * 80)

lbl_files = list(target_path.glob("lbl_itos*"))
for filepath in lbl_files:
    print(f"\n{filepath.name}")
    print("-" * 80)
    try:
        if filepath.suffix == '.pkl':
            lbl_itos = pickle.load(open(filepath, "rb"))
            print(f"  Type: {type(lbl_itos)}")
            if isinstance(lbl_itos, dict):
                print(f"  Keys: {list(lbl_itos.keys())}")
                for key, value in lbl_itos.items():
                    if isinstance(value, (list, np.ndarray)):
                        print(f"    {key}: {len(value)} items")
                        print(f"      First 5: {value[:5]}")
                    else:
                        print(f"    {key}: {value}")
            elif isinstance(lbl_itos, (list, np.ndarray)):
                print(f"  Length: {len(lbl_itos)}")
                print(f"  First 10 items: {lbl_itos[:10]}")
        elif filepath.suffix == '.npy':
            lbl_itos = np.load(filepath, allow_pickle=True)
            print(f"  Type: {type(lbl_itos)}")
            print(f"  Shape: {lbl_itos.shape}")
            print(f"  First 10 items: {lbl_itos[:10]}")
    except Exception as e:
        print(f"  ❌ Error loading file: {e}")

print("\n" + "=" * 80)
print("Summary:")
print("=" * 80)

# Load both DataFrames and compare
df_path = target_path / "df.pkl"
df_memmap_path = target_path / "df_memmap.pkl"

if df_path.exists() and df_memmap_path.exists():
    try:
        df_original = pickle.load(open(df_path, "rb"))
        df_memmap = pickle.load(open(df_memmap_path, "rb"))

        print(f"\nColumn comparison:")
        print(f"  df.pkl columns: {len(df_original.columns)}")
        print(f"  df_memmap.pkl columns: {len(df_memmap.columns)}")

        # Find missing columns
        cols_in_original = set(df_original.columns)
        cols_in_memmap = set(df_memmap.columns)

        missing_in_memmap = cols_in_original - cols_in_memmap
        extra_in_memmap = cols_in_memmap - cols_in_original

        if missing_in_memmap:
            print(f"\n  ❌ Columns in df.pkl but MISSING in df_memmap.pkl:")
            for col in sorted(missing_in_memmap):
                print(f"      - {col}")
        else:
            print(f"\n  ✓ All columns from df.pkl are present in df_memmap.pkl")

        if extra_in_memmap:
            print(f"\n  Extra columns in df_memmap.pkl:")
            for col in sorted(extra_in_memmap):
                print(f"      - {col}")

    except Exception as e:
        print(f"  ❌ Error comparing DataFrames: {e}")
else:
    print(f"\n  Cannot compare: One or both DataFrame files are missing")

print("\n" + "=" * 80)
