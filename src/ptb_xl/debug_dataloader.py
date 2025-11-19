"""
Debug script for PTB-XL dataloader
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ptb_xl.ptbxl_dataloader import prepare_ptbxl_data, create_ptbxl_dataloaders

# Set your paths here
data_dir = "./data_folder_ptb_xl/"  # Replace with your actual path
save_dir = "./processed_ptb_xl_fs100"  # Replace with your actual path

print("=" * 80)
print("PTB-XL DataLoader Debug")
print("=" * 80)

# Check if data directory exists
data_path = Path(data_dir)
print(f"\n1. Checking data directory: {data_path}")
if not data_path.exists():
    print(f"   ERROR: Data directory does not exist!")
    print(f"   Please download PTB-XL data to this location.")
    sys.exit(1)
else:
    print(f"   OK: Data directory exists")
    # Check for key files
    csv_file = data_path / "ptbxl_database.csv"
    if csv_file.exists():
        print(f"   OK: Found ptbxl_database.csv")
    else:
        print(f"   ERROR: ptbxl_database.csv not found!")
        print(f"   Files in directory:")
        for f in list(data_path.glob("*"))[:10]:
            print(f"      {f.name}")
        sys.exit(1)

# Try to prepare data
print(f"\n2. Attempting to prepare PTB-XL data...")
print(f"   Source: {data_dir}")
print(f"   Target: {save_dir}")

try:
    df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl = prepare_ptbxl_data(
        data_folder_ptb_xl=data_dir,
        target_folder_ptb_xl=save_dir,
        target_fs=100,
        min_cnt=0,
        recreate_data=True,
        delete_npys=True
    )
    print(f"   OK: Data preparation completed")
    print(f"   DataFrame shape: {df_ptb_xl.shape}")
    print(f"   DataFrame columns: {list(df_ptb_xl.columns)[:10]}...")
    print(f"   Number of label types: {len(lbl_itos_ptb_xl)}")

    # Check which label columns exist
    print(f"\n   Label columns in DataFrame:")
    label_cols = [col for col in df_ptb_xl.columns if 'label' in col]
    for col in label_cols[:20]:  # Show first 20 label columns
        print(f"      - {col}")

except Exception as e:
    print(f"   ERROR during data preparation:")
    print(f"   {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check if memmap files were created
print(f"\n3. Checking created files in {save_dir}...")
save_path = Path(save_dir)
if save_path.exists():
    files = list(save_path.glob("*"))
    for f in files:
        print(f"   - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
else:
    print(f"   ERROR: Target directory was not created!")
    sys.exit(1)

# Try to create dataloaders
print(f"\n4. Attempting to create dataloaders...")

try:
    train_loader, val_loader, test_loader, lbl_itos = create_ptbxl_dataloaders(
        target_folder_ptb_xl=save_dir,
        input_size=1000,
        batch_size=8,
        num_workers=0,  # Use 0 for debugging
        chunkify_train=False,
        chunkify_valtest=False,
        ptb_xl_label="label_diag"
    )
    print(f"   OK: DataLoaders created successfully")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    print(f"   Number of classes: {len(lbl_itos)}")

except Exception as e:
    print(f"   ERROR during dataloader creation:")
    print(f"   {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

    # Additional debugging: check what was loaded
    print(f"\n   Debugging: Checking loaded DataFrame...")
    try:
        from ptb_xl.clinical_ts.timeseries_utils import load_dataset
        df_mapped, lbl_itos_loaded, _, _ = load_dataset(save_dir)
        print(f"   Loaded DataFrame shape: {df_mapped.shape}")
        print(f"   Loaded DataFrame columns:")
        for col in df_mapped.columns:
            print(f"      - {col}")
    except Exception as e2:
        print(f"   ERROR loading dataset: {e2}")

    sys.exit(1)

# Try to load one batch
print(f"\n5. Testing batch loading...")
try:
    for batch_idx, (data, labels) in enumerate(train_loader):
        print(f"   Batch {batch_idx}:")
        print(f"      Data shape: {data.shape}")
        print(f"      Labels shape: {labels.shape}")
        if batch_idx == 0:
            break
    print(f"   OK: Successfully loaded a batch!")

except Exception as e:
    print(f"   ERROR during batch loading:")
    print(f"   {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("All tests passed! DataLoader is working correctly.")
print("=" * 80)
