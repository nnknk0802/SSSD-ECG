"""
Example: Fix for Label Dimension Mismatch Error

This example demonstrates how to fix the common error:
RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x44 and 71x128)

This error occurs when the number of classes in your data doesn't match
the number of classes configured in the model.

Solution: Use the num_classes parameter when initializing SSSDECG.
"""

import sys
sys.path.append("../sssd_standalone")
sys.path.append("../src/ptb_xl")

import torch
from model_wrapper import SSSDECG
from ptbxl_dataloader import get_ptbxl_dataloaders

# Path to your processed data
save_dir = "/export/work/users/nonaka/project/SynECG/dataset/v251119_ptbxl_for_sssd"

# 1. Load the dataloader
train_loader, val_loader, test_loader, lbl_itos, mean, std = get_ptbxl_dataloaders(
    data_folder_ptb_xl="./dummy",
    target_folder_ptb_xl=save_dir,
    recreate_data=False,
    batch_size=8,
)

# Get the actual number of classes from the data
num_classes = len(lbl_itos)
print(f"Number of classes in data: {num_classes}")

# 2. Initialize model with the correct number of classes
# IMPORTANT: Pass num_classes parameter to override the config value
sssd_cfg = "../sssd_standalone/config/config_SSSD_ECG.json"
model = SSSDECG(config_path=sssd_cfg, num_classes=num_classes)

# 3. Setup optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# 4. Training loop (now working without dimension mismatch error!)
print("\nStarting training...")
for batch_idx, (x, y) in enumerate(train_loader):
    loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")

    # Just test first batch for demonstration
    if batch_idx == 0:
        print("\n✓ Training successful! No dimension mismatch error.")
        break
