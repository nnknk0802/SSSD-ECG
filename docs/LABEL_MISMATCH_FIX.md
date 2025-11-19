# Label Dimension Mismatch Fix

## Problem Description

When using the SSSD-ECG model with PTB-XL data, you may encounter the following error:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x44 and 71x128)
```

This error occurs in `SSSD_ECG.py` at line 201:
```python
label_embed = label @ self.embedding.weight if self.embedding is not None else None
```

### Root Cause

The error happens when there's a mismatch between:
- **Model configuration**: Expects 71 classes (as specified in `config_SSSD_ECG.json`)
- **Actual data**: Has 44 classes (when using PTB-XL with `label_diag`)

The shape mismatch:
- `label`: shape `(batch_size, 44)` - multi-hot encoded labels from PTB-XL
- `self.embedding.weight`: shape `(71, 128)` - embedding matrix from config

For matrix multiplication `A @ B` to work, the inner dimensions must match (44 ≠ 71), causing the error.

### Why Does This Happen?

1. The original SSSD-ECG paper used 71 diagnostic classes
2. PTB-XL dataset has different numbers of classes depending on:
   - Label type used (`label_diag`, `label_form`, `label_rhythm`)
   - Filtering criteria (`min_cnt` parameter)
   - Dataset version and preprocessing

For example:
- `label_diag` (diagnostic labels): typically 44 classes
- `label_all` (all labels): can have 71+ classes

## Solution

### Option 1: Override num_classes Parameter (Recommended)

The `SSSDECG` class now accepts a `num_classes` parameter to override the config value:

```python
from model_wrapper import SSSDECG
from ptbxl_dataloader import get_ptbxl_dataloaders

# Load data
train_loader, val_loader, test_loader, lbl_itos, mean, std = get_ptbxl_dataloaders(
    data_folder_ptb_xl="./data",
    target_folder_ptb_xl="./processed",
    recreate_data=False,
    batch_size=8,
)

# Get actual number of classes from data
num_classes = len(lbl_itos)
print(f"Using {num_classes} classes")

# Initialize model with correct number of classes
model = SSSDECG(
    config_path="config/config_SSSD_ECG.json",
    num_classes=num_classes  # ← Override config value
)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
for x, y in train_loader:
    loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Option 2: Update Configuration File

Alternatively, you can modify `config/config_SSSD_ECG.json`:

```json
{
    "wavenet_config": {
        ...
        "label_embed_classes": 44  // ← Change from 71 to 44
    }
}
```

**Note**: This requires creating separate config files for different label types.

## Complete Example

See `examples/fix_label_mismatch_example.py` for a complete working example.

## Different Label Types

PTB-XL supports multiple label types:

| Label Type | Typical # Classes | Description |
|------------|------------------|-------------|
| `label_diag` | 44 | Diagnostic labels (e.g., MI, STTC) |
| `label_form` | ~20 | Morphological labels |
| `label_rhythm` | ~12 | Rhythm labels |
| `label_all` | 71+ | All diagnostic labels combined |

To use different label types, specify when creating the dataloader:

```python
train_loader, val_loader, test_loader, lbl_itos, mean, std = get_ptbxl_dataloaders(
    data_folder_ptb_xl="./data",
    target_folder_ptb_xl="./processed",
    ptb_xl_label="label_diag",  # ← Specify label type
    recreate_data=False
)

num_classes = len(lbl_itos)
model = SSSDECG(config_path="config.json", num_classes=num_classes)
```

## Implementation Details

The fix modifies `model_wrapper.py`:

```python
class SSSDECG(nn.Module):
    def __init__(self, config_path=None, device=None, num_classes=None):
        # ... load config ...

        # Override number of classes if provided
        if num_classes is not None:
            self.model_config["label_embed_classes"] = num_classes
            print(f"Using {num_classes} classes (overriding config value)")

        # Initialize model with updated config
        self.model = SSSD_ECG_Base(**self.model_config).to(self.device)
```

This ensures the embedding layer is created with the correct dimensions:
```python
# In SSSD_ECG.py
self.embedding = nn.Embedding(label_embed_classes, label_embed_dim)
# Now uses the correct label_embed_classes value
```

## Backward Compatibility

The fix is fully backward compatible:
- If `num_classes` is not specified, uses the config value (71)
- Existing code continues to work without modification
- Only projects with different class counts need to use the parameter

## Troubleshooting

If you still encounter errors:

1. **Verify the number of classes in your data:**
   ```python
   print(f"Number of classes: {len(lbl_itos)}")
   print(f"Classes: {lbl_itos}")
   ```

2. **Check label shape:**
   ```python
   for x, y in train_loader:
       print(f"Data shape: {x.shape}")
       print(f"Label shape: {y.shape}")  # Should be (batch_size, num_classes)
       break
   ```

3. **Verify model configuration:**
   ```python
   model = SSSDECG(num_classes=44)
   print(f"Model expects {model.model_config['label_embed_classes']} classes")
   ```

## References

- PTB-XL Dataset: https://physionet.org/content/ptb-xl/
- SSSD-ECG Paper: https://doi.org/10.1016/j.compbiomed.2023.107115
