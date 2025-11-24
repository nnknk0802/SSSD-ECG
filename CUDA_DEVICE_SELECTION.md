# CUDA Device Selection

This document explains how to use specific CUDA devices (cuda:0, cuda:1, etc.) with the SSSD-ECG model.

## Problem Fixed

Previously, the model was hardcoded to use `cuda:0` only. Attempting to use other CUDA devices (like `cuda:1`) would result in errors because `.cuda()` calls were hardcoded throughout the codebase.

## Solution

All hardcoded `.cuda()` calls have been replaced with device-aware `.to(device)` calls. The device parameter now properly propagates through all functions.

## Usage

### Specifying CUDA Device During Initialization

You can now specify any CUDA device when initializing the model:

```python
from model_wrapper import SSSDECG

# Use the first CUDA device (cuda:0)
model = SSSDECG(config_path="config/config_SSSD_ECG.json", device="cuda:0")

# Use the second CUDA device (cuda:1)
model = SSSDECG(config_path="config/config_SSSD_ECG.json", device="cuda:1")

# Use a specific CUDA device by index
device_id = 2
model = SSSDECG(config_path="config/config_SSSD_ECG.json", device=f"cuda:{device_id}")

# Use CPU
model = SSSDECG(config_path="config/config_SSSD_ECG.json", device="cpu")

# Auto-select (default: cuda if available, else cpu)
model = SSSDECG(config_path="config/config_SSSD_ECG.json")
```

### Using torch.device Objects

You can also pass `torch.device` objects:

```python
import torch

# Create a torch.device object
device = torch.device("cuda:1")
model = SSSDECG(config_path="config/config_SSSD_ECG.json", device=device)
```

### Training Example

```python
import torch
from model_wrapper import SSSDECG

# Initialize model on cuda:1
model = SSSDECG(config_path="config/config_SSSD_ECG.json", device="cuda:1")

# Your training data should also be on the same device
x = torch.randn(8, 8, 1000).to("cuda:1")  # batch_size=8, channels=8, length=1000
y = torch.randint(0, 71, (8,)).to("cuda:1")  # class labels

# Forward pass - automatically uses cuda:1
loss = model(x, y)

# Backward pass
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Generation Example

```python
import torch
from model_wrapper import SSSDECG

# Initialize model on cuda:1
model = SSSDECG(config_path="config/config_SSSD_ECG.json", device="cuda:1")

# Load checkpoint
model.load_checkpoint("checkpoint.pkl")

# Generate samples - automatically uses cuda:1
labels = torch.tensor([0, 5, 10, 15, 20]).to("cuda:1")
samples = model.generate(labels=labels)

print(f"Generated samples shape: {samples.shape}")
print(f"Samples device: {samples.device}")  # Should be cuda:1
```

### Multi-GPU Training

For distributed training across multiple GPUs, you can initialize separate model instances:

```python
import torch
from model_wrapper import SSSDECG

# Model on GPU 0
model_0 = SSSDECG(config_path="config/config_SSSD_ECG.json", device="cuda:0")

# Model on GPU 1
model_1 = SSSDECG(config_path="config/config_SSSD_ECG.json", device="cuda:1")
```

Or use PyTorch's `DataParallel` or `DistributedDataParallel`:

```python
import torch
import torch.nn as nn
from model_wrapper import SSSDECG

# Initialize model on cuda:0
model = SSSDECG(config_path="config/config_SSSD_ECG.json", device="cuda:0")

# Wrap with DataParallel to use all available GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
```

## Changes Made

The following files were modified to support arbitrary CUDA device selection:

### sssd_standalone directory:
- `utils/util.py`: Added `device` parameter to `std_normal`, `calc_diffusion_step_embedding`, `sampling_label`, and `training_loss_label` functions
- `model_wrapper.py`: Updated to pass `device` parameter to utility functions
- `models/SSSD_ECG.py`: Updated `Residual_group.forward()` to pass device to `calc_diffusion_step_embedding`

### src/sssd directory:
- `utils/util.py`: Same changes as sssd_standalone
- `model_wrapper.py`: Same changes as sssd_standalone
- `models/SSSD_ECG.py`: Same changes as sssd_standalone

## Testing

To verify that different CUDA devices work correctly, use the provided test script:

```bash
python test_cuda_devices.py
```

This script will:
1. Detect all available CUDA devices
2. Test model initialization on each device
3. Test training forward pass
4. Test generation
5. Verify all tensors are on the correct device

## Backward Compatibility

All changes are backward compatible. If you don't specify a device, the model will default to using CUDA if available (same as before), but now you have the flexibility to choose specific devices.

```python
# Old code still works
model = SSSDECG(config_path="config/config_SSSD_ECG.json")

# New code with explicit device selection
model = SSSDECG(config_path="config/config_SSSD_ECG.json", device="cuda:1")
```
