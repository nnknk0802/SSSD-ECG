# PTB-XL Dataset DataLoader

This directory contains tools for downloading, preprocessing, and loading the PTB-XL dataset.

## Download and Pre-process the PTB-XL Dataset

### 1. Download the data

Locate inside the data directory:
```bash
cd data_folder_ptb_xl/
```

Download the data:
```bash
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.1/
```

Note: You can also download pre-processed signals and labels from [here](https://mega.nz/folder/UfUDFYjS#YYUJ3CCUGb6ZNmJdCZLseg).

### 2. Create DataLoaders

You have two options:

#### Option A: Using the Python Module (Recommended)

Use the `ptbxl_dataloader.py` module for easy dataloader creation:

```python
from ptb_xl.ptbxl_dataloader import get_ptbxl_dataloaders

# Create dataloaders with full pipeline
train_loader, val_loader, test_loader, lbl_itos, mean, std = get_ptbxl_dataloaders(
    data_folder_ptb_xl="./data_folder_ptb_xl/",
    target_folder_ptb_xl="./processed_ptb_xl_fs100",
    target_fs=100,  # 100 Hz or 500 Hz
    batch_size=8,
    num_workers=4,
    recreate_data=True  # Set to False if data is already prepared
)

# Use the dataloaders
for data, labels in train_loader:
    # data shape: [batch_size, channels, timesteps]
    # labels shape: [batch_size, num_classes]
    pass
```

See `example_ptbxl_dataloader.py` for more examples.

#### Option B: Using the Jupyter Notebook

Open the jupyter notebook `ecg_data_preprocessing.ipynb` and run the notebook. The PTB-XL dataloaders are at the end.

## Module Features

The `ptbxl_dataloader.py` module provides:

- **`prepare_ptbxl_data()`**: Prepare and preprocess raw PTB-XL data
- **`create_ptbxl_dataloaders()`**: Create PyTorch dataloaders from prepared data
- **`get_ptbxl_dataloaders()`**: Full pipeline (prepare + create dataloaders)

### Key Parameters

- `target_fs`: Sampling rate (100 Hz or 500 Hz)
- `input_size`: Sample length (default: 1000 for 10 seconds at 100 Hz)
- `batch_size`: Batch size for training
- `ptb_xl_label`: Label type (`"label_diag"`, `"label_form"`, `"label_rhythm"`, etc.)
- `min_cnt`: Minimum count for label filtering
- `chunkify_train/valtest`: Whether to split samples into chunks

### Example Usage

```bash
# Run example script
python example_ptbxl_dataloader.py --example 1

# Or use as a module
python -m ptb_xl.ptbxl_dataloader \
    --data_folder ./data_folder_ptb_xl/ \
    --target_folder ./processed_ptb_xl_fs100 \
    --target_fs 100 \
    --batch_size 8 \
    --recreate_data
```

## Data Format

The dataloader returns:
- **Data**: Shape `[batch_size, 12, 1000]` for 12-lead ECG signals
- **Labels**: Shape `[batch_size, num_classes]` for multi-label classification

The dataloaders are highly functional and allow you to access patient metadata for further research.
