"""
Example of using sssd_standalone from an external project.

This demonstrates different ways to import and use sssd_standalone
when it's located in a subdirectory of your project.
"""

# ==========================================
# Method 1: Using sys.path (when not installed)
# ==========================================
def method1_sys_path():
    import sys
    from pathlib import Path

    # Add parent directory of sssd_standalone to path
    # Assuming structure: ./run.py and ./codes/data/models/sssd_standalone
    standalone_parent = Path(__file__).parent.parent.parent / "codes/data/models"
    sys.path.insert(0, str(standalone_parent))

    from sssd_standalone import SSSDECG, ECGDataset

    # Use relative path to config
    config_path = standalone_parent / "sssd_standalone/config/config_SSSD_ECG.json"
    model = SSSDECG(config_path=str(config_path))

    return model


# ==========================================
# Method 2: After installing with pip install -e .
# ==========================================
def method2_installed():
    # Simply import after installation
    from sssd_standalone import SSSDECG, ECGDataset, create_dataloaders
    import torch

    # Initialize model
    model = SSSDECG()

    # Create dummy data for testing
    batch_size = 4
    channels = 8
    length = 1000
    num_classes = 71

    x = torch.randn(batch_size, channels, length)
    y = torch.randint(0, num_classes, (batch_size,))

    # Training
    loss = model(x, y)
    print(f"Loss: {loss.item():.4f}")

    # Inference
    samples = model.generate(num_samples=5, return_numpy=True)
    print(f"Generated samples shape: {samples.shape}")

    return model


# ==========================================
# Method 3: With explicit config path
# ==========================================
def method3_with_config():
    from pathlib import Path

    # Method 1 import approach
    import sys
    standalone_parent = Path(__file__).parent.parent.parent / "codes/data/models"
    sys.path.insert(0, str(standalone_parent))

    from sssd_standalone import SSSDECG

    # Construct config path
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "codes/data/models/sssd_standalone/config/config_SSSD_ECG.json"

    # Initialize with explicit config
    model = SSSDECG(config_path=str(config_path))

    return model


# ==========================================
# Full example with data loading
# ==========================================
def full_example():
    """Complete example with data loading and training."""
    import sys
    from pathlib import Path
    import torch
    from torch.utils.data import DataLoader

    # Setup path
    standalone_parent = Path(__file__).parent.parent.parent / "codes/data/models"
    sys.path.insert(0, str(standalone_parent))

    from sssd_standalone import SSSDECG, ECGDataset

    # Paths
    config_path = standalone_parent / "sssd_standalone/config/config_SSSD_ECG.json"

    # Initialize model
    model = SSSDECG(config_path=str(config_path))
    print(f"Model initialized with {model.get_model_size() / 1e6:.2f}M parameters")

    # Create dataset (assuming you have data files)
    # dataset = ECGDataset(
    #     data_path="your_data.npy",
    #     labels_path="your_labels.npy",
    #     segment_length=1000
    # )

    # For demo, create dummy data
    import numpy as np
    dummy_data = np.random.randn(100, 8, 1000).astype(np.float32)
    dummy_labels = np.random.randint(0, 71, (100,))

    dataset = ECGDataset(
        data_path=dummy_data,
        labels_path=dummy_labels,
        segment_length=1000
    )

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Training loop (1 epoch for demo)
    print("\nTraining for 1 epoch...")
    model.train()
    for i, (x, y) in enumerate(dataloader):
        loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            print(f"  Batch {i}, Loss: {loss.item():.4f}")

    # Generate samples
    print("\nGenerating samples...")
    model.eval()
    samples = model.generate(num_samples=5, return_numpy=True)
    print(f"Generated shape: {samples.shape}")  # (5, 8, 1000)

    return model


if __name__ == "__main__":
    print("="*60)
    print("SSSD-ECG External Usage Examples")
    print("="*60)

    print("\n[1] Testing Method 1: sys.path approach")
    try:
        model1 = method1_sys_path()
        print("✓ Method 1 successful")
    except Exception as e:
        print(f"✗ Method 1 failed: {e}")

    print("\n[2] Testing Method 2: Installed package")
    try:
        model2 = method2_installed()
        print("✓ Method 2 successful")
    except Exception as e:
        print(f"✗ Method 2 failed: {e}")

    print("\n[3] Testing Full Example")
    try:
        model3 = full_example()
        print("✓ Full example successful")
    except Exception as e:
        print(f"✗ Full example failed: {e}")

    print("\n" + "="*60)
    print("Examples completed!")
