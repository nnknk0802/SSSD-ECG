"""
Example usage of the refactored SSSD-ECG model.

This script demonstrates:
1. Loading data using the ECGDataset class
2. Training with the SSSDECG model wrapper
3. Generating samples with the trained model
"""

import torch
import numpy as np
from torch.utils.data import DataLoader

from model_wrapper import SSSDECG
from dataset import ECGDataset, PTBXLDataset, create_dataloaders


def example_basic_training():
    """
    Example 1: Basic training loop
    """
    print("=" * 60)
    print("Example 1: Basic Training")
    print("=" * 60)

    # Initialize model
    model = SSSDECG(config_path="config/config_SSSD_ECG.json")
    print(f"Model initialized with {model.get_model_size() / 1e6:.2f}M parameters")

    # Create dummy data for demonstration
    # In practice, you would load real data from .npy files
    num_samples = 100
    num_channels = 8
    seq_length = 1000
    num_classes = 71

    dummy_data = np.random.randn(num_samples, num_channels, seq_length).astype(np.float32)
    dummy_labels = np.random.randint(0, num_classes, size=(num_samples,))

    # Create dataset and dataloader
    dataset = ECGDataset(
        data_path=dummy_data,
        labels_path=dummy_labels,
        segment_length=seq_length
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        drop_last=True
    )

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Training loop
    print("\nTraining for 10 iterations...")
    model.train()

    for iteration in range(10):
        for x, y in dataloader:
            # Compute loss
            loss = model(x, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Iteration {iteration + 1}: Loss = {loss.item():.4f}")
            break  # Only one batch per iteration for demo

    print("Training complete!")


def example_inference():
    """
    Example 2: Generating samples
    """
    print("\n" + "=" * 60)
    print("Example 2: Generating Samples")
    print("=" * 60)

    # Initialize model
    model = SSSDECG(config_path="config/config_SSSD_ECG.json")

    # Generate samples with random labels
    print("\nGenerating 5 samples with random labels...")
    samples = model.generate(num_samples=5, return_numpy=True)
    print(f"Generated samples shape: {samples.shape}")  # (5, 8, 1000)

    # Generate samples with specific labels
    print("\nGenerating samples with specific labels...")
    labels = torch.tensor([0, 5, 10, 15, 20])  # Class indices
    samples = model.generate(labels=labels, return_numpy=True)
    print(f"Generated samples shape: {samples.shape}")  # (5, 8, 1000)

    # Generate with one-hot encoded labels
    print("\nGenerating samples with one-hot labels...")
    num_classes = 71
    labels_onehot = torch.zeros(5, num_classes)
    labels_onehot[0, 0] = 1  # Class 0
    labels_onehot[1, 10] = 1  # Class 10
    labels_onehot[2, 20] = 1  # Class 20
    labels_onehot[3, 30] = 1  # Class 30
    labels_onehot[4, 40] = 1  # Class 40

    samples = model.generate(labels=labels_onehot, return_numpy=True)
    print(f"Generated samples shape: {samples.shape}")


def example_save_load_checkpoint():
    """
    Example 3: Saving and loading checkpoints
    """
    print("\n" + "=" * 60)
    print("Example 3: Save/Load Checkpoints")
    print("=" * 60)

    # Initialize model and optimizer
    model = SSSDECG(config_path="config/config_SSSD_ECG.json")
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Save checkpoint
    checkpoint_path = "example_checkpoint.pkl"
    print(f"\nSaving checkpoint to {checkpoint_path}...")
    model.save_checkpoint(checkpoint_path, optimizer=optimizer, epoch=100)

    # Create a new model and load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    new_model = SSSDECG(config_path="config/config_SSSD_ECG.json")
    new_model.load_checkpoint(checkpoint_path)

    print("Checkpoint loaded successfully!")

    # Clean up
    import os
    os.remove(checkpoint_path)
    print(f"Removed example checkpoint file")


def example_with_real_data():
    """
    Example 4: Training with real PTB-XL data
    (Only works if you have the data files)
    """
    print("\n" + "=" * 60)
    print("Example 4: Training with Real Data")
    print("=" * 60)

    # File paths (adjust these to your actual data location)
    train_data_path = "ptbxl_train_data.npy"
    train_labels_path = "ptbxl_train_labels.npy"

    # Check if files exist
    import os
    if not os.path.exists(train_data_path) or not os.path.exists(train_labels_path):
        print("\nSkipping: Data files not found")
        print(f"Expected files:")
        print(f"  - {train_data_path}")
        print(f"  - {train_labels_path}")
        return

    # Create dataloaders using the convenience function
    train_loader, _ = create_dataloaders(
        train_data_path=train_data_path,
        train_labels_path=train_labels_path,
        batch_size=8,
        num_workers=0,  # Use 0 for demo, increase for real training
        shuffle_train=True,
        lead_indices=[0, 1, 6, 7, 8, 9, 10, 11],  # 8 leads
        segment_length=1000
    )

    print(f"\nDataset size: {len(train_loader.dataset)}")
    print(f"Number of batches: {len(train_loader)}")

    # Initialize model
    model = SSSDECG(config_path="config/config_SSSD_ECG.json")
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Train for a few iterations
    print("\nTraining for 5 batches...")
    model.train()

    for batch_idx, (x, y) in enumerate(train_loader):
        if batch_idx >= 5:
            break

        loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Batch {batch_idx + 1}: Loss = {loss.item():.4f}, "
              f"Data shape = {x.shape}, Labels shape = {y.shape}")

    print("Training complete!")


def example_custom_dataset():
    """
    Example 5: Using PTBXLDataset for specialized preprocessing
    """
    print("\n" + "=" * 60)
    print("Example 5: Custom Dataset Class")
    print("=" * 60)

    # Create dummy data
    num_samples = 50
    num_channels = 12  # Full 12-lead ECG
    seq_length = 1000
    num_classes = 71

    dummy_data = np.random.randn(num_samples, num_channels, seq_length).astype(np.float32)
    dummy_labels = np.random.randint(0, num_classes, size=(num_samples,))

    # Use PTBXLDataset which automatically selects 8 leads
    dataset = PTBXLDataset(
        data_path=dummy_data,
        labels_path=dummy_labels,
        use_8_leads=True  # Automatically select I, II, V1-V6
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Check the output
    x, y = next(iter(dataloader))
    print(f"\nBatch shape: {x.shape}")  # Should be (4, 8, 1000)
    print(f"Labels shape: {y.shape}")
    print(f"Selected 8 leads from original 12-lead ECG")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("SSSD-ECG Model Usage Examples")
    print("=" * 60)

    # Run examples
    example_basic_training()
    example_inference()
    example_save_load_checkpoint()
    example_with_real_data()
    example_custom_dataset()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    main()
