"""
Clean training script using the new SSSDECG wrapper interface.

Usage:
    python train_new.py --config config/config_SSSD_ECG.json
    python train_new.py --data_path ptbxl_train_data.npy --labels_path ptbxl_train_labels.npy
"""

import os
import argparse
import json
import torch
from tqdm import tqdm

from model_wrapper import SSSDECG
from dataset import create_dataloaders


def train(
    model,
    train_loader,
    optimizer,
    output_directory,
    n_iters,
    iters_per_ckpt=4000,
    iters_per_logging=100,
    start_iter=0
):
    """
    Train the SSSD-ECG model.

    Args:
        model (SSSDECG): The model to train
        train_loader (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer
        output_directory (str): Directory to save checkpoints
        n_iters (int): Total number of iterations to train
        iters_per_ckpt (int): Iterations between checkpoint saves
        iters_per_logging (int): Iterations between logging
        start_iter (int): Starting iteration (for resuming)
    """
    model.train()

    n_iter = start_iter
    epoch = 0

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    print(f"Starting training from iteration {start_iter}")
    print(f"Target iterations: {n_iters}")
    print(f"Output directory: {output_directory}")

    while n_iter < n_iters:
        epoch += 1
        print(f"\n--- Epoch {epoch} ---")

        # Progress bar for the epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, (x, y) in enumerate(pbar):
            # Forward pass and compute loss
            optimizer.zero_grad()
            loss = model(x, y)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Logging
            if n_iter % iters_per_logging == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "iter": n_iter})
                print(f"Iteration {n_iter}: loss = {loss.item():.4f}")

            # Save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_path = os.path.join(output_directory, f"{n_iter}.pkl")
                model.save_checkpoint(checkpoint_path, optimizer=optimizer, epoch=n_iter)
                print(f"Saved checkpoint at iteration {n_iter}")

            n_iter += 1

            # Check if we've reached the target iterations
            if n_iter >= n_iters:
                break

        pbar.close()

    # Save final checkpoint
    final_checkpoint_path = os.path.join(output_directory, f"{n_iter}.pkl")
    model.save_checkpoint(final_checkpoint_path, optimizer=optimizer, epoch=n_iter)
    print(f"\nTraining completed! Final checkpoint saved at iteration {n_iter}")


def main(args):
    # Load configuration if provided
    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)

        train_config = config.get("train_config", {})
        trainset_config = config.get("trainset_config", {})

        # Override with command line arguments if provided
        data_path = args.data_path or trainset_config.get("data_path", "ptbxl_train_data.npy")
        labels_path = args.labels_path or trainset_config.get("labels_path", "ptbxl_train_labels.npy")
        output_directory = args.output_dir or train_config.get("output_directory", "checkpoints")
        learning_rate = args.lr or train_config.get("learning_rate", 2e-4)
        batch_size = args.batch_size or train_config.get("batch_size", 8)
        n_iters = args.n_iters or train_config.get("n_iters", 100000)
        iters_per_ckpt = args.iters_per_ckpt or train_config.get("iters_per_ckpt", 4000)
        iters_per_logging = args.iters_per_logging or train_config.get("iters_per_logging", 100)
        segment_length = trainset_config.get("segment_length", 1000)
    else:
        # Use command line arguments only
        data_path = args.data_path
        labels_path = args.labels_path
        output_directory = args.output_dir
        learning_rate = args.lr
        batch_size = args.batch_size
        n_iters = args.n_iters
        iters_per_ckpt = args.iters_per_ckpt
        iters_per_logging = args.iters_per_logging
        segment_length = 1000

    print("=" * 50)
    print("Training Configuration:")
    print("=" * 50)
    print(f"Data path: {data_path}")
    print(f"Labels path: {labels_path}")
    print(f"Output directory: {output_directory}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Total iterations: {n_iters}")
    print(f"Checkpoint interval: {iters_per_ckpt}")
    print(f"Logging interval: {iters_per_logging}")
    print("=" * 50)

    # Initialize model
    print("\nInitializing model...")
    model = SSSDECG(config_path=args.config)
    model.print_model_size()

    # Create data loaders
    print("\nCreating data loaders...")
    # Use 8 leads: I, II, V1-V6 (indices 0, 1, 6, 7, 8, 9, 10, 11)
    lead_indices = [0, 1, 6, 7, 8, 9, 10, 11]

    train_loader, _ = create_dataloaders(
        train_data_path=data_path,
        train_labels_path=labels_path,
        batch_size=batch_size,
        num_workers=args.num_workers,
        shuffle_train=True,
        lead_indices=lead_indices,
        segment_length=segment_length
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load checkpoint if specified
    start_iter = 0
    if args.checkpoint is not None:
        print(f"\nLoading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epoch' in checkpoint:
            start_iter = checkpoint['epoch']
        print(f"Resumed from iteration {start_iter}")

    # Train
    print("\nStarting training...\n")
    train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        output_directory=output_directory,
        n_iters=n_iters,
        iters_per_ckpt=iters_per_ckpt,
        iters_per_logging=iters_per_logging,
        start_iter=start_iter
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SSSD-ECG model")

    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/config_SSSD_ECG.json",
        help="Path to configuration JSON file"
    )

    # Data
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to training data .npy file"
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        default=None,
        help="Path to training labels .npy file"
    )

    # Training
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size"
    )
    parser.add_argument(
        "--n_iters",
        type=int,
        default=None,
        help="Number of iterations to train"
    )
    parser.add_argument(
        "--iters_per_ckpt",
        type=int,
        default=None,
        help="Iterations between checkpoint saves"
    )
    parser.add_argument(
        "--iters_per_logging",
        type=int,
        default=None,
        help="Iterations between logging"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )

    # Resume training
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()
    main(args)
