"""
Example usage of DDIM-accelerated ECG generation
This demonstrates how to use the SSSD-ECG model with DDIM for fast generation.
"""
import sys
sys.path.insert(0, 'src/sssd')

import torch
from model_wrapper import SSSDECG
import numpy as np

def main():
    print("="*60)
    print("DDIM-Accelerated ECG Generation Example")
    print("="*60)

    # Initialize the model
    print("\n1. Initializing model...")
    model = SSSDECG(
        config_path="src/sssd/config/config_SSSD_ECG.json",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"   Using device: {model.device}")
    print(f"   Model parameters: {model.get_model_size() / 1e6:.2f}M")

    # Load checkpoint (replace with your checkpoint path)
    # model.load_checkpoint("path/to/your/checkpoint.pkl")
    print("\n   Note: Checkpoint loading skipped (add your checkpoint path)")

    # Example 1: Generate with DDIM (fast, default)
    print("\n2. Generating with DDIM (50 steps)...")
    num_samples = 5
    labels = torch.randint(0, 71, (num_samples,), device=model.device)

    samples_ddim = model.generate(
        labels=labels,
        use_ddim=True,
        ddim_timesteps=50,
        ddim_eta=0.0,
        return_numpy=True
    )
    print(f"   Generated shape: {samples_ddim.shape}")
    print(f"   Expected speedup: ~4x faster than DDPM")

    # Example 2: Generate with very fast DDIM (20 steps)
    print("\n3. Generating with fast DDIM (20 steps)...")
    samples_fast = model.generate(
        labels=labels,
        use_ddim=True,
        ddim_timesteps=20,
        ddim_eta=0.0,
        return_numpy=True
    )
    print(f"   Generated shape: {samples_fast.shape}")
    print(f"   Expected speedup: ~10x faster than DDPM")

    # Example 3: Generate with high-quality DDIM (100 steps)
    print("\n4. Generating with high-quality DDIM (100 steps)...")
    samples_hq = model.generate(
        labels=labels,
        use_ddim=True,
        ddim_timesteps=100,
        ddim_eta=0.0,
        return_numpy=True
    )
    print(f"   Generated shape: {samples_hq.shape}")
    print(f"   Expected speedup: ~2x faster than DDPM")

    # Example 4: Generate with original DDPM (for comparison)
    print("\n5. Generating with DDPM (200 steps, original)...")
    samples_ddpm = model.generate(
        labels=labels,
        use_ddim=False,
        return_numpy=True
    )
    print(f"   Generated shape: {samples_ddpm.shape}")
    print(f"   This is the baseline (1x speed)")

    # Example 5: Batch generation with multi-label conditioning
    print("\n6. Batch generation with multi-label conditioning...")
    num_classes = 71
    batch_size = 10

    # Create random multi-label conditioning (multi-hot encoding)
    labels_multilabel = torch.zeros(batch_size, num_classes, device=model.device)
    for i in range(batch_size):
        # Randomly select 2-4 active labels per sample
        num_active = torch.randint(2, 5, (1,)).item()
        active_indices = torch.randperm(num_classes)[:num_active]
        labels_multilabel[i, active_indices] = 1.0

    samples_batch = model.generate(
        labels=labels_multilabel,
        use_ddim=True,
        ddim_timesteps=50,
        return_numpy=True
    )
    print(f"   Generated shape: {samples_batch.shape}")
    print(f"   Batch size: {batch_size}")

    print("\n" + "="*60)
    print("Summary of DDIM Options:")
    print("="*60)
    print("  - 20 steps:  Very fast (~10x), good for prototyping")
    print("  - 50 steps:  Balanced (~4x), recommended default")
    print("  - 100 steps: High quality (~2x), closer to DDPM")
    print("  - 200 steps: Original DDPM (baseline)")
    print("\nRecommendation: Start with 50 steps, adjust based on your")
    print("quality requirements and compute budget.")
    print("="*60)

if __name__ == "__main__":
    main()
