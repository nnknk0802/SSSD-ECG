"""
Test script to compare DDIM and DDPM sampling speeds
"""
import sys
sys.path.insert(0, 'src/sssd')

import torch
import time
from model_wrapper import SSSDECG

def test_ddim_vs_ddpm():
    """Compare DDIM and DDPM generation speeds"""

    # Initialize model
    print("Initializing SSSD-ECG model...")
    model = SSSDECG(
        config_path="src/sssd/config/config_SSSD_ECG.json",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model.model.eval()

    device = model.device
    print(f"Using device: {device}")

    # Prepare test labels (single sample)
    num_samples = 1
    num_classes = 71
    test_labels = torch.randint(0, num_classes, (num_samples,), device=device)

    print(f"\nGenerating {num_samples} sample(s) with different configurations:\n")

    # Test configurations
    configs = [
        {"name": "DDPM (200 steps)", "use_ddim": False, "ddim_timesteps": None},
        {"name": "DDIM (50 steps)", "use_ddim": True, "ddim_timesteps": 50},
        {"name": "DDIM (20 steps)", "use_ddim": True, "ddim_timesteps": 20},
        {"name": "DDIM (100 steps)", "use_ddim": True, "ddim_timesteps": 100},
    ]

    results = []

    for config in configs:
        print(f"Testing: {config['name']}")

        # Warmup run
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()

        if config['use_ddim']:
            samples = model.generate(
                labels=test_labels,
                use_ddim=True,
                ddim_timesteps=config['ddim_timesteps'],
                ddim_eta=0.0
            )
        else:
            samples = model.generate(
                labels=test_labels,
                use_ddim=False
            )

        if device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed_time = time.time() - start_time

        print(f"  Time: {elapsed_time:.2f}s")
        print(f"  Output shape: {samples.shape}")
        print()

        results.append({
            'name': config['name'],
            'time': elapsed_time,
            'shape': samples.shape
        })

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    baseline_time = results[0]['time']

    for result in results:
        speedup = baseline_time / result['time']
        print(f"{result['name']:25s}: {result['time']:6.2f}s  (speedup: {speedup:.2f}x)")

    print("\nRecommendation:")
    print("  - For fast generation: use DDIM with 20-50 steps")
    print("  - For best quality: use DDIM with 100 steps or DDPM")
    print("  - Default (50 steps) provides good balance")

if __name__ == "__main__":
    test_ddim_vs_ddpm()
