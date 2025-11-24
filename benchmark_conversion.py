#!/usr/bin/env python3
"""
Benchmark script for 8-lead to 12-lead ECG conversion.
"""

import numpy as np
import time
from src.sssd.convert_8_to_12_lead import convert_8_to_12_lead


def benchmark_single_sample():
    """Benchmark conversion time for a single sample."""
    print("=== Single Sample Benchmark ===")
    signal_length = 1000
    data_8lead = np.random.randn(8, signal_length).astype(np.float32)

    # Warm-up
    for _ in range(10):
        convert_8_to_12_lead(data_8lead)

    # Benchmark
    num_iterations = 1000
    start_time = time.time()
    for _ in range(num_iterations):
        data_12lead = convert_8_to_12_lead(data_8lead)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_sample = total_time / num_iterations

    print(f"Total time for {num_iterations} conversions: {total_time:.4f} seconds")
    print(f"Average time per sample: {avg_time_per_sample*1000:.4f} ms")
    print(f"Average time per sample: {avg_time_per_sample*1000000:.2f} µs")
    print(f"Throughput: {num_iterations/total_time:.2f} samples/second")


def benchmark_batch():
    """Benchmark conversion time for batches of different sizes."""
    print("\n=== Batch Benchmark ===")
    signal_length = 1000
    batch_sizes = [1, 10, 50, 100, 500, 1000]

    for batch_size in batch_sizes:
        data_8lead = np.random.randn(batch_size, 8, signal_length).astype(np.float32)

        # Warm-up
        for _ in range(5):
            convert_8_to_12_lead(data_8lead)

        # Benchmark
        num_iterations = 100
        start_time = time.time()
        for _ in range(num_iterations):
            data_12lead = convert_8_to_12_lead(data_8lead)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_batch = total_time / num_iterations
        avg_time_per_sample = total_time / (num_iterations * batch_size)

        print(f"\nBatch size: {batch_size}")
        print(f"  Time per batch: {avg_time_per_batch*1000:.4f} ms")
        print(f"  Time per sample: {avg_time_per_sample*1000:.4f} ms ({avg_time_per_sample*1000000:.2f} µs)")
        print(f"  Throughput: {batch_size*num_iterations/total_time:.2f} samples/second")


def benchmark_different_lengths():
    """Benchmark conversion time for different signal lengths."""
    print("\n=== Different Signal Length Benchmark ===")
    signal_lengths = [100, 500, 1000, 2500, 5000]
    batch_size = 100

    for signal_length in signal_lengths:
        data_8lead = np.random.randn(batch_size, 8, signal_length).astype(np.float32)

        # Warm-up
        for _ in range(5):
            convert_8_to_12_lead(data_8lead)

        # Benchmark
        num_iterations = 100
        start_time = time.time()
        for _ in range(num_iterations):
            data_12lead = convert_8_to_12_lead(data_8lead)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_sample = total_time / (num_iterations * batch_size)

        print(f"\nSignal length: {signal_length}")
        print(f"  Time per sample: {avg_time_per_sample*1000:.4f} ms ({avg_time_per_sample*1000000:.2f} µs)")
        print(f"  Throughput: {batch_size*num_iterations/total_time:.2f} samples/second")


if __name__ == "__main__":
    print("8-Lead to 12-Lead ECG Conversion Benchmark")
    print("=" * 60)

    benchmark_single_sample()
    benchmark_batch()
    benchmark_different_lengths()

    print("\n" + "=" * 60)
    print("Benchmark completed!")
