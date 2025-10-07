"""
Batch benchmark script for comparing baseline vs VRAM-optimized aggregator.

This script allows users to easily configure multiple experimental conditions
and automatically runs comparisons across different:
- Batch sizes (B)
- Sequence lengths (S)
- Image resolutions (H, W)
- Model configurations (embed_dim, depth, etc.)

Results are saved to a CSV file for easy analysis.

Usage:
    python scripts/benchmark_vram_sweep.py
    python scripts/benchmark_vram_sweep.py --output results.csv
    python scripts/benchmark_vram_sweep.py --device cpu
"""

import os
import time
import argparse
import csv
from datetime import datetime
import torch

from vggt.models.aggregator import Aggregator
from vggt.models.aggregator_vram_optimized import AggregatorVramOptimized, LayerSelectView
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead


# ============================================================================
# USER CONFIGURATION - Modify these parameters for your experiments
# ============================================================================

# Test configurations: each dict defines one experiment
# You can add/remove/modify configurations as needed
EXPERIMENT_CONFIGS = [
    # Small scale tests - quick validation
    {
        "name": "small_2frames",
        "batch": 1,
        "seq": 2,
        "img": 112,
        "patch": 14,
        "embed_dim": 128,
        "depth": 24,
        "heads": 16,
    },
    {
        "name": "small_4frames",
        "batch": 1,
        "seq": 4,
        "img": 112,
        "patch": 14,
        "embed_dim": 128,
        "depth": 24,
        "heads": 16,
    },

    # Medium scale - different batch vs seq combinations
    {
        "name": "medium_B1_S8",
        "batch": 1,
        "seq": 8,
        "img": 224,
        "patch": 14,
        "embed_dim": 128,
        "depth": 24,
        "heads": 16,
    },
    {
        "name": "medium_B2_S4",
        "batch": 2,
        "seq": 4,
        "img": 224,
        "patch": 14,
        "embed_dim": 128,
        "depth": 24,
        "heads": 16,
    },
    {
        "name": "medium_B4_S2",
        "batch": 4,
        "seq": 2,
        "img": 224,
        "patch": 14,
        "embed_dim": 128,
        "depth": 24,
        "heads": 16,
    },

    # Resolution tests - same total frames, different resolutions
    {
        "name": "res_112",
        "batch": 1,
        "seq": 4,
        "img": 112,
        "patch": 14,
        "embed_dim": 128,
        "depth": 24,
        "heads": 16,
    },
    {
        "name": "res_224",
        "batch": 1,
        "seq": 4,
        "img": 224,
        "patch": 14,
        "embed_dim": 128,
        "depth": 24,
        "heads": 16,
    },
    {
        "name": "res_448",
        "batch": 1,
        "seq": 4,
        "img": 448,
        "patch": 14,
        "embed_dim": 128,
        "depth": 24,
        "heads": 16,
    },

    # Stress tests - large scale
    {
        "name": "1frames_large_res",
        "batch": 1,
        "seq": 4,
        "img": 518,
        "patch": 14,
        "embed_dim": 128,
        "depth": 24,
        "heads": 16,
    },
    {
        "name": "2frames_large_res",
        "batch": 1,
        "seq": 2,
        "img": 518,
        "patch": 14,
        "embed_dim": 128,
        "depth": 24,
        "heads": 16,
    },
    {
        "name": "4frames_large_res",
        "batch": 1,
        "seq": 4,
        "img": 518,
        "patch": 14,
        "embed_dim": 128,
        "depth": 24,
        "heads": 16,
    },
    {
        "name": "8frames_large_res",
        "batch": 1,
        "seq": 8,
        "img": 518,
        "patch": 14,
        "embed_dim": 128,
        "depth": 24,
        "heads": 16,
    },
    {
        "name": "16frames_large_res",
        "batch": 1,
        "seq": 16,
        "img": 518,
        "patch": 14,
        "embed_dim": 128,
        "depth": 24,
        "heads": 16,
    },
    {
        "name": "32frames_large_res",
        "batch": 1,
        "seq": 32,
        "img": 518,
        "patch": 14,
        "embed_dim": 128,
        "depth": 24,
        "heads": 16,
    },
    {
        "name": "64frames_large_res",
        "batch": 1,
        "seq": 64,
        "img": 518,
        "patch": 14,
        "embed_dim": 128,
        "depth": 24,
        "heads":16,
    },
    {
        "name": "128frames_large_res",
        "batch": 1,
        "seq": 128,
        "img": 518,
        "patch": 14,
        "embed_dim": 128,
        "depth": 24,
        "heads": 16,
    },
    {
        "name": "256frames_large_res",
        "batch": 1,
        "seq": 256,
        "img": 518,
        "patch": 14,
        "embed_dim": 128,
        "depth": 24,
        "heads": 16,
    },
]

# Layer selection for optimized aggregator
SELECTED_LAYERS = [4, 11, 17, 23]

# Number of warmup runs before timing (to avoid cold start effects)
NUM_WARMUP = 2

# Number of timed runs to average
NUM_RUNS = 5

# ============================================================================
# Implementation - No need to modify below unless changing functionality
# ============================================================================


def tensor_list_nbytes(tensors):
    """Calculate total bytes used by a list of tensors."""
    total = 0
    for t in tensors:
        if isinstance(t, torch.Tensor):
            total += t.element_size() * t.nelement()
    return total


def format_bytes(num_bytes):
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def run_benchmark(config, device, verbose=True):
    """
    Run a single benchmark with given config.

    Returns dict with timing and memory stats.
    """
    torch.manual_seed(0)

    B = config["batch"]
    S = config["seq"]
    img_size = config["img"]
    patch_size = config["patch"]
    embed_dim = config["embed_dim"]
    depth = config["depth"]
    num_heads = config["heads"]

    # Adjust image size to be multiple of patch size
    H = W = img_size - (img_size % patch_size)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {config['name']}")
        print(f"  Config: B={B}, S={S}, H=W={H}, patch={patch_size}")
        print(f"  Model: embed_dim={embed_dim}, depth={depth}, heads={num_heads}")
        print(f"  Total frames: {B * S}")

    # Create synthetic data
    images = torch.rand(B, S, 3, H, W, device=device)

    # Build models
    agg_full = Aggregator(
        img_size=H,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_embed="conv",
    ).to(device).eval()

    agg_opt = AggregatorVramOptimized(
        img_size=H,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_embed="conv",
        selected_layer_idx=SELECTED_LAYERS,
    ).to(device).eval()

    # Align weights
    agg_opt.load_state_dict(agg_full.state_dict(), strict=False)

    # Build heads for end-to-end test
    cam_head = CameraHead(dim_in=2 * embed_dim).to(device).eval()
    dpt_head = DPTHead(
        dim_in=2 * embed_dim,
        output_dim=2,
        features=min(128, embed_dim),
        out_channels=[min(128, embed_dim)] * 4,
        intermediate_layer_idx=SELECTED_LAYERS,
        pos_embed=False,
        patch_size=patch_size,
    ).to(device).eval()

    # Warmup runs
    if verbose:
        print(f"  Warmup ({NUM_WARMUP} runs)...")
    for _ in range(NUM_WARMUP):
        with torch.no_grad():
            _ = agg_full(images)
            _ = agg_opt(images)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark baseline (full)
    if verbose:
        print(f"  Benchmarking baseline ({NUM_RUNS} runs)...")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    full_times = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        with torch.no_grad():
            full_list, patch_start = agg_full(images)
            full_bytes = tensor_list_nbytes(full_list)
            _ = cam_head(full_list)
            _ = dpt_head(full_list, images=images, patch_start_idx=patch_start)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        full_times.append(t1 - t0)

    mem_full = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    full_time_mean = sum(full_times) / len(full_times)
    full_time_std = (sum((t - full_time_mean) ** 2 for t in full_times) / len(full_times)) ** 0.5

    # Benchmark optimized
    if verbose:
        print(f"  Benchmarking optimized ({NUM_RUNS} runs)...")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    opt_times = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        with torch.no_grad():
            view, patch_start2, kept = agg_opt(images)
            assert isinstance(view, LayerSelectView)
            opt_bytes = tensor_list_nbytes(view._selected_outputs)
            _ = cam_head(view)
            _ = dpt_head(view, images=images, patch_start_idx=patch_start2)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        opt_times.append(t1 - t0)

    mem_opt = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    opt_time_mean = sum(opt_times) / len(opt_times)
    opt_time_std = (sum((t - opt_time_mean) ** 2 for t in opt_times) / len(opt_times)) ** 0.5

    # Compute speedup and memory reduction
    speedup = full_time_mean / opt_time_mean if opt_time_mean > 0 else 0
    mem_reduction_pct = (1 - mem_opt / mem_full) * 100 if mem_full > 0 else 0
    storage_reduction_pct = (1 - opt_bytes / full_bytes) * 100 if full_bytes > 0 else 0

    if verbose:
        print(f"\n  Results:")
        print(f"    Timing (mean ± std):")
        print(f"      Baseline:  {full_time_mean*1000:.2f} ± {full_time_std*1000:.2f} ms")
        print(f"      Optimized: {opt_time_mean*1000:.2f} ± {opt_time_std*1000:.2f} ms")
        print(f"      Speedup:   {speedup:.2f}x")
        if device.type == "cuda":
            print(f"    Peak CUDA memory:")
            print(f"      Baseline:  {format_bytes(mem_full)}")
            print(f"      Optimized: {format_bytes(mem_opt)}")
            print(f"      Reduction: {mem_reduction_pct:.1f}%")
        print(f"    Stored layer outputs:")
        print(f"      Baseline:  {format_bytes(full_bytes)}")
        print(f"      Optimized: {format_bytes(opt_bytes)}")
        print(f"      Reduction: {storage_reduction_pct:.1f}%")

    # Return results as dict
    return {
        "name": config["name"],
        "batch": B,
        "seq": S,
        "total_frames": B * S,
        "img_size": H,
        "patch_size": patch_size,
        "embed_dim": embed_dim,
        "depth": depth,
        "num_heads": num_heads,
        "full_time_ms": full_time_mean * 1000,
        "full_time_std_ms": full_time_std * 1000,
        "opt_time_ms": opt_time_mean * 1000,
        "opt_time_std_ms": opt_time_std * 1000,
        "speedup": speedup,
        "full_mem_bytes": mem_full,
        "opt_mem_bytes": mem_opt,
        "mem_reduction_pct": mem_reduction_pct,
        "full_storage_bytes": full_bytes,
        "opt_storage_bytes": opt_bytes,
        "storage_reduction_pct": storage_reduction_pct,
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Batch benchmark for VRAM optimization")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to run on (cuda/cpu)")
    ap.add_argument("--output", type=str, default=None,
                    help="Output CSV file (default: auto-generated with timestamp)")
    ap.add_argument("--verbose", action="store_true", default=True,
                    help="Print detailed progress")
    ap.add_argument("--quiet", action="store_true", default=False,
                    help="Suppress detailed output")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    verbose = args.verbose and not args.quiet

    # Generate output filename if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"benchmark_vram_sweep_{timestamp}.csv"
    else:
        output_file = args.output

    print(f"{'='*60}")
    print(f"VRAM Optimization Benchmark Sweep")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Number of experiments: {len(EXPERIMENT_CONFIGS)}")
    print(f"Selected layers: {SELECTED_LAYERS}")
    print(f"Warmup runs: {NUM_WARMUP}")
    print(f"Timed runs per config: {NUM_RUNS}")
    print(f"Output file: {output_file}")
    print(f"{'='*60}")

    # Run all benchmarks
    results = []
    for i, config in enumerate(EXPERIMENT_CONFIGS, 1):
        print(f"\n[{i}/{len(EXPERIMENT_CONFIGS)}] ", end="")
        try:
            result = run_benchmark(config, device, verbose=verbose)
            results.append(result)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            print(f"  Skipping config: {config['name']}")
            continue

    # Write results to CSV
    if results:
        fieldnames = [
            "name", "batch", "seq", "total_frames", "img_size", "patch_size",
            "embed_dim", "depth", "num_heads",
            "full_time_ms", "full_time_std_ms", "opt_time_ms", "opt_time_std_ms", "speedup",
            "full_mem_bytes", "opt_mem_bytes", "mem_reduction_pct",
            "full_storage_bytes", "opt_storage_bytes", "storage_reduction_pct",
        ]

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n{'='*60}")
        print(f"Results saved to: {output_file}")
        print(f"Total experiments completed: {len(results)}/{len(EXPERIMENT_CONFIGS)}")

        # Print summary statistics
        if len(results) > 0:
            avg_speedup = sum(r["speedup"] for r in results) / len(results)
            avg_mem_reduction = sum(r["mem_reduction_pct"] for r in results) / len(results)
            avg_storage_reduction = sum(r["storage_reduction_pct"] for r in results) / len(results)

            print(f"\nSummary (averaged across {len(results)} experiments):")
            print(f"  Average speedup: {avg_speedup:.2f}x")
            if device.type == "cuda":
                print(f"  Average peak memory reduction: {avg_mem_reduction:.1f}%")
            print(f"  Average storage reduction: {avg_storage_reduction:.1f}%")

        print(f"{'='*60}")
    else:
        print("\nNo results to save (all experiments failed)")


if __name__ == "__main__":
    main()
