#!/usr/bin/env python3
"""
Test script for KITTI-360 dataset loader validation.

Usage:
    python test_kitti360_loading.py [--visualize]

This script:
1. Loads sample sequences from KITTI-360
2. Validates data format and shapes
3. Computes depth coverage statistics
4. Analyzes building ratio distribution
5. Optionally visualizes samples
"""

import os
import sys
import numpy as np
import torch
import argparse
from pathlib import Path
from collections import defaultdict

# Add training directory to path
sys.path.insert(0, str(Path(__file__).parent / "training"))

from data.datasets.kitti360 import KITTI360Dataset
from omegaconf import OmegaConf


def create_dummy_config():
    """Create a minimal config for testing"""
    config = OmegaConf.create({
        'img_size': 518,
        'patch_size': 14,
        'debug': False,
        'training': True,
        'inside_random': False,  # Disable random sampling for reproducibility
        'rescale': True,
        'rescale_aug': False,
        'landscape_check': False,
        'load_depth': True,
        'augs': {
            'scales': None,
            'aspects': [1.0, 1.0],
        }
    })
    return config


def analyze_depth_coverage(depths, point_masks):
    """Analyze depth map coverage statistics"""
    stats = {
        'mean_coverage': [],
        'median_coverage': [],
        'valid_points_per_frame': [],
        'frames_with_sufficient_points': 0,
    }

    for depth, mask in zip(depths, point_masks):
        if isinstance(depth, torch.Tensor):
            depth = depth.numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()

        total_pixels = mask.size
        valid_pixels = mask.sum()
        coverage = (valid_pixels / total_pixels) * 100

        stats['mean_coverage'].append(coverage)
        stats['median_coverage'].append(coverage)
        stats['valid_points_per_frame'].append(valid_pixels)

        if valid_pixels >= 2000:
            stats['frames_with_sufficient_points'] += 1

    stats['mean_coverage'] = np.mean(stats['mean_coverage'])
    stats['median_coverage'] = np.median(stats['median_coverage'])
    stats['mean_valid_points'] = np.mean(stats['valid_points_per_frame'])
    stats['median_valid_points'] = np.median(stats['valid_points_per_frame'])
    stats['min_valid_points'] = np.min(stats['valid_points_per_frame'])
    stats['max_valid_points'] = np.max(stats['valid_points_per_frame'])
    stats['sufficient_points_ratio'] = stats['frames_with_sufficient_points'] / len(depths)

    return stats


def analyze_semantic_weights(semantic_weights_list):
    """Analyze semantic weight distribution"""
    all_weights = []
    for weights in semantic_weights_list:
        if isinstance(weights, torch.Tensor):
            weights = weights.numpy()
        all_weights.append(weights.flatten())

    all_weights = np.concatenate(all_weights)

    stats = {
        'mean_weight': np.mean(all_weights),
        'median_weight': np.median(all_weights),
        'min_weight': np.min(all_weights),
        'max_weight': np.max(all_weights),
        'std_weight': np.std(all_weights),
        'zero_weight_ratio': (all_weights == 0).sum() / len(all_weights),
        'high_weight_ratio': (all_weights >= 1.5).sum() / len(all_weights),  # Building pixels
    }

    return stats


def test_dataset_loading():
    """Test basic dataset loading and data format"""
    print("=" * 80)
    print("KITTI-360 Dataset Loader Validation")
    print("=" * 80)

    # Create dataset
    print("\n[1/6] Initializing dataset...")
    config = create_dummy_config()

    try:
        dataset = KITTI360Dataset(
            root_dir="/home/zerun/data/dataset/KITTI-360",
            split="train",
            common_conf=config,
            sampling_strategy="adaptive",
            building_sampling_weights=(0.5, 0.3, 0.2),
            accumulation_frames=4,
            min_valid_points=2000,
            semantic_weight_enabled=True,
            filter_buildings_only=False,
            depth_range=(0.1, 80.0),
            len_train=100,  # Small for testing
        )
        print(f"✓ Dataset initialized successfully")
        print(f"  Total frames: {len(dataset.frames)}")
        print(f"  Sequences: {len(dataset.sequences)}")
        print(f"  Dataset length: {len(dataset)}")

        if dataset.sampling_strategy == 'adaptive':
            print(f"\n  Sampling buckets:")
            print(f"    Building-rich: {len(dataset.bucket_building_rich)} frames")
            print(f"    Mixed scenes:  {len(dataset.bucket_mixed)} frames")
            print(f"    Road-dominant: {len(dataset.bucket_road)} frames")

    except Exception as e:
        print(f"✗ Failed to initialize dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test data loading
    print("\n[2/6] Testing data loading...")
    try:
        # Test get_data method
        batch = dataset.get_data(seq_index=0, img_per_seq=4, aspect_ratio=1.0)

        print(f"✓ Successfully loaded batch")
        print(f"  Sequence: {batch['seq_name']}")
        print(f"  Frame indices: {batch['ids']}")
        print(f"  Number of frames: {batch['frame_num']}")

    except Exception as e:
        print(f"✗ Failed to load batch: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Validate data shapes
    print("\n[3/6] Validating data shapes...")
    try:
        num_frames = batch['frame_num']
        H, W = batch['images'][0].shape[:2]

        checks = [
            ('images', len(batch['images']), num_frames),
            ('depths', len(batch['depths']), num_frames),
            ('extrinsics', len(batch['extrinsics']), num_frames),
            ('intrinsics', len(batch['intrinsics']), num_frames),
            ('cam_points', len(batch['cam_points']), num_frames),
            ('world_points', len(batch['world_points']), num_frames),
            ('point_masks', len(batch['point_masks']), num_frames),
            ('semantic_weights', len(batch['semantic_weights']), num_frames),
        ]

        all_valid = True
        for name, actual, expected in checks:
            status = "✓" if actual == expected else "✗"
            print(f"  {status} {name}: {actual} frames (expected {expected})")
            all_valid = all_valid and (actual == expected)

        # Check image shape
        img_shape = batch['images'][0].shape
        print(f"\n  Image shape: {img_shape}")
        print(f"  Depth shape: {batch['depths'][0].shape}")
        print(f"  Extrinsics shape: {batch['extrinsics'][0].shape}")
        print(f"  Intrinsics shape: {batch['intrinsics'][0].shape}")

        if not all_valid:
            print("\n✗ Shape validation failed")
            return False

        print("\n✓ All shapes validated successfully")

    except Exception as e:
        print(f"✗ Shape validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Analyze depth coverage
    print("\n[4/6] Analyzing depth coverage...")
    try:
        depth_stats = analyze_depth_coverage(batch['depths'], batch['point_masks'])

        print(f"  Mean coverage: {depth_stats['mean_coverage']:.2f}%")
        print(f"  Median coverage: {depth_stats['median_coverage']:.2f}%")
        print(f"  Mean valid points/frame: {depth_stats['mean_valid_points']:.0f}")
        print(f"  Valid points range: [{depth_stats['min_valid_points']:.0f}, {depth_stats['max_valid_points']:.0f}]")
        print(f"  Frames with ≥2000 points: {depth_stats['sufficient_points_ratio']*100:.1f}%")

        if depth_stats['mean_coverage'] < 1.0:
            print(f"\n  ⚠ Warning: Coverage is very low ({depth_stats['mean_coverage']:.2f}%)")
            print(f"    This is similar to ETH3D. Expected 30-70% for KITTI-360.")
        elif depth_stats['mean_coverage'] >= 30.0:
            print(f"\n  ✓ Good coverage! Much better than ETH3D (0.1%)")

    except Exception as e:
        print(f"✗ Depth coverage analysis failed: {e}")
        import traceback
        traceback.print_exc()

    # Analyze semantic weights
    print("\n[5/6] Analyzing semantic weights...")
    try:
        if 'semantic_weights' in batch:
            sem_stats = analyze_semantic_weights(batch['semantic_weights'])

            print(f"  Mean weight: {sem_stats['mean_weight']:.3f}")
            print(f"  Weight range: [{sem_stats['min_weight']:.3f}, {sem_stats['max_weight']:.3f}]")
            print(f"  High-weight pixels (building): {sem_stats['high_weight_ratio']*100:.1f}%")
            print(f"  Zero-weight pixels: {sem_stats['zero_weight_ratio']*100:.1f}%")

            if sem_stats['high_weight_ratio'] > 0.1:
                print(f"\n  ✓ Semantic weighting is working (building pixels detected)")
            else:
                print(f"\n  ⚠ Warning: Very few high-weight pixels detected")
        else:
            print(f"  ⚠ No semantic weights in batch")

    except Exception as e:
        print(f"✗ Semantic weight analysis failed: {e}")
        import traceback
        traceback.print_exc()

    # Test multiple batches
    print("\n[6/6] Testing multiple batch loads...")
    try:
        coverage_stats = []
        num_test_batches = min(5, len(dataset))
        for i in range(num_test_batches):
            # Use None for seq_index to trigger random sampling
            batch = dataset.get_data(seq_index=None, img_per_seq=4, aspect_ratio=1.0)
            stats = analyze_depth_coverage(batch['depths'], batch['point_masks'])
            coverage_stats.append(stats['mean_coverage'])

        print(f"  Loaded {num_test_batches} batches successfully")
        print(f"  Coverage across batches: {np.mean(coverage_stats):.2f}% ± {np.std(coverage_stats):.2f}%")
        print(f"\n✓ Multi-batch loading successful")

    except Exception as e:
        print(f"✗ Multi-batch loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)

    return True


def visualize_samples(num_samples=3):
    """Visualize sample data (requires matplotlib)"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
    except ImportError:
        print("Matplotlib not available for visualization")
        return

    print("\n" + "=" * 80)
    print("Visualizing samples...")
    print("=" * 80)

    # Create dataset
    config = create_dummy_config()
    dataset = KITTI360Dataset(
        root_dir="/home/zerun/data/dataset/KITTI-360",
        split="train",
        common_conf=config,
        sampling_strategy="adaptive",
        building_sampling_weights=(0.5, 0.3, 0.2),
        accumulation_frames=4,
        min_valid_points=2000,
        semantic_weight_enabled=True,
        filter_buildings_only=False,
        depth_range=(0.1, 80.0),
        len_train=100,
    )

    for sample_idx in range(num_samples):
        batch = dataset.get_data(seq_index=sample_idx, img_per_seq=2, aspect_ratio=1.0)

        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)

        for frame_idx in range(2):
            # RGB image
            ax = fig.add_subplot(gs[frame_idx, 0])
            img = batch['images'][frame_idx]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            ax.imshow(img)
            ax.set_title(f"Frame {frame_idx}: RGB")
            ax.axis('off')

            # Depth map
            ax = fig.add_subplot(gs[frame_idx, 1])
            depth = batch['depths'][frame_idx]
            mask = batch['point_masks'][frame_idx]
            depth_vis = np.copy(depth)
            depth_vis[~mask] = 0
            im = ax.imshow(depth_vis, cmap='plasma', vmin=0, vmax=50)
            ax.set_title(f"Frame {frame_idx}: Depth (coverage: {mask.mean()*100:.1f}%)")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

            # Semantic weights
            ax = fig.add_subplot(gs[frame_idx, 2])
            weights = batch['semantic_weights'][frame_idx]
            im = ax.imshow(weights, cmap='hot', vmin=0, vmax=2.0)
            ax.set_title(f"Frame {frame_idx}: Semantic Weights")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        plt.suptitle(f"Sample {sample_idx + 1}: {batch['seq_name']}", fontsize=14)
        plt.tight_layout()

        output_path = f"kitti360_sample_{sample_idx + 1}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization: {output_path}")
        plt.close()

    print("✓ Visualization complete")


def main():
    parser = argparse.ArgumentParser(description="Test KITTI-360 dataset loader")
    parser.add_argument('--visualize', action='store_true', help="Generate visualizations")
    parser.add_argument('--num-samples', type=int, default=3, help="Number of samples to visualize")
    args = parser.parse_args()

    # Run tests
    success = test_dataset_loading()

    # Visualize if requested
    if args.visualize and success:
        visualize_samples(num_samples=args.num_samples)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
