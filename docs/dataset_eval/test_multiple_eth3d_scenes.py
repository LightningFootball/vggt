#!/usr/bin/env python3
"""
Test multiple ETH3D scenes to get comprehensive statistics.
"""

import sys
sys.path.insert(0, '/home/zerun/workspace/vggt/training')

import torch
import numpy as np
from data.datasets.eth3d import ETH3DDataset

def main():
    print("=" * 80)
    print("Comprehensive ETH3D Dataset Sparsity Test")
    print("=" * 80)
    print()

    # Initialize dataset
    dataset = ETH3DDataset(
        root_dir='/home/zerun/data/dataset/ETH3D/Stereo/High-res_multi-view',
        split='train',
        img_size=518,
        sequence_length=8,
        use_building_scenes_only=True,
        train_val_split=0.85,
    )

    print(f"Total sequences in dataset: {len(dataset)}\n")

    # Test first 10 sequences (or all if less than 10)
    num_test = min(10, len(dataset))

    all_coverages = []
    all_point_counts = []

    for idx in range(num_test):
        print(f"Testing sequence {idx+1}/{num_test}...")
        batch = dataset[idx]

        point_masks = batch['point_masks']
        S = point_masks.shape[0]

        seq_coverages = []
        seq_point_counts = []

        for i in range(S):
            mask_np = point_masks[i].numpy()
            valid_pixels = mask_np.sum()
            total_pixels = mask_np.size
            coverage = valid_pixels / total_pixels * 100

            seq_coverages.append(coverage)
            seq_point_counts.append(valid_pixels)
            all_coverages.append(coverage)
            all_point_counts.append(valid_pixels)

        print(f"  Sequence: {batch['seq_name']}")
        print(f"  Frames: {S}")
        print(f"  Avg coverage: {np.mean(seq_coverages):.4f}%")
        print(f"  Avg points: {np.mean(seq_point_counts):.1f}")
        print(f"  Min points: {min(seq_point_counts)}")
        print(f"  Max points: {max(seq_point_counts)}")
        print()

    # Overall statistics
    print("=" * 80)
    print("📊 OVERALL STATISTICS")
    print("=" * 80)
    print()

    print(f"Total frames tested: {len(all_point_counts)}")
    print()

    print("Coverage Statistics:")
    print(f"  Mean: {np.mean(all_coverages):.4f}%")
    print(f"  Median: {np.median(all_coverages):.4f}%")
    print(f"  Min: {np.min(all_coverages):.4f}%")
    print(f"  Max: {np.max(all_coverages):.4f}%")
    print(f"  Std: {np.std(all_coverages):.4f}%")
    print()

    print("Valid Points per Frame:")
    print(f"  Mean: {np.mean(all_point_counts):.1f}")
    print(f"  Median: {np.median(all_point_counts):.1f}")
    print(f"  Min: {int(np.min(all_point_counts))}")
    print(f"  Max: {int(np.max(all_point_counts))}")
    print(f"  Std: {np.std(all_point_counts):.1f}")
    print()

    # Training viability check
    print("=" * 80)
    print("⚠️  TRAINING VIABILITY CHECK")
    print("=" * 80)
    print()

    frames_above_10 = sum(1 for c in all_point_counts if c > 10)
    frames_above_100 = sum(1 for c in all_point_counts if c > 100)
    frames_above_1000 = sum(1 for c in all_point_counts if c > 1000)

    total_frames = len(all_point_counts)

    print(f"Frames with >10 points: {frames_above_10}/{total_frames} ({frames_above_10/total_frames*100:.1f}%)")
    print(f"Frames with >100 points: {frames_above_100}/{total_frames} ({frames_above_100/total_frames*100:.1f}%)")
    print(f"Frames with >1000 points: {frames_above_1000}/{total_frames} ({frames_above_1000/total_frames*100:.1f}%)")
    print()

    # Compare with Co3D
    total_pixels = 518 * 518
    co3d_dense_coverage = 90.0  # %
    co3d_points = int(total_pixels * co3d_dense_coverage / 100)

    print("=" * 80)
    print("📈 COMPARISON WITH CO3D")
    print("=" * 80)
    print()

    print(f"Co3D typical supervision:")
    print(f"  Coverage: ~{co3d_dense_coverage}%")
    print(f"  Points per frame: ~{co3d_points:,}")
    print()

    print(f"ETH3D actual supervision:")
    print(f"  Coverage: {np.mean(all_coverages):.4f}%")
    print(f"  Points per frame: ~{int(np.mean(all_point_counts))}")
    print()

    supervision_ratio = co3d_dense_coverage / np.mean(all_coverages)
    print(f"❌ ETH3D provides {supervision_ratio:.0f}x LESS supervision than Co3D")
    print()

    # Final recommendation
    print("=" * 80)
    print("🎯 FINAL RECOMMENDATION")
    print("=" * 80)
    print()

    if np.mean(all_coverages) < 1.0:
        print("❌ ETH3D is NOT SUITABLE for VGGT LoRA fine-tuning as-is")
        print()
        print("Reasons:")
        print(f"  1. Coverage ({np.mean(all_coverages):.4f}%) is {supervision_ratio:.0f}x lower than Co3D")
        print(f"  2. Only {int(np.mean(all_point_counts))} supervision points vs Co3D's ~{co3d_points:,}")
        print(f"  3. Gradient signal will be extremely sparse and unstable")
        print()
        print("✅ Viable Alternatives:")
        print()
        print("  Option 1 (Recommended): Use Dense Depth Datasets")
        print("    - Replica (indoor, synthetic, perfect depth)")
        print("    - ScanNet (indoor, real-world, dense depth)")
        print("    - Matterport3D (indoor, real-world, dense depth)")
        print("    - KITTI-360 (outdoor, LiDAR depth)")
        print()
        print("  Option 2: Generate Pseudo-Dense Depth for ETH3D")
        print("    - Use Depth Anything V2 / ZoeDepth / Metric3D")
        print("    - Align scale/shift with COLMAP sparse points")
        print("    - Risk: pseudo-GT quality may be lower")
        print()
        print("  Option 3: Modify Training for Sparse Supervision")
        print("    - Redesign loss to only supervise sparse keypoints")
        print("    - Add unsupervised losses (photometric consistency, etc.)")
        print("    - Major code changes required")
    else:
        print("✅ Coverage is marginally acceptable")

    print("=" * 80)

if __name__ == '__main__':
    main()
