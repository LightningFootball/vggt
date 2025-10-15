#!/usr/bin/env python3
"""
Test actual ETH3D dataset loading to see real sparse depth maps.
"""

import sys
sys.path.insert(0, '/home/zerun/workspace/vggt/training')

import torch
import numpy as np
from data.datasets.eth3d import ETH3DDataset

def analyze_sparsity(batch):
    """Analyze the actual sparse depth maps"""

    images = batch['images']  # [S, 3, H, W]
    depths = batch['depths']  # [S, H, W]
    point_masks = batch['point_masks']  # [S, H, W]

    S = images.shape[0]

    for i in range(S):
        # Get numpy arrays
        depth_np = depths[i].numpy()
        mask_np = point_masks[i].numpy()

        # Print statistics
        total_pixels = mask_np.size
        valid_pixels = mask_np.sum()
        coverage = valid_pixels / total_pixels * 100

        print(f"Frame {i}:")
        print(f"  Total pixels: {total_pixels:,}")
        print(f"  Valid depth pixels: {valid_pixels:,}")
        print(f"  Coverage: {coverage:.4f}%")
        if valid_pixels > 0:
            print(f"  Depth range: [{depth_np[mask_np].min():.2f}, {depth_np[mask_np].max():.2f}]")
        else:
            print(f"  Depth range: N/A (no valid points)")
        print()

def main():
    print("=" * 80)
    print("Testing ETH3D Dataset Actual Loading")
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

    print(f"Dataset size: {len(dataset)} sequences\n")

    # Test first sequence
    print("Loading first sequence...")
    batch = dataset[0]

    print("\n📊 Batch Information:")
    print(f"  Sequence name: {batch['seq_name']}")
    print(f"  Number of frames: {batch['images'].shape[0]}")
    print(f"  Image shape: {batch['images'].shape}")
    print(f"  Depth shape: {batch['depths'].shape}")
    print(f"  Extrinsics shape: {batch['extrinsics'].shape}")
    print(f"  Intrinsics shape: {batch['intrinsics'].shape}")
    print()

    # Analyze sparsity
    print("=" * 80)
    print("🔍 Sparsity Analysis")
    print("=" * 80)
    print()

    point_masks = batch['point_masks']
    total_frames = point_masks.shape[0]

    all_valid_counts = []
    for i in range(total_frames):
        valid_count = point_masks[i].sum().item()
        all_valid_counts.append(valid_count)

    print(f"Valid points per frame:")
    print(f"  Min: {min(all_valid_counts)}")
    print(f"  Max: {max(all_valid_counts)}")
    print(f"  Mean: {np.mean(all_valid_counts):.1f}")
    print(f"  Median: {np.median(all_valid_counts):.1f}")
    print()

    # Check against training thresholds
    print("=" * 80)
    print("⚠️  Training Threshold Check (from loss.py)")
    print("=" * 80)
    print()

    # Camera loss threshold (loss.py:98)
    camera_valid_frames = sum(1 for c in all_valid_counts if c > 10)
    print(f"Camera loss valid frames (>10 points): {camera_valid_frames}/{total_frames}")

    # Depth loss threshold (loss.py:262)
    depth_valid_frames = sum(1 for c in all_valid_counts if c >= 10)
    print(f"Depth loss valid frames (≥10 points): {depth_valid_frames}/{total_frames}")

    # Original Co3D threshold would be >100
    co3d_style_valid = sum(1 for c in all_valid_counts if c > 100)
    print(f"Co3D-style valid frames (>100 points): {co3d_style_valid}/{total_frames}")
    print()

    # Coverage comparison
    total_pixels = 518 * 518
    avg_coverage = np.mean(all_valid_counts) / total_pixels * 100

    print("=" * 80)
    print("📈 Coverage Comparison")
    print("=" * 80)
    print()
    print(f"ETH3D actual coverage: {avg_coverage:.4f}%")
    print(f"Co3D typical coverage: ~90% (dense depth from MVS)")
    print(f"VKitti typical coverage: ~100% (synthetic dense depth)")
    print()
    print(f"❌ ETH3D provides ~{90/avg_coverage:.0f}x LESS supervision than Co3D")
    print()

    # Detailed frame analysis
    print("=" * 80)
    print("🖼️  Detailed Frame Analysis")
    print("=" * 80)
    print()

    analyze_sparsity(batch)

    # Final verdict
    print("\n" + "=" * 80)
    print("🎯 FINAL VERDICT")
    print("=" * 80)
    print()

    if avg_coverage < 1.0:
        print("❌ CONFIRMED: ETH3D has INSUFFICIENT depth supervision for VGGT training")
        print()
        print("Evidence:")
        print(f"  1. Coverage: {avg_coverage:.4f}% (vs Co3D's ~90%)")
        print(f"  2. Avg points per frame: {np.mean(all_valid_counts):.0f} (vs Co3D's ~240,000)")
        print(f"  3. Supervision pixels: ~{int(np.mean(all_valid_counts))}/{total_pixels} ({avg_coverage:.4f}%)")
        print()
        print("This is NOT a code error - it's the fundamental limitation of COLMAP sparse")
        print("reconstruction vs MVS/synthetic dense depth.")
        print()
        print("✅ Recommended Actions:")
        print("  1. Switch to Replica/ScanNet/Matterport3D (dense depth)")
        print("  2. Or use Depth Anything V2 to generate pseudo-dense depth")
        print("  3. Or redesign loss for sparse supervision (major work)")
    else:
        print("✅ Coverage is sufficient")

    print("=" * 80)

if __name__ == '__main__':
    main()
