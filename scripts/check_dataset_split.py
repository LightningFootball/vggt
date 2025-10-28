#!/usr/bin/env python3
"""
Quick script to verify KITTI-360 dataset split statistics.
"""

import os
import sys
from pathlib import Path

def check_split_files(data_root: str):
    """Check and display KITTI-360 split file statistics."""

    split_dir = Path(data_root) / "data_2d_semantics" / "train"
    test_dir = Path(data_root) / "data_2d_test"

    if not split_dir.exists():
        print(f"❌ Error: Split directory not found: {split_dir}")
        print(f"   Make sure KITTI-360 is properly downloaded to: {data_root}")
        return False

    print("=" * 80)
    print("KITTI-360 Dataset Split Information")
    print("=" * 80)
    print(f"Data root: {data_root}")
    print()

    # Check for split files (train/val are frame-level splits within training sequences)
    train_file = split_dir / "2013_05_28_drive_train_frames.txt"
    val_file = split_dir / "2013_05_28_drive_val_frames.txt"

    print("Train/Val split files (from training sequences):")
    print(f"  train: {'✓ exists' if train_file.exists() else '✗ missing'} - {train_file}")
    print(f"  val:   {'✓ exists' if val_file.exists() else '✗ missing'} - {val_file}")
    print()

    print("Test sequences directory:")
    print(f"  test:  {'✓ exists' if test_dir.exists() else '✗ missing'} - {test_dir}")
    print()

    # Count frames in train/val splits
    if train_file.exists():
        with open(train_file, 'r') as f:
            train_count = len([line for line in f if line.strip()])
        print(f"Training frames (from training sequences): {train_count:,}")
    else:
        print(f"❌ Training split file not found!")
        return False

    if val_file.exists():
        with open(val_file, 'r') as f:
            val_count = len([line for line in f if line.strip()])
        print(f"Validation frames (from training sequences): {val_count:,}")
    else:
        print(f"❌ Validation split file not found!")
        return False

    # Count frames in test sequences
    test_count = 0
    if test_dir.exists():
        test_sequences = sorted([d for d in test_dir.iterdir() if d.is_dir()])
        print(f"\nTest sequences (held-out sequences):")
        for seq_dir in test_sequences:
            image_dir = seq_dir / "image_00" / "data_rect"
            if image_dir.exists():
                seq_frames = len(list(image_dir.glob("*.png")))
                test_count += seq_frames
                print(f"  {seq_dir.name}: {seq_frames:,} frames")
        if test_count > 0:
            print(f"Total test frames: {test_count:,}")
        else:
            print(f"  (No images found in test sequences)")
    else:
        print(f"❌ Test directory not found!")
        test_count = 0

    print()
    total = train_count + val_count + test_count
    print(f"Grand total: {total:,} frames")
    print()

    # Calculate percentages
    if total > 0:
        train_pct = (train_count / total) * 100
        val_pct = (val_count / total) * 100
        test_pct = (test_count / total) * 100

        print("Split percentages:")
        print(f"  Training:   {train_pct:5.1f}% ({train_count:,} frames)")
        print(f"  Validation: {val_pct:5.1f}% ({val_count:,} frames)")
        print(f"  Test:       {test_pct:5.1f}% ({test_count:,} frames)")
    print()

    print("=" * 80)
    print("Summary:")
    print("=" * 80)
    print("✓ KITTI-360 has THREE splits: train, val, and test")
    print("✓ Train/val are frame-level splits within 9 training sequences")
    print("✓ Test uses 2 completely separate sequences (0008, 0018)")
    print()
    print("For FINAL model evaluation, use:")
    print("  --split test  (held-out sequences, ~900 frames)")
    print()
    print("For validation during development, use:")
    print("  --split val  (~12k frames from training sequences)")
    print()
    print("For overfitting analysis/debugging, use:")
    print("  --split train  (~49k frames from training sequences)")
    print("=" * 80)

    return True

def main():
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        data_root = "/home/zerun/data/dataset/KITTI-360"

    if not os.path.exists(data_root):
        print(f"❌ Error: Data root does not exist: {data_root}")
        print(f"Usage: {sys.argv[0]} [DATA_ROOT]")
        return 1

    success = check_split_files(data_root)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
