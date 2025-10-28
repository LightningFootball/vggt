# KITTI-360 Dataset Split Explanation

## Summary - CORRECTED ✅

**KITTI-360 DOES have a separate test split!** The dataset has three splits:

1. **`train`**: Training set (~49,000 frames from sequences 0000, 0002-0007, 0009-0010)
2. **`val`**: Validation set (~12,000 frames from sequences 0000, 0002-0007, 0009-0010)
3. **`test`**: Test set (~910 frames from sequences 0008, 0018) ← **Use this for final evaluation**

## Dataset Structure

KITTI-360 has a hierarchical split structure:

### Training/Validation Sequences (9 sequences)
Located in `data_2d_raw/`:
- 2013_05_28_drive_0000_sync
- 2013_05_28_drive_0002_sync
- 2013_05_28_drive_0003_sync
- 2013_05_28_drive_0004_sync
- 2013_05_28_drive_0005_sync
- 2013_05_28_drive_0006_sync
- 2013_05_28_drive_0007_sync
- 2013_05_28_drive_0009_sync
- 2013_05_28_drive_0010_sync

These sequences are further split into:
- **train frames**: Listed in `data_2d_semantics/train/2013_05_28_drive_train_frames.txt` (~49,004 frames)
- **val frames**: Listed in `data_2d_semantics/train/2013_05_28_drive_val_frames.txt` (~12,276 frames)

### Test Sequences (2 sequences) ✅
Located in `data_2d_test/`:
- 2013_05_28_drive_0008_sync (~566 frames)
- 2013_05_28_drive_0018_sync (~344 frames)
- **Total: ~910 frames**

These are completely held-out sequences with no overlap with training/validation data.

## What Should I Use?

### ✅ For Final Model Evaluation
```bash
--split test
```

**This is the correct choice for evaluation!** The `test` split uses completely held-out sequences (0008, 0018).
- Use this for final benchmarking
- Use this for reporting results in papers
- Use this for comparing different models

### For Validation During Development
```bash
--split val
```

Use this for:
- Hyperparameter tuning
- Model selection
- Quick evaluation during development

### ❌ DON'T Use Training Set for Evaluation
```bash
--split train  # Don't use this for evaluation!
```

The training set should only be used for:
- Debugging data loading issues
- Checking if the model is overfitting (should perform better on train than val/test)
- Sanity checks

## Verification

You can verify this structure:

```bash
# Check train/val split files
$ ls /home/zerun/data/dataset/KITTI-360/data_2d_semantics/train/
2013_05_28_drive_train_frames.txt  # Training frames (~49k)
2013_05_28_drive_val_frames.txt    # Validation frames (~12k)

# Check test sequences
$ ls /home/zerun/data/dataset/KITTI-360/data_2d_test/
2013_05_28_drive_0008_sync/  # Test sequence 1 (~566 frames)
2013_05_28_drive_0018_sync/  # Test sequence 2 (~344 frames)

# Count frames in test sequences
$ ls /home/zerun/data/dataset/KITTI-360/data_2d_test/2013_05_28_drive_0008_sync/image_00/data_rect/ | wc -l
566

$ ls /home/zerun/data/dataset/KITTI-360/data_2d_test/2013_05_28_drive_0018_sync/image_00/data_rect/ | wc -l
344
```

You can also run the verification script:
```bash
python scripts/check_dataset_split.py
```

## Comparison with Other Datasets

### Datasets with separate test sequences (like KITTI-360):
- **KITTI-360**: train (~49k) + val (~12k) + test (~0.9k, sequences 0008/0018)
- **KITTI**: Similar structure with held-out test sequences
- **Waymo Open**: train + val + test (separate sequences)

### Datasets with train/val only:
- **NYUv2**: train (795) + test (654)
- **ScanNet**: train + val

## Conclusion - CORRECTED ✅

**You should use `--split test` for final evaluation!**

The test split uses completely different sequences (0008, 0018) that were never seen during training or validation. This provides a true out-of-distribution test set for fair evaluation.

### Recommended Workflow:
1. **Development**: Use `--split val` for quick iteration
2. **Final Evaluation**: Use `--split test` for results to report
3. **Debugging**: Use `--split train` to check overfitting

## Updated Command

For final evaluation, use:
```bash
python scripts/evaluate_kitti360_buildings.py \
    --data-root /home/zerun/data/dataset/KITTI-360 \
    --log-dir logs/lora_kitti360_strategy_b_r16 \
    --device cuda:0 \
    --split test \    # ← Use test split for final evaluation
    --seq-stride 8 \
    --verbose
```
