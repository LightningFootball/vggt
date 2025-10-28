# KITTI-360 Evaluation Quick Guide

## TL;DR

### Final Evaluation on Test Set (For Reporting Results)
```bash
python scripts/evaluate_kitti360_buildings.py \
    --data-root /home/zerun/data/dataset/KITTI-360 \
    --log-dir logs/lora_kitti360_strategy_b_r16 \
    --device cuda:0 \
    --split test \     # Test sequences 0008, 0018 (910 frames)
    --seq-stride 8 \
    --verbose
```
**Time**: ~2 min/checkpoint (test set is small)
**Sequences**: ~110 non-overlapping windows from held-out sequences
**Memory**: <50GB RAM, ~5-8GB VRAM

### Validation During Development
```bash
python scripts/evaluate_kitti360_buildings.py \
    --data-root /home/zerun/data/dataset/KITTI-360 \
    --log-dir logs/lora_kitti360_strategy_b_r16 \
    --device cuda:0 \
    --split val \      # Validation frames from training sequences (12k frames)
    --seq-stride 8 \
    --verbose
```
**Time**: ~32 min/checkpoint (~2.7 hours for 5 checkpoints)
**Sequences**: ~1,500 non-overlapping windows
**Memory**: <50GB RAM, ~5-8GB VRAM

## Understanding the Current Behavior

### Your Current Command
```bash
python scripts/evaluate_kitti360_buildings.py \
    --data-root /home/zerun/data/dataset/KITTI-360 \
    --log-dir logs/lora_kitti360_strategy_b_r16 \
    --device cuda:0 \
    --verbose
```

**What's happening:**
- Using **validation set** (`split=val` by default)
- Using **stride=8** (changed from old default of 4 for faster evaluation)
- Processing ~1,500 sequences (down from 3,058 with old stride=4)
- Taking ~1.3s per sequence
- **Total time per checkpoint**: ~32 minutes (down from 66 minutes)
- **Total time for all checkpoints**: Depends on number of checkpoints in `logs/lora_kitti360_strategy_b_r16/ckpts/`

### Why Was It Slow Before?

**Old defaults (stride=4)**:
- Created ~3,058 overlapping sequences
- Each sequence takes ~1.3 seconds
- Total: 3,058 × 1.3s ≈ **66 minutes per checkpoint**
- If you have 5 checkpoints: 66 × 5 = **330 minutes (5.5 hours)**

**New defaults (stride=8)**:
- Creates ~1,529 non-overlapping sequences
- Each sequence takes ~1.3 seconds
- Total: 1,529 × 1.3s ≈ **32 minutes per checkpoint**
- If you have 5 checkpoints: 32 × 5 = **160 minutes (2.7 hours)**

## Which Dataset Split Should I Use?

**Important**: KITTI-360 only has two splits: `train` and `val`. There is no separate `test` split.

### Validation Set (`--split val`) ✅ **RECOMMENDED FOR EVALUATION**
- **This is the test set**: Used for final model evaluation and comparison
- **Size**: ~5,000 frames from held-out sequences
- **Use for**:
  - ✅ Model evaluation and benchmarking
  - ✅ Comparing different checkpoints
  - ✅ Reporting final results in papers
- **This is the default and what you should use**

### Training Set (`--split train`)
- **Not for evaluation**: Only for debugging/analysis
- **Size**: ~50,000 frames from training sequences
- **Use for**:
  - Checking if model is overfitting
  - Debugging data loading issues
  - Sanity checks (model should perform better on train than val)
- **Takes 10x longer than validation set**
- **Don't use for final evaluation metrics**

## Speed vs. Thoroughness Trade-off

### Stride Comparison

| --seq-stride | Overlap | Sequences | Speed      | Use Case                    |
|--------------|---------|-----------|------------|-----------------------------|
| 8 (new default) | 0%   | ~1,500    | Fast (32min) | Development, quick tests |
| 4 (old default) | 50%  | ~3,000    | Medium (65min) | Final evaluation, papers |
| 2            | 75%     | ~6,000    | Slow (130min) | Very thorough analysis   |

### Recommendation
1. **During development**: Use `--seq-stride 8` (default)
2. **Before submitting paper**: Use `--seq-stride 4`
3. **For ablation studies**: Use `--seq-stride 8` to save time

## Quick Tests

### Test with 10 sequences only
```bash
python scripts/evaluate_kitti360_buildings.py \
    --data-root /home/zerun/data/dataset/KITTI-360 \
    --log-dir logs/lora_kitti360_strategy_b_r16 \
    --device cuda:0 \
    --max-seqs 10 \
    --verbose
```
**Time**: ~13 seconds

### Monitor memory usage
```bash
python scripts/test_memory_usage.py \
    --data-root /home/zerun/data/dataset/KITTI-360 \
    --log-dir logs/lora_kitti360_strategy_b_r16 \
    --max-memory 50.0 \
    --max-seqs 10
```

## Output Files

After evaluation completes, you'll find:

```
evaluate/kitti360_b/
├── baseline_metrics.json          # Metrics for base model
├── checkpoint_1000_metrics.json   # Metrics for checkpoint at step 1000
├── checkpoint_2000_metrics.json   # Metrics for checkpoint at step 2000
├── ...
└── metrics_overall.csv            # Summary table of all checkpoints
```

## Common Issues

### Still too slow?
- Reduce sequences: `--max-seqs 500`
- Use faster stride: `--seq-stride 16` (non-standard)
- Evaluate fewer checkpoints: Delete some from `logs/.../ckpts/`

### Running out of memory?
- Should not happen with new optimizations
- If it does: Use `--max-seqs` to limit batch
- Check with: `python scripts/test_memory_usage.py`

### Want more thorough evaluation?
- Use smaller stride: `--seq-stride 2` or `--seq-stride 1`
- Warning: Will take much longer (4x-8x)

## Technical Details

### Sequence Window Generation
- Each sequence has 8 frames (`--seq-len 8`)
- Windows slide along temporal dimension with configurable stride
- Example with stride=8:
  - Window 1: frames [0, 1, 2, 3, 4, 5, 6, 7]
  - Window 2: frames [8, 9, 10, 11, 12, 13, 14, 15]
  - Window 3: frames [16, 17, 18, 19, 20, 21, 22, 23]
  - No overlap → faster, but less comprehensive

- Example with stride=4:
  - Window 1: frames [0, 1, 2, 3, 4, 5, 6, 7]
  - Window 2: frames [4, 5, 6, 7, 8, 9, 10, 11]
  - Window 3: frames [8, 9, 10, 11, 12, 13, 14, 15]
  - 50% overlap → slower, more comprehensive

### Memory Optimization
- Generator mode loads one sequence at a time
- GPU tensors are deleted after each sequence
- Model is unloaded between checkpoints
- Total memory stays under 50GB regardless of dataset size
