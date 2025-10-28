# Memory Optimization and Speed Enhancement for KITTI-360 Evaluation

## Overview

The `evaluate_kitti360_buildings.py` script has been optimized to:
1. Limit memory usage to under 50GB during evaluation
2. Provide faster evaluation with configurable sequence overlap
3. Support both validation and training set evaluation

## Key Optimizations

1. **Generator Mode (default)**: Samples are loaded on-demand instead of all at once
   - Reduces CPU memory usage significantly
   - Only one sequence kept in memory at a time

2. **GPU Memory Cleanup**: Aggressive memory management
   - Tensors are deleted after each sample
   - `torch.cuda.empty_cache()` called regularly
   - Models are unloaded between checkpoints

3. **Streaming Data Loading**: Data is processed in a streaming fashion
   - No accumulation of samples in memory
   - Constant memory footprint regardless of dataset size

## Usage

### Basic Usage (Memory Efficient, Fast - Default)

**Recommended for fast evaluation (non-overlapping windows):**
```bash
python scripts/evaluate_kitti360_buildings.py \
    --data-root /home/zerun/data/dataset/KITTI-360 \
    --log-dir logs/lora_kitti360_strategy_b_r16 \
    --device cuda:0 \
    --split val \
    --seq-stride 8 \
    --verbose
```

**For more thorough evaluation (50% overlap, 2x slower):**
```bash
python scripts/evaluate_kitti360_buildings.py \
    --data-root /home/zerun/data/dataset/KITTI-360 \
    --log-dir logs/lora_kitti360_strategy_b_r16 \
    --device cuda:0 \
    --split val \
    --seq-stride 4 \
    --verbose
```

### Disable Generator Mode (High Memory Usage)

If you need the old behavior (all samples loaded at once):

```bash
python scripts/evaluate_kitti360_buildings.py \
    --data-root /home/zerun/data/dataset/KITTI-360 \
    --log-dir logs/lora_kitti360_strategy_b_r16 \
    --device cuda:0 \
    --no-generator \
    --verbose
```

### Test Memory Usage

To verify that memory usage stays under 50GB:

```bash
python scripts/test_memory_usage.py \
    --data-root /home/zerun/data/dataset/KITTI-360 \
    --log-dir logs/lora_kitti360_strategy_b_r16 \
    --max-memory 50.0 \
    --max-seqs 10 \
    --device cuda:0
```

This will run the evaluation on 10 sequences and monitor memory usage, failing if it exceeds 50GB.

## Memory Usage Comparison

### Before Optimization
- **CPU Memory**: ~80-150GB (all samples loaded)
- **GPU Memory**: ~15-20GB (accumulated tensors)
- **Issue**: Out-of-memory errors on machines with <128GB RAM

### After Optimization
- **CPU Memory**: ~10-20GB (generator mode)
- **GPU Memory**: ~5-8GB (cleaned after each sample)
- **Benefit**: Runs reliably on machines with 64GB+ RAM

## Command-Line Options

### Speed/Thoroughness Trade-off
- `--split {train,val}`: Dataset split to use (default: `val`)
  - **`val`**: Validation/test set for evaluation (RECOMMENDED, this is the test set)
  - **`train`**: Training set for debugging only (not for evaluation)
- `--seq-stride N`: Stride between sequences (default: `8`)
  - `stride=8` (seq_len=8): Non-overlapping windows, **fastest**, ~1500 sequences on val set
  - `stride=4` (seq_len=8): 50% overlap, **2x slower**, ~3000 sequences on val set
  - `stride=2` (seq_len=8): 75% overlap, **4x slower**, ~6000 sequences on val set

### Memory Management
- `--no-generator`: Disable memory-efficient generator mode (not recommended)
- `--max-seqs N`: Limit evaluation to N sequences (useful for testing)

### Other Options
- `--seq-len N`: Frames per sequence (default: 8)
- `--device DEVICE`: Device to use (default: `cuda:0`)
- `--verbose`: Show detailed logging including memory usage

## Evaluation Speed Estimates

Based on typical performance (1.3s/sequence):

| Stride | Sequences (val) | Time/Checkpoint | Total Time (5 ckpts) |
|--------|----------------|-----------------|----------------------|
| 8      | ~1,500         | ~32 min         | ~2.7 hours          |
| 4      | ~3,000         | ~65 min         | ~5.4 hours          |
| 2      | ~6,000         | ~130 min        | ~10.8 hours         |

**Recommendation**: Use `--seq-stride 8` for initial evaluation, use `--seq-stride 4` for final/published results.

## Monitoring Memory Usage

The script logs GPU memory usage at key points when using `--verbose`:

```
INFO - GPU Memory before evaluation: 4.23 GB
INFO - First batch complete. GPU Memory: 7.45 GB
INFO - Freed base model memory. GPU Memory: 4.23 GB
INFO - Freed model memory. GPU Memory: 4.23 GB
```

## Troubleshooting

### Still Running Out of Memory?

1. **Reduce batch size**: Use `--max-seqs` to process fewer sequences
2. **Check GPU memory**: Use `nvidia-smi` to monitor GPU usage
3. **Close other programs**: Free up system memory
4. **Use CPU**: Set `--device cpu` (slower but uses less GPU memory)

### Performance Impact

Generator mode has minimal performance impact:
- **Throughput**: ~5-10% slower due to on-demand loading
- **Latency**: Negligible difference per sample
- **Disk I/O**: Slightly higher, but cached by OS

## Technical Details

### Generator Implementation

The `sample_generator()` function yields samples one at a time:

```python
def sample_generator(dataset, seq_len, stride, max_sequences):
    for seq_name, windows in sequences.items():
        for frame_indices in windows:
            # Load and yield one sample
            sample = create_sample(frame_indices)
            yield sample
            # Sample is garbage collected after use
```

### Memory Cleanup Strategy

After processing each sample:

```python
# Delete tensors
del data, outputs, pred_depths
if pred_extrinsics is not None:
    del pred_extrinsics, pred_intrinsics

# Clear GPU cache
torch.cuda.empty_cache()
```

After each model:

```python
# Unload model
del model

# Clear GPU cache
torch.cuda.empty_cache()
```

## Related Files

- `scripts/evaluate_kitti360_buildings.py` - Main evaluation script
- `scripts/test_memory_usage.py` - Memory monitoring utility
- `training/data/datasets/kitti360.py` - KITTI-360 dataset implementation
