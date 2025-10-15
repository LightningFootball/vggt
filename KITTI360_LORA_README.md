# KITTI-360 LoRA Fine-tuning for VGGT

This directory contains implementation for fine-tuning VGGT on KITTI-360 dataset using LoRA (Low-Rank Adaptation) with building-focused strategies.

## 📁 Files Created

### 1. Dataset Loader
- **`training/data/datasets/kitti360.py`** (733 lines)
  - KITTI-360 dataset loader with adaptive sampling
  - Multi-frame LiDAR accumulation for dense depth
  - Semantic-based pixel weighting
  - Compatible with VGGT training pipeline

### 2. Loss Modifications
- **`training/loss.py`** (modified)
  - Added semantic weighting support in `compute_depth_loss()`
  - Curriculum learning (facade boost in early epochs)
  - Epoch-aware weight scheduling

### 3. Training Configurations
- **`training/config/lora_kitti360_strategy_a.yaml`**
  - **Strategy A**: LoRA on depth head only (fastest, recommended start)
  - 100 epochs, LR=1e-4, LoRA rank=16

- **`training/config/lora_kitti360_strategy_b.yaml`**
  - **Strategy B**: Depth head + late 8 transformer layers
  - 150 epochs, LR=5e-5, LoRA rank=16

- **`training/config/lora_kitti360_strategy_c.yaml`**
  - **Strategy C**: Depth head + full 24 transformer layers
  - 200 epochs, LR=2e-5, LoRA rank=32

### 4. Testing Script
- **`test_kitti360_loading.py`**
  - Validates dataset loading
  - Analyzes depth coverage and semantic weights
  - Generates visualizations

---

## 🚀 Quick Start

### 1. Verify Dataset Structure
```bash
# Your KITTI-360 should look like:
/home/zerun/data/dataset/KITTI-360/
├── data_2d_raw/              # RGB images
├── data_3d_raw/              # LiDAR scans
├── data_poses/               # Camera poses
├── data_2d_semantics/        # Semantic segmentation
└── calibration/              # Camera calibration
```

### 2. Test Dataset Loading
```bash
# Basic test (no visualization)
python test_kitti360_loading.py

# With visualization (requires matplotlib)
python test_kitti360_loading.py --visualize --num-samples 5
```

Expected output:
```
✓ Dataset initialized successfully
  Total frames: XXXX
  Mean coverage: 40-70%  # Much better than ETH3D (0.1%)
  Mean valid points/frame: 50,000-130,000
```

### 3. Start Training

**Strategy A (Recommended First)**:
```bash
# Single GPU (use the provided script)
bash train_single_gpu.sh lora_kitti360_strategy_a

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 training/launch.py --config lora_kitti360_strategy_a
```

**Note**: The `train_single_gpu.sh` script automatically sets the required environment variables (`RANK`, `LOCAL_RANK`, etc.) for single GPU training.

**Strategy B (If Strategy A succeeds)**:
```bash
torchrun --nproc_per_node=4 training/launch.py --config lora_kitti360_strategy_b
```

**Strategy C (Advanced)**:
```bash
torchrun --nproc_per_node=4 training/launch.py --config lora_kitti360_strategy_c
```

---

## 📊 Key Features

### Adaptive Sampling Strategy
- **Building-rich scenes** (≥30% building pixels): 50% sampling weight
- **Mixed scenes** (10-30%): 30% sampling weight
- **Road-dominant scenes** (<10%): 20% sampling weight

### Semantic Pixel Weighting
```python
building/wall:  2.0x weight  # Focus on building facades
road/sidewalk:  0.7x weight  # Basic supervision
car:            0.5x weight  # Reduce moving objects
sky:            0.2x weight  # Background
```

### Curriculum Learning
- **First 40% epochs**: Boost facade weights by 1.5x (2.0 → 3.0)
- **Remaining 60%**: Gradually decay boost (3.0 → 2.0)

### Multi-frame LiDAR Accumulation
- Accumulates ±2 frames (4 total) for dense depth
- Expected coverage: 40-70% (vs ETH3D 0.1%)
- ~50,000-130,000 points/frame (vs ETH3D 293 points/frame)

---

## 🎯 LoRA Strategy Comparison

| Strategy | Target Modules | Epochs | LR | Rank | Speed | Quality | Use Case |
|----------|---------------|--------|-------|------|-------|---------|----------|
| **A** | Depth head only | 100 | 1e-4 | 16 | Fastest | Good | Quick validation |
| **B** | Depth + Late 8 layers | 150 | 5e-5 | 16 | Medium | Better | If A succeeds |
| **C** | Depth + All 24 layers | 200 | 2e-5 | 32 | Slowest | Best | Maximum quality |

**Recommendation**: Start with A → if successful, try B → if B improves, try C

---

## 🔧 Configuration Guide

### Adjust Sampling Weights
Edit `lora_kitti360_strategy_a.yaml`:
```yaml
dataset:
  building_sampling_weights: [0.6, 0.3, 0.1]  # More buildings
  # or
  building_sampling_weights: [0.4, 0.4, 0.2]  # More balanced
```

### Adjust Semantic Weights
Edit `training/data/datasets/kitti360.py`:
```python
DEFAULT_SEMANTIC_WEIGHTS = {
    11: 2.5,   # building (increase focus)
    7:  0.5,   # road (reduce more)
    # ...
}
```

### Adjust Curriculum Learning
Edit `lora_kitti360_strategy_a.yaml`:
```yaml
loss:
  depth:
    facade_boost_ratio: 0.3  # Boost first 30% epochs (default 40%)
```

### Adjust LiDAR Accumulation
Edit `lora_kitti360_strategy_a.yaml`:
```yaml
dataset:
  accumulation_frames: 6  # ±3 frames (default ±2)
  min_valid_points: 3000  # Stricter filtering (default 2000)
```

---

## 📈 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir logs/lora_kitti360_strategy_a/tensorboard
```

### Key Metrics to Watch
- `loss_conf_depth`: Main depth loss (should decrease steadily)
- `loss_grad_depth`: Gradient smoothness (for facade planarity)
- `loss_camera`: Camera pose loss (should be stable)

### Expected Loss Values
- Initial: `loss_objective` ~0.5-1.0
- Converged: `loss_objective` ~0.1-0.3

---

## 🐛 Troubleshooting

### Issue: "Mean coverage: 0.1%" (Too low)
**Cause**: LiDAR accumulation not working
**Fix**:
```python
# Check if lidar files exist
ls /home/zerun/data/dataset/KITTI-360/data_3d_raw/*/velodyne_points/data/ | head
# Increase accumulation
accumulation_frames: 6  # Try ±3 frames
```

### Issue: "Zero-weight pixels: 99%"
**Cause**: Semantic masks not loading
**Fix**:
```bash
# Check semantic files exist
ls /home/zerun/data/dataset/KITTI-360/data_2d_semantics/train/*/image_00/semantic/ | head
# Disable semantic weighting temporarily
semantic_weight_enabled: False
```

### Issue: OOM (Out of Memory)
**Fix**:
```yaml
max_img_per_gpu: 2  # Reduce from 4
accum_steps: 2      # Add gradient accumulation
```

### Issue: Loss not decreasing
**Possible causes**:
1. Learning rate too high → Reduce to `5e-5`
2. Too sparse depth → Check coverage with test script
3. Wrong checkpoint → Verify `resume_checkpoint_path`

---

## 📝 Comparison with ETH3D

| Metric | ETH3D | KITTI-360 (This Implementation) |
|--------|-------|----------------------------------|
| Depth source | COLMAP sparse | LiDAR accumulated |
| Coverage | 0.1% | 40-70% |
| Points/frame | 293 | 50,000-130,000 |
| Training frames | ~100 | 49,004 |
| Building focus | ❌ None | ✅ Adaptive + weighted |
| **Total supervision** | Baseline | **~50,000× more** |

---

## 📚 References

- **VGGT Paper**: [Visual Geometry Grounded Transformer](https://arxiv.org/abs/2501.xxxxx)
- **KITTI-360 Paper**: [KITTI-360: A Novel Dataset and Benchmarks for Urban Scene Understanding in 2D and 3D](https://arxiv.org/abs/2109.13410)
- **KITTI-360 Dataset**: https://www.cvlibs.net/datasets/kitti-360/
- **LoRA Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

---

## 🤝 Contributing

If you encounter issues or have improvements:
1. Check the test script output for diagnostics
2. Review TensorBoard logs
3. Adjust configuration parameters as described above
4. Document your findings

---

## 📄 License

This code follows the same license as the original VGGT repository.

**Note on checkpoints**:
- Original checkpoint (`facebook/VGGT-1B`): Non-commercial use only
- Commercial checkpoint (`facebook/VGGT-1B-Commercial`): Allows commercial use (excluding military)

---

## ✅ Next Steps

1. ✅ Verify dataset with `python test_kitti360_loading.py`
2. ✅ Check test output shows 30-70% coverage
3. ✅ Start training with Strategy A
4. ⏳ Monitor training for 10-20 epochs
5. ⏳ If successful, proceed to Strategy B
6. ⏳ Evaluate on validation set
7. ⏳ Fine-tune hyperparameters if needed

Good luck with your LoRA fine-tuning! 🚀
