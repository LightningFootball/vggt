# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VGGT (Visual Geometry Grounded Transformer) is a CVPR 2025 Best Paper that performs feed-forward 3D scene reconstruction from one to hundreds of images. The model predicts camera poses, depth maps, point maps, and point tracks within seconds without requiring traditional SfM pipelines.

## Core Architecture

The codebase is organized around a transformer-based architecture:

- **`vggt/models/vggt.py`**: Main VGGT model class that orchestrates all components
- **`vggt/models/aggregator.py`**: Core transformer with alternating attention mechanism over input frames
- **`vggt/heads/`**: Task-specific prediction heads (camera, depth, point, track)
- **`vggt/layers/`**: Vision transformer components (attention, blocks, embeddings, RoPE)
- **`vggt/utils/`**: Geometry utilities, pose encoding, rotation handling
- **`vggt/dependency/`**: Track prediction modules and geometric projections

## Development Commands

### Basic Setup
```bash
# Install core dependencies
pip install -r requirements.txt

# Install demo dependencies (for visualization)
pip install -r requirements_demo.txt

# Install as editable package
pip install -e .
```

### Running Demos
```bash
# Gradio web interface demo
python demo_gradio.py

# Viser 3D viewer
python demo_viser.py --image_folder path/to/images

# Export to COLMAP format
python demo_colmap.py --scene_dir=/path/to/scene
python demo_colmap.py --scene_dir=/path/to/scene --use_ba  # with bundle adjustment
```

### Training
```bash
# Fine-tune on Co3D dataset (requires dataset setup)
cd training
torchrun --nproc_per_node=4 launch.py
```

## Key Usage Patterns

### Basic Model Usage
```python
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
images = load_and_preprocess_images(image_paths).to(device)
with torch.no_grad():
    predictions = model(images)
```

### Individual Component Access
```python
# Access specific prediction heads
aggregated_tokens_list, ps_idx = model.aggregator(images)
pose_enc = model.camera_head(aggregated_tokens_list)[-1]
depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
```

## Important Conventions

- **Coordinate System**: Follows OpenCV camera-from-world convention
- **Image Format**: Input images should be in range [0, 1], shape [S, 3, H, W] or [B, S, 3, H, W]
- **Masking**: Unwanted pixels can be masked by setting values to 0 or 1
- **Memory**: Use `model.train()` to enable gradient checkpointing for reduced memory usage

## Configuration Files

- **`training/config/default.yaml`**: Main training configuration
- **`training/config/default_dataset.yaml`**: Dataset-specific settings
- **`pyproject.toml`**: Package configuration with demo dependencies
- **`requirements.txt`**: Core PyTorch dependencies
- **`requirements_demo.txt`**: Visualization and demo dependencies

## Model Checkpoints

- `facebook/VGGT-1B`: Main research checkpoint (non-commercial)
- `facebook/VGGT-1B-Commercial`: Commercial-use checkpoint (requires approval)

## Training Data Structure

Training expects data in this format:
```
SCENE_DIR/
├── images/           # Input images only
└── sparse/          # COLMAP format outputs
    ├── cameras.bin
    ├images.bin
    └── points3D.bin
```

## GPU Memory Considerations

Runtime scales with number of input frames:
- 1 frame: ~1.9GB, 0.04s (H100)
- 100 frames: ~21GB, 3.1s (H100)
- 200 frames: ~41GB, 8.8s (H100)

Adjust `max_img_per_gpu` and `accum_steps` in training config for memory optimization.