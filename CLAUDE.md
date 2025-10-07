# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VGGT (Visual Geometry Grounded Transformer) is a feed-forward neural network that predicts 3D scene attributes (camera parameters, depth maps, point maps, and 3D point tracks) from one or multiple image views. The model consists of an aggregator module and multiple prediction heads (camera, depth, point, track).

## Development Environment Setup

```bash
# Basic installation
pip install -r requirements.txt

# Demo dependencies (for visualization)
pip install -r requirements_demo.txt

# Training dependencies
pip install -r requirements_training.txt

# Install as package (required for training)
pip install -e .
```

## Common Commands

### Running Demos

```bash
# Gradio web interface
python demo_gradio.py

# Viser 3D viewer
python demo_viser.py --image_folder path/to/your/images/folder

# Export to COLMAP format (feedforward only)
python demo_colmap.py --scene_dir=/YOUR/SCENE_DIR/

# With bundle adjustment
python demo_colmap.py --scene_dir=/YOUR/SCENE_DIR/ --use_ba

# Faster BA with reduced parameters
python demo_colmap.py --scene_dir=/YOUR/SCENE_DIR/ --use_ba --max_query_pts=2048 --query_frame_num=5
```

### Training

```bash
# Fine-tune on Co3D with 4 GPUs using DDP
torchrun --nproc_per_node=4 training/launch.py

# Use custom config
torchrun --nproc_per_node=4 training/launch.py --config your_config_name
```

**Important:** Before training:
1. Update paths in `training/config/default.yaml`:
   - `CO3D_DIR`: Path to Co3D dataset
   - `CO3D_ANNOTATION_DIR`: Path to annotation files
   - `resume_checkpoint_path`: Path to pre-trained checkpoint

## Architecture

### Core Components

- **`vggt/models/aggregator.py`**: Vision Transformer-based aggregator that processes multi-view images and produces aggregated tokens
- **`vggt/heads/`**: Prediction heads for different tasks:
  - `camera_head.py`: Predicts camera extrinsics and intrinsics
  - `dpt_head.py`: Dense Prediction Transformer for depth/point maps
  - `track_head.py`: Point tracking across frames
- **`vggt/layers/`**: Transformer building blocks (attention, MLP, vision transformer)
- **`vggt/utils/`**: Geometry utilities, pose encoding/decoding, image loading

### Training Framework

- **`training/data/`**: Dataset implementations and dataloaders
  - Supports Co3D, VKitti, and custom datasets
  - `dynamic_dataloader.py`: Handles variable sequence lengths
  - `composed_dataset.py`: Combines multiple datasets with configurable sampling ratios
- **`training/trainer.py`**: Main training loop with DDP support
- **`training/loss.py`**: Multi-task loss computation (camera, depth, point, track)
- **`training/train_utils/`**: Optimization, checkpointing, gradient clipping, logging

### Model Forward Pass

The model expects images in shape `[S, 3, H, W]` or `[B, S, 3, H, W]` (range [0, 1]):
1. Aggregator processes images → aggregated tokens
2. Camera head → pose encodings (extrinsics + intrinsics)
3. Depth/Point heads → dense predictions
4. Track head → point trajectories (if query_points provided)

## Training Configuration

Key parameters in `training/config/default.yaml`:
- `max_img_per_gpu`: Batch size per GPU (reduce if OOM)
- `accum_steps`: Gradient accumulation steps
- `frozen_module_names`: List of modules to freeze (e.g., `["*aggregator*"]`)
- Learning rate: Critical hyperparameter, try `[5e-6, 1e-5, 5e-5, 1e-4, 5e-4]` based on total batch size
- Loss weights: `camera.weight`, `depth.weight`, `point.weight`, `track.weight`

The default config fine-tunes with aggregator frozen, training only camera and depth heads.

## Coordinate System Conventions

- Camera poses follow **OpenCV convention** (camera-from-world)
- Extrinsic matrices: world → camera transformation
- Depth maps aligned with corresponding camera poses

## COLMAP Integration

For scenes in `/YOUR/SCENE_DIR/images/`, the model outputs:
```
SCENE_DIR/
├── images/
└── sparse/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

These files work directly with gsplat/NeRF tools:
```bash
cd gsplat
python examples/simple_trainer.py default --data_factor 1 --data_dir /YOUR/SCENE_DIR/ --result_dir /YOUR/RESULT_DIR/
```

## Data Requirements

- Images should be in `scene_dir/images/` folder containing only image files
- For training, Co3D annotations required from [Hugging Face](https://huggingface.co/datasets/JianyuanWang/co3d_anno)
- Masking unwanted pixels: set to 0 or 1 (reflections, sky, water) - simple bounding boxes work

## Performance Notes

- Use bfloat16 on Ampere+ GPUs (Compute Capability ≥8.0) for better performance
- Flash Attention 3 recommended for speed (compile from source)
- Typical reconstruction time: <1 second; visualization may take longer
- Multiple dataset training: control sampling ratio via `len_train` parameter

## License

Model checkpoint licensing:
- **Original checkpoint** (`facebook/VGGT-1B`): Non-commercial use only
- **Commercial checkpoint** (`facebook/VGGT-1B-Commercial`): Allows commercial use (excluding military applications)
