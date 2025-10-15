#!/bin/bash
# Minimal debugging - only enable DSA without blocking mode
# Usage: bash train_single_gpu_minimal_debug.sh [config_name]

CONFIG=${1:-lora_kitti360_strategy_a}

echo "=========================================="
echo "VGGT LoRA Fine-tuning - Minimal Debug Mode"
echo "=========================================="
echo "Config: $CONFIG"
echo "=========================================="

# Set environment variables for single GPU
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Only enable Device-Side Assertions (no blocking)
echo "Enabling minimal CUDA debugging..."
export TORCH_USE_CUDA_DSA=1            # Catch memory errors without blocking

# Memory debugging (helps with OOM and fragmentation)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

echo "Debug flags enabled:"
echo "  TORCH_USE_CUDA_DSA=1 (Device-side assertions)"
echo "  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "  (CUDA_LAUNCH_BLOCKING disabled to avoid deadlock)"
echo "=========================================="

# Run training
/home/zerun/miniconda3/envs/vggt-train/bin/python training/launch.py --config $CONFIG
