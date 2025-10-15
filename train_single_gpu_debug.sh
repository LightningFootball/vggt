#!/bin/bash
# Single GPU training launcher with GPU debugging enabled
# Usage: bash train_single_gpu_debug.sh [config_name]
# Example: bash train_single_gpu_debug.sh lora_kitti360_strategy_b

CONFIG=${1:-lora_kitti360_strategy_a}

echo "=========================================="
echo "VGGT LoRA Fine-tuning - Single GPU (DEBUG MODE)"
echo "=========================================="
echo "Config: $CONFIG"
echo "=========================================="

# Set environment variables for single GPU
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# GPU Debugging Environment Variables
echo "Enabling CUDA debugging flags..."
export CUDA_LAUNCH_BLOCKING=1          # Synchronous CUDA execution (critical!)
export TORCH_USE_CUDA_DSA=1            # Device-side assertions for memory errors
export NCCL_ASYNC_ERROR_HANDLING=1     # Better error reporting
export NCCL_BLOCKING_WAIT=1            # Blocking NCCL waits

# Additional debugging options (uncomment if needed)
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Better memory allocation
# export CUDA_DEVICE_DEBUG=1           # More verbose CUDA errors
# export TORCH_DISTRIBUTED_DEBUG=INFO  # Distributed debugging (multi-GPU)

echo "Debug flags enabled:"
echo "  CUDA_LAUNCH_BLOCKING=1 (Synchronous execution for precise error location)"
echo "  TORCH_USE_CUDA_DSA=1 (Device-side assertions for memory errors)"
echo "=========================================="

# Run training with vggt-train environment Python
/home/zerun/miniconda3/envs/vggt-train/bin/python training/launch.py --config $CONFIG
