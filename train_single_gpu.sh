#!/bin/bash
# Single GPU training launcher for VGGT LoRA fine-tuning
# Usage: bash train_single_gpu.sh [config_name]
# Example: bash train_single_gpu.sh lora_kitti360_strategy_a

CONFIG=${1:-lora_kitti360_strategy_a}

echo "=========================================="
echo "VGGT LoRA Fine-tuning - Single GPU"
echo "=========================================="
echo "Config: $CONFIG"
echo "=========================================="

# Set environment variables for single GPU (provide sane defaults if unset)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RANK="${RANK:-0}"
export LOCAL_RANK="${LOCAL_RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# Enable Flash Attention by default for better performance
export VGGT_ENABLE_FLASH_ATTENTION="${VGGT_ENABLE_FLASH_ATTENTION:-1}"

# Run training with vggt-train environment Python
/home/zerun/miniconda3/envs/vggt-train/bin/python training/launch.py --config $CONFIG
