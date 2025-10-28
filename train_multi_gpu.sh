#!/bin/bash
# Multi-GPU training launcher for VGGT LoRA fine-tuning
# Usage: bash train_multi_gpu.sh [config_name] [num_gpus]
# Example: bash train_multi_gpu.sh lora_kitti360_strategy_b_2gpu 2

CONFIG=${1:-lora_kitti360_strategy_b_2gpu}
NUM_GPUS=${2:-2}

echo "=========================================="
echo "VGGT LoRA Fine-tuning - Multi-GPU"
echo "=========================================="
echo "Config: $CONFIG"
echo "Number of GPUs: $NUM_GPUS"
echo "=========================================="

# Set CUDA devices (use first NUM_GPUS GPUs, or respect CUDA_VISIBLE_DEVICES if set)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
fi

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Run training with torchrun for DDP
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=${MASTER_PORT:-29500} \
    training/launch.py --config $CONFIG
