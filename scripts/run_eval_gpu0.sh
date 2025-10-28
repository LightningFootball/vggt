#!/bin/bash
# Evaluation wrapper script that sets up GPU environment correctly
# This mimics the environment used in train_single_gpu.sh

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Set distributed training environment (even for single GPU)
export RANK="${RANK:-0}"
export LOCAL_RANK="${LOCAL_RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# Set library paths if needed
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

echo "=========================================="
echo "VGGT Evaluation - GPU Environment"
echo "=========================================="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Python: /home/zerun/miniconda3/envs/vggt-train/bin/python"
echo "=========================================="

# Run evaluation script
/home/zerun/miniconda3/envs/vggt-train/bin/python scripts/evaluate_kitti360_buildings.py \
    --data-root /home/zerun/data/dataset/KITTI-360 \
    --log-dir logs/lora_kitti360_strategy_b_r16 \
    --device cuda:0 \
    "$@"
