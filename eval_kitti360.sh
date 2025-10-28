#!/bin/bash
# Evaluation launcher for KITTI-360 - mimics train_single_gpu.sh environment
# Usage: bash eval_kitti360.sh

CONFIG="lora_kitti360_strategy_b"
LOG_DIR="logs/lora_kitti360_strategy_b_r16"
DATA_ROOT="/home/zerun/data/dataset/KITTI-360"

echo "=========================================="
echo "VGGT Evaluation - KITTI-360 Buildings"
echo "=========================================="
echo "Config: $CONFIG"
echo "Log dir: $LOG_DIR"
echo "Data root: $DATA_ROOT"
echo "=========================================="

# Set environment variables for single GPU (same as training)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RANK="${RANK:-0}"
export LOCAL_RANK="${LOCAL_RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-29500}"

echo "Environment:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  LOCAL_RANK: $LOCAL_RANK"
echo "=========================================="

# Run evaluation with vggt-train environment Python (same as training)
/home/zerun/miniconda3/envs/vggt-train/bin/python scripts/evaluate_kitti360_buildings.py \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    --log-dir "$LOG_DIR" \
    --device cuda:0 \
    "$@"
