#!/bin/bash
# Training script for LoRA fine-tuning on ETH3D dataset
#
# Usage:
#   bash scripts/train_lora_eth3d.sh strategy_a   # Strategy A: Depth Head only
#   bash scripts/train_lora_eth3d.sh strategy_b   # Strategy B: Aggregator late + Depth Head (Recommended)
#   bash scripts/train_lora_eth3d.sh strategy_c   # Strategy C: Full LoRA

set -e  # Exit on error

# Default strategy
STRATEGY=${1:-strategy_a}

# Validate strategy
if [[ ! "$STRATEGY" =~ ^strategy_[abc]$ ]]; then
    echo "Error: Invalid strategy '$STRATEGY'"
    echo "Usage: $0 [strategy_a|strategy_b|strategy_c]"
    exit 1
fi

CONFIG_NAME="lora_eth3d_${STRATEGY}"

echo "========================================="
echo "LoRA Fine-tuning on ETH3D"
echo "========================================="
echo "Strategy: $STRATEGY"
echo "Config:   $CONFIG_NAME"
echo "========================================="
echo ""

# Check if PEFT is installed
python3 -c "import peft" 2>/dev/null || {
    echo "Error: PEFT library not found!"
    echo "Installing PEFT..."
    pip install peft
}

# Check if pretrained checkpoint path is set
CHECKPOINT_PATH=$(grep "resume_checkpoint_path:" training/config/${CONFIG_NAME}.yaml | awk '{print $2}')
if [[ "$CHECKPOINT_PATH" == "/YOUR/PATH/TO/PRETRAINED/VGGT/CHECKPOINT" ]]; then
    echo "========================================="
    echo "WARNING: Please update resume_checkpoint_path in"
    echo "  training/config/${CONFIG_NAME}.yaml"
    echo ""
    echo "You need to download the pretrained VGGT checkpoint from:"
    echo "  https://huggingface.co/facebook/VGGT-1B"
    echo "  or"
    echo "  https://huggingface.co/facebook/VGGT-1B-Commercial"
    echo "========================================="
    echo ""
    read -p "Press Enter to continue anyway (will fail), or Ctrl+C to abort..."
fi

# Print GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Training command
echo "Starting training..."
echo "Command: torchrun --nproc_per_node=1 training/launch.py --config $CONFIG_NAME"
echo ""

# Use torchrun even for single GPU to set up distributed environment variables
torchrun --nproc_per_node=1 training/launch.py --config $CONFIG_NAME "${@:2}"

echo ""
echo "========================================="
echo "Training completed!"
echo "========================================="
echo "Logs and checkpoints saved to: logs/${CONFIG_NAME}/"
echo ""
echo "To monitor training, run:"
echo "  tensorboard --logdir logs/${CONFIG_NAME}/tensorboard"
echo ""
echo "To evaluate the trained model, run:"
echo "  python scripts/evaluate_eth3d.py --checkpoint logs/${CONFIG_NAME}/ckpts/checkpoint.pth --config ${CONFIG_NAME}"
