#!/bin/bash
# Quick test script to verify the evaluation runs without errors

echo "Testing evaluation script with 2 sequences..."
python scripts/evaluate_kitti360_buildings.py \
    --data-root /home/zerun/data/dataset/KITTI-360 \
    --log-dir logs/lora_kitti360_strategy_b_r16 \
    --device cuda:0 \
    --max-seqs 2 \
    --verbose

if [ $? -eq 0 ]; then
    echo "✓ Test passed! Script runs without errors."
else
    echo "✗ Test failed! Check error messages above."
fi
