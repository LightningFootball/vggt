#!/bin/bash
# Quick script to preprocess KITTI-360 test set

echo "Preprocessing KITTI-360 test set..."
echo "This will generate depth maps for test sequences (0008, 0018)"
echo ""

python scripts/preprocess_kitti360_lora.py \
    --config training/config/lora_kitti360_strategy_b.yaml \
    --stages depths \
    --split test \
    --workers 8 \
    --skip-visualization

echo ""
echo "Test set preprocessing complete!"
echo ""
echo "You can now run evaluation with:"
echo "  python scripts/evaluate_kitti360_buildings.py \\"
echo "      --data-root /home/zerun/data/dataset/KITTI-360 \\"
echo "      --log-dir logs/lora_kitti360_strategy_b_r16 \\"
echo "      --device cuda:0 \\"
echo "      --split test \\"
echo "      --verbose"
