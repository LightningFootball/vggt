#!/bin/bash
# Test CUDA environment with all NVIDIA libraries in path

NVIDIA_LIB_BASE="/home/zerun/miniconda3/envs/vggt-train/lib/python3.12/site-packages/nvidia"

# Build library path from all NVIDIA lib directories
LIB_PATHS=""
for dir in cuda_runtime cublas cufft curand cusolver cusparse cudnn nccl nvtx cuda_cupti cuda_nvrtc; do
    if [ -d "$NVIDIA_LIB_BASE/$dir/lib" ]; then
        LIB_PATHS="$NVIDIA_LIB_BASE/$dir/lib:$LIB_PATHS"
    fi
done

export LD_LIBRARY_PATH="${LIB_PATHS}:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=========================================="
echo "Testing CUDA Environment"
echo "=========================================="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:0:200}..."
echo "=========================================="

/home/zerun/miniconda3/envs/vggt-train/bin/python -c "
import torch
print('PyTorch CUDA Test:')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('  ✗ No GPUs detected')
"
