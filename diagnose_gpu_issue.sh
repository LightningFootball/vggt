#!/bin/bash
# GPU Issue Diagnostic Script
# This script runs progressive tests to identify the source of GPU crashes

echo "========================================"
echo "GPU Issue Diagnostic Tool"
echo "========================================"

CONFIG=${1:-lora_kitti360_strategy_b}

echo "Testing configuration: $CONFIG"
echo ""

# Test 1: Check GPU health
echo "[Test 1] Checking GPU status..."
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv
echo ""

# Test 2: Check for ECC errors
echo "[Test 2] Checking for GPU ECC errors..."
nvidia-smi --query-gpu=ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total --format=csv
echo ""

# Test 3: Check CUDA environment
echo "[Test 3] Checking CUDA environment..."
/home/zerun/miniconda3/envs/vggt-train/bin/python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'cuDNN version: {torch.backends.cudnn.version()}')
print(f'Number of GPUs: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"
echo ""

# Test 4: Quick memory allocation test
echo "[Test 4] Testing GPU memory allocation..."
/home/zerun/miniconda3/envs/vggt-train/bin/python -c "
import torch
try:
    # Try allocating 20GB
    x = torch.randn(5000, 5000, 20, device='cuda', dtype=torch.bfloat16)
    print(f'Successfully allocated {x.numel() * 2 / 1e9:.2f} GB')
    print(f'Current GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
    del x
    torch.cuda.empty_cache()
    print('Memory test passed!')
except Exception as e:
    print(f'Memory allocation failed: {e}')
"
echo ""

# Test 5: Check for kernel log errors
echo "[Test 5] Checking kernel logs for GPU errors (last 20 lines)..."
echo "(This may require sudo privileges)"
sudo dmesg | grep -i 'gpu\|nvidia\|cuda' | tail -20 || echo "No sudo access or no GPU errors found"
echo ""

# Test 6: Test DDP initialization only (no training)
echo "[Test 6] Testing DDP initialization without training..."
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=29501

/home/zerun/miniconda3/envs/vggt-train/bin/python -c "
import torch
import torch.distributed as dist
from datetime import timedelta

try:
    dist.init_process_group(
        backend='nccl',
        timeout=timedelta(minutes=5)
    )
    print('DDP initialization successful')

    # Test basic CUDA operations
    x = torch.randn(100, 100, device='cuda')
    y = x @ x.T
    print(f'Basic CUDA operations successful')

    dist.destroy_process_group()
    print('DDP cleanup successful')
except Exception as e:
    print(f'DDP test failed: {e}')
    import traceback
    traceback.print_exc()
"
echo ""

echo "========================================"
echo "Diagnostic complete!"
echo ""
echo "Next steps:"
echo "1. If Test 1-3 fail: GPU driver or CUDA installation issue"
echo "2. If Test 4 fails: Potential GPU hardware issue"
echo "3. If Test 5 shows errors: Check kernel logs with 'sudo dmesg | grep -i nvidia'"
echo "4. If Test 6 fails: DDP/NCCL configuration issue"
echo ""
echo "To run minimal debug training, use:"
echo "  ./train_single_gpu_minimal_debug.sh $CONFIG"
echo "========================================"
