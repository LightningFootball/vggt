#!/usr/bin/env python3
import torch
import os

print("="*60)
print("CUDA Environment Test")
print("="*60)
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Test actual GPU computation
    print("\nTesting GPU computation...")
    x = torch.randn(1000, 1000, device='cuda:0')
    y = x @ x
    print(f"✓ GPU computation successful on {y.device}")
else:
    print("\n✗ CUDA not available - cannot use GPU")
    import sys
    sys.exit(1)
