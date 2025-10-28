#!/usr/bin/env python3
"""Test if we can force use CUDA even when torch.cuda.is_available() is False"""

import torch
import os

print("="*60)
print("Force CUDA Test (like trainer.py does)")
print("="*60)
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
print(f"PyTorch version: {torch.  __version__}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

print("\n" + "="*60)
print("Attempting to create CUDA device anyway...")
print("="*60)

try:
    # This is what trainer.py does
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    print(f"✓ Created device: {device}")

    print("\nAttempting GPU computation...")
    x = torch.randn(100, 100, device=device)
    y = x @ x
    print(f"✓ Successfully computed on device: {y.device}")
    print(f"✓ Result shape: {y.shape}")
    print(f"✓ Result mean: {y.mean().item():.4f}")

    print("\n" + "="*60)
    print("SUCCESS: GPU is actually working!")
    print("="*60)

except Exception as e:
    print(f"\n✗ Failed: {e}")
    print("\nThis means GPU is truly not accessible")
    import traceback
    traceback.print_exc()
