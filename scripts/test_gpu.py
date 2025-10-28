#!/usr/bin/env python3
"""Quick GPU availability test"""

import torch

print("=" * 60)
print("GPU Availability Test")
print("=" * 60)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")

    # Test GPU 0 specifically
    print("\n" + "=" * 60)
    print("Testing GPU 0 (RTX 5090)")
    print("=" * 60)

    torch.cuda.set_device(0)
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

    # Create a tensor on GPU 0
    x = torch.randn(1000, 1000, device='cuda:0')
    y = torch.randn(1000, 1000, device='cuda:0')
    z = x @ y

    print(f"\nTest tensor created on: {z.device}")
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

    print("\n✓ GPU 0 is working correctly!")
else:
    print("\n✗ CUDA is not available. Please check your PyTorch installation.")

print("\n" + "=" * 60)
