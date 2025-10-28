#!/usr/bin/env python3
"""
Quick script to test memory usage of the optimized evaluation script.
This script monitors system memory usage during evaluation.
"""

import argparse
import os
import subprocess
import sys
import time
import psutil
from pathlib import Path


def get_memory_usage():
    """Get current process memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def monitor_command(cmd: list, max_memory_gb: float = 50.0):
    """
    Run a command and monitor its memory usage.

    Args:
        cmd: Command to run as a list of strings
        max_memory_gb: Maximum allowed memory usage in GB

    Returns:
        tuple: (success: bool, peak_memory_gb: float)
    """
    print(f"Starting command: {' '.join(cmd)}")
    print(f"Maximum allowed memory: {max_memory_gb:.2f} GB")

    # Start the process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Monitor memory usage
    peak_memory = 0.0
    ps_process = psutil.Process(process.pid)

    try:
        while True:
            # Check if process is still running
            if process.poll() is not None:
                break

            # Get memory usage including child processes
            try:
                memory_usage = ps_process.memory_info().rss / (1024 ** 3)

                # Include child processes
                children = ps_process.children(recursive=True)
                for child in children:
                    try:
                        memory_usage += child.memory_info().rss / (1024 ** 3)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                peak_memory = max(peak_memory, memory_usage)

                # Print memory status every 10 seconds
                if int(time.time()) % 10 == 0:
                    print(f"Current memory usage: {memory_usage:.2f} GB (peak: {peak_memory:.2f} GB)")

                # Check if memory limit exceeded
                if memory_usage > max_memory_gb:
                    print(f"ERROR: Memory usage ({memory_usage:.2f} GB) exceeded limit ({max_memory_gb:.2f} GB)!")
                    process.terminate()
                    process.wait(timeout=10)
                    return False, peak_memory

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

            time.sleep(1)

        # Wait for process to complete
        return_code = process.wait()

        print(f"\nProcess completed with return code: {return_code}")
        print(f"Peak memory usage: {peak_memory:.2f} GB")

        if return_code == 0:
            print(f"SUCCESS: Evaluation completed within memory limit!")
            return True, peak_memory
        else:
            print(f"FAILED: Process exited with error code {return_code}")
            return False, peak_memory

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        process.terminate()
        process.wait(timeout=10)
        return False, peak_memory


def main():
    parser = argparse.ArgumentParser(description="Test memory usage of evaluation script")
    parser.add_argument("--data-root", type=str, required=True, help="Path to KITTI-360 root")
    parser.add_argument("--log-dir", type=str, required=True, help="Path to training log dir")
    parser.add_argument("--max-memory", type=float, default=50.0, help="Maximum memory in GB (default: 50)")
    parser.add_argument("--max-seqs", type=int, default=5, help="Number of sequences to test (default: 5)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    args = parser.parse_args()

    # Build the evaluation command
    script_path = Path(__file__).parent / "evaluate_kitti360_buildings.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--data-root", args.data_root,
        "--log-dir", args.log_dir,
        "--device", args.device,
        "--max-seqs", str(args.max_seqs),
        "--verbose",
    ]

    print("=" * 80)
    print("Memory Usage Test for KITTI-360 Evaluation")
    print("=" * 80)
    print(f"Data root: {args.data_root}")
    print(f"Log dir: {args.log_dir}")
    print(f"Max sequences: {args.max_seqs}")
    print(f"Memory limit: {args.max_memory:.2f} GB")
    print(f"Device: {args.device}")
    print("=" * 80)
    print()

    success, peak_memory = monitor_command(cmd, args.max_memory)

    print()
    print("=" * 80)
    if success:
        print(f"✓ TEST PASSED")
        print(f"  Peak memory usage: {peak_memory:.2f} GB (limit: {args.max_memory:.2f} GB)")
        print(f"  Memory overhead: {(peak_memory / args.max_memory * 100):.1f}%")
    else:
        print(f"✗ TEST FAILED")
        print(f"  Peak memory usage: {peak_memory:.2f} GB (limit: {args.max_memory:.2f} GB)")
    print("=" * 80)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
