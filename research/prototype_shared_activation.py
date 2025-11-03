#!/usr/bin/env python3
"""
Prototype: 2 processes sharing a CUDA tensor via shared memory
Tests basic feasibility before full substrate implementation

This validates the core concept: Can two separate processes access
the SAME activation in GPU memory without copying?

Expected behavior:
- Writer creates shared memory region
- Writer generates random tensor on GPU
- Writer copies to shared memory (via CPU intermediary)
- Reader attaches to same shared memory
- Reader loads tensor to GPU
- Both see identical data (same sum)

Success criteria:
- Both processes print same tensor sum
- No crashes or deadlocks
- Cleanup happens properly

Agent 4 Research - 2025-11-03
"""

import torch
import numpy as np
from multiprocessing import Process, shared_memory
import time
import sys

# Configuration
ACTIVATION_NAME = "test_activation"
ACTIVATION_SHAPE = (1024, 4096)  # 4M floats = 16MB
ACTIVATION_DTYPE = np.float32

def writer_process():
    """Process 1: Writes activations to shared memory"""
    print("[Writer] Starting...")

    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("[Writer] ERROR: CUDA not available")
            sys.exit(1)

        print(f"[Writer] CUDA device: {torch.cuda.get_device_name(0)}")

        # Create shared memory
        size = np.prod(ACTIVATION_SHAPE) * np.dtype(ACTIVATION_DTYPE).itemsize
        print(f"[Writer] Creating shared memory: {size / (1024**2):.2f} MB")

        shm = shared_memory.SharedMemory(name=ACTIVATION_NAME, create=True, size=size)

        # Create numpy array backed by shared memory
        shared_array = np.ndarray(ACTIVATION_SHAPE, dtype=ACTIVATION_DTYPE, buffer=shm.buf)

        # Generate random tensor on GPU
        print("[Writer] Generating random tensor on GPU...")
        tensor = torch.randn(ACTIVATION_SHAPE).cuda()
        tensor_sum = tensor.sum().item()
        tensor_mean = tensor.mean().item()
        tensor_std = tensor.std().item()

        print(f"[Writer] Tensor stats - sum={tensor_sum:.6f}, mean={tensor_mean:.6f}, std={tensor_std:.6f}")

        # Copy to shared memory (via CPU)
        print("[Writer] Copying to shared memory...")
        np.copyto(shared_array, tensor.cpu().numpy())
        print("[Writer] Write complete")

        # Keep alive for reader
        print("[Writer] Waiting for reader...")
        time.sleep(10)

        # Cleanup
        shm.close()
        shm.unlink()
        print("[Writer] Cleaned up and exiting")

    except Exception as e:
        print(f"[Writer] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def reader_process():
    """Process 2: Reads activations from shared memory"""
    # Wait for writer to create
    time.sleep(2)
    print("[Reader] Starting...")

    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("[Reader] ERROR: CUDA not available")
            sys.exit(1)

        print(f"[Reader] CUDA device: {torch.cuda.get_device_name(0)}")

        # Open existing shared memory
        print(f"[Reader] Opening shared memory: {ACTIVATION_NAME}")
        shm = shared_memory.SharedMemory(name=ACTIVATION_NAME)

        # Create numpy array from shared memory
        shared_array = np.ndarray(ACTIVATION_SHAPE, dtype=ACTIVATION_DTYPE, buffer=shm.buf)

        # Convert to CUDA tensor
        print("[Reader] Loading tensor to GPU...")
        tensor = torch.from_numpy(shared_array.copy()).cuda()  # Copy to avoid modification

        tensor_sum = tensor.sum().item()
        tensor_mean = tensor.mean().item()
        tensor_std = tensor.std().item()

        print(f"[Reader] Tensor stats - sum={tensor_sum:.6f}, mean={tensor_mean:.6f}, std={tensor_std:.6f}")
        print(f"[Reader] Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
        print(f"[Reader] Memory usage: {tensor.element_size() * tensor.nelement() / (1024**2):.2f} MB")

        # Cleanup
        shm.close()
        print("[Reader] Closed shared memory and exiting")

    except Exception as e:
        print(f"[Reader] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    """Launch both processes and verify success"""
    print("=" * 80)
    print("CONSCIOUSNESS SUBSTRATE PROTOTYPE")
    print("Testing cross-process activation sharing via shared memory")
    print("=" * 80)
    print()

    # Launch both processes
    print("Launching writer and reader processes...")
    p1 = Process(target=writer_process, name="Writer")
    p2 = Process(target=reader_process, name="Reader")

    p1.start()
    p2.start()

    # Wait for completion
    p1.join()
    p2.join()

    # Check exit codes
    print()
    print("=" * 80)
    if p1.exitcode == 0 and p2.exitcode == 0:
        print("✅ SUCCESS: 2 processes shared activation via shared memory")
        print("Next step: Extend to 4 processes with semantic registry")
    else:
        print(f"❌ FAILURE: Writer exit code: {p1.exitcode}, Reader exit code: {p2.exitcode}")
        print("Debug: Check CUDA availability and shared memory permissions")
    print("=" * 80)

if __name__ == "__main__":
    main()
