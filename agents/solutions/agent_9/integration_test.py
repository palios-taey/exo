#!/usr/bin/env python3
"""
Integration Tests for NVPTXCompilerV2 with Tinygrad

Tests compiler integration with actual tinygrad Tensor operations.

Run: python3 integration_test.py
"""

import sys
import os
from pathlib import Path

# Add solution directory
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("NVPTXCompilerV2 Integration Tests with Tinygrad")
print("=" * 70)
print()

# Patch compiler BEFORE importing tinygrad
print("[1/5] Patching NVPTXCompiler...")
try:
    from nvptx_compiler_v2 import NVPTXCompilerV2
    import tinygrad.runtime.support.compiler_cuda as cuda_compiler
    cuda_compiler.NVPTXCompiler = NVPTXCompilerV2
    print("✓ NVPTXCompilerV2 patched successfully")
except Exception as e:
    print(f"✗ Failed to patch: {e}")
    sys.exit(1)

print()
print("[2/5] Importing tinygrad...")
try:
    from tinygrad import Tensor, Device
    from tinygrad.helpers import DEBUG
    print(f"✓ Tinygrad imported (Device.DEFAULT={Device.DEFAULT})")
except ImportError as e:
    print(f"✗ Failed to import tinygrad: {e}")
    sys.exit(1)

# Set device to CUDA
print()
print("[3/5] Setting device to CUDA...")
try:
    Device.DEFAULT = "CUDA:0"
    print(f"✓ Device set to {Device.DEFAULT}")
except Exception as e:
    print(f"✗ Failed to set device: {e}")
    sys.exit(1)

# Test 1: Simple tensor operations
print()
print("[4/5] Test 1: Simple tensor addition...")
try:
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    c = (a + b).realize()  # Forces kernel compilation

    result = c.numpy().tolist()
    expected = [5.0, 7.0, 9.0]

    if result == expected:
        print(f"✓ Tensor addition: {result} == {expected}")
    else:
        print(f"✗ Tensor addition failed: {result} != {expected}")
        sys.exit(1)
except Exception as e:
    print(f"✗ Tensor addition failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Matrix multiplication
print()
print("[5/5] Test 2: Matrix multiplication...")
try:
    A = Tensor([[1.0, 2.0], [3.0, 4.0]])
    B = Tensor([[5.0, 6.0], [7.0, 8.0]])
    C = (A @ B).realize()

    result = C.numpy().tolist()
    # Expected: [[19, 22], [43, 50]]
    expected = [[19.0, 22.0], [43.0, 50.0]]

    if result == expected:
        print(f"✓ Matrix multiplication correct")
        print(f"  Result: {result}")
    else:
        print(f"✗ Matrix multiplication failed")
        print(f"  Got: {result}")
        print(f"  Expected: {expected}")
        sys.exit(1)
except Exception as e:
    print(f"✗ Matrix multiplication failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("✓ ALL INTEGRATION TESTS PASSED")
print("=" * 70)
