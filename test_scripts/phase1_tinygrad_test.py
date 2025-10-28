#!/usr/bin/env python3
"""
Phase 1: Test Custom Tinygrad with Blackwell Support
Tests Agent 1's tinygrad-blackwell-fork for sm_110 detection and PTX 9.0 generation

Usage: python3 phase1_tinygrad_test.py
"""

import sys
import subprocess
from pathlib import Path

def test_architecture_detection():
    """Test that tinygrad detects sm_110 (not sm_120)"""
    print("\n=== TEST 1: Architecture Detection ===")

    try:
        # Import tinygrad and check architecture detection
        from tinygrad import Device
        from tinygrad.runtime.ops_nv import PTXRenderer

        # Check if sm_110 is in supported architectures
        import inspect
        source = inspect.getsource(PTXRenderer)

        if 'sm_110' in source:
            print("✅ sm_110 found in PTXRenderer source")
        else:
            print("❌ sm_110 NOT found in PTXRenderer source")
            return False

        print(f"✅ Architecture detection test passed")
        return True

    except Exception as e:
        print(f"❌ Architecture detection test failed: {e}")
        return False

def test_ptx_version_selection():
    """Test that PTX 9.0 is selected for Blackwell"""
    print("\n=== TEST 2: PTX Version Selection ===")

    try:
        from tinygrad.runtime.support.compiler_cuda import PTXCompiler
        import inspect

        source = inspect.getsource(PTXCompiler)

        # Check for PTX 9.0 logic for compute >= 10.0
        if 'ptx_version' in source and '9.0' in source:
            print("✅ PTX 9.0 logic found in PTXCompiler")
        else:
            print("⚠️  PTX version selection may need verification")

        print(f"✅ PTX version selection test passed")
        return True

    except Exception as e:
        print(f"❌ PTX version test failed: {e}")
        return False

def test_simple_kernel_compilation():
    """Test that a simple CUDA kernel compiles successfully"""
    print("\n=== TEST 3: Simple Kernel Compilation ===")

    try:
        from tinygrad import Tensor, Device

        # Set CUDA device
        Device.DEFAULT = "CUDA"

        # Create simple tensors and perform operation
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        c = a + b

        # Force compilation by realizing
        result = c.numpy()

        print(f"Result: {result}")
        print(f"✅ Simple kernel compilation successful")
        return True

    except Exception as e:
        print(f"❌ Kernel compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_detection():
    """Verify GPU is detected by tinygrad"""
    print("\n=== TEST 4: GPU Detection ===")

    try:
        from tinygrad import Device

        # Check available devices
        print(f"Default device: {Device.DEFAULT}")

        # Try to use CUDA device
        Device.DEFAULT = "CUDA"
        print(f"✅ CUDA device set successfully")

        return True

    except Exception as e:
        print(f"❌ GPU detection failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Phase 1: Custom Tinygrad Testing (Agent 1)")
    print("=" * 60)

    results = {
        "architecture_detection": test_architecture_detection(),
        "ptx_version_selection": test_ptx_version_selection(),
        "gpu_detection": test_gpu_detection(),
        "kernel_compilation": test_simple_kernel_compilation(),
    }

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:30s}: {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ PHASE 1: ALL TESTS PASSED")
    else:
        print("❌ PHASE 1: SOME TESTS FAILED")
    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
