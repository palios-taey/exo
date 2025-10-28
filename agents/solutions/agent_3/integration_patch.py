#!/usr/bin/env python3
"""
Integration Patch for NVCCCompiler

Monkey-patches tinygrad to use NVCCCompiler for CUDA 13.0 / Blackwell GPUs

Usage:
    Add to top of exo/inference/tinygrad/inference.py:

    import sys
    sys.path.insert(0, '/home/mira/exo/agents/solutions/agent_3')
    from integration_patch import patch_nvptx_compiler
    patch_nvptx_compiler()

Author: Solution Agent 3
Date: 2025-10-23
"""

import sys
import os


def patch_nvptx_compiler():
    """
    Monkey-patch tinygrad's NVPTXCompiler to use our NVCCCompiler

    Must be called BEFORE any tinygrad imports that initialize CUDA device

    This replaces the broken NVPTXCompiler (expects PTX, receives CUDA C)
    with our NVCCCompiler (compiles CUDA C → PTX → CUBIN via nvcc)
    """

    # Step 1: Import our NVCCCompiler
    try:
        from nvcc_compiler import NVCCCompiler
    except ImportError as e:
        print(f"[NVCC PATCH] ERROR: Could not import NVCCCompiler: {e}", file=sys.stderr)
        print("[NVCC PATCH] Make sure nvcc_compiler.py is in Python path", file=sys.stderr)
        raise

    # Step 2: Verify nvcc is available
    import shutil
    if not shutil.which('nvcc'):
        print("[NVCC PATCH] WARNING: nvcc not found in PATH", file=sys.stderr)
        print("[NVCC PATCH] Compilation will fail when kernels are compiled", file=sys.stderr)
        print("[NVCC PATCH] Add to PATH: export PATH=/usr/local/cuda/bin:$PATH", file=sys.stderr)

    # Step 3: Monkey-patch tinygrad's compiler_cuda module
    # This must happen BEFORE tinygrad.runtime.ops_cuda imports

    try:
        import tinygrad.runtime.support.compiler_cuda as compiler_cuda

        # Store original for reference (and potential rollback)
        if not hasattr(compiler_cuda, '_original_NVPTXCompiler'):
            compiler_cuda._original_NVPTXCompiler = compiler_cuda.NVPTXCompiler

        # Replace NVPTXCompiler with our NVCCCompiler
        compiler_cuda.NVPTXCompiler = NVCCCompiler

        print("[NVCC PATCH] Successfully patched NVPTXCompiler → NVCCCompiler", file=sys.stderr)

    except ImportError as e:
        print(f"[NVCC PATCH] ERROR: Could not import tinygrad.runtime.support.compiler_cuda: {e}", file=sys.stderr)
        print("[NVCC PATCH] This module must be imported after tinygrad is installed", file=sys.stderr)
        raise

    # Step 4: Verify patch applied
    try:
        from tinygrad.runtime.support.compiler_cuda import NVPTXCompiler as PatchedCompiler
        if PatchedCompiler is NVCCCompiler:
            print("[NVCC PATCH] Verification: NVPTXCompiler is now NVCCCompiler ✅", file=sys.stderr)
        else:
            print("[NVCC PATCH] WARNING: Patch may not have applied correctly", file=sys.stderr)
    except Exception as e:
        print(f"[NVCC PATCH] Could not verify patch: {e}", file=sys.stderr)


def unpatch_nvptx_compiler():
    """
    Restore original NVPTXCompiler (for testing/rollback)

    Only works if patch was applied in same Python session
    """
    try:
        import tinygrad.runtime.support.compiler_cuda as compiler_cuda

        if hasattr(compiler_cuda, '_original_NVPTXCompiler'):
            compiler_cuda.NVPTXCompiler = compiler_cuda._original_NVPTXCompiler
            print("[NVCC PATCH] Restored original NVPTXCompiler", file=sys.stderr)
        else:
            print("[NVCC PATCH] No original NVPTXCompiler found to restore", file=sys.stderr)

    except ImportError:
        print("[NVCC PATCH] Could not import tinygrad to unpatch", file=sys.stderr)


# Auto-patch if this file is imported directly
# (but not if imported via integration in inference.py)
if __name__ != '__main__':
    # Being imported as module
    pass
else:
    # Being run directly for testing
    print("[NVCC PATCH] Running standalone test")
    patch_nvptx_compiler()
    print("\n[Test] Attempting to import tinygrad...")
    try:
        from tinygrad.runtime.support.compiler_cuda import NVPTXCompiler
        print(f"[Test] NVPTXCompiler class: {NVPTXCompiler}")
        print(f"[Test] NVPTXCompiler module: {NVPTXCompiler.__module__}")

        # Try to instantiate
        compiler = NVPTXCompiler('sm_110')
        print(f"[Test] ✅ Successfully instantiated: {compiler}")

    except Exception as e:
        print(f"[Test] ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
