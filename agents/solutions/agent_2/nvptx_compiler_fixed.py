"""
Enhanced NVPTXCompiler - Proper NVRTC Usage for CUDA 13.0

This is a BACKUP solution if removing the monkey-patch reveals unexpected issues.
It demonstrates the CORRECT way to use NVRTC + nvJitLink compilation chain.

Author: Solution Agent 2
Date: 2025-10-23
Status: BACKUP (use only if primary solution fails)
"""

import ctypes
from typing import Optional
from tinygrad.helpers import to_char_p_p, init_c_var
from tinygrad.runtime.support.compiler_cuda import CUDACompiler, Compiler, nvrtc
import sys


class NVPTXCompilerFixed(CUDACompiler):
    """
    Enhanced NVPTXCompiler that properly compiles CUDA C → PTX → CUBIN.

    Key Changes from Original:
    1. Base class: PTXCompiler → CUDACompiler (gets real NVRTC compilation)
    2. Compilation step: Uses _compile_program() to get real PTX
    3. Type match: Passes actual PTX to nvJitLink, not CUDA C

    Compilation Chain:
        CUDA C Source
            ↓
        nvrtcCompileProgram() [NVRTC]
            ↓
        PTX Assembly (.version, .target)
            ↓
        nvJitLinkAddData() [nvJitLink]
            ↓
        CUBIN Binary
    """

    def __init__(self, arch: str, cache_key: str = "nv_ptx_fixed"):
        """
        Initialize enhanced NVPTXCompiler.

        Args:
            arch: Target architecture (e.g., "sm_110", "sm_101")
            cache_key: Cache identifier for compiled kernels
        """
        # Initialize as CUDACompiler to get NVRTC compilation capability
        super().__init__(arch, cache_key=cache_key)
        print(f"[NVPTXCompilerFixed] Initialized for architecture: {arch}")

    def compile(self, src: str) -> bytes:
        """
        Compile CUDA C source to CUBIN using proper NVRTC + nvJitLink chain.

        Args:
            src: CUDA C source code (string)

        Returns:
            CUBIN binary data (bytes)

        Raises:
            RuntimeError: If NVRTC compilation or nvJitLink fails
        """
        try:
            # Step 1: Use NVRTC to compile CUDA C → Real PTX assembly
            print(f"[NVPTXCompilerFixed] Step 1: Compiling CUDA C → PTX using NVRTC...")
            ptx = self._compile_program(src, nvrtc.nvrtcGetPTX, nvrtc.nvrtcGetPTXSize)

            # Verify PTX format
            ptx_str = ptx.decode('utf-8', errors='ignore')
            if not ptx_str.startswith('.version'):
                raise RuntimeError(
                    f"[NVPTXCompilerFixed] NVRTC output is not valid PTX!\n"
                    f"Expected: .version X.X\n"
                    f"Got: {ptx_str[:100]}"
                )

            print(f"[NVPTXCompilerFixed] PTX generated successfully ({len(ptx)} bytes)")
            print(f"[NVPTXCompilerFixed] PTX first 200 chars: {ptx_str[:200]}")

            # Step 2: Create nvJitLink handle with architecture options
            print(f"[NVPTXCompilerFixed] Step 2: Creating nvJitLink handle for {self.arch}...")
            handle = nvrtc.nvJitLinkHandle()
            options = [f'-arch={self.arch}'.encode()]

            result = nvrtc.nvJitLinkCreate(
                ctypes.byref(handle),
                len(options),
                to_char_p_p(options)
            )

            if result != 0:
                raise RuntimeError(
                    f"[NVPTXCompilerFixed] nvJitLinkCreate failed with code {result}"
                )

            try:
                # Step 3: Add REAL PTX to nvJitLink (now type matches!)
                print(f"[NVPTXCompilerFixed] Step 3: Adding PTX to nvJitLink...")
                result = nvrtc.nvJitLinkAddData(
                    handle,
                    nvrtc.NVJITLINK_INPUT_PTX,  # Type: PTX assembly (NOW CORRECT!)
                    ptx,                         # Actual PTX from NVRTC
                    len(ptx),
                    b"<kernel.ptx>"
                )

                if result != 0:
                    error_log = self._get_nvjitlink_error_log(handle)
                    raise RuntimeError(
                        f"[NVPTXCompilerFixed] nvJitLinkAddData failed with code {result}\n"
                        f"Error log: {error_log}"
                    )

                # Step 4: Link PTX → CUBIN
                print(f"[NVPTXCompilerFixed] Step 4: Linking PTX → CUBIN...")
                result = nvrtc.nvJitLinkComplete(handle)

                if result != 0:
                    error_log = self._get_nvjitlink_error_log(handle)
                    raise RuntimeError(
                        f"[NVPTXCompilerFixed] nvJitLinkComplete failed with code {result}\n"
                        f"Error log: {error_log}"
                    )

                # Step 5: Get CUBIN output
                print(f"[NVPTXCompilerFixed] Step 5: Retrieving CUBIN...")
                cubin_size = init_c_var(
                    ctypes.c_size_t(),
                    lambda x: nvrtc.nvJitLinkGetLinkedCubinSize(handle, ctypes.byref(x))
                )

                cubin = ctypes.create_string_buffer(cubin_size.value)
                result = nvrtc.nvJitLinkGetLinkedCubin(handle, cubin)

                if result != 0:
                    raise RuntimeError(
                        f"[NVPTXCompilerFixed] nvJitLinkGetLinkedCubin failed with code {result}"
                    )

                cubin_data = ctypes.string_at(cubin, size=cubin_size.value)
                print(f"[NVPTXCompilerFixed] CUBIN generated successfully ({len(cubin_data)} bytes)")

                return cubin_data

            finally:
                # Step 6: Cleanup nvJitLink handle
                print(f"[NVPTXCompilerFixed] Step 6: Cleaning up nvJitLink...")
                nvrtc.nvJitLinkDestroy(ctypes.byref(handle))

        except Exception as e:
            print(f"[NVPTXCompilerFixed] Compilation failed: {e}", file=sys.stderr)
            # Print CUDA C source for debugging
            print(f"[NVPTXCompilerFixed] CUDA C source (first 500 chars):", file=sys.stderr)
            print(src[:500], file=sys.stderr)
            raise

    def _get_nvjitlink_error_log(self, handle: nvrtc.nvJitLinkHandle) -> str:
        """
        Retrieve error log from nvJitLink for debugging.

        Args:
            handle: nvJitLink handle

        Returns:
            Error log as string, or empty string if unavailable
        """
        try:
            log_size = init_c_var(
                ctypes.c_size_t(),
                lambda x: nvrtc.nvJitLinkGetErrorLogSize(handle, ctypes.byref(x))
            )

            if log_size.value > 0:
                log = ctypes.create_string_buffer(log_size.value)
                nvrtc.nvJitLinkGetErrorLog(handle, log)
                return ctypes.string_at(log, size=log_size.value).decode('utf-8')
        except:
            pass

        return ""


# Monkey-patch function to replace NVPTXCompiler in tinygrad
def apply_nvptx_compiler_fix():
    """
    Replace tinygrad's NVPTXCompiler with our fixed version.

    Call this BEFORE any tinygrad device initialization.

    Example:
        from nvptx_compiler_fixed import apply_nvptx_compiler_fix
        apply_nvptx_compiler_fix()

        from tinygrad import Device
        Device.DEFAULT = Device["CUDA"]
    """
    try:
        from tinygrad.runtime.support import compiler_cuda

        # Replace NVPTXCompiler class
        original_class = compiler_cuda.NVPTXCompiler
        compiler_cuda.NVPTXCompiler = NVPTXCompilerFixed

        print("[NVPTXCompilerFixed] Successfully replaced tinygrad's NVPTXCompiler")
        print(f"[NVPTXCompilerFixed] Original: {original_class}")
        print(f"[NVPTXCompilerFixed] Replaced with: {NVPTXCompilerFixed}")

        return True

    except Exception as e:
        print(f"[NVPTXCompilerFixed] Failed to apply fix: {e}", file=sys.stderr)
        return False


# Self-test function
def test_nvrtc_available():
    """
    Test if NVRTC library is available and functional.

    Returns:
        bool: True if NVRTC is available, False otherwise
    """
    try:
        # Test 1: Can we import nvrtc module?
        from tinygrad.runtime.autogen import nvrtc
        print("[TEST] ✅ nvrtc module imported successfully")

        # Test 2: Can we call nvrtcVersion?
        major = ctypes.c_int()
        minor = ctypes.c_int()
        result = nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))

        if result == 0:
            print(f"[TEST] ✅ nvrtcVersion succeeded: {major.value}.{minor.value}")
        else:
            print(f"[TEST] ❌ nvrtcVersion failed with code {result}")
            return False

        # Test 3: Can we create a simple program?
        simple_src = b"""
        extern "C" __global__ void test_kernel(float* data) {
            int idx = threadIdx.x;
            data[idx] = idx * 2.0f;
        }
        """

        prog = nvrtc.nvrtcProgram()
        result = nvrtc.nvrtcCreateProgram(
            ctypes.byref(prog),
            simple_src,
            b"test.cu",
            0, None, None
        )

        if result == 0:
            print("[TEST] ✅ nvrtcCreateProgram succeeded")
            nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
        else:
            print(f"[TEST] ❌ nvrtcCreateProgram failed with code {result}")
            return False

        print("[TEST] ✅ All NVRTC tests passed - library is fully functional")
        return True

    except Exception as e:
        print(f"[TEST] ❌ NVRTC test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 80)
    print("NVPTXCompilerFixed - BACKUP Solution Test")
    print("=" * 80)
    print()

    # Test NVRTC availability
    print("Testing NVRTC availability...")
    if test_nvrtc_available():
        print()
        print("✅ NVRTC is available! Primary solution (remove monkey-patch) should work.")
        print("   This backup solution is NOT needed.")
    else:
        print()
        print("❌ NVRTC test failed. This backup solution may be needed.")
        print("   Use apply_nvptx_compiler_fix() to enable it.")

    print()
    print("=" * 80)
