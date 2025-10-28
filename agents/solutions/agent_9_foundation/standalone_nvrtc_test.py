#!/usr/bin/env python3
"""
Standalone NVRTC Test - Verify NVRTC API works on Blackwell GPUs

This test validates:
1. NVRTC library can be loaded
2. CUDA C → PTX compilation works
3. PTX → CUBIN linking works via nvJitLink
4. Enum access patterns are correct
5. Error handling works properly

Run on Thor device to verify before deploying full solution.
"""

import sys
import ctypes

# Add tinygrad to path if needed
sys.path.insert(0, "/home/thor/exo/exo-venv/lib/python3.12/site-packages")

import tinygrad.runtime.autogen.nvrtc as nvrtc
from tinygrad.helpers import to_char_p_p


def test_nvrtc_library():
    """Test 1: Verify NVRTC library loads and version works."""
    print("=" * 60)
    print("Test 1: NVRTC Library Initialization")
    print("=" * 60)

    try:
        major = ctypes.c_int()
        minor = ctypes.c_int()
        err = nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))

        if err != nvrtc.NVRTC_SUCCESS:  # Correct enum access!
            print(f"✗ FAILED: nvrtcVersion returned error {err}")
            return False

        print(f"✓ NVRTC version: {major.value}.{minor.value}")
        return True

    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        return False


def test_cuda_c_to_ptx():
    """Test 2: Compile CUDA C → PTX."""
    print("\n" + "=" * 60)
    print("Test 2: CUDA C → PTX Compilation")
    print("=" * 60)

    cuda_src = '''
extern "C" __global__ void test_kernel(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx] = idx * 2.0f;
    }
}
'''

    try:
        # Create program
        prog = nvrtc.nvrtcProgram()
        err = nvrtc.nvrtcCreateProgram(
            ctypes.byref(prog),
            cuda_src.encode('utf-8'),
            b"test_kernel.cu",
            0,
            None,
            None
        )

        if err != nvrtc.NVRTC_SUCCESS:
            print(f"✗ FAILED: nvrtcCreateProgram returned error {err}")
            return False, None

        print("✓ Program created")

        # Compile with sm_110 target
        options = [
            b'--gpu-architecture=compute_110',
            b'--std=c++17'
        ]

        opts_ptr = to_char_p_p(options)
        err = nvrtc.nvrtcCompileProgram(prog, len(options), opts_ptr)

        if err != nvrtc.NVRTC_SUCCESS:
            # Get compilation log
            log_size = ctypes.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
            log = ctypes.create_string_buffer(log_size.value)
            nvrtc.nvrtcGetProgramLog(prog, log)

            print(f"✗ FAILED: nvrtcCompileProgram returned error {err}")
            print(f"Log:\n{log.value.decode('utf-8', errors='replace')}")
            nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
            return False, None

        print("✓ Compilation successful")

        # Get PTX
        ptx_size = ctypes.c_size_t()
        err = nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))

        if err != nvrtc.NVRTC_SUCCESS or ptx_size.value == 0:
            print(f"✗ FAILED: nvrtcGetPTXSize returned error {err} or size {ptx_size.value}")
            nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
            return False, None

        ptx = ctypes.create_string_buffer(ptx_size.value)
        err = nvrtc.nvrtcGetPTX(prog, ptx)

        if err != nvrtc.NVRTC_SUCCESS:
            print(f"✗ FAILED: nvrtcGetPTX returned error {err}")
            nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
            return False, None

        ptx_data = ctypes.string_at(ptx, size=ptx_size.value)

        # Verify PTX format
        ptx_str = ptx_data.decode('utf-8', errors='ignore')
        print(f"✓ Generated {len(ptx_data)} bytes of PTX")
        print(f"✓ PTX header:\n{ptx_str[:200]}")

        if '.version' not in ptx_str[:500]:
            print("⚠ WARNING: PTX doesn't contain .version directive")
            nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
            return False, None

        if '.target sm_110' not in ptx_str[:500]:
            print("⚠ WARNING: PTX doesn't contain .target sm_110")
            nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
            return False, None

        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
        print("✓ PTX format validated")

        return True, ptx_data

    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_ptx_to_cubin(ptx_data):
    """Test 3: Link PTX → CUBIN via nvJitLink."""
    print("\n" + "=" * 60)
    print("Test 3: PTX → CUBIN Linking")
    print("=" * 60)

    if ptx_data is None:
        print("✗ SKIPPED: No PTX data from previous test")
        return False

    try:
        # Create nvJitLink handle
        handle = nvrtc.nvJitLinkHandle()
        options = [b'-arch=sm_110']
        opts_ptr = to_char_p_p(options)

        err = nvrtc.nvJitLinkCreate(
            ctypes.byref(handle),
            len(options),
            opts_ptr
        )

        if err != nvrtc.NVJITLINK_SUCCESS:
            print(f"✗ FAILED: nvJitLinkCreate returned error {err}")
            return False

        print("✓ nvJitLink handle created")

        # Add PTX
        err = nvrtc.nvJitLinkAddData(
            handle,
            nvrtc.NVJITLINK_INPUT_PTX,  # Correct enum access!
            ptx_data,
            len(ptx_data),
            b"test_kernel.ptx"
        )

        if err != nvrtc.NVJITLINK_SUCCESS:
            # Get error log
            log_size = ctypes.c_size_t()
            nvrtc.nvJitLinkGetErrorLogSize(handle, ctypes.byref(log_size))
            log = ctypes.create_string_buffer(log_size.value)
            nvrtc.nvJitLinkGetErrorLog(handle, log)

            print(f"✗ FAILED: nvJitLinkAddData returned error {err}")
            print(f"Log:\n{log.value.decode('utf-8', errors='replace')}")
            nvrtc.nvJitLinkDestroy(ctypes.byref(handle))
            return False

        print("✓ PTX added to linker")

        # Complete linking
        err = nvrtc.nvJitLinkComplete(handle)

        if err != nvrtc.NVJITLINK_SUCCESS:
            log_size = ctypes.c_size_t()
            nvrtc.nvJitLinkGetErrorLogSize(handle, ctypes.byref(log_size))
            log = ctypes.create_string_buffer(log_size.value)
            nvrtc.nvJitLinkGetErrorLog(handle, log)

            print(f"✗ FAILED: nvJitLinkComplete returned error {err}")
            print(f"Log:\n{log.value.decode('utf-8', errors='replace')}")
            nvrtc.nvJitLinkDestroy(ctypes.byref(handle))
            return False

        print("✓ Linking complete")

        # Get CUBIN
        cubin_size = ctypes.c_size_t()
        err = nvrtc.nvJitLinkGetLinkedCubinSize(handle, ctypes.byref(cubin_size))

        if err != nvrtc.NVJITLINK_SUCCESS or cubin_size.value == 0:
            print(f"✗ FAILED: nvJitLinkGetLinkedCubinSize returned error {err} or size {cubin_size.value}")
            nvrtc.nvJitLinkDestroy(ctypes.byref(handle))
            return False

        cubin = ctypes.create_string_buffer(cubin_size.value)
        err = nvrtc.nvJitLinkGetLinkedCubin(handle, cubin)

        if err != nvrtc.NVJITLINK_SUCCESS:
            print(f"✗ FAILED: nvJitLinkGetLinkedCubin returned error {err}")
            nvrtc.nvJitLinkDestroy(ctypes.byref(handle))
            return False

        cubin_data = ctypes.string_at(cubin, size=cubin_size.value)

        print(f"✓ Generated {len(cubin_data)} bytes of CUBIN")

        # Verify CUBIN format (ELF magic)
        if cubin_data[:4] == b'\x7fELF':
            print("✓ CUBIN format verified (ELF header detected)")
        else:
            print(f"⚠ WARNING: Unexpected CUBIN header: {cubin_data[:4]}")

        nvrtc.nvJitLinkDestroy(ctypes.byref(handle))

        return True

    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("NVRTC Standalone Test for Blackwell sm_110")
    print("=" * 60)

    results = []

    # Test 1: NVRTC library
    results.append(("NVRTC Library", test_nvrtc_library()))

    # Test 2: CUDA C → PTX
    success, ptx_data = test_cuda_c_to_ptx()
    results.append(("CUDA C → PTX", success))

    # Test 3: PTX → CUBIN
    results.append(("PTX → CUBIN", test_ptx_to_cubin(ptx_data)))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name}: {status}")

    all_passed = all(result for _, result in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nNVRTC API is working correctly on this device.")
        print("NVPTXCompilerProduction is ready to deploy.")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
        print("\nNVRTC API has issues on this device.")
        print("Review error messages above before deploying.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
