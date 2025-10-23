#!/usr/bin/env python3
"""
NVPTXCompilerV2: Production CUDA 13.0 Compiler for Blackwell GPUs

ROOT CAUSE FIX:
- Tinygrad's CUDARenderer generates CUDA C source code
- Original NVPTXCompiler expected PTX assembly input
- PTXCompiler only does string substitution (TARGET/VERSION), not compilation
- This caused type mismatch: nvJitLink expects PTX, receives CUDA C

SOLUTION:
- Two-stage compilation pipeline:
  Stage 1: CUDA C → PTX (via NVRTC API)
  Stage 2: PTX → CUBIN (via nvJitLink API)

ARCHITECTURE TARGETS:
- Jetson Thor: sm_110 (compute capability 11.0)
- PTX Version: 8.5+ (Blackwell support)
- CUDA Version: 13.0.48

AUTHOR: Solution Agent 9
DATE: 2025-10-23
STATUS: Production-ready
"""

import ctypes
import os
import sys
from typing import Optional, List, Tuple
from pathlib import Path

# Tinygrad imports (will be available when tinygrad is installed)
from tinygrad.device import Compiler, CompileError
from tinygrad.helpers import to_char_p_p, init_c_var, DEBUG

# NVRTC and nvJitLink bindings (from tinygrad)
import tinygrad.runtime.autogen.nvrtc as nvrtc


class NVPTXCompilerV2(Compiler):
    """
    Complete CUDA C → CUBIN compiler for CUDA 13.0 + Blackwell.

    Replaces broken NVPTXCompiler that expected PTX but received CUDA C.

    Pipeline:
        CUDA C (from tinygrad CUDARenderer)
            ↓ [Stage 1: NVRTC]
        PTX assembly (.version 8.5, .target sm_110)
            ↓ [Stage 2: nvJitLink]
        CUBIN binary (executable GPU code)
    """

    def __init__(self, arch: str):
        """
        Initialize compiler for target architecture.

        Args:
            arch: Target GPU architecture (e.g., "sm_110" for Blackwell)
        """
        self.arch = arch
        super().__init__(f"compile_nvptx_v2_{arch}")

        # Extract compute capability for NVRTC flags
        # "sm_110" → compute_110
        self.compute_arch = arch.replace("sm_", "compute_")

        # Determine PTX version based on architecture
        # sm_110 (Blackwell) requires PTX 8.5+
        # sm_89 (Hopper) requires PTX 7.8+
        # Older: PTX 7.5
        arch_num = int(arch.replace("sm_", ""))
        if arch_num >= 110:  # Blackwell
            self.ptx_version = "8.5"
        elif arch_num >= 89:  # Hopper
            self.ptx_version = "7.8"
        else:
            self.ptx_version = "7.5"

        if DEBUG >= 1:
            print(f"[NVPTXCompilerV2] Initialized for {arch} (PTX {self.ptx_version})", file=sys.stderr)

    def compile(self, src: str) -> bytes:
        """
        Main compilation entry point.

        Accepts CUDA C source from tinygrad, returns CUBIN binary.

        Args:
            src: CUDA C source code (NOT PTX assembly!)

        Returns:
            bytes: CUBIN binary ready for GPU execution

        Raises:
            CompileError: If compilation or linking fails
        """
        try:
            # Stage 1: Compile CUDA C → PTX
            if DEBUG >= 2:
                print(f"[NVPTXCompilerV2] Stage 1: CUDA C → PTX", file=sys.stderr)
                print(f"[NVPTXCompilerV2] Input length: {len(src)} chars", file=sys.stderr)
                if DEBUG >= 3:
                    print(f"[NVPTXCompilerV2] First 500 chars:\n{src[:500]}", file=sys.stderr)

            ptx = self._compile_cuda_to_ptx(src)

            if DEBUG >= 2:
                print(f"[NVPTXCompilerV2] PTX length: {len(ptx)} bytes", file=sys.stderr)
                if DEBUG >= 3:
                    # Verify PTX format
                    ptx_str = ptx.decode('utf-8', errors='ignore')[:500]
                    print(f"[NVPTXCompilerV2] PTX header:\n{ptx_str}", file=sys.stderr)
                    if not ptx_str.startswith('.version'):
                        print("[NVPTXCompilerV2] WARNING: PTX doesn't start with .version!", file=sys.stderr)

            # Stage 2: Link PTX → CUBIN
            if DEBUG >= 2:
                print(f"[NVPTXCompilerV2] Stage 2: PTX → CUBIN", file=sys.stderr)

            cubin = self._link_ptx_to_cubin(ptx)

            if DEBUG >= 1:
                print(f"[NVPTXCompilerV2] ✓ Compilation successful: {len(cubin)} bytes CUBIN", file=sys.stderr)

            return cubin

        except CompileError:
            # Re-raise CompileError as-is (already formatted)
            raise
        except Exception as e:
            # Wrap unexpected errors
            self._save_failed_source(src, str(e))
            raise CompileError(f"NVPTXCompilerV2 unexpected error: {type(e).__name__}: {e}")

    def _compile_cuda_to_ptx(self, cuda_src: str) -> bytes:
        """
        Stage 1: Compile CUDA C source to PTX assembly using NVRTC.

        Args:
            cuda_src: CUDA C source code (with #define, extern "C", __global__)

        Returns:
            bytes: PTX assembly (starts with .version, .target)

        Raises:
            CompileError: If NVRTC compilation fails
        """
        # Create NVRTC program
        prog = nvrtc.nvrtcProgram()
        err = nvrtc.nvrtcCreateProgram(
            ctypes.byref(prog),
            cuda_src.encode('utf-8'),
            b"<tinygrad_kernel>",  # Kernel name for error messages
            0,  # No headers
            None,
            None
        )

        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise CompileError(f"nvrtcCreateProgram failed: {self._get_nvrtc_error_name(err)}")

        try:
            # Compile options for Blackwell
            options = self._get_compile_options()

            if DEBUG >= 3:
                print(f"[NVPTXCompilerV2] NVRTC options: {options}", file=sys.stderr)

            # Compile CUDA C → PTX
            opts_c = to_char_p_p([opt.encode('utf-8') for opt in options])
            err = nvrtc.nvrtcCompileProgram(prog, len(options), opts_c)

            if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                # Get compilation log
                log = self._get_program_log(prog)
                self._save_failed_source(cuda_src, log)
                raise CompileError(
                    f"NVRTC compilation failed ({self._get_nvrtc_error_name(err)}):\n\n"
                    f"Source saved to: /tmp/nvptx_failed_*.cu\n\n"
                    f"Compilation log:\n{log}"
                )

            # Extract PTX
            ptx_size = init_c_var(
                ctypes.c_size_t(),
                lambda x: nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(x))
            )

            if ptx_size.value == 0:
                raise CompileError("NVRTC generated empty PTX")

            ptx = ctypes.create_string_buffer(ptx_size.value)
            err = nvrtc.nvrtcGetPTX(prog, ptx)

            if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                raise CompileError(f"nvrtcGetPTX failed: {self._get_nvrtc_error_name(err)}")

            return ctypes.string_at(ptx, size=ptx_size.value)

        finally:
            # Always clean up NVRTC program
            nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

    def _link_ptx_to_cubin(self, ptx: bytes) -> bytes:
        """
        Stage 2: Link PTX assembly to CUBIN binary using nvJitLink.

        Args:
            ptx: PTX assembly bytes (from NVRTC or PTXRenderer)

        Returns:
            bytes: CUBIN binary (executable GPU code)

        Raises:
            CompileError: If linking fails
        """
        # Create nvJitLink handle
        handle = nvrtc.nvJitLinkHandle()
        err = nvrtc.nvJitLinkCreate(
            ctypes.byref(handle),
            1,  # Number of options
            to_char_p_p([f'-arch={self.arch}'.encode('utf-8')])
        )

        if err != nvrtc.nvJitLinkResult.NVJITLINK_SUCCESS:
            raise CompileError(f"nvJitLinkCreate failed: {self._get_jitlink_error_name(err)}")

        try:
            # Add PTX to linker
            if DEBUG >= 3:
                print(f"[NVPTXCompilerV2] Adding PTX to nvJitLink ({len(ptx)} bytes)", file=sys.stderr)

            err = nvrtc.nvJitLinkAddData(
                handle,
                nvrtc.nvJitLinkInputType.NVJITLINK_INPUT_PTX,  # ← CRITICAL: Correct type!
                ptx,
                len(ptx),
                b"<tinygrad_kernel.ptx>"
            )

            if err != nvrtc.nvJitLinkResult.NVJITLINK_SUCCESS:
                # Get error log
                log = self._get_jitlink_error_log(handle)
                self._save_failed_ptx(ptx, log)
                raise CompileError(
                    f"nvJitLink failed to add PTX ({self._get_jitlink_error_name(err)}):\n\n"
                    f"PTX saved to: /tmp/nvptx_failed_*.ptx\n\n"
                    f"Link error:\n{log}"
                )

            # Complete linking
            err = nvrtc.nvJitLinkComplete(handle)

            if err != nvrtc.nvJitLinkResult.NVJITLINK_SUCCESS:
                log = self._get_jitlink_error_log(handle)
                raise CompileError(f"nvJitLinkComplete failed:\n{log}")

            # Extract CUBIN
            cubin_size = init_c_var(
                ctypes.c_size_t(),
                lambda x: nvrtc.nvJitLinkGetLinkedCubinSize(handle, ctypes.byref(x))
            )

            if cubin_size.value == 0:
                raise CompileError("nvJitLink generated empty CUBIN")

            cubin = ctypes.create_string_buffer(cubin_size.value)
            err = nvrtc.nvJitLinkGetLinkedCubin(handle, cubin)

            if err != nvrtc.nvJitLinkResult.NVJITLINK_SUCCESS:
                raise CompileError(f"nvJitLinkGetLinkedCubin failed: {self._get_jitlink_error_name(err)}")

            return ctypes.string_at(cubin, size=cubin_size.value)

        finally:
            # Always clean up nvJitLink handle
            nvrtc.nvJitLinkDestroy(ctypes.byref(handle))

    def _get_compile_options(self) -> List[str]:
        """
        Generate NVRTC compilation options for target architecture.

        Returns:
            List of nvcc-style compiler flags
        """
        options = [
            f'--gpu-architecture={self.compute_arch}',  # e.g., compute_110
            '--std=c++17',  # C++17 for modern CUDA features
            '--use_fast_math',  # Aggressive math optimizations
            '--extra-device-vectorization',  # Enable device vectorization
            '--restrict',  # Enable restrict keyword
        ]

        # Blackwell-specific optimizations
        if self.arch >= "sm_110":
            options.extend([
                '--device-c',  # Device-side compilation
                '--relocatable-device-code=false',  # Faster, no relocation needed
            ])

        # Add debug symbols if DEBUG >= 3
        if DEBUG >= 3:
            options.append('--device-debug')
            options.append('--generate-line-info')

        return options

    def _get_program_log(self, prog: nvrtc.nvrtcProgram) -> str:
        """Get compilation log from NVRTC program."""
        try:
            log_size = init_c_var(
                ctypes.c_size_t(),
                lambda x: nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(x))
            )

            if log_size.value == 0:
                return "(no log available)"

            log = ctypes.create_string_buffer(log_size.value)
            nvrtc.nvrtcGetProgramLog(prog, log)
            return ctypes.string_at(log, size=log_size.value).decode('utf-8', errors='replace')
        except:
            return "(failed to retrieve log)"

    def _get_jitlink_error_log(self, handle: nvrtc.nvJitLinkHandle) -> str:
        """Get error log from nvJitLink handle."""
        try:
            log_size = init_c_var(
                ctypes.c_size_t(),
                lambda x: nvrtc.nvJitLinkGetErrorLogSize(handle, ctypes.byref(x))
            )

            if log_size.value == 0:
                return "(no error log)"

            log = ctypes.create_string_buffer(log_size.value)
            nvrtc.nvJitLinkGetErrorLog(handle, log)
            return ctypes.string_at(log, size=log_size.value).decode('utf-8', errors='replace')
        except:
            return "(failed to retrieve log)"

    def _get_nvrtc_error_name(self, err: int) -> str:
        """Convert NVRTC error code to readable name."""
        error_names = {
            0: "NVRTC_SUCCESS",
            1: "NVRTC_ERROR_OUT_OF_MEMORY",
            2: "NVRTC_ERROR_PROGRAM_CREATION_FAILURE",
            3: "NVRTC_ERROR_INVALID_INPUT",
            4: "NVRTC_ERROR_INVALID_PROGRAM",
            5: "NVRTC_ERROR_INVALID_OPTION",
            6: "NVRTC_ERROR_COMPILATION",
            7: "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE",
            8: "NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION",
            9: "NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION",
            10: "NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID",
            11: "NVRTC_ERROR_INTERNAL_ERROR",
        }
        return error_names.get(err, f"UNKNOWN_ERROR_{err}")

    def _get_jitlink_error_name(self, err: int) -> str:
        """Convert nvJitLink error code to readable name."""
        error_names = {
            0: "NVJITLINK_SUCCESS",
            1: "NVJITLINK_ERROR_UNRECOGNIZED_OPTION",
            2: "NVJITLINK_ERROR_MISSING_ARCH",
            3: "NVJITLINK_ERROR_INVALID_INPUT",
            4: "NVJITLINK_ERROR_PTX_COMPILE",
            5: "NVJITLINK_ERROR_NVVM_COMPILE",
            6: "NVJITLINK_ERROR_NOSUPPORT_FOR_ARCH",
            7: "NVJITLINK_ERROR_INTERNAL",
            8: "NVJITLINK_ERROR_UNRECOGNIZED_INPUT",
            9: "NVJITLINK_ERROR_THREADPOOL",
        }
        return error_names.get(err, f"UNKNOWN_ERROR_{err}")

    def _save_failed_source(self, src: str, error: str):
        """Save failed CUDA C source for debugging."""
        try:
            import tempfile
            import time

            timestamp = int(time.time())
            cu_file = f"/tmp/nvptx_failed_{timestamp}.cu"
            err_file = f"/tmp/nvptx_failed_{timestamp}.log"

            Path(cu_file).write_text(src)
            Path(err_file).write_text(error)

            if DEBUG >= 1:
                print(f"[NVPTXCompilerV2] Failed source saved to {cu_file}", file=sys.stderr)
                print(f"[NVPTXCompilerV2] Error log saved to {err_file}", file=sys.stderr)
        except:
            pass  # Don't fail if we can't save debug files

    def _save_failed_ptx(self, ptx: bytes, error: str):
        """Save failed PTX for debugging."""
        try:
            import tempfile
            import time

            timestamp = int(time.time())
            ptx_file = f"/tmp/nvptx_failed_{timestamp}.ptx"
            err_file = f"/tmp/nvptx_failed_{timestamp}_link.log"

            Path(ptx_file).write_bytes(ptx)
            Path(err_file).write_text(error)

            if DEBUG >= 1:
                print(f"[NVPTXCompilerV2] Failed PTX saved to {ptx_file}", file=sys.stderr)
                print(f"[NVPTXCompilerV2] Link error saved to {err_file}", file=sys.stderr)
        except:
            pass

    def disassemble(self, lib: bytes):
        """
        Disassemble CUBIN for debugging (optional).

        Uses cuobjdump if available, otherwise prints size only.
        """
        if DEBUG >= 2:
            print(f"[NVPTXCompilerV2] CUBIN size: {len(lib)} bytes", file=sys.stderr)

        try:
            import subprocess
            import tempfile

            with tempfile.NamedTemporaryFile(suffix='.cubin', delete=False) as f:
                f.write(lib)
                cubin_path = f.name

            result = subprocess.run(
                ['cuobjdump', '--dump-sass', cubin_path],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0 and DEBUG >= 3:
                print(f"[NVPTXCompilerV2] Disassembly:\n{result.stdout[:1000]}", file=sys.stderr)

            os.unlink(cubin_path)

        except (FileNotFoundError, subprocess.TimeoutExpired):
            # cuobjdump not available or timed out
            pass
        except Exception as e:
            if DEBUG >= 2:
                print(f"[NVPTXCompilerV2] Disassembly failed: {e}", file=sys.stderr)


# ============================================================================
# COMPATIBILITY LAYER
# ============================================================================

# For drop-in replacement of tinygrad's NVPTXCompiler
if __name__ != "__main__":
    # When imported as module, provide aliases
    PTXCompilerFixed = NVPTXCompilerV2  # Alternative name
    NVPTXCompiler = NVPTXCompilerV2  # Direct replacement


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    print("NVPTXCompilerV2 Standalone Test")
    print("=" * 60)

    # Test CUDA C source
    test_src = '''
    extern "C" __global__ void test_kernel(float* output, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            output[idx] = idx * 2.0f;
        }
    }
    '''

    print(f"\nTest source ({len(test_src)} chars):")
    print(test_src)

    try:
        # Test compilation
        compiler = NVPTXCompilerV2('sm_110')
        print(f"\nCompiling for {compiler.arch} (PTX {compiler.ptx_version})...")

        cubin = compiler.compile(test_src)
        print(f"✓ SUCCESS: Generated {len(cubin)} bytes of CUBIN")

        # Verify CUBIN format (ELF magic)
        if cubin[:4] == b'\x7fELF':
            print("✓ CUBIN format verified (ELF header detected)")
        else:
            print(f"⚠ WARNING: Unexpected CUBIN header: {cubin[:4]}")

        # Try disassembly
        compiler.disassemble(cubin)

    except CompileError as e:
        print(f"\n✗ COMPILATION FAILED:\n{e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR:\n{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All tests passed!")
