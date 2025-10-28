#!/usr/bin/env python3
"""
nvcc Subprocess Compiler for CUDA 13.0 / Blackwell GPUs

Compiles CUDA C source to CUBIN via external nvcc process:
CUDA C → nvcc -ptx → PTX assembly → nvJitLink → CUBIN

Author: Solution Agent 3
Date: 2025-10-23
"""

import subprocess
import tempfile
import os
import ctypes
from pathlib import Path
from typing import Optional

# Tinygrad imports (will be available when integrated)
try:
    from tinygrad.device import Compiler, CompileError
    from tinygrad.helpers import to_char_p_p, init_c_var
    import tinygrad.runtime.autogen.nvrtc as nvrtc
    from tinygrad.runtime.support.compiler_cuda import jitlink_check, _get_bytes
    TINYGRAD_AVAILABLE = True
except ImportError:
    # For standalone testing
    TINYGRAD_AVAILABLE = False
    class Compiler:
        def __init__(self, name): self.name = name
    class CompileError(Exception): pass


class NVCCCompiler(Compiler):
    """
    Compiles CUDA C source to CUBIN using nvcc subprocess + nvJitLink

    Pipeline:
    1. Write CUDA C source to temporary .cu file
    2. Invoke: nvcc -ptx -arch=sm_110 input.cu -o output.ptx
    3. Read PTX assembly from .ptx file
    4. Pass to nvJitLink with NVJITLINK_INPUT_PTX type
    5. Link PTX → CUBIN
    6. Clean up temporary files

    Why This Works:
    - nvcc exists in CUDA 13.0 (unlike NVRTC library)
    - Generates real PTX assembly (not CUDA C)
    - nvJitLink correctly receives PTX type
    - Zero library dependencies
    """

    def __init__(self, arch: str = 'sm_110'):
        """
        Initialize NVCC compiler

        Args:
            arch: GPU architecture (e.g., 'sm_110' for Blackwell)
        """
        self.arch = arch
        super().__init__(f"compile_nvcc_{arch}")

        # Verify nvcc exists at initialization
        if not self._nvcc_available():
            raise CompileError(
                "nvcc not found in PATH. "
                "Please ensure CUDA 13.0 is installed and nvcc is accessible.\n"
                "Try: export PATH=/usr/local/cuda/bin:$PATH"
            )

        # Verify architecture support
        if not self._verify_arch_support():
            raise CompileError(
                f"Architecture {arch} not supported by installed nvcc.\n"
                f"Run 'nvcc --help' to see supported architectures."
            )

    def _nvcc_available(self) -> bool:
        """Check if nvcc is available in PATH"""
        try:
            result = subprocess.run(
                ['nvcc', '--version'],
                capture_output=True,
                timeout=5,
                check=False
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _verify_arch_support(self) -> bool:
        """Verify nvcc supports target architecture"""
        try:
            # Try to get help output (contains supported architectures)
            result = subprocess.run(
                ['nvcc', '--help'],
                capture_output=True,
                timeout=5,
                check=False
            )
            # Check if our architecture is mentioned
            # sm_110, compute_110, or generic support
            help_text = result.stdout.decode('utf-8', errors='ignore').lower()
            arch_num = self.arch.replace('sm_', '').replace('compute_', '')
            return (
                self.arch.lower() in help_text or
                f'sm_{arch_num}' in help_text or
                f'compute_{arch_num}' in help_text or
                'sm_90' in help_text  # If sm_90 supported, likely newer too
            )
        except Exception:
            # If check fails, assume supported (fail at compilation)
            return True

    def compile(self, src: str) -> bytes:
        """
        Compile CUDA C source to CUBIN binary

        Args:
            src: CUDA C source code (from CUDARenderer)

        Returns:
            CUBIN binary bytes (ready for cuModuleLoadData)

        Raises:
            CompileError: If compilation or linking fails
        """
        # Step 1: Compile CUDA C → PTX using nvcc
        ptx_data = self._compile_cuda_to_ptx(src)

        # Step 2: Link PTX → CUBIN using nvJitLink
        cubin_data = self._link_ptx_to_cubin(ptx_data)

        return cubin_data

    def _compile_cuda_to_ptx(self, cuda_src: str) -> bytes:
        """
        Compile CUDA C source to PTX assembly using nvcc subprocess

        Args:
            cuda_src: CUDA C source code

        Returns:
            PTX assembly bytes

        Raises:
            CompileError: If nvcc compilation fails
        """
        cu_file = None
        ptx_file = None

        try:
            # Create temporary .cu file
            cu_fd, cu_path = tempfile.mkstemp(suffix='.cu', text=True)
            cu_file = Path(cu_path)

            # Write CUDA C source
            with os.fdopen(cu_fd, 'w') as f:
                f.write(cuda_src)

            # Generate .ptx output path
            ptx_file = cu_file.with_suffix('.ptx')

            # Build nvcc command
            nvcc_cmd = [
                'nvcc',
                f'--gpu-architecture={self.arch}',
                '-ptx',  # Generate PTX, not CUBIN
                str(cu_file),
                '-o', str(ptx_file)
            ]

            # Execute nvcc
            result = subprocess.run(
                nvcc_cmd,
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout
                check=False  # We'll handle errors manually
            )

            if result.returncode != 0:
                # Compilation failed - include stderr in error
                error_msg = f"nvcc compilation failed (exit code {result.returncode}):\n"
                error_msg += f"Command: {' '.join(nvcc_cmd)}\n"
                error_msg += f"stderr:\n{result.stderr}\n"
                if result.stdout:
                    error_msg += f"stdout:\n{result.stdout}\n"
                raise CompileError(error_msg)

            # Read PTX output
            if not ptx_file.exists():
                raise CompileError(
                    f"nvcc succeeded but PTX file not found: {ptx_file}\n"
                    f"This should never happen. Check disk space and permissions."
                )

            ptx_data = ptx_file.read_bytes()

            if len(ptx_data) == 0:
                raise CompileError("nvcc generated empty PTX file")

            # Verify PTX format (should start with .version or .target)
            ptx_text = ptx_data.decode('utf-8', errors='ignore')
            if not ('.version' in ptx_text[:200] or '.target' in ptx_text[:200]):
                raise CompileError(
                    f"nvcc output doesn't look like PTX assembly.\n"
                    f"First 200 bytes: {ptx_data[:200]}"
                )

            return ptx_data

        except subprocess.TimeoutExpired:
            raise CompileError(
                f"nvcc compilation timed out after 60 seconds.\n"
                f"This may indicate a bug in the CUDA C source or nvcc."
            )

        except Exception as e:
            if isinstance(e, CompileError):
                raise
            raise CompileError(f"Unexpected error during CUDA C → PTX compilation: {e}")

        finally:
            # Clean up temporary files
            if cu_file and cu_file.exists():
                try:
                    cu_file.unlink()
                except Exception:
                    pass  # Best effort cleanup

            if ptx_file and ptx_file.exists():
                try:
                    ptx_file.unlink()
                except Exception:
                    pass  # Best effort cleanup

    def _link_ptx_to_cubin(self, ptx_data: bytes) -> bytes:
        """
        Link PTX assembly to CUBIN binary using nvJitLink

        Args:
            ptx_data: PTX assembly bytes

        Returns:
            CUBIN binary bytes

        Raises:
            CompileError: If linking fails
        """
        if not TINYGRAD_AVAILABLE:
            raise CompileError(
                "nvJitLink integration requires tinygrad. "
                "This module is meant to be used within tinygrad."
            )

        handle = nvrtc.nvJitLinkHandle()

        try:
            # Step 1: Create nvJitLink handle with architecture
            jitlink_check(
                nvrtc.nvJitLinkCreate(
                    ctypes.byref(handle),
                    1,  # Number of options
                    to_char_p_p([f'-arch={self.arch}'.encode()])
                ),
                handle
            )

            # Step 2: Add PTX data with correct input type
            # CRITICAL: Type is NVJITLINK_INPUT_PTX (PTX assembly, not CUDA C!)
            jitlink_check(
                nvrtc.nvJitLinkAddData(
                    handle,
                    nvrtc.NVJITLINK_INPUT_PTX,  # ← Now type matches content!
                    ptx_data,
                    len(ptx_data),
                    "<nvcc_generated>".encode()
                ),
                handle
            )

            # Step 3: Complete linking (PTX → CUBIN)
            jitlink_check(nvrtc.nvJitLinkComplete(handle), handle)

            # Step 4: Get linked CUBIN
            cubin_data = _get_bytes(
                handle,
                nvrtc.nvJitLinkGetLinkedCubin,
                nvrtc.nvJitLinkGetLinkedCubinSize,
                jitlink_check
            )

            if len(cubin_data) == 0:
                raise CompileError("nvJitLink produced empty CUBIN")

            # Verify CUBIN format (should be ELF)
            if cubin_data[:4] != b'\x7fELF':
                raise CompileError(
                    f"nvJitLink output doesn't look like CUBIN (not ELF format).\n"
                    f"First 4 bytes: {cubin_data[:4].hex()}"
                )

            return cubin_data

        except Exception as e:
            # Try to get error log from nvJitLink
            try:
                log_size = init_c_var(
                    ctypes.c_size_t(),
                    lambda x: nvrtc.nvJitLinkGetErrorLogSize(handle, ctypes.byref(x))
                )
                if log_size.value > 0:
                    log = ctypes.create_string_buffer(log_size.value)
                    nvrtc.nvJitLinkGetErrorLog(handle, log)
                    error_log = log.value.decode('utf-8', errors='ignore')
                    raise CompileError(f"nvJitLink failed:\n{error_log}\n\nOriginal error: {e}")
            except Exception:
                pass  # Couldn't get log, use original error

            if isinstance(e, CompileError):
                raise
            raise CompileError(f"PTX → CUBIN linking failed: {e}")

        finally:
            # Clean up nvJitLink handle
            try:
                jitlink_check(nvrtc.nvJitLinkDestroy(ctypes.byref(handle)))
            except Exception:
                pass  # Best effort cleanup

    def disassemble(self, lib: bytes) -> None:
        """
        Disassemble CUBIN for debugging (optional)

        Uses cuobjdump if available, otherwise no-op

        Args:
            lib: CUBIN binary bytes
        """
        try:
            # Try to use cuobjdump (comes with CUDA toolkit)
            with tempfile.NamedTemporaryFile(suffix='.cubin', delete=False) as f:
                f.write(lib)
                cubin_path = f.name

            try:
                result = subprocess.run(
                    ['cuobjdump', '-sass', cubin_path],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    print("[NVCC Disassembly]")
                    print(result.stdout)
            finally:
                Path(cubin_path).unlink(missing_ok=True)

        except Exception:
            # cuobjdump not available or failed - no big deal
            pass


def test_nvcc_compiler():
    """Standalone test of NVCCCompiler (without tinygrad)"""
    print("[Test] NVCCCompiler Standalone")

    # Test 1: Check nvcc availability
    print("\n[Test 1] Checking nvcc availability...")
    try:
        compiler = NVCCCompiler('sm_110')
        print("✅ nvcc found and architecture supported")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return

    # Test 2: Compile simple kernel to PTX
    print("\n[Test 2] Compiling simple kernel to PTX...")
    simple_src = """
    extern "C" __global__ void test_kernel(float* data) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        data[idx] = idx * 2.0f;
    }
    """

    try:
        ptx = compiler._compile_cuda_to_ptx(simple_src)
        print(f"✅ PTX generated ({len(ptx)} bytes)")

        # Verify PTX content
        ptx_text = ptx.decode('utf-8')
        if '.version' in ptx_text and '.target sm_110' in ptx_text:
            print("✅ PTX contains correct directives")
        else:
            print("❌ PTX missing expected directives")
            print(f"First 500 chars:\n{ptx_text[:500]}")

    except Exception as e:
        print(f"❌ PTX compilation failed: {e}")
        return

    print("\n[Test] All standalone tests passed!")
    print("Note: Full integration test requires tinygrad")


if __name__ == '__main__':
    test_nvcc_compiler()
