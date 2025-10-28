"""
nvcc Subprocess Compiler

Compiles CUDA C → PTX using nvcc subprocess, then links PTX → CUBIN using nvJitLink.
This is the most reliable path for CUDA 13.0 + Blackwell.

Based on Solution A from Agent 10 research.
"""

import subprocess
import tempfile
import os
from pathlib import Path


class NVCCSubprocessCompiler:
    """
    Compile CUDA C → PTX → CUBIN via nvcc subprocess + nvJitLink
    """

    def __init__(self, arch: str, config: dict, logger):
        self.arch = arch
        self.config = config
        self.logger = logger

        # Build nvcc flags from config
        self.nvcc_flags = self._build_nvcc_flags()

    def _build_nvcc_flags(self) -> list:
        """Build nvcc command-line flags from configuration"""
        flags = [
            f'--gpu-architecture={self.arch}',
            f'-O{self.config.get("optimization_level", 3)}'
        ]

        if self.config.get('use_fast_math', True):
            flags.append('--use_fast_math')

        if self.config.get('debug_symbols', False):
            flags.append('-lineinfo')

        # Add extra flags from config
        extra_flags = self.config.get('nvcc_extra_flags', [])
        flags.extend(extra_flags)

        return flags

    def compile(self, src: str) -> bytes:
        """
        Complete pipeline: CUDA C → PTX (nvcc) → CUBIN (nvJitLink)

        Args:
            src: CUDA C source code

        Returns:
            CUBIN binary bytes
        """
        # Step 1: Compile CUDA C → PTX using nvcc
        ptx = self._compile_cuda_to_ptx(src)

        # Step 2: Link PTX → CUBIN using nvJitLink
        cubin = self._link_ptx_to_cubin(ptx)

        return cubin

    def _compile_cuda_to_ptx(self, cuda_src: str) -> bytes:
        """Compile CUDA C source to PTX assembly using nvcc"""
        # Write CUDA C to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
            f.write(cuda_src)
            cu_file = Path(f.name)

        ptx_file = cu_file.with_suffix('.ptx')

        try:
            # Build nvcc command
            cmd = ['nvcc', '-ptx'] + self.nvcc_flags + [
                '-o', str(ptx_file),
                str(cu_file)
            ]

            self.logger.debug(f"nvcc command: {' '.join(cmd)}")

            # Execute nvcc
            timeout = self.config.get('subprocess_timeout', 30)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )

            if result.returncode != 0:
                self.logger.error(f"nvcc compilation failed: {result.stderr}")
                raise RuntimeError(f"nvcc failed: {result.stderr}")

            # Read PTX output
            ptx = ptx_file.read_bytes()

            if self.config.get('log_ptx_output', False):
                self.logger.debug(f"PTX output (first 500 bytes): {ptx[:500]}")

            self.logger.info(f"nvcc compilation successful. PTX size: {len(ptx)} bytes")
            return ptx

        finally:
            # Cleanup temp files
            cu_file.unlink(missing_ok=True)
            ptx_file.unlink(missing_ok=True)

    def _link_ptx_to_cubin(self, ptx: bytes) -> bytes:
        """Link PTX assembly to CUBIN binary using nvJitLink"""
        try:
            # Import nvJitLink APIs
            import ctypes
            from tinygrad.helpers import to_char_p_p
            import tinygrad.runtime.autogen.nvrtc as nvrtc
            from tinygrad.runtime.support.compiler_cuda import jitlink_check, _get_bytes

            # Create nvJitLink handle
            handle = nvrtc.nvJitLinkHandle()
            jitlink_check(nvrtc.nvJitLinkCreate(
                handle, 1,
                to_char_p_p([f'-arch={self.arch}'.encode()])
            ), handle)

            # Add PTX to linker
            jitlink_check(nvrtc.nvJitLinkAddData(
                handle,
                nvrtc.NVJITLINK_INPUT_PTX,  # Correct type!
                ptx,
                len(ptx),
                b"<kernel>"
            ), handle)

            # Link
            jitlink_check(nvrtc.nvJitLinkComplete(handle), handle)

            # Extract CUBIN
            cubin = _get_bytes(
                handle,
                nvrtc.nvJitLinkGetLinkedCubin,
                nvrtc.nvJitLinkGetLinkedCubinSize,
                jitlink_check
            )

            # Cleanup
            jitlink_check(nvrtc.nvJitLinkDestroy(handle))

            if self.config.get('log_cubin_size', True):
                self.logger.info(f"nvJitLink successful. CUBIN size: {len(cubin)} bytes")

            return cubin

        except Exception as e:
            self.logger.error(f"nvJitLink failed: {e}")
            raise

    @classmethod
    def is_available(cls) -> bool:
        """Check if nvcc is available in PATH"""
        try:
            result = subprocess.run(['which', 'nvcc'],
                                  capture_output=True,
                                  timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def __str__(self):
        return f"NVCCSubprocessCompiler(arch={self.arch}, flags={self.nvcc_flags})"
