"""
Enhanced PTXCompiler with Blackwell sm_110 Optimizations

This module provides an enhanced PTXCompiler that:
1. Supports PTX ISA 8.5 for Blackwell architecture (sm_110)
2. Maintains backward compatibility with Hopper (sm_89, PTX 7.8)
3. Preserves support for older architectures (PTX 7.5)

Usage:
    1. Ensure PTX=1 environment variable is set BEFORE tinygrad imports
    2. Import and monkey-patch tinygrad's PTXCompiler with this enhanced version
    3. NVPTXCompiler will automatically use enhanced version

Architecture Mapping:
    - sm_110+ (Blackwell): PTX ISA 8.5
    - sm_89+ (Hopper): PTX ISA 7.8
    - sm_75+ (older): PTX ISA 7.5
"""

from tinygrad.runtime.support.compiler_cuda import PTXCompiler, NVPTXCompiler
from tinygrad.device import Compiler
import tinygrad.runtime.autogen.nvrtc as nvrtc
from tinygrad.runtime.support.compiler_cuda import jitlink_check, _get_bytes
from tinygrad.helpers import to_char_p_p


class EnhancedPTXCompiler(PTXCompiler):
    """
    Enhanced PTXCompiler with Blackwell-specific optimizations.

    Key Enhancement:
        - PTX ISA 8.5 support for sm_110 (Blackwell)
        - Correct version selection based on architecture
        - Maintains all existing PTXCompiler functionality
    """

    def __init__(self, arch: str, cache_key="ptx_enhanced"):
        self.arch = arch
        # Call Compiler.__init__ directly to set cache key
        Compiler.__init__(self, f"compile_{cache_key}_{self.arch}")

    def compile(self, src: str) -> bytes:
        """
        Compile PTX template to PTX assembly with correct version.

        Args:
            src: PTX template with TARGET and VERSION placeholders

        Returns:
            PTX assembly bytes with architecture and version set

        PTX Version Selection Logic:
            - Blackwell (sm_110+): 8.5 (PTX ISA 9.0 features)
            - Hopper (sm_89+): 7.8 (PTX ISA 8.x features)
            - Ampere/Turing (sm_75+): 7.5 (PTX ISA 7.5 features)

        Blackwell-Specific Features in PTX 8.5:
            - Enhanced tensor core instructions
            - Improved memory access patterns
            - Compute data compression support
            - Structured sparsity instructions
        """
        # Select PTX version based on architecture
        if self.arch >= "sm_110":
            # Blackwell requires PTX ISA 8.5 (included in CUDA 13.0)
            version = "8.5"
            print(f"[PTX ENHANCED] Using PTX {version} for Blackwell {self.arch}")
        elif self.arch >= "sm_89":
            # Hopper requires PTX ISA 7.8
            version = "7.8"
            print(f"[PTX ENHANCED] Using PTX {version} for Hopper {self.arch}")
        else:
            # Older architectures use PTX ISA 7.5
            version = "7.5"
            print(f"[PTX ENHANCED] Using PTX {version} for legacy {self.arch}")

        # Replace placeholders in PTX template
        ptx_assembly = src.replace("TARGET", self.arch).replace("VERSION", version)

        # Validate PTX format (should start with .version)
        if not ptx_assembly.startswith(".version"):
            raise ValueError(f"Invalid PTX format: expected '.version', got: {ptx_assembly[:50]}")

        return ptx_assembly.encode()

    def disassemble(self, lib: bytes):
        """Disassemble CUBIN to PTX/SASS using nvdisasm"""
        from tinygrad.runtime.support.compiler_cuda import cuda_disassemble
        cuda_disassemble(lib, self.arch)


class EnhancedNVPTXCompiler(NVPTXCompiler):
    """
    Enhanced NVPTXCompiler using EnhancedPTXCompiler.

    This compiler:
    1. Uses EnhancedPTXCompiler for PTX generation (with Blackwell PTX 8.5)
    2. Links PTX → CUBIN via nvJitLink (CUDA 13.0 compatible)
    3. No NVRTC dependency (removed in CUDA 13.0)

    Compilation Pipeline:
        PTX Template → EnhancedPTXCompiler → PTX Assembly → nvJitLink → CUBIN
    """

    def __init__(self, arch: str):
        # Initialize with enhanced PTX compiler
        self.arch = arch
        self.ptx_compiler = EnhancedPTXCompiler(arch)
        Compiler.__init__(self, f"compile_nv_ptx_enhanced_{self.arch}")

    def compile(self, src: str) -> bytes:
        """
        Complete compilation pipeline: PTX Template → CUBIN

        Args:
            src: PTX template from PTXRenderer

        Returns:
            CUBIN binary ready for cuModuleLoadData

        Pipeline:
            1. EnhancedPTXCompiler: Template → PTX Assembly (with correct version)
            2. nvJitLink: PTX Assembly → CUBIN (via nvJitLinkAddData)
            3. Extract linked CUBIN from nvJitLink

        Error Handling:
            - Validates PTX format before passing to nvJitLink
            - Provides detailed error logs on compilation failure
            - Uses jitlink_check for all nvJitLink API calls
        """
        # Step 1: Compile PTX template to PTX assembly
        print(f"[NVPTX ENHANCED] Compiling PTX template for {self.arch}")
        ptx_assembly = self.ptx_compiler.compile(src)

        # Debug: Show first 200 bytes of PTX (verify format)
        print(f"[NVPTX ENHANCED] PTX assembly first 200 bytes: {ptx_assembly[:200]}")

        # Validate PTX format
        if not ptx_assembly.startswith(b".version"):
            raise ValueError(f"EnhancedPTXCompiler produced invalid PTX: {ptx_assembly[:100]}")

        # Step 2: Create nvJitLink handle
        handle = nvrtc.nvJitLinkHandle()
        jitlink_check(
            nvrtc.nvJitLinkCreate(
                handle,
                1,  # Number of options
                to_char_p_p([f'-arch={self.arch}'.encode()])
            ),
            handle
        )

        try:
            # Step 3: Add PTX assembly to linker
            jitlink_check(
                nvrtc.nvJitLinkAddData(
                    handle,
                    nvrtc.NVJITLINK_INPUT_PTX,  # Type: PTX assembly (correct!)
                    ptx_assembly,                # PTX bytes from EnhancedPTXCompiler
                    len(ptx_assembly),           # Length
                    "<null>".encode()            # Name
                ),
                handle
            )

            # Step 4: Complete linking (PTX → CUBIN)
            jitlink_check(nvrtc.nvJitLinkComplete(handle), handle)

            # Step 5: Extract CUBIN
            cubin = _get_bytes(
                handle,
                nvrtc.nvJitLinkGetLinkedCubin,
                nvrtc.nvJitLinkGetLinkedCubinSize,
                jitlink_check
            )

            print(f"[NVPTX ENHANCED] Successfully compiled CUBIN: {len(cubin)} bytes")

            return cubin

        finally:
            # Cleanup nvJitLink handle
            jitlink_check(nvrtc.nvJitLinkDestroy(handle))


def apply_enhancements():
    """
    Apply enhanced compilers to tinygrad runtime.

    This function monkey-patches tinygrad's PTXCompiler and NVPTXCompiler
    with enhanced versions that support Blackwell PTX 8.5.

    Usage:
        import os
        os.environ['PTX'] = '1'  # MUST be set BEFORE tinygrad imports

        # After tinygrad imports but before device initialization
        from ptx_renderer_enhanced import apply_enhancements
        apply_enhancements()

    Effects:
        - PTXCompiler → EnhancedPTXCompiler (PTX 8.5 for Blackwell)
        - NVPTXCompiler → EnhancedNVPTXCompiler (uses enhanced version)
        - All existing code continues to work (backward compatible)
    """
    import tinygrad.runtime.support.compiler_cuda as cuda_compiler

    # Monkey-patch with enhanced versions
    cuda_compiler.PTXCompiler = EnhancedPTXCompiler
    cuda_compiler.NVPTXCompiler = EnhancedNVPTXCompiler

    print("[PTX ENHANCEMENTS] Applied Blackwell PTX 8.5 support")
    print("[PTX ENHANCEMENTS] PTXCompiler → EnhancedPTXCompiler")
    print("[PTX ENHANCEMENTS] NVPTXCompiler → EnhancedNVPTXCompiler")


# Convenience: Allow direct import and patch
if __name__ != "__main__":
    # Module imported, not executed
    # User can call apply_enhancements() manually
    pass
