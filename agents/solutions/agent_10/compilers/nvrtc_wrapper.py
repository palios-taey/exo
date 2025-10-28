"""
NVRTC Wrapper Compiler

Uses NVRTC runtime compilation (if available) to compile CUDA C → PTX,
then links PTX → CUBIN using nvJitLink.

Note: NVRTC removed in CUDA 13.0, so this is primarily for CUDA <13.0 environments.
"""


class NVRTCWrapperCompiler:
    """
    Compile CUDA C → PTX → CUBIN via NVRTC + nvJitLink
    """

    def __init__(self, arch: str, config: dict, logger):
        self.arch = arch
        self.config = config
        self.logger = logger

        # Check NVRTC availability
        if not self.is_available():
            raise RuntimeError("NVRTC library not available")

    def compile(self, src: str) -> bytes:
        """
        Complete pipeline: CUDA C → PTX (NVRTC) → CUBIN (nvJitLink)
        """
        # Import NVRTC APIs
        import ctypes
        from tinygrad.helpers import to_char_p_p
        import tinygrad.runtime.autogen.nvrtc as nvrtc
        from tinygrad.runtime.support.compiler_cuda import jitlink_check, _get_bytes, nvrtc_check

        # Step 1: Create NVRTC program
        prog = nvrtc.nvrtcProgram()
        nvrtc_check(nvrtc.nvrtcCreateProgram(
            ctypes.byref(prog),
            src.encode(),
            b"<kernel>",
            0, None, None
        ))

        # Step 2: Compile CUDA C → PTX
        compile_opts = [
            f'--gpu-architecture={self.arch}'.encode(),
        ]

        if self.config.get('use_fast_math', True):
            compile_opts.append(b'--use_fast_math')

        if self.config.get('debug_symbols', False):
            compile_opts.append(b'-lineinfo')

        nvrtc_check(nvrtc.nvrtcCompileProgram(
            prog,
            len(compile_opts),
            to_char_p_p(compile_opts)
        ))

        # Step 3: Get PTX
        ptx_size = ctypes.c_size_t()
        nvrtc_check(nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size)))

        ptx = ctypes.create_string_buffer(ptx_size.value)
        nvrtc_check(nvrtc.nvrtcGetPTX(prog, ptx))

        # Cleanup NVRTC program
        nvrtc_check(nvrtc.nvrtcDestroyProgram(ctypes.byref(prog)))

        if self.config.get('log_ptx_output', False):
            self.logger.debug(f"PTX from NVRTC (first 500 bytes): {ptx.raw[:500]}")

        self.logger.info(f"NVRTC compilation successful. PTX size: {ptx_size.value} bytes")

        # Step 4: Link PTX → CUBIN using nvJitLink
        handle = nvrtc.nvJitLinkHandle()
        jitlink_check(nvrtc.nvJitLinkCreate(
            handle, 1,
            to_char_p_p([f'-arch={self.arch}'.encode()])
        ), handle)

        jitlink_check(nvrtc.nvJitLinkAddData(
            handle,
            nvrtc.NVJITLINK_INPUT_PTX,
            ptx.raw,
            ptx_size.value,
            b"<kernel>"
        ), handle)

        jitlink_check(nvrtc.nvJitLinkComplete(handle), handle)

        cubin = _get_bytes(
            handle,
            nvrtc.nvJitLinkGetLinkedCubin,
            nvrtc.nvJitLinkGetLinkedCubinSize,
            jitlink_check
        )

        jitlink_check(nvrtc.nvJitLinkDestroy(handle))

        if self.config.get('log_cubin_size', True):
            self.logger.info(f"nvJitLink successful. CUBIN size: {len(cubin)} bytes")

        return cubin

    @classmethod
    def is_available(cls) -> bool:
        """Check if NVRTC library is available"""
        try:
            import ctypes.util
            lib_path = ctypes.util.find_library('nvrtc')
            return lib_path is not None
        except:
            return False

    def __str__(self):
        return f"NVRTCWrapperCompiler(arch={self.arch})"
