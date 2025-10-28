"""
nvJitLink Direct Compiler (EXPERIMENTAL)

Attempts to pass CUDA C source directly to nvJitLink.
This may not be supported - research needed to confirm if nvJitLink
accepts NVJITLINK_INPUT_CUDA or similar type.

Based on Solution C from Agent 10 research.
"""


class NVJitLinkDirectCompiler:
    """
    EXPERIMENTAL: Try to pass CUDA C directly to nvJitLink
    """

    def __init__(self, arch: str, config: dict, logger):
        self.arch = arch
        self.config = config
        self.logger = logger

        self.logger.warning("nvJitLink Direct is EXPERIMENTAL - may not be supported")

    def compile(self, src: str) -> bytes:
        """
        Attempt to pass CUDA C directly to nvJitLink

        This is experimental and may fail if nvJitLink doesn't support CUDA C input.
        """
        try:
            import ctypes
            from tinygrad.helpers import to_char_p_p
            import tinygrad.runtime.autogen.nvrtc as nvrtc
            from tinygrad.runtime.support.compiler_cuda import jitlink_check, _get_bytes

            handle = nvrtc.nvJitLinkHandle()
            jitlink_check(nvrtc.nvJitLinkCreate(
                handle, 1,
                to_char_p_p([f'-arch={self.arch}'.encode()])
            ), handle)

            # Try different input types to see what works
            # NVJITLINK_INPUT_CUDA doesn't exist in current API, but we try INPUT_ANY
            input_type = getattr(nvrtc, 'NVJITLINK_INPUT_ANY', 10)

            self.logger.debug(f"Trying nvJitLink with input type: {input_type}")

            jitlink_check(nvrtc.nvJitLinkAddData(
                handle,
                input_type,
                src.encode(),
                len(src),
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

            self.logger.info(f"nvJitLink Direct succeeded! CUBIN size: {len(cubin)} bytes")
            return cubin

        except Exception as e:
            self.logger.error(f"nvJitLink Direct failed (as expected): {e}")
            self.logger.info("Falling back to nvcc subprocess...")
            # Fall back to nvcc
            from .nvcc_subprocess import NVCCSubprocessCompiler
            fallback = NVCCSubprocessCompiler(self.arch, self.config, self.logger)
            return fallback.compile(src)

    @classmethod
    def is_available(cls) -> bool:
        """Always claim available - will fail at runtime if not supported"""
        return True  # We'll try it and fall back if it fails

    def __str__(self):
        return f"NVJitLinkDirectCompiler(arch={self.arch}, EXPERIMENTAL)"
