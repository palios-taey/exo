"""
Compiler implementations for configuration-based selector

Provides 4 compilation paths:
1. PTX Direct (PTX=1 environment variable)
2. nvcc Subprocess (CUDA C → nvcc → PTX → nvJitLink)
3. NVRTC Wrapper (CUDA C → NVRTC → PTX → nvJitLink)
4. nvJitLink Direct (experimental)
"""

from .ptx_direct import PTXDirectCompiler
from .nvcc_subprocess import NVCCSubprocessCompiler
from .nvrtc_wrapper import NVRTCWrapperCompiler
from .nvjitlink_direct import NVJitLinkDirectCompiler


def get_compiler(compiler_name: str, arch: str, config: dict, logger):
    """Factory function to get compiler instance"""

    if compiler_name == 'ptx':
        return PTXDirectCompiler(arch, config, logger)
    elif compiler_name == 'nvcc':
        return NVCCSubprocessCompiler(arch, config, logger)
    elif compiler_name == 'nvrtc':
        return NVRTCWrapperCompiler(arch, config, logger)
    elif compiler_name == 'direct':
        return NVJitLinkDirectCompiler(arch, config, logger)
    else:
        raise ValueError(f"Unknown compiler: {compiler_name}")


__all__ = [
    'get_compiler',
    'PTXDirectCompiler',
    'NVCCSubprocessCompiler',
    'NVRTCWrapperCompiler',
    'NVJitLinkDirectCompiler'
]
