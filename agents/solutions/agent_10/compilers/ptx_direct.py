"""
PTX Direct Compiler (PTX=1 Path)

Uses tinygrad's PTXRenderer + PTXCompiler path by setting PTX=1 environment variable.
This is the simplest path but relies on string replacement for PTX generation.

Pros: No subprocess overhead, simplest implementation
Cons: PTXCompiler only does string substitution (TARGET, VERSION)
"""

import os


class PTXDirectCompiler:
    """
    PTX Direct path: Forces tinygrad to use PTXRenderer + PTXCompiler
    """

    def __init__(self, arch: str, config: dict, logger):
        self.arch = arch
        self.config = config
        self.logger = logger

        # Force PTX=1 environment variable
        os.environ['PTX'] = '1'
        self.logger.info("PTX Direct: Set PTX=1 environment variable")

    def compile(self, src: str) -> bytes:
        """
        This compiler doesn't actually compile - it just ensures PTX=1 is set.
        Tinygrad's PTXCompiler will handle the actual (string replacement) compilation.
        """
        # Log PTX version being used
        ptx_version = self.config.get('ptx_version', '8.5')
        self.logger.debug(f"PTX Direct using version: {ptx_version}")

        # In reality, tinygrad's PTXCompiler will be called directly
        # This class exists to configure the environment
        raise NotImplementedError("PTX Direct uses tinygrad's PTXCompiler directly")

    @classmethod
    def is_available(cls) -> bool:
        """PTX direct is always available (uses environment variable)"""
        return True

    def __str__(self):
        return f"PTXDirectCompiler(arch={self.arch}, PTX=1)"
