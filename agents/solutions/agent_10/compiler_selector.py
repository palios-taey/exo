#!/usr/bin/env python3
"""
Configuration-Based Compiler Selector for CUDA 13.0 + Blackwell

This module provides intelligent compiler selection for tinygrad's CUDA backend,
supporting multiple compilation paths with user configuration and auto-detection.

Author: Solution Agent 10
Date: 2025-10-23
"""

import os
import sys
import yaml
import logging
import subprocess
import socket
from typing import Optional, Dict, List, Any
from pathlib import Path


class CompilerConfig:
    """Load and validate compiler configuration"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self._validate_config()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        # Search paths for config file
        search_paths = [
            config_path,
            os.environ.get('COMPILER_CONFIG_PATH'),
            os.path.join(os.getcwd(), 'compiler_config.yaml'),
            os.path.join(os.path.dirname(__file__), 'compiler_config.yaml'),
            os.path.expanduser('~/exo/compiler_config.yaml'),
        ]

        for path in search_paths:
            if path and os.path.exists(path):
                with open(path, 'r') as f:
                    return yaml.safe_load(f)

        # Default config if no file found
        return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'compiler': {
                'preferred': 'auto',
                'architecture': 'sm_110',
                'fallback': ['nvcc', 'ptx', 'nvrtc']
            },
            'detection': {
                'check_nvcc': True,
                'check_nvrtc': True,
                'cuda_version_min': '13.0',
                'verify_architecture': True,
                'fail_on_no_compiler': True
            },
            'compilation': {
                'ptx_version': '8.5',
                'optimization_level': 3,
                'use_fast_math': True,
                'debug_symbols': False,
                'subprocess_timeout': 30
            },
            'logging': {
                'level': 'INFO',
                'log_compilation': True,
                'log_selection': True,
                'log_ptx_output': False,
                'log_cubin_size': True,
                'log_file': '/tmp/compiler_selector.log'
            }
        }

    def _validate_config(self):
        """Validate configuration values"""
        valid_compilers = ['auto', 'ptx', 'nvcc', 'nvrtc', 'direct']
        preferred = self.config['compiler']['preferred']

        if preferred not in valid_compilers:
            raise ValueError(f"Invalid compiler.preferred: {preferred}. "
                           f"Must be one of {valid_compilers}")

        # Validate architecture format (e.g., sm_110)
        arch = self.config['compiler']['architecture']
        if not arch.startswith('sm_'):
            raise ValueError(f"Invalid architecture format: {arch}. "
                           f"Must start with 'sm_' (e.g., sm_110)")

    def get_device_config(self) -> Dict[str, Any]:
        """Get device-specific configuration override"""
        if 'devices' not in self.config:
            return self.config

        # Get current device IP or hostname
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)

        # Check for device-specific override
        devices = self.config.get('devices', {})
        if ip_address in devices:
            # Merge device config with defaults
            device_config = self.config.copy()
            device_config.update(devices[ip_address])
            return device_config
        elif hostname in devices:
            device_config = self.config.copy()
            device_config.update(devices[hostname])
            return device_config
        elif 'default' in devices:
            device_config = self.config.copy()
            device_config.update(devices['default'])
            return device_config

        return self.config


class EnvironmentDetector:
    """Detect available compilers and CUDA environment"""

    def __init__(self, config: CompilerConfig):
        self.config = config.config
        self.logger = logging.getLogger(__name__)

    def detect_cuda_version(self) -> Optional[str]:
        """Detect CUDA version from nvcc or nvidia-smi"""
        try:
            result = subprocess.run(['nvcc', '--version'],
                                  capture_output=True, text=True,
                                  timeout=5)
            if result.returncode == 0:
                # Parse: "Cuda compilation tools, release 13.0, V13.0.48"
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        version = line.split('release')[1].split(',')[0].strip()
                        self.logger.info(f"Detected CUDA version: {version}")
                        return version
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Try nvidia-smi as fallback
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version',
                                   '--format=csv,noheader'],
                                  capture_output=True, text=True,
                                  timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.logger.info(f"Detected CUDA driver version: {version}")
                return version
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        self.logger.warning("Could not detect CUDA version")
        return None

    def check_nvcc(self) -> bool:
        """Check if nvcc is available"""
        if not self.config['detection']['check_nvcc']:
            return False

        try:
            result = subprocess.run(['which', 'nvcc'],
                                  capture_output=True, text=True,
                                  timeout=5)
            available = result.returncode == 0
            if available:
                self.logger.info(f"nvcc found at: {result.stdout.strip()}")
            else:
                self.logger.warning("nvcc not found in PATH")
            return available
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.logger.warning("Could not check for nvcc")
            return False

    def check_nvrtc(self) -> bool:
        """Check if NVRTC library is available"""
        if not self.config['detection']['check_nvrtc']:
            return False

        try:
            import ctypes.util
            lib_path = ctypes.util.find_library('nvrtc')
            if lib_path:
                self.logger.info(f"NVRTC library found: {lib_path}")
                return True
            else:
                self.logger.warning("NVRTC library not found")
                return False
        except Exception as e:
            self.logger.warning(f"Could not check for NVRTC: {e}")
            return False

    def verify_architecture(self, arch: str) -> bool:
        """Verify architecture matches actual GPU"""
        if not self.config['detection']['verify_architecture']:
            return True

        try:
            result = subprocess.run(['nvidia-smi',
                                   '--query-gpu=compute_cap',
                                   '--format=csv,noheader'],
                                  capture_output=True, text=True,
                                  timeout=5)
            if result.returncode == 0:
                compute_cap = result.stdout.strip().replace('.', '')
                expected_arch = f"sm_{compute_cap}"
                if expected_arch == arch:
                    self.logger.info(f"Architecture verified: {arch}")
                    return True
                else:
                    self.logger.warning(f"Architecture mismatch: config={arch}, "
                                      f"actual={expected_arch}")
                    return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.logger.warning("Could not verify architecture")
            return True  # Don't fail if we can't verify

        return True

    def detect_best_compiler(self) -> str:
        """Auto-detect best available compiler"""
        cuda_version = self.detect_cuda_version()
        has_nvcc = self.check_nvcc()
        has_nvrtc = self.check_nvrtc()

        # Decision logic
        if cuda_version and cuda_version >= "13.0":
            # CUDA 13.0+: NVRTC removed, prefer nvcc
            if has_nvcc:
                self.logger.info("Auto-detected compiler: nvcc (best for CUDA 13.0+)")
                return "nvcc"
            elif has_nvrtc:
                self.logger.warning("NVRTC detected on CUDA 13.0+ (unexpected)")
                return "nvrtc"
            else:
                self.logger.warning("No compiler detected, falling back to PTX direct")
                return "ptx"
        else:
            # CUDA <13.0: NVRTC preferred
            if has_nvrtc:
                self.logger.info("Auto-detected compiler: nvrtc (best for CUDA <13.0)")
                return "nvrtc"
            elif has_nvcc:
                self.logger.info("Auto-detected compiler: nvcc")
                return "nvcc"
            else:
                self.logger.warning("No compiler detected, falling back to PTX direct")
                return "ptx"


class CompilerSelector:
    """Main compiler selection logic"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = CompilerConfig(config_path)
        self.detector = EnvironmentDetector(self.config)
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging based on configuration"""
        log_config = self.config.config['logging']
        level = getattr(logging, log_config['level'].upper())

        # Configure logging
        logging.basicConfig(
            level=level,
            format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        self.logger = logging.getLogger(__name__)

        # Add file handler if specified
        log_file = log_config.get('log_file')
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(
                logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
            )
            self.logger.addHandler(file_handler)

    def select_compiler(self) -> str:
        """Select compiler based on configuration and environment"""
        device_config = self.config.get_device_config()
        compiler_config = device_config['compiler']
        preferred = compiler_config['preferred']

        self.logger.info(f"Compiler selection started. Preferred: {preferred}")

        # Auto-detect if requested
        if preferred == 'auto':
            selected = self.detector.detect_best_compiler()
        else:
            selected = preferred

        # Verify selected compiler is available
        if not self._verify_compiler_available(selected):
            self.logger.warning(f"Preferred compiler '{selected}' not available. "
                              f"Trying fallback...")
            selected = self._select_fallback(compiler_config['fallback'])

        # Verify architecture if requested
        arch = compiler_config['architecture']
        if not self.detector.verify_architecture(arch):
            self.logger.error(f"Architecture verification failed: {arch}")
            if self.config.config['detection']['fail_on_no_compiler']:
                raise RuntimeError(f"Architecture mismatch: {arch}")

        self.logger.info(f"Compiler selected: {selected} (arch: {arch})")
        return selected

    def _verify_compiler_available(self, compiler: str) -> bool:
        """Verify that selected compiler is actually available"""
        if compiler == 'ptx':
            return True  # PTX direct always available (uses environment variable)
        elif compiler == 'nvcc':
            return self.detector.check_nvcc()
        elif compiler == 'nvrtc':
            return self.detector.check_nvrtc()
        elif compiler == 'direct':
            # Experimental: assume available, will fail at runtime if not
            self.logger.warning("Direct CUDA input is experimental")
            return True
        else:
            self.logger.error(f"Unknown compiler: {compiler}")
            return False

    def _select_fallback(self, fallback_list: List[str]) -> str:
        """Select first available fallback compiler"""
        for fallback in fallback_list:
            if self._verify_compiler_available(fallback):
                self.logger.info(f"Selected fallback compiler: {fallback}")
                return fallback

        # No compiler available
        if self.config.config['detection']['fail_on_no_compiler']:
            raise RuntimeError("No compiler available and fail_on_no_compiler=True")

        self.logger.error("No compiler available! Using PTX direct as last resort")
        return 'ptx'

    def get_compiler_instance(self, compiler: str):
        """Get actual compiler instance for selected path"""
        from compilers import get_compiler

        arch = self.config.config['compiler']['architecture']
        compilation_config = self.config.config['compilation']

        return get_compiler(
            compiler,
            arch,
            compilation_config,
            self.logger
        )


def get_active_compiler() -> str:
    """Convenience function to get currently active compiler"""
    selector = CompilerSelector()
    return selector.select_compiler()


def patch_tinygrad():
    """Patch tinygrad to use selected compiler"""
    selector = CompilerSelector()
    compiler_name = selector.select_compiler()

    # Get compiler instance
    compiler = selector.get_compiler_instance(compiler_name)

    # Patch tinygrad
    if compiler_name == 'ptx':
        # Set environment variable for PTX direct path
        os.environ['PTX'] = '1'
        selector.logger.info("Set PTX=1 for PTX direct path")
    else:
        # Monkey-patch tinygrad's compiler classes
        import tinygrad.runtime.support.compiler_cuda as cuda_compiler

        # Replace NVPTXCompiler with our implementation
        cuda_compiler.NVPTXCompiler = compiler.__class__
        selector.logger.info(f"Patched NVPTXCompiler with {compiler.__class__.__name__}")

    selector.logger.info("Tinygrad patching complete")


if __name__ == '__main__':
    # Test compiler selection
    selector = CompilerSelector()
    compiler = selector.select_compiler()
    print(f"Selected compiler: {compiler}")
