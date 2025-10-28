#!/usr/bin/env python3
"""
CUDA Capability Detection for Hybrid Compiler

Detects:
- CUDA version
- Compute capability
- Available compilation tools (nvcc, NVRTC, nvJitLink)
- Optimal compilation path recommendation

Author: Solution Agent 5
Date: 2025-10-23
"""

import subprocess
import ctypes
import ctypes.util
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List


class CUDACapability:
    """Detect CUDA environment capabilities"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.capabilities = {}

    def detect_all(self) -> Dict[str, Any]:
        """Run all detection checks"""
        self.capabilities = {
            'cuda_version': self._detect_cuda_version(),
            'compute_capability': self._detect_compute_capability(),
            'nvcc': self._detect_nvcc(),
            'nvrtc': self._detect_nvrtc(),
            'nvjitlink': self._detect_nvjitlink(),
            'ptx_isa': self._detect_ptx_isa(),
            'architecture': self._detect_architecture(),
            'recommendations': self._generate_recommendations()
        }

        return self.capabilities

    def _log(self, message: str):
        """Print log message if verbose"""
        if self.verbose:
            print(f"[DETECT] {message}", file=sys.stderr)

    def _detect_cuda_version(self) -> Dict[str, Any]:
        """Detect CUDA toolkit version"""
        result = {
            'detected': False,
            'version': None,
            'method': None
        }

        # Method 1: nvcc --version
        try:
            proc = subprocess.run(
                ['nvcc', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if proc.returncode == 0:
                # Parse version from output
                # Example: "Cuda compilation tools, release 13.0, V13.0.48"
                for line in proc.stdout.split('\n'):
                    if 'release' in line.lower():
                        parts = line.split(',')
                        for part in parts:
                            if 'release' in part.lower():
                                version = part.split()[-1]
                                result['detected'] = True
                                result['version'] = version
                                result['method'] = 'nvcc'
                                self._log(f"CUDA version: {version} (via nvcc)")
                                return result

        except Exception as e:
            self._log(f"nvcc version check failed: {e}")

        # Method 2: nvidia-smi
        try:
            proc = subprocess.run(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if proc.returncode == 0:
                driver_version = proc.stdout.strip()
                result['detected'] = True
                result['version'] = f"driver_{driver_version}"
                result['method'] = 'nvidia-smi'
                self._log(f"CUDA driver version: {driver_version}")
                return result

        except Exception as e:
            self._log(f"nvidia-smi version check failed: {e}")

        self._log("CUDA version detection failed")
        return result

    def _detect_compute_capability(self) -> Dict[str, Any]:
        """Detect GPU compute capability"""
        result = {
            'detected': False,
            'capability': None,
            'arch': None,
            'gpu_name': None
        }

        # Method 1: nvidia-smi
        try:
            proc = subprocess.run(
                ['nvidia-smi', '--query-gpu=compute_cap,name', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if proc.returncode == 0:
                line = proc.stdout.strip().split(',')
                if len(line) >= 2:
                    cap = line[0].strip()
                    name = line[1].strip()

                    # Convert to sm_XXX format
                    cap_parts = cap.split('.')
                    if len(cap_parts) == 2:
                        major, minor = cap_parts
                        arch = f"sm_{major}{minor}"

                        result['detected'] = True
                        result['capability'] = cap
                        result['arch'] = arch
                        result['gpu_name'] = name

                        self._log(f"Compute capability: {cap} ({arch}) - {name}")
                        return result

        except Exception as e:
            self._log(f"nvidia-smi compute capability check failed: {e}")

        # Method 2: Try to detect via CUDA API (requires pycuda or similar)
        try:
            import ctypes
            cuda = ctypes.CDLL('libcuda.so.1')

            # cuInit
            result_code = cuda.cuInit(0)
            if result_code == 0:
                # cuDeviceGetCount
                device_count = ctypes.c_int()
                cuda.cuDeviceGetCount(ctypes.byref(device_count))

                if device_count.value > 0:
                    # cuDeviceGet
                    device = ctypes.c_int()
                    cuda.cuDeviceGet(ctypes.byref(device), 0)

                    # cuDeviceGetAttribute (compute capability major/minor)
                    major = ctypes.c_int()
                    minor = ctypes.c_int()

                    # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
                    # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
                    cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, device)
                    cuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, device)

                    cap = f"{major.value}.{minor.value}"
                    arch = f"sm_{major.value}{minor.value}"

                    result['detected'] = True
                    result['capability'] = cap
                    result['arch'] = arch
                    result['gpu_name'] = "Unknown (via CUDA API)"

                    self._log(f"Compute capability: {cap} ({arch}) via CUDA API")
                    return result

        except Exception as e:
            self._log(f"CUDA API compute capability check failed: {e}")

        self._log("Compute capability detection failed")
        return result

    def _detect_nvcc(self) -> Dict[str, Any]:
        """Detect nvcc compiler"""
        result = {
            'available': False,
            'path': None,
            'version': None,
            'supported_archs': []
        }

        # Check if nvcc is in PATH
        try:
            which_proc = subprocess.run(
                ['which', 'nvcc'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if which_proc.returncode == 0:
                nvcc_path = which_proc.stdout.strip()
                result['available'] = True
                result['path'] = nvcc_path

                # Get version
                version_proc = subprocess.run(
                    ['nvcc', '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if version_proc.returncode == 0:
                    for line in version_proc.stdout.split('\n'):
                        if 'release' in line.lower():
                            result['version'] = line.strip()

                # Get supported architectures
                help_proc = subprocess.run(
                    ['nvcc', '--help'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if help_proc.returncode == 0:
                    # Parse --gpu-architecture options
                    in_arch_section = False
                    for line in help_proc.stdout.split('\n'):
                        if '--gpu-architecture' in line:
                            in_arch_section = True
                        elif in_arch_section and 'sm_' in line:
                            # Extract sm_XXX
                            words = line.strip().split()
                            for word in words:
                                if word.startswith('sm_'):
                                    result['supported_archs'].append(word)

                self._log(f"nvcc available: {nvcc_path}")
                self._log(f"  Supported archs: {', '.join(result['supported_archs'][:5])}...")
                return result

        except Exception as e:
            self._log(f"nvcc detection failed: {e}")

        self._log("nvcc not available")
        return result

    def _detect_nvrtc(self) -> Dict[str, Any]:
        """Detect NVRTC library"""
        result = {
            'available': False,
            'library_path': None,
            'version': None
        }

        # Try to find libnvrtc.so
        try:
            lib_path = ctypes.util.find_library('nvrtc')
            if lib_path:
                result['available'] = True
                result['library_path'] = lib_path

                # Try to load and get version
                try:
                    nvrtc = ctypes.CDLL(lib_path)

                    # Try nvrtcVersion
                    major = ctypes.c_int()
                    minor = ctypes.c_int()

                    ret = nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))

                    if ret == 0:
                        result['version'] = f"{major.value}.{minor.value}"
                        self._log(f"NVRTC available: {lib_path}")
                        self._log(f"  Version: {result['version']}")
                    else:
                        self._log(f"NVRTC library found but version check failed: {ret}")

                except Exception as e:
                    self._log(f"NVRTC library found but loading failed: {e}")
                    result['available'] = False

                return result

        except Exception as e:
            self._log(f"NVRTC detection failed: {e}")

        # Manual search in common locations
        common_paths = [
            '/usr/local/cuda/lib64/libnvrtc.so',
            '/usr/local/cuda/targets/sbsa-linux/lib/libnvrtc.so',
            '/usr/lib/x86_64-linux-gnu/libnvrtc.so'
        ]

        for path in common_paths:
            if Path(path).exists():
                result['available'] = True
                result['library_path'] = path
                self._log(f"NVRTC found at: {path}")
                return result

        self._log("NVRTC not available")
        return result

    def _detect_nvjitlink(self) -> Dict[str, Any]:
        """Detect nvJitLink library"""
        result = {
            'available': False,
            'library_path': None
        }

        # Try to find libnvJitLink.so
        try:
            lib_path = ctypes.util.find_library('nvJitLink')
            if lib_path:
                result['available'] = True
                result['library_path'] = lib_path
                self._log(f"nvJitLink available: {lib_path}")
                return result

        except Exception as e:
            self._log(f"nvJitLink detection failed: {e}")

        # Manual search in common locations
        common_paths = [
            '/usr/local/cuda/lib64/libnvJitLink.so',
            '/usr/local/cuda/targets/sbsa-linux/lib/libnvJitLink.so',
            '/usr/lib/x86_64-linux-gnu/libnvJitLink.so'
        ]

        for path in common_paths:
            if Path(path).exists():
                result['available'] = True
                result['library_path'] = path
                self._log(f"nvJitLink found at: {path}")
                return result

        self._log("nvJitLink not available")
        return result

    def _detect_ptx_isa(self) -> Dict[str, Any]:
        """Detect supported PTX ISA version"""
        result = {
            'supported_versions': [],
            'recommended': None
        }

        # PTX ISA versions by CUDA version
        cuda_ptx_map = {
            '13.0': '9.0',
            '12.9': '8.5',
            '12.8': '8.5',
            '12.0': '8.0',
            '11.8': '7.8',
            '11.0': '7.0'
        }

        cuda_version = self.capabilities.get('cuda_version', {}).get('version')
        if cuda_version:
            # Extract major.minor
            if 'V' in cuda_version:
                version = cuda_version.split('V')[1].split('.')[0:2]
                version_str = '.'.join(version)
            else:
                version_str = cuda_version.split('.')[0:2]
                version_str = '.'.join(version_str)

            if version_str in cuda_ptx_map:
                result['recommended'] = cuda_ptx_map[version_str]
                result['supported_versions'] = [
                    v for k, v in cuda_ptx_map.items()
                    if float(k) <= float(version_str)
                ]

                self._log(f"PTX ISA: Recommended {result['recommended']} "
                          f"for CUDA {version_str}")

        return result

    def _detect_architecture(self) -> Dict[str, Any]:
        """Detect hardware architecture"""
        result = {
            'platform': None,
            'machine': None
        }

        try:
            # Get platform (Linux, Darwin, etc.)
            platform_proc = subprocess.run(
                ['uname', '-s'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if platform_proc.returncode == 0:
                result['platform'] = platform_proc.stdout.strip()

            # Get machine (x86_64, aarch64, etc.)
            machine_proc = subprocess.run(
                ['uname', '-m'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if machine_proc.returncode == 0:
                result['machine'] = machine_proc.stdout.strip()

            self._log(f"Platform: {result['platform']} / {result['machine']}")

        except Exception as e:
            self._log(f"Architecture detection failed: {e}")

        return result

    def _generate_recommendations(self) -> Dict[str, Any]:
        """Generate compilation path recommendations based on capabilities"""
        recommendations = {
            'optimal_path': None,
            'available_paths': [],
            'warnings': [],
            'required_fixes': []
        }

        # Check each path's viability
        if self.capabilities.get('nvcc', {}).get('available'):
            recommendations['available_paths'].append({
                'path': 'nvcc',
                'priority': 3,
                'reason': 'Most robust, guaranteed correctness'
            })
            if recommendations['optimal_path'] is None:
                recommendations['optimal_path'] = 'nvcc'

        if self.capabilities.get('nvrtc', {}).get('available'):
            recommendations['available_paths'].append({
                'path': 'nvrtc',
                'priority': 2,
                'reason': 'Official NVIDIA API, good performance'
            })
            if recommendations['optimal_path'] is None:
                recommendations['optimal_path'] = 'nvrtc'

        # PTX=1 path is always available (string replacement)
        recommendations['available_paths'].append({
            'path': 'ptx1',
            'priority': 1,
            'reason': 'Fastest, uses tinygrad built-in PTX generation'
        })
        if recommendations['optimal_path'] is None:
            recommendations['optimal_path'] = 'ptx1'

        # Generate warnings
        if not self.capabilities.get('cuda_version', {}).get('detected'):
            recommendations['warnings'].append(
                "CUDA version could not be detected"
            )

        if not self.capabilities.get('compute_capability', {}).get('detected'):
            recommendations['warnings'].append(
                "GPU compute capability could not be detected"
            )
            recommendations['required_fixes'].append(
                "Install nvidia-smi or verify GPU is accessible"
            )

        if not self.capabilities.get('nvjitlink', {}).get('available'):
            recommendations['warnings'].append(
                "nvJitLink library not found - required for all paths"
            )
            recommendations['required_fixes'].append(
                "Install CUDA 13.0+ toolkit with nvJitLink support"
            )

        # Architecture-specific recommendations
        compute_cap = self.capabilities.get('compute_capability', {})
        if compute_cap.get('detected'):
            cap_value = float(compute_cap['capability'])

            if cap_value >= 11.0:
                recommendations['warnings'].append(
                    f"Blackwell GPU detected (compute {compute_cap['capability']})"
                )
                recommendations['warnings'].append(
                    "Ensure CUDA 13.0+ and PTX ISA 9.0 support"
                )

        return recommendations

    def print_report(self):
        """Print human-readable capability report"""
        if not self.capabilities:
            self.detect_all()

        print("\n" + "=" * 80)
        print("CUDA CAPABILITY DETECTION REPORT")
        print("=" * 80)

        # CUDA Version
        cuda = self.capabilities['cuda_version']
        print(f"\nCUDA Version:")
        print(f"  Detected: {cuda['detected']}")
        if cuda['detected']:
            print(f"  Version: {cuda['version']}")
            print(f"  Method: {cuda['method']}")

        # Compute Capability
        compute = self.capabilities['compute_capability']
        print(f"\nCompute Capability:")
        print(f"  Detected: {compute['detected']}")
        if compute['detected']:
            print(f"  Capability: {compute['capability']}")
            print(f"  Architecture: {compute['arch']}")
            print(f"  GPU: {compute['gpu_name']}")

        # nvcc
        nvcc = self.capabilities['nvcc']
        print(f"\nnvcc Compiler:")
        print(f"  Available: {nvcc['available']}")
        if nvcc['available']:
            print(f"  Path: {nvcc['path']}")
            print(f"  Version: {nvcc['version']}")
            if nvcc['supported_archs']:
                print(f"  Supported archs: {', '.join(nvcc['supported_archs'][:10])}")

        # NVRTC
        nvrtc = self.capabilities['nvrtc']
        print(f"\nNVRTC Library:")
        print(f"  Available: {nvrtc['available']}")
        if nvrtc['available']:
            print(f"  Path: {nvrtc['library_path']}")
            if nvrtc['version']:
                print(f"  Version: {nvrtc['version']}")

        # nvJitLink
        nvjitlink = self.capabilities['nvjitlink']
        print(f"\nnvJitLink Library:")
        print(f"  Available: {nvjitlink['available']}")
        if nvjitlink['available']:
            print(f"  Path: {nvjitlink['library_path']}")

        # PTX ISA
        ptx = self.capabilities['ptx_isa']
        print(f"\nPTX ISA:")
        if ptx['recommended']:
            print(f"  Recommended: {ptx['recommended']}")
            print(f"  Supported: {', '.join(ptx['supported_versions'])}")

        # Architecture
        arch = self.capabilities['architecture']
        print(f"\nPlatform:")
        print(f"  OS: {arch['platform']}")
        print(f"  Machine: {arch['machine']}")

        # Recommendations
        rec = self.capabilities['recommendations']
        print(f"\nRecommendations:")
        print(f"  Optimal path: {rec['optimal_path']}")
        print(f"  Available paths:")
        for path in sorted(rec['available_paths'], key=lambda x: x['priority'], reverse=True):
            print(f"    - {path['path']} (priority {path['priority']}): {path['reason']}")

        if rec['warnings']:
            print(f"\n  Warnings:")
            for warning in rec['warnings']:
                print(f"    ⚠️  {warning}")

        if rec['required_fixes']:
            print(f"\n  Required fixes:")
            for fix in rec['required_fixes']:
                print(f"    ❌ {fix}")

        print("\n" + "=" * 80)

    def save_json(self, filename: str):
        """Save capabilities to JSON file"""
        if not self.capabilities:
            self.detect_all()

        with open(filename, 'w') as f:
            json.dump(self.capabilities, f, indent=2)

        print(f"\n[DETECT] Saved capabilities to: {filename}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Detect CUDA capabilities')
    parser.add_argument('--json', help='Save report to JSON file')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    args = parser.parse_args()

    detector = CUDACapability(verbose=not args.quiet)
    detector.detect_all()
    detector.print_report()

    if args.json:
        detector.save_json(args.json)
