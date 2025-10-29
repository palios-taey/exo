#!/usr/bin/env python3
"""
Test 1: Basic Imports and Environment Validation
Tests that all required modules can be imported and CUDA is available.
"""

import sys
import os
from datetime import datetime

def print_header(message):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {message}")
    print("=" * 70)

def print_success(message):
    """Print success message"""
    print(f"✓ {message}")

def print_failure(message):
    """Print failure message"""
    print(f"✗ {message}")

def print_info(key, value):
    """Print info key-value pair"""
    print(f"  {key}: {value}")

def test_basic_imports():
    """Test basic Python imports"""
    print_header("Testing Basic Python Imports")

    try:
        import numpy as np
        print_success("numpy imported")
        print_info("Version", np.__version__)
    except ImportError as e:
        print_failure(f"numpy failed: {e}")
        return False

    try:
        import asyncio
        print_success("asyncio imported")
    except ImportError as e:
        print_failure(f"asyncio failed: {e}")
        return False

    try:
        import argparse
        print_success("argparse imported")
    except ImportError as e:
        print_failure(f"argparse failed: {e}")
        return False

    return True

def test_exo_imports():
    """Test exo framework imports"""
    print_header("Testing Exo Framework Imports")

    try:
        from exo.orchestration.node import Node
        print_success("exo.orchestration.node imported")
    except ImportError as e:
        print_failure(f"exo.orchestration.node failed: {e}")
        return False

    try:
        from exo.networking.grpc.grpc_server import GRPCServer
        print_success("exo.networking.grpc.grpc_server imported")
    except ImportError as e:
        print_failure(f"exo.networking.grpc.grpc_server failed: {e}")
        return False

    try:
        from exo.networking.udp.udp_discovery import UDPDiscovery
        print_success("exo.networking.udp.udp_discovery imported")
    except ImportError as e:
        print_failure(f"exo.networking.udp.udp_discovery failed: {e}")
        return False

    try:
        from exo.api import ChatGPTAPI
        print_success("exo.api imported")
    except ImportError as e:
        print_failure(f"exo.api failed: {e}")
        return False

    return True

def test_tinygrad_imports():
    """Test tinygrad imports"""
    print_header("Testing Tinygrad Imports")

    try:
        from tinygrad import Device, Tensor
        print_success("tinygrad core imported")
        print_info("Available devices", Device._devices)
    except ImportError as e:
        print_failure(f"tinygrad failed: {e}")
        return False

    try:
        from tinygrad.helpers import getenv
        print_success("tinygrad.helpers imported")
    except ImportError as e:
        print_failure(f"tinygrad.helpers failed: {e}")
        return False

    return True

def test_cuda_availability():
    """Test CUDA availability via tinygrad"""
    print_header("Testing CUDA Availability")

    try:
        from tinygrad import Device

        # Check if CUDA is in available devices
        available_devices = Device._devices
        print_info("Available devices", available_devices)

        if 'CUDA' in available_devices or 'GPU' in available_devices:
            print_success("CUDA device available in tinygrad")

            # Try to access CUDA device
            try:
                cuda_device = Device['CUDA']
                print_success("Successfully accessed CUDA device")
                print_info("CUDA device", cuda_device)
                return True
            except Exception as e:
                print_failure(f"Failed to access CUDA device: {e}")
                return False
        else:
            print_failure("CUDA not in available devices")
            print_info("DEVICE env var", os.getenv('DEVICE', 'not set'))
            print_info("CUDA env var", os.getenv('CUDA', 'not set'))
            return False

    except Exception as e:
        print_failure(f"CUDA availability check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_variables():
    """Test required environment variables"""
    print_header("Testing Environment Variables")

    env_vars = {
        'DEVICE': os.getenv('DEVICE', 'NOT SET'),
        'CUDA_PATH': os.getenv('CUDA_PATH', 'NOT SET'),
        'CUDA_HOME': os.getenv('CUDA_HOME', 'NOT SET'),
        'PATH': os.getenv('PATH', 'NOT SET')[:100] + '...',  # Truncate PATH
        'LD_LIBRARY_PATH': os.getenv('LD_LIBRARY_PATH', 'NOT SET'),
    }

    for key, value in env_vars.items():
        print_info(key, value)

    # Check if DEVICE is set to CUDA
    if os.getenv('DEVICE') == 'CUDA':
        print_success("DEVICE environment variable set to CUDA")
        return True
    else:
        print_failure("DEVICE environment variable not set to CUDA")
        print_info("Suggestion", "Set DEVICE=CUDA before running exo")
        return False

def test_package_versions():
    """Test and report package versions"""
    print_header("Package Versions")

    packages = [
        'numpy',
        'tinygrad',
        'grpcio',
        'transformers',
        'safetensors',
        'accelerate',
    ]

    for package_name in packages:
        try:
            package = __import__(package_name)
            version = getattr(package, '__version__', 'unknown')
            print_info(package_name, version)
        except ImportError:
            print_info(package_name, "NOT INSTALLED")

    return True

def test_nvidia_smi():
    """Test nvidia-smi availability"""
    print_header("Testing nvidia-smi")

    import subprocess

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            print_success("nvidia-smi available")
            gpu_info = result.stdout.strip()
            for line in gpu_info.split('\n'):
                print_info("GPU", line)
            return True
        else:
            print_failure(f"nvidia-smi failed: {result.stderr}")
            return False

    except FileNotFoundError:
        print_failure("nvidia-smi not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        print_failure("nvidia-smi timed out")
        return False
    except Exception as e:
        print_failure(f"nvidia-smi error: {e}")
        return False

def main():
    """Run all import tests"""
    print("\n" + "=" * 70)
    print("  EXO DISTRIBUTED INFERENCE - TEST 1: IMPORTS & ENVIRONMENT")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    tests = [
        ("Basic Imports", test_basic_imports),
        ("Exo Imports", test_exo_imports),
        ("Tinygrad Imports", test_tinygrad_imports),
        ("Environment Variables", test_environment_variables),
        ("CUDA Availability", test_cuda_availability),
        ("Package Versions", test_package_versions),
        ("nvidia-smi", test_nvidia_smi),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_failure(f"Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Print summary
    print_header("Test Summary")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed

    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")

    print(f"\nTotal: {total} tests")
    print(f"Passed: {passed} tests")
    print(f"Failed: {failed} tests")
    print(f"Success rate: {(passed/total)*100:.1f}%")

    if failed == 0:
        print("\n✓ ALL TESTS PASSED - Environment ready for exo distributed inference")
        return 0
    else:
        print(f"\n✗ {failed} TEST(S) FAILED - Fix issues before proceeding")
        return 1

if __name__ == "__main__":
    sys.exit(main())
