#!/usr/bin/env python3
"""
Unit Tests for NVPTXCompilerV2

Tests compilation pipeline in isolation without full tinygrad integration.

Test Categories:
1. NVRTC API Tests: Verify NVRTC library loading and basic compilation
2. nvJitLink API Tests: Verify linking PTX → CUBIN
3. Complete Pipeline Tests: CUDA C → PTX → CUBIN
4. Error Handling Tests: Invalid input, wrong architecture, etc.
5. Blackwell-Specific Tests: sm_110 features

Run: python3 unit_tests.py
"""

import sys
import os
import unittest
from pathlib import Path

# Add solution directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from nvptx_compiler_v2 import NVPTXCompilerV2
    from tinygrad.device import CompileError
except ImportError as e:
    print(f"ERROR: Failed to import NVPTXCompilerV2: {e}")
    print("Make sure tinygrad is installed: pip install tinygrad")
    sys.exit(1)


class TestNVRTCAvailability(unittest.TestCase):
    """Test that NVRTC library is available and functional."""

    def test_nvrtc_import(self):
        """Verify NVRTC module imports successfully."""
        try:
            import tinygrad.runtime.autogen.nvrtc as nvrtc
            self.assertIsNotNone(nvrtc)
        except ImportError:
            self.fail("NVRTC module not available")

    def test_nvrtc_version(self):
        """Verify NVRTC version can be queried."""
        import tinygrad.runtime.autogen.nvrtc as nvrtc
        import ctypes

        major = ctypes.c_int()
        minor = ctypes.c_int()

        try:
            err = nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
            self.assertEqual(err, 0, "nvrtcVersion should succeed")
            self.assertGreaterEqual(major.value, 13, "Should be CUDA 13.0+")
            print(f"  NVRTC version: {major.value}.{minor.value}")
        except Exception as e:
            self.fail(f"nvrtcVersion failed: {e}")


class TestSimpleCompilation(unittest.TestCase):
    """Test basic CUDA C → CUBIN compilation."""

    def setUp(self):
        """Initialize compiler for testing."""
        self.compiler = NVPTXCompilerV2('sm_110')

    def test_empty_kernel(self):
        """Test compilation of minimal valid kernel."""
        src = '''
        extern "C" __global__ void empty_kernel() {
            // Does nothing
        }
        '''

        try:
            cubin = self.compiler.compile(src)
            self.assertIsInstance(cubin, bytes)
            self.assertGreater(len(cubin), 0, "CUBIN should not be empty")
            # Verify ELF magic number
            self.assertEqual(cubin[:4], b'\x7fELF', "CUBIN should be ELF format")
        except CompileError as e:
            self.fail(f"Compilation failed: {e}")

    def test_simple_add_kernel(self):
        """Test kernel with basic arithmetic."""
        src = '''
        extern "C" __global__ void add_kernel(float* a, float* b, float* c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
        '''

        try:
            cubin = self.compiler.compile(src)
            self.assertGreater(len(cubin), 100, "CUBIN should be substantial")
        except CompileError as e:
            self.fail(f"Add kernel compilation failed: {e}")

    def test_defines_and_constants(self):
        """Test kernel with #define and constants."""
        src = '''
        #define BLOCK_SIZE 256

        extern "C" __global__ void test_defines(float* data, int n) {
            int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            if (idx < n) {
                data[idx] = 3.14159f;
            }
        }
        '''

        try:
            cubin = self.compiler.compile(src)
            self.assertIsInstance(cubin, bytes)
        except CompileError as e:
            self.fail(f"Define kernel compilation failed: {e}")


class TestPTXGeneration(unittest.TestCase):
    """Test Stage 1: CUDA C → PTX compilation."""

    def setUp(self):
        """Initialize compiler."""
        self.compiler = NVPTXCompilerV2('sm_110')

    def test_ptx_format(self):
        """Verify PTX has correct format."""
        src = 'extern "C" __global__ void test() {}'

        ptx = self.compiler._compile_cuda_to_ptx(src)

        # PTX should be text
        ptx_str = ptx.decode('utf-8')

        # Check required PTX directives
        self.assertIn('.version', ptx_str, "PTX must have .version directive")
        self.assertIn('.target', ptx_str, "PTX must have .target directive")
        self.assertIn('sm_110', ptx_str, "PTX must target sm_110")

        # Should NOT contain CUDA C syntax
        self.assertNotIn('extern "C"', ptx_str, "PTX should not contain extern C")
        self.assertNotIn('#define', ptx_str, "PTX should not contain #define")

    def test_ptx_version(self):
        """Verify PTX version is appropriate for architecture."""
        ptx = self.compiler._compile_cuda_to_ptx('extern "C" __global__ void test() {}')
        ptx_str = ptx.decode('utf-8')

        # For sm_110 (Blackwell), should use PTX 8.5+
        self.assertIn('.version 8.', ptx_str, "Blackwell requires PTX 8.x")


class TestNVJitLinkStage(unittest.TestCase):
    """Test Stage 2: PTX → CUBIN linking."""

    def setUp(self):
        """Initialize compiler."""
        self.compiler = NVPTXCompilerV2('sm_110')

    def test_link_valid_ptx(self):
        """Test linking of valid PTX."""
        # Generate PTX first
        src = 'extern "C" __global__ void test(float* x) { x[0] = 1.0f; }'
        ptx = self.compiler._compile_cuda_to_ptx(src)

        # Link to CUBIN
        try:
            cubin = self.compiler._link_ptx_to_cubin(ptx)
            self.assertIsInstance(cubin, bytes)
            self.assertGreater(len(cubin), 0)
            self.assertEqual(cubin[:4], b'\x7fELF')
        except CompileError as e:
            self.fail(f"PTX linking failed: {e}")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and reporting."""

    def setUp(self):
        """Initialize compiler."""
        self.compiler = NVPTXCompilerV2('sm_110')

    def test_syntax_error(self):
        """Test that syntax errors are reported clearly."""
        src = '''
        extern "C" __global__ void broken() {
            this is not valid C++;
        }
        '''

        with self.assertRaises(CompileError) as context:
            self.compiler.compile(src)

        error_msg = str(context.exception)
        # Should mention compilation failure
        self.assertIn('NVRTC', error_msg)

    def test_undefined_function(self):
        """Test error reporting for undefined functions."""
        src = '''
        extern "C" __global__ void uses_undefined() {
            undefined_function();
        }
        '''

        with self.assertRaises(CompileError):
            self.compiler.compile(src)

    def test_invalid_ptx(self):
        """Test that invalid PTX is rejected by nvJitLink."""
        invalid_ptx = b"This is not valid PTX assembly"

        with self.assertRaises(CompileError) as context:
            self.compiler._link_ptx_to_cubin(invalid_ptx)

        error_msg = str(context.exception)
        self.assertIn('nvJitLink', error_msg)


class TestBlackwellFeatures(unittest.TestCase):
    """Test Blackwell-specific compilation."""

    def setUp(self):
        """Initialize compiler for Blackwell."""
        self.compiler = NVPTXCompilerV2('sm_110')

    def test_architecture_targeting(self):
        """Verify compilation targets sm_110."""
        self.assertEqual(self.compiler.arch, 'sm_110')
        self.assertEqual(self.compiler.compute_arch, 'compute_110')
        self.assertEqual(self.compiler.ptx_version, '8.5')

    def test_shared_memory_kernel(self):
        """Test kernel using shared memory (Blackwell: 228KB/SM)."""
        src = '''
        extern "C" __global__ void shared_test(float* output) {
            __shared__ float shared_data[256];
            int idx = threadIdx.x;
            shared_data[idx] = idx * 1.0f;
            __syncthreads();
            output[idx] = shared_data[idx];
        }
        '''

        try:
            cubin = self.compiler.compile(src)
            self.assertGreater(len(cubin), 0)
        except CompileError as e:
            self.fail(f"Shared memory kernel failed: {e}")

    def test_tensor_core_types(self):
        """Test compilation with tensor core data types."""
        src = '''
        #include <cuda_fp16.h>

        extern "C" __global__ void fp16_kernel(__half* a, __half* b, __half* c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = __hadd(a[idx], b[idx]);
            }
        }
        '''

        try:
            cubin = self.compiler.compile(src)
            self.assertGreater(len(cubin), 0)
        except CompileError as e:
            self.fail(f"FP16 kernel failed: {e}")


class TestCompilerOptions(unittest.TestCase):
    """Test NVRTC compilation options."""

    def setUp(self):
        """Initialize compiler."""
        self.compiler = NVPTXCompilerV2('sm_110')

    def test_default_options(self):
        """Verify default compilation options."""
        options = self.compiler._get_compile_options()

        # Check required options
        self.assertIn('--gpu-architecture=compute_110', options)
        self.assertIn('--std=c++17', options)
        self.assertIn('--use_fast_math', options)

    def test_architecture_specific_options(self):
        """Test that Blackwell gets appropriate options."""
        options = self.compiler._get_compile_options()

        # Blackwell-specific
        self.assertIn('--device-c', options)
        self.assertIn('--relocatable-device-code=false', options)


class TestDebugOutput(unittest.TestCase):
    """Test debug logging and error file saving."""

    def setUp(self):
        """Initialize compiler."""
        self.compiler = NVPTXCompilerV2('sm_110')

    def test_failed_source_saved(self):
        """Test that failed sources are saved to /tmp."""
        # Force a compilation error
        src = 'extern "C" __global__ void broken() { syntax error; }'

        try:
            self.compiler.compile(src)
        except CompileError:
            # Check if debug files were created
            tmp_files = list(Path('/tmp').glob('nvptx_failed_*.cu'))
            self.assertGreater(len(tmp_files), 0, "Failed source should be saved")

            # Clean up
            for f in tmp_files:
                try:
                    f.unlink()
                except:
                    pass


class TestPerformance(unittest.TestCase):
    """Test compilation performance."""

    def setUp(self):
        """Initialize compiler."""
        self.compiler = NVPTXCompilerV2('sm_110')

    def test_compilation_speed(self):
        """Measure compilation time for typical kernel."""
        import time

        src = '''
        extern "C" __global__ void matmul(float* A, float* B, float* C, int N) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            if (row < N && col < N) {
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += A[row * N + k] * B[k * N + col];
                }
                C[row * N + col] = sum;
            }
        }
        '''

        start = time.time()
        cubin = self.compiler.compile(src)
        elapsed = time.time() - start

        print(f"\n  Compilation time: {elapsed:.3f}s")
        print(f"  CUBIN size: {len(cubin)} bytes")

        # Should complete in reasonable time (< 5 seconds)
        self.assertLess(elapsed, 5.0, "Compilation should be reasonably fast")


def run_tests():
    """Run all tests with detailed output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestNVRTCAvailability))
    suite.addTests(loader.loadTestsFromTestCase(TestSimpleCompilation))
    suite.addTests(loader.loadTestsFromTestCase(TestPTXGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestNVJitLinkStage))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestBlackwellFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestCompilerOptions))
    suite.addTests(loader.loadTestsFromTestCase(TestDebugOutput))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))

    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    print("=" * 70)
    print("NVPTXCompilerV2 Unit Tests")
    print("=" * 70)
    print()

    sys.exit(run_tests())
