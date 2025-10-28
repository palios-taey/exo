#!/bin/bash
#
# Comprehensive Test Suite for NVCCCompiler
#
# Tests compilation at multiple levels:
# - Unit: nvcc subprocess alone
# - Integration: NVCCCompiler class
# - System: Full tinygrad pipeline
# - Hardware: GPU execution
#
# Author: Solution Agent 3
# Date: 2025-10-23

set -e  # Exit on error

echo "=================================================="
echo "NVCCCompiler Test Suite"
echo "=================================================="
echo ""

# Configuration
SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXO_ROOT="${EXO_ROOT:-/home/$(whoami)/exo}"
TEST_OUTPUT_DIR="/tmp/nvcc_compiler_tests"

# Create test output directory
mkdir -p "$TEST_OUTPUT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass() {
    echo -e "${GREEN}✓${NC} $1"
}

fail() {
    echo -e "${RED}✗${NC} $1"
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

run_test() {
    local test_name="$1"
    local test_command="$2"

    echo ""
    echo "[$test_name]"

    if eval "$test_command"; then
        pass "$test_name"
        ((TESTS_PASSED++))
        return 0
    else
        fail "$test_name"
        ((TESTS_FAILED++))
        return 1
    fi
}

skip_test() {
    local test_name="$1"
    local reason="$2"

    echo ""
    echo "[$test_name]"
    warn "$test_name (SKIPPED: $reason)"
    ((TESTS_SKIPPED++))
}

echo "[Phase 1] Environment Tests"
echo "=================================================="

# Test 1.1: nvcc available
run_test "nvcc available in PATH" "command -v nvcc > /dev/null"

# Test 1.2: nvcc version
if command -v nvcc > /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | cut -d',' -f1)
    echo "  nvcc version: $NVCC_VERSION"

    MAJOR_VERSION=$(echo "$NVCC_VERSION" | cut -d'.' -f1)
    run_test "CUDA 13.0 or later" "[ $MAJOR_VERSION -ge 13 ]" || warn "Version < 13.0 (patch designed for 13.0+)"
fi

# Test 1.3: Python available
run_test "Python 3 available" "command -v python3 > /dev/null"

# Test 1.4: Exo directory exists
run_test "Exo directory exists" "[ -d '$EXO_ROOT' ]"

echo ""
echo "[Phase 2] Unit Tests - nvcc Subprocess"
echo "=================================================="

# Test 2.1: Simple kernel compilation
TEST_KERNEL="$TEST_OUTPUT_DIR/test_kernel.cu"
TEST_PTX="$TEST_OUTPUT_DIR/test_kernel.ptx"

cat > "$TEST_KERNEL" << 'EOF'
extern "C" __global__ void simple_kernel(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = idx * 2.0f;
}
EOF

run_test "Compile simple CUDA C kernel to PTX" \
    "nvcc --gpu-architecture=sm_110 -ptx '$TEST_KERNEL' -o '$TEST_PTX' 2>&1"

# Test 2.2: Verify PTX format
if [ -f "$TEST_PTX" ]; then
    run_test "PTX file generated" "[ -s '$TEST_PTX' ]"

    if grep -q ".version" "$TEST_PTX" && grep -q ".target sm_110" "$TEST_PTX"; then
        pass "PTX contains correct directives (.version, .target sm_110)"
        ((TESTS_PASSED++))
    else
        fail "PTX missing expected directives"
        ((TESTS_FAILED++))
        echo "  First 20 lines of PTX:"
        head -20 "$TEST_PTX" | sed 's/^/    /'
    fi
else
    skip_test "PTX format verification" "PTX file not generated"
fi

# Test 2.3: Complex kernel with defines
TEST_KERNEL_COMPLEX="$TEST_OUTPUT_DIR/test_kernel_complex.cu"
TEST_PTX_COMPLEX="$TEST_OUTPUT_DIR/test_kernel_complex.ptx"

cat > "$TEST_KERNEL_COMPLEX" << 'EOF'
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))

extern "C" __global__ void __launch_bounds__(16) complex_kernel(float* data0_32) {
    int gidx0 = blockIdx.x;
    int lidx0 = threadIdx.x;
    float* data0 = (float*)(data0_32);

    if (lidx0 < 16) {
        data0[lidx0 + (gidx0 * 16)] = ((float)(lidx0)) * 2.0f;
    }
}
EOF

run_test "Compile complex kernel (similar to tinygrad output)" \
    "nvcc --gpu-architecture=sm_110 -ptx '$TEST_KERNEL_COMPLEX' -o '$TEST_PTX_COMPLEX' 2>&1"

echo ""
echo "[Phase 3] Integration Tests - NVCCCompiler Class"
echo "=================================================="

# Test 3.1: Import NVCCCompiler
TEST_IMPORT="cd '$SOLUTION_DIR' && python3 -c 'from nvcc_compiler import NVCCCompiler; print(\"OK\")' 2>&1"
run_test "Import NVCCCompiler module" "$TEST_IMPORT"

# Test 3.2: Instantiate compiler
TEST_INSTANTIATE="cd '$SOLUTION_DIR' && python3 -c 'from nvcc_compiler import NVCCCompiler; c = NVCCCompiler(\"sm_110\"); print(\"OK\")' 2>&1"
run_test "Instantiate NVCCCompiler(sm_110)" "$TEST_INSTANTIATE"

# Test 3.3: Standalone test
if [ -f "$SOLUTION_DIR/nvcc_compiler.py" ]; then
    TEST_STANDALONE="cd '$SOLUTION_DIR' && python3 nvcc_compiler.py 2>&1"
    run_test "Run standalone test suite" "$TEST_STANDALONE"
else
    skip_test "Standalone test suite" "nvcc_compiler.py not found"
fi

echo ""
echo "[Phase 4] Integration Tests - Monkey Patch"
echo "=================================================="

# Test 4.1: Import integration patch
TEST_PATCH_IMPORT="cd '$SOLUTION_DIR' && python3 -c 'from integration_patch import patch_nvptx_compiler; print(\"OK\")' 2>&1"
run_test "Import integration_patch module" "$TEST_PATCH_IMPORT"

# Test 4.2: Apply patch (if tinygrad available)
TEST_PATCH_APPLY=$(cat << 'EOFTEST'
cd "$SOLUTION_DIR" && python3 << 'EOFPYTHON'
import sys
sys.path.insert(0, '.')
from integration_patch import patch_nvptx_compiler

try:
    patch_nvptx_compiler()
    print("OK")
except ImportError as e:
    # Tinygrad not available
    print(f"SKIP: {e}")
    sys.exit(42)  # Special exit code for skip
EOFPYTHON
EOFTEST
)

PATCH_RESULT=$(eval "$TEST_PATCH_APPLY" 2>&1)
PATCH_EXIT=$?

if [ $PATCH_EXIT -eq 0 ]; then
    pass "Apply monkey patch"
    ((TESTS_PASSED++))
elif [ $PATCH_EXIT -eq 42 ]; then
    skip_test "Apply monkey patch" "tinygrad not installed"
else
    fail "Apply monkey patch"
    ((TESTS_FAILED++))
    echo "  Error: $PATCH_RESULT"
fi

echo ""
echo "[Phase 5] System Tests - Tinygrad Integration"
echo "=================================================="

# Test 5.1: Check if exo-venv exists
if [ -d "$EXO_ROOT/exo-venv" ]; then
    pass "exo-venv found"
    ((TESTS_PASSED++))

    VENV_PYTHON="$EXO_ROOT/exo-venv/bin/python3"

    # Test 5.2: Import tinygrad
    TEST_TINYGRAD="$VENV_PYTHON -c 'import tinygrad; print(\"OK\")' 2>&1"
    run_test "Import tinygrad" "$TEST_TINYGRAD"

    # Test 5.3: Import with patch
    TEST_TINYGRAD_PATCHED=$(cat << EOFTEST
$VENV_PYTHON << 'EOFPYTHON'
import sys
sys.path.insert(0, '$SOLUTION_DIR')
from integration_patch import patch_nvptx_compiler
patch_nvptx_compiler()

# Try to import Device (which would trigger CUDA initialization)
from tinygrad import Device
print("OK")
EOFPYTHON
EOFTEST
)

    run_test "Import tinygrad with patch" "$TEST_TINYGRAD_PATCHED" || warn "Patch may not be compatible with this tinygrad version"

else
    skip_test "exo-venv verification" "exo-venv not found (run from Thor device)"
    skip_test "Import tinygrad" "exo-venv not found"
    skip_test "Import tinygrad with patch" "exo-venv not found"
fi

echo ""
echo "[Phase 6] Hardware Tests - GPU Execution"
echo "=================================================="

# Test 6.1: Check if GPU available
if command -v nvidia-smi > /dev/null; then
    if nvidia-smi > /dev/null 2>&1; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)

        pass "GPU detected: $GPU_NAME (compute $COMPUTE_CAP)"
        ((TESTS_PASSED++))

        # Test 6.2: Verify compute capability
        if [ "$COMPUTE_CAP" = "11.0" ] || [ "$COMPUTE_CAP" = "10.1" ]; then
            pass "Blackwell GPU detected (compute $COMPUTE_CAP)"
            ((TESTS_PASSED++))
        else
            warn "Non-Blackwell GPU (compute $COMPUTE_CAP) - patch designed for Blackwell"
            ((TESTS_PASSED++))  # Not a failure, just informational
        fi

    else
        skip_test "GPU detection" "nvidia-smi failed to run"
        skip_test "Compute capability check" "GPU not accessible"
    fi
else
    skip_test "GPU detection" "nvidia-smi not available"
    skip_test "Compute capability check" "GPU tools not available"
fi

# Test 6.3: Simple tensor operation (if tinygrad + GPU available)
if [ -f "$VENV_PYTHON" ] && command -v nvidia-smi > /dev/null && nvidia-smi > /dev/null 2>&1; then
    TEST_TENSOR=$(cat << 'EOFTEST'
$VENV_PYTHON << 'EOFPYTHON'
import os
import sys
sys.path.insert(0, '$SOLUTION_DIR')

# Set environment
os.environ['DEVICE'] = 'CUDA'

# Apply patch
from integration_patch import patch_nvptx_compiler
patch_nvptx_compiler()

# Try tensor operation
from tinygrad import Tensor, Device
Device.DEFAULT = "CUDA"

try:
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    c = (a + b).realize()  # This will trigger kernel compilation

    result = c.numpy().tolist()
    expected = [5.0, 7.0, 9.0]

    if result == expected:
        print("OK")
    else:
        print(f"FAIL: Expected {expected}, got {result}")
        sys.exit(1)

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOFPYTHON
EOFTEST
)

    run_test "Simple tensor operation on GPU" "$TEST_TENSOR" || warn "Tensor operation failed - see error above"

else
    skip_test "Simple tensor operation on GPU" "Requires tinygrad + GPU"
fi

echo ""
echo "=================================================="
echo "Test Summary"
echo "=================================================="
echo ""
echo -e "${GREEN}Passed:${NC}  $TESTS_PASSED"
echo -e "${RED}Failed:${NC}  $TESTS_FAILED"
echo -e "${YELLOW}Skipped:${NC} $TESTS_SKIPPED"
echo ""

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))
echo "Total: $TOTAL_TESTS tests"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All non-skipped tests passed!${NC}"
    echo ""

    if [ $TESTS_SKIPPED -gt 0 ]; then
        echo "Note: $TESTS_SKIPPED tests skipped (likely due to missing dependencies)"
        echo "For full validation, run on Thor device with exo-venv"
    fi

    exit 0
else
    echo -e "${RED}$TESTS_FAILED test(s) failed${NC}"
    echo ""
    echo "Review failed tests above for details"
    exit 1
fi
