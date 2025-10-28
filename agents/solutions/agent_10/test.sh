#!/bin/bash
# Test Configuration-Based Compiler Selector
# Solution Agent 10

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Testing Configuration-Based Compiler Selector${NC}"
echo "=============================================="

# Parse test mode
TEST_MODE="${1:---basic}"

test_config_loading() {
    echo -e "\n${YELLOW}Test 1: Configuration Loading${NC}"
    python3 << 'EOF'
import sys
sys.path.insert(0, '/home/mira/exo/agents/solutions/agent_10')

from compiler_selector import CompilerConfig

config = CompilerConfig()
print(f"✓ Config loaded successfully")
print(f"  Preferred compiler: {config.config['compiler']['preferred']}")
print(f"  Architecture: {config.config['compiler']['architecture']}")
EOF
}

test_environment_detection() {
    echo -e "\n${YELLOW}Test 2: Environment Detection${NC}"
    python3 << 'EOF'
import sys
sys.path.insert(0, '/home/mira/exo/agents/solutions/agent_10')

from compiler_selector import CompilerConfig, EnvironmentDetector

config = CompilerConfig()
detector = EnvironmentDetector(config)

print(f"✓ Detector initialized")
cuda_version = detector.detect_cuda_version()
print(f"  CUDA version: {cuda_version}")
has_nvcc = detector.check_nvcc()
print(f"  nvcc available: {has_nvcc}")
has_nvrtc = detector.check_nvrtc()
print(f"  NVRTC available: {has_nvrtc}")
EOF
}

test_compiler_selection() {
    echo -e "\n${YELLOW}Test 3: Compiler Selection${NC}"
    python3 << 'EOF'
import sys
sys.path.insert(0, '/home/mira/exo/agents/solutions/agent_10')

from compiler_selector import CompilerSelector

selector = CompilerSelector()
compiler = selector.select_compiler()

print(f"✓ Compiler selected: {compiler}")
EOF
}

test_compiler_imports() {
    echo -e "\n${YELLOW}Test 4: Compiler Module Imports${NC}"
    python3 << 'EOF'
import sys
sys.path.insert(0, '/home/mira/exo/agents/solutions/agent_10')

from compilers import (
    PTXDirectCompiler,
    NVCCSubprocessCompiler,
    NVRTCWrapperCompiler,
    NVJitLinkDirectCompiler,
    get_compiler
)

print("✓ All compiler modules imported successfully")

# Test availability checks
print(f"  PTX Direct available: {PTXDirectCompiler.is_available()}")
print(f"  nvcc available: {NVCCSubprocessCompiler.is_available()}")
print(f"  NVRTC available: {NVRTCWrapperCompiler.is_available()}")
print(f"  nvJitLink Direct available: {NVJitLinkDirectCompiler.is_available()}")
EOF
}

test_simple_compilation() {
    echo -e "\n${YELLOW}Test 5: Simple Kernel Compilation${NC}"
    python3 << 'EOF'
import sys
sys.path.insert(0, '/home/mira/exo/agents/solutions/agent_10')

from compiler_selector import CompilerSelector

# Simple CUDA kernel
cuda_src = '''
extern "C" __global__ void test_kernel(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = idx * 2.0f;
}
'''

selector = CompilerSelector()
compiler_name = selector.select_compiler()

if compiler_name in ['nvcc', 'nvrtc']:
    compiler = selector.get_compiler_instance(compiler_name)
    print(f"✓ Testing compilation with {compiler_name}...")
    try:
        cubin = compiler.compile(cuda_src)
        print(f"✓ Compilation succeeded! CUBIN size: {len(cubin)} bytes")
    except Exception as e:
        print(f"✗ Compilation failed: {e}")
else:
    print(f"  Skipping compilation test for {compiler_name} (requires tinygrad integration)")
EOF
}

test_all_compilers() {
    echo -e "\n${YELLOW}Test 6: All Compiler Paths${NC}"

    for compiler in ptx nvcc nvrtc direct; do
        echo -e "\n  Testing $compiler..."
        TINYGRAD_COMPILER=$compiler python3 -c "
import sys
sys.path.insert(0, '/home/mira/exo/agents/solutions/agent_10')
import os
os.environ['TINYGRAD_COMPILER'] = '$compiler'
from compiler_selector import CompilerSelector
selector = CompilerSelector()
selected = selector.select_compiler()
print(f'    Selected: {selected}')
        "
    done
}

# Run tests based on mode
if [ "$TEST_MODE" = "--basic" ] || [ "$TEST_MODE" = "-b" ]; then
    echo "Running basic tests..."
    test_config_loading
    test_environment_detection
    test_compiler_selection
    test_compiler_imports

elif [ "$TEST_MODE" = "--full" ] || [ "$TEST_MODE" = "-f" ]; then
    echo "Running full test suite..."
    test_config_loading
    test_environment_detection
    test_compiler_selection
    test_compiler_imports
    test_simple_compilation

elif [ "$TEST_MODE" = "--all-paths" ] || [ "$TEST_MODE" = "-a" ]; then
    echo "Testing all compiler paths..."
    test_all_compilers

elif [ "$TEST_MODE" = "--integration" ] || [ "$TEST_MODE" = "-i" ]; then
    echo "Running integration tests with tinygrad..."
    test_simple_compilation

else
    echo "Unknown test mode: $TEST_MODE"
    echo "Usage: ./test.sh [--basic|--full|--all-paths|--integration]"
    exit 1
fi

echo -e "\n${GREEN}All tests completed!${NC}"
echo ""
echo "Test modes:"
echo "  --basic (-b)       : Basic functionality tests"
echo "  --full (-f)        : Full test suite including compilation"
echo "  --all-paths (-a)   : Test all compiler paths"
echo "  --integration (-i) : Integration tests with tinygrad"
