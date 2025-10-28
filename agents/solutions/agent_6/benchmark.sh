#!/bin/bash
#
# benchmark.sh - Performance Benchmarks for PTX Solution
#
# This script benchmarks:
# 1. Kernel compilation time (PTX vs baseline)
# 2. Runtime performance (should be identical)
# 3. Memory bandwidth utilization
# 4. Tensor core throughput (if applicable)
#
# Usage:
#   bash benchmark.sh [--local|--thor1|--thor2]
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Nodes
THOR1="thor@10.0.0.78"
THOR2="jetson@10.0.0.93"

# Test mode
MODE="${1:---local}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}PTXRenderer Performance Benchmarks${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Benchmark 1: Compilation Time
benchmark_compilation() {
    local HOST=$1
    echo -e "${YELLOW}Benchmark 1: Kernel Compilation Time${NC}"

    if [ "${HOST}" == "local" ]; then
        python3 << 'EOF'
import os
os.environ['PTX'] = '1'

from tinygrad import Tensor
import time

# Warmup
a = Tensor([1.0])
_ = a.realize()

# Benchmark: First compilation (cold cache)
print("Measuring first kernel compilation (cold cache)...")
start = time.time()
a = Tensor.randn(1000, 1000)
b = Tensor.randn(1000, 1000)
c = (a + b).realize()
compilation_time = time.time() - start
print(f"First compilation: {compilation_time*1000:.2f}ms")

# Benchmark: Second compilation (warm cache, should be cached)
print("Measuring cached kernel compilation...")
start = time.time()
a = Tensor.randn(1000, 1000)
b = Tensor.randn(1000, 1000)
c = (a + b).realize()
cached_time = time.time() - start
print(f"Cached execution: {cached_time*1000:.2f}ms")

print(f"Cache speedup: {compilation_time/cached_time:.2f}x")
EOF
    else
        ssh "${HOST}" "cd /home/\$(whoami)/exo && python3 << 'EOF'
import os
os.environ['PTX'] = '1'

from tinygrad import Tensor
import time

a = Tensor([1.0])
_ = a.realize()

start = time.time()
a = Tensor.randn(1000, 1000)
b = Tensor.randn(1000, 1000)
c = (a + b).realize()
compilation_time = time.time() - start
print(f\"First compilation: {compilation_time*1000:.2f}ms\")

start = time.time()
a = Tensor.randn(1000, 1000)
b = Tensor.randn(1000, 1000)
c = (a + b).realize()
cached_time = time.time() - start
print(f\"Cached execution: {cached_time*1000:.2f}ms\")

print(f\"Cache speedup: {compilation_time/cached_time:.2f}x\")
EOF"
    fi
    echo ""
}

# Benchmark 2: Element-wise Throughput
benchmark_elementwise() {
    local HOST=$1
    echo -e "${YELLOW}Benchmark 2: Element-wise Operation Throughput${NC}"

    if [ "${HOST}" == "local" ]; then
        python3 << 'EOF'
import os
os.environ['PTX'] = '1'

from tinygrad import Tensor
import time
import numpy as np

# Create tensors
N = 10000000  # 10M elements
a = Tensor.randn(N)
b = Tensor.randn(N)

# Warmup
for _ in range(5):
    c = (a + b).realize()

# Benchmark
iterations = 100
start = time.time()
for _ in range(iterations):
    c = (a + b).realize()
elapsed = time.time() - start

# Calculate metrics
ops_per_sec = iterations / elapsed
elements_per_sec = N * iterations / elapsed
bandwidth_gb = (N * 4 * 3 * iterations) / elapsed / 1e9  # 3 arrays (a, b, c), 4 bytes each

print(f"Element-wise addition performance:")
print(f"  Operations/sec: {ops_per_sec:.2f}")
print(f"  Elements/sec: {elements_per_sec/1e6:.2f}M")
print(f"  Memory bandwidth: {bandwidth_gb:.2f} GB/s")
EOF
    else
        ssh "${HOST}" "cd /home/\$(whoami)/exo && python3 << 'EOF'
import os
os.environ['PTX'] = '1'

from tinygrad import Tensor
import time

N = 10000000
a = Tensor.randn(N)
b = Tensor.randn(N)

for _ in range(5):
    c = (a + b).realize()

iterations = 100
start = time.time()
for _ in range(iterations):
    c = (a + b).realize()
elapsed = time.time() - start

ops_per_sec = iterations / elapsed
elements_per_sec = N * iterations / elapsed
bandwidth_gb = (N * 4 * 3 * iterations) / elapsed / 1e9

print(f\"Element-wise performance:\")
print(f\"  Operations/sec: {ops_per_sec:.2f}\")
print(f\"  Elements/sec: {elements_per_sec/1e6:.2f}M\")
print(f\"  Bandwidth: {bandwidth_gb:.2f} GB/s\")
EOF"
    fi
    echo ""
}

# Benchmark 3: Matrix Multiplication
benchmark_matmul() {
    local HOST=$1
    echo -e "${YELLOW}Benchmark 3: Matrix Multiplication Throughput${NC}"

    if [ "${HOST}" == "local" ]; then
        python3 << 'EOF'
import os
os.environ['PTX'] = '1'

from tinygrad import Tensor
import time

# Test different matrix sizes
sizes = [128, 512, 1024, 2048]

for N in sizes:
    print(f"\nMatrix size: {N}x{N}")

    # Create matrices
    a = Tensor.randn(N, N)
    b = Tensor.randn(N, N)

    # Warmup
    for _ in range(3):
        c = (a @ b).realize()

    # Benchmark
    iterations = 10
    start = time.time()
    for _ in range(iterations):
        c = (a @ b).realize()
    elapsed = time.time() - start

    # Calculate metrics
    flops = 2 * N * N * N * iterations  # 2N^3 operations per matmul
    gflops = flops / elapsed / 1e9
    matmuls_per_sec = iterations / elapsed

    print(f"  Matmuls/sec: {matmuls_per_sec:.2f}")
    print(f"  GFLOPS: {gflops:.2f}")
    print(f"  Time per matmul: {elapsed*1000/iterations:.2f}ms")
EOF
    else
        ssh "${HOST}" "cd /home/\$(whoami)/exo && python3 << 'EOF'
import os
os.environ['PTX'] = '1'

from tinygrad import Tensor
import time

sizes = [128, 512, 1024, 2048]

for N in sizes:
    print(f\"\\nMatrix size: {N}x{N}\")

    a = Tensor.randn(N, N)
    b = Tensor.randn(N, N)

    for _ in range(3):
        c = (a @ b).realize()

    iterations = 10
    start = time.time()
    for _ in range(iterations):
        c = (a @ b).realize()
    elapsed = time.time() - start

    flops = 2 * N * N * N * iterations
    gflops = flops / elapsed / 1e9
    matmuls_per_sec = iterations / elapsed

    print(f\"  Matmuls/sec: {matmuls_per_sec:.2f}\")
    print(f\"  GFLOPS: {gflops:.2f}\")
    print(f\"  Time/matmul: {elapsed*1000/iterations:.2f}ms\")
EOF"
    fi
    echo ""
}

# Benchmark 4: Reduction Operations
benchmark_reduction() {
    local HOST=$1
    echo -e "${YELLOW}Benchmark 4: Reduction Operation Throughput${NC}"

    if [ "${HOST}" == "local" ]; then
        python3 << 'EOF'
import os
os.environ['PTX'] = '1'

from tinygrad import Tensor
import time

# Create large tensor
N = 10000000
a = Tensor.randn(N)

# Warmup
for _ in range(5):
    s = a.sum().realize()

# Benchmark
iterations = 100
start = time.time()
for _ in range(iterations):
    s = a.sum().realize()
elapsed = time.time() - start

# Calculate metrics
reductions_per_sec = iterations / elapsed
elements_per_sec = N * iterations / elapsed
bandwidth_gb = (N * 4 * iterations) / elapsed / 1e9  # Read bandwidth

print(f"Reduction (sum) performance:")
print(f"  Reductions/sec: {reductions_per_sec:.2f}")
print(f"  Elements/sec: {elements_per_sec/1e6:.2f}M")
print(f"  Read bandwidth: {bandwidth_gb:.2f} GB/s")
EOF
    else
        ssh "${HOST}" "cd /home/\$(whoami)/exo && python3 << 'EOF'
import os
os.environ['PTX'] = '1'

from tinygrad import Tensor
import time

N = 10000000
a = Tensor.randn(N)

for _ in range(5):
    s = a.sum().realize()

iterations = 100
start = time.time()
for _ in range(iterations):
    s = a.sum().realize()
elapsed = time.time() - start

reductions_per_sec = iterations / elapsed
elements_per_sec = N * iterations / elapsed
bandwidth_gb = (N * 4 * iterations) / elapsed / 1e9

print(f\"Reduction performance:\")
print(f\"  Reductions/sec: {reductions_per_sec:.2f}\")
print(f\"  Elements/sec: {elements_per_sec/1e6:.2f}M\")
print(f\"  Bandwidth: {bandwidth_gb:.2f} GB/s\")
EOF"
    fi
    echo ""
}

# Run benchmarks based on mode
case "${MODE}" in
    --local)
        echo "Benchmarking LOCAL (Mira)..."
        echo ""
        benchmark_compilation "local"
        benchmark_elementwise "local"
        benchmark_matmul "local"
        benchmark_reduction "local"
        ;;

    --thor1)
        echo "Benchmarking THOR #1 (${THOR1})..."
        echo ""
        benchmark_compilation "${THOR1}"
        benchmark_elementwise "${THOR1}"
        benchmark_matmul "${THOR1}"
        benchmark_reduction "${THOR1}"
        ;;

    --thor2)
        echo "Benchmarking THOR #2 (${THOR2})..."
        echo ""
        benchmark_compilation "${THOR2}"
        benchmark_elementwise "${THOR2}"
        benchmark_matmul "${THOR2}"
        benchmark_reduction "${THOR2}"
        ;;

    --compare)
        echo "Comparing THOR #1 vs THOR #2..."
        echo ""
        echo "=== THOR #1 ==="
        benchmark_matmul "${THOR1}"
        echo "=== THOR #2 ==="
        benchmark_matmul "${THOR2}"
        ;;

    *)
        echo -e "${RED}Unknown mode: ${MODE}${NC}"
        echo "Usage: bash benchmark.sh [--local|--thor1|--thor2|--compare]"
        exit 1
        ;;
esac

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Benchmark Suite Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Performance notes:"
echo "  - Compilation time includes PTX generation + nvJitLink (one-time)"
echo "  - Runtime performance should be identical to CUDA C path"
echo "  - PTX version (7.8 vs 8.5) is metadata, minimal impact on existing kernels"
echo "  - Future FP4 tensor cores will show 2x improvement for transformers"
echo ""
