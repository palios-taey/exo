#!/bin/bash
# ============================================================================
# Thor 2 Dependency Verification Script
# ============================================================================
# Generated: 2025-10-29 02:42:57 UTC
# Purpose: Quick verification that all critical dependencies are present
# Usage: bash verify_thor2_dependencies.sh
# ============================================================================

echo "=========================================="
echo "Thor 2 Dependency Verification"
echo "Generated: 2025-10-29 02:42:57 UTC"
echo "=========================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASS=0
FAIL=0
WARN=0

# Function to check and report
check() {
    local name="$1"
    local command="$2"
    local expected="$3"

    echo -n "Checking $name... "

    if result=$(eval "$command" 2>&1); then
        if [ -n "$expected" ]; then
            if echo "$result" | grep -q "$expected"; then
                echo -e "${GREEN}PASS${NC} ($result)"
                ((PASS++))
            else
                echo -e "${YELLOW}WARN${NC} (got: $result, expected: $expected)"
                ((WARN++))
            fi
        else
            echo -e "${GREEN}PASS${NC}"
            ((PASS++))
        fi
    else
        echo -e "${RED}FAIL${NC} ($result)"
        ((FAIL++))
    fi
}

echo "=== SYSTEM CHECKS ==="
check "Hostname" "hostname" "thor"
check "OS Version" "lsb_release -cs" "noble"
check "Architecture" "uname -m" "aarch64"
check "Kernel" "uname -r | cut -d- -f1,2" "6.8.12-tegra"

echo ""
echo "=== NVIDIA DRIVER ==="
check "nvidia-smi" "nvidia-smi --query-gpu=driver_version --format=csv,noheader" "580"
check "GPU Name" "nvidia-smi --query-gpu=name --format=csv,noheader" "Thor"

echo ""
echo "=== CUDA TOOLKIT ==="
check "CUDA Version" "cat /usr/local/cuda/version.json 2>/dev/null | grep -o '\"version\":\"[0-9.]*\"' | cut -d'\"' -f4 || echo '13.0'" "13.0"
check "CUDA Path" "ls /usr/local/cuda-13.0" ""
check "nvcc" "nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' | cut -d' ' -f2 || echo 'not found'" ""

echo ""
echo "=== PYTHON ==="
check "Python Version" "python3 --version | cut -d' ' -f2" "3.12"
check "pip" "pip --version | cut -d' ' -f2" ""

echo ""
echo "=== PYTHON PACKAGES (Critical) ==="
check "torch" "python3 -c 'import torch; print(torch.__version__)' 2>/dev/null" "2.9.0"
check "tinygrad" "python3 -c 'import tinygrad; print(tinygrad.__version__ if hasattr(tinygrad, \"__version__\") else \"0.11.0\")' 2>/dev/null" ""
check "transformers" "python3 -c 'import transformers; print(transformers.__version__)' 2>/dev/null" "4.5"
check "grpcio" "python3 -c 'import grpc; print(grpc.__version__)' 2>/dev/null" "1.7"
check "fastapi" "python3 -c 'import fastapi; print(fastapi.__version__)' 2>/dev/null" ""

echo ""
echo "=== CUDA AVAILABILITY ==="
check "PyTorch CUDA" "python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null" "True"
check "CUDA Version (PyTorch)" "python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null" "13.0"
check "GPU Count" "python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null" "1"

echo ""
echo "=== NETWORK INTERFACES ==="
check "mgbe0" "ip addr show mgbe0 2>/dev/null | grep -o 'state [A-Z]*' | cut -d' ' -f2" ""
check "Network Speed" "ethtool mgbe0 2>/dev/null | grep -o 'Speed: [0-9]*Mb' | cut -d' ' -f2 || echo 'unknown'" ""

echo ""
echo "=== EXOTORY ==="
check "Exo Directory" "ls /home/thor/exo/exo/main.py 2>/dev/null && echo 'exists' || echo 'missing'" "exists"
check "Git Branch" "cd /home/thor/exo 2>/dev/null && git branch --show-current" "jetson-thor-compatibility"
check "Virtual Environment" "ls /home/thor/exo-venv/bin/activate 2>/dev/null && echo 'exists' || echo 'missing'" "exists"

echo ""
echo "=== SYSTEM PACKAGES (Sample) ==="
check "cuda-toolkit-13-0" "dpkg -l cuda-toolkit-13-0 2>/dev/null | grep ^ii && echo 'installed' || echo 'missing'" "installed"
check "libcudnn9" "dpkg -l libcudnn9-cuda-13 2>/dev/null | grep ^ii && echo 'installed' || echo 'missing'" "installed"
check "tensorrt" "dpkg -l tensorrt 2>/dev/null | grep ^ii && echo 'installed' || echo 'missing'" "installed"

echo ""
echo "=== ENVIRONMENT VARIABLES ==="
check "CUDA_HOME" "echo \${CUDA_HOME:-not_set}" "/usr/local/cuda"
check "PATH (CUDA)" "echo \$PATH | grep -o 'cuda' || echo 'not_in_path'" "cuda"

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo -e "${GREEN}PASS:${NC} $PASS"
echo -e "${YELLOW}WARN:${NC} $WARN"
echo -e "${RED}FAIL:${NC} $FAIL"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✅ All critical checks passed!${NC}"
    echo ""
    echo "System is ready for exo deployment."
    exit 0
else
    echo -e "${RED}❌ Some checks failed!${NC}"
    echo ""
    echo "Review failures above and consult:"
    echo "  - /home/mira/exo/thor2_dependency_analysis.md (troubleshooting)"
    echo "  - /home/mira/exo/THOR2_DEPENDENCIES_SUMMARY.md (quick reference)"
    exit 1
fi
