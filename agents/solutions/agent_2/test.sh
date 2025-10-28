#!/bin/bash
#
# Test script for NVRTC Proper Usage Fix (Solution Agent 2)
#
# This script:
# 1. Tests server startup (NVRTC imports)
# 2. Tests kernel compilation (CUDA C → PTX → CUBIN)
# 3. Tests inference quality (actual model output)
# 4. Collects diagnostic information
#
# Usage:
#   ./test.sh [--node thor|jetson|all]
#
# Author: Solution Agent 2
# Date: 2025-10-23

set -e

# Configuration
THOR_NODE="thor@10.0.0.78"
THOR_PORT="52415"
JETSON_NODE="jetson@10.0.0.93"
JETSON_PORT="52416"
MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
TARGET_NODE="all"
if [[ "$1" == "--node" ]]; then
    TARGET_NODE="$2"
fi

# Test functions
test_nvrtc_library() {
    local node=$1
    log_info "Testing NVRTC library availability on $node..."

    # Check library files
    local lib_check=$(ssh "$node" "find /usr/local/cuda-13.0 -name 'libnvrtc.so*' 2>/dev/null | wc -l")
    if [[ "$lib_check" -gt 0 ]]; then
        log_success "NVRTC library files found on $node"
        ssh "$node" "find /usr/local/cuda-13.0 -name 'libnvrtc.so*' 2>/dev/null"
    else
        log_error "NVRTC library not found on $node"
        return 1
    fi

    # Test Python can load it
    local python_check=$(ssh "$node" "python3 -c 'import ctypes.util; lib=ctypes.util.find_library(\"nvrtc\"); print(\"found\" if lib else \"not found\")' 2>/dev/null")
    if [[ "$python_check" == "found" ]]; then
        log_success "Python can load NVRTC on $node"
    else
        log_error "Python cannot load NVRTC on $node"
        return 1
    fi

    return 0
}

test_server_startup() {
    local node=$1
    local port=$2
    local node_name=${node%%@*}

    log_info "Testing server startup on $node_name..."

    # Check if already running
    local pid=$(ssh "$node" "pgrep -f 'exo.main.*--node-port $port' || echo ''")
    if [[ -n "$pid" ]]; then
        log_info "Server already running on $node_name (PID: $pid)"
        return 0
    fi

    log_info "Server not running. Please start it manually:"
    echo "  ssh $node 'cd ~/exo && python3 -m exo.main --node-port $port > /tmp/${node_name}_exo.log 2>&1 &'"
    return 1
}

test_nvrtc_logs() {
    local node=$1
    local node_name=${node%%@*}
    local log_file="/tmp/${node_name}_exo.log"

    log_info "Checking logs on $node_name for NVRTC usage..."

    # Check for monkey-patch messages (should NOT be present)
    local patch_msgs=$(ssh "$node" "grep -c 'NVRTC MONKEY-PATCH' '$log_file' 2>/dev/null || echo 0")
    if [[ "$patch_msgs" -eq 0 ]]; then
        log_success "No monkey-patch messages found (expected)"
    else
        log_warning "Found $patch_msgs monkey-patch messages (unexpected!)"
    fi

    # Check for compilation messages
    local compile_msgs=$(ssh "$node" "grep -i 'compil' '$log_file' 2>/dev/null | tail -5 || echo 'No compilation messages'")
    if [[ "$compile_msgs" != "No compilation messages" ]]; then
        log_info "Recent compilation activity:"
        echo "$compile_msgs" | while read -r line; do
            echo "    $line"
        done
    else
        log_warning "No compilation messages found yet"
    fi

    # Check for NVRTC errors
    local nvrtc_errors=$(ssh "$node" "grep -i 'nvrtc.*error' '$log_file' 2>/dev/null | tail -5 || echo ''")
    if [[ -z "$nvrtc_errors" ]]; then
        log_success "No NVRTC errors found"
    else
        log_error "NVRTC errors detected:"
        echo "$nvrtc_errors" | while read -r line; do
            echo "    $line"
        done
        return 1
    fi

    # Check for nvJitLink errors
    local jitlink_errors=$(ssh "$node" "grep -i 'nvjitlink.*error\|bad input' '$log_file' 2>/dev/null | tail -5 || echo ''")
    if [[ -z "$jitlink_errors" ]]; then
        log_success "No nvJitLink errors found"
    else
        log_error "nvJitLink errors detected:"
        echo "$jitlink_errors" | while read -r line; do
            echo "    $line"
        done
        return 1
    fi

    return 0
}

test_inference() {
    local port=$1
    local node_name=$2

    log_info "Testing inference on $node_name (port $port)..."

    # Simple inference request
    local request='{
        "model": "'$MODEL'",
        "messages": [{"role": "user", "content": "Count to 5"}],
        "temperature": 0.7,
        "max_tokens": 50
    }'

    log_info "Sending inference request..."
    local response=$(curl -s -X POST "http://localhost:$port/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$request" \
        --max-time 60)

    # Check if response is valid JSON
    if echo "$response" | jq empty 2>/dev/null; then
        log_success "Valid JSON response received"

        # Extract generated text
        local content=$(echo "$response" | jq -r '.choices[0].message.content // "ERROR: No content"')
        log_info "Generated content:"
        echo "    $content"

        # Check for actual content (not error)
        if [[ "$content" != "ERROR: No content" ]] && [[ -n "$content" ]]; then
            log_success "Inference successful on $node_name"
            return 0
        else
            log_error "Inference produced empty/error response"
            return 1
        fi
    else
        log_error "Invalid JSON response"
        echo "$response" | head -20
        return 1
    fi
}

collect_diagnostics() {
    local node=$1
    local node_name=${node%%@*}
    local log_file="/tmp/${node_name}_exo.log"
    local output_file="/tmp/${node_name}_diagnostics_$(date +%Y%m%d_%H%M%S).txt"

    log_info "Collecting diagnostics from $node_name..."

    ssh "$node" "cat > '$output_file' <<'DIAGNOSTICS_END'
# Diagnostics for $node_name - $(date)

## System Info
$(uname -a)

## CUDA Version
$(nvcc --version 2>/dev/null || echo 'nvcc not found')

## NVRTC Library Files
$(find /usr/local/cuda-13.0 -name 'libnvrtc.so*' 2>/dev/null || echo 'Not found')

## Python NVRTC Test
$(python3 -c 'import ctypes.util; print(\"NVRTC:\", ctypes.util.find_library(\"nvrtc\"))' 2>&1)

## Modified Files
$(ls -la ~/exo/exo/inference/tinygrad/inference.py* 2>/dev/null)

## Patch Removal Verification
$(grep -c 'NVRTC MONKEY-PATCH' ~/exo/exo/inference/tinygrad/inference.py 2>/dev/null || echo '0 (patch removed)')

## Recent Log (last 100 lines)
$(tail -100 '$log_file' 2>/dev/null || echo 'Log file not found')

## Process Status
$(pgrep -f 'exo.main' -a || echo 'No exo processes')

## GPU Info
$(nvidia-smi 2>/dev/null || echo 'nvidia-smi not available')

DIAGNOSTICS_END
"

    log_success "Diagnostics saved to $node:$output_file"
    echo "  Download with: scp $node:$output_file ./"
}

# Main test execution
log_success "============================================"
log_success "  NVRTC PROPER USAGE FIX - TEST SUITE"
log_success "============================================"
echo

# Test Thor
if [[ "$TARGET_NODE" == "all" ]] || [[ "$TARGET_NODE" == "thor" ]]; then
    echo
    log_info "=== Testing Thor Node ==="
    echo

    test_nvrtc_library "$THOR_NODE" || log_error "NVRTC library test failed on Thor"
    echo

    test_server_startup "$THOR_NODE" "$THOR_PORT" || log_warning "Server not running on Thor"
    echo

    test_nvrtc_logs "$THOR_NODE" || log_error "Log analysis failed on Thor"
    echo

    # Only test inference if server is running
    if ssh "$THOR_NODE" "pgrep -f 'exo.main.*--node-port $THOR_PORT' > /dev/null"; then
        test_inference "$THOR_PORT" "Thor" || log_error "Inference test failed on Thor"
    else
        log_warning "Skipping inference test (server not running)"
    fi
    echo

    collect_diagnostics "$THOR_NODE"
fi

# Test Jetson
if [[ "$TARGET_NODE" == "all" ]] || [[ "$TARGET_NODE" == "jetson" ]]; then
    echo
    log_info "=== Testing Jetson Node ==="
    echo

    test_nvrtc_library "$JETSON_NODE" || log_error "NVRTC library test failed on Jetson"
    echo

    test_server_startup "$JETSON_NODE" "$JETSON_PORT" || log_warning "Server not running on Jetson"
    echo

    test_nvrtc_logs "$JETSON_NODE" || log_error "Log analysis failed on Jetson"
    echo

    # Only test inference if server is running
    if ssh "$JETSON_NODE" "pgrep -f 'exo.main.*--node-port $JETSON_PORT' > /dev/null"; then
        test_inference "$JETSON_PORT" "Jetson" || log_error "Inference test failed on Jetson"
    else
        log_warning "Skipping inference test (server not running)"
    fi
    echo

    collect_diagnostics "$JETSON_NODE"
fi

echo
log_success "============================================"
log_success "  TEST SUITE COMPLETE"
log_success "============================================"
echo

log_info "Review diagnostics files for detailed information"
log_info "If tests failed, check logs and consider rollback:"
echo "  ./rollback.sh"
echo

log_success "Done!"
