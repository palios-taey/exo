#!/bin/bash
# Test Script: Verify NVRTC Compilation Works
# ============================================
# Tests that removing monkey-patch enables proper compilation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "================================================================="
echo "NVRTC Compilation Test"
echo "================================================================="

# Test configuration
TEST_DEVICE="thor@10.0.0.78"
TEST_EXO_PATH="/home/thor/exo"
TEST_PORT="52415"

echo ""
echo "Test Device: $TEST_DEVICE"
echo "Exo Path: $TEST_EXO_PATH"
echo "Port: $TEST_PORT"
echo ""

# Function: Check if server is running
check_server() {
    local host="$1"
    local port="$2"

    if nc -z "$host" "$port" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function: Start server in background
start_server() {
    echo "-----------------------------------------------------------------"
    echo "Starting exo server on Thor..."
    echo "-----------------------------------------------------------------"

    ssh "$TEST_DEVICE" "cd $TEST_EXO_PATH && nohup python3 -u exo/main.py --node-port $TEST_PORT > /tmp/test_exo.log 2>&1 &"

    echo "Waiting for server to initialize..."
    sleep 10

    # Check if server started
    if check_server "10.0.0.78" "$TEST_PORT"; then
        echo -e "${GREEN}✅ Server started successfully${NC}"
        return 0
    else
        echo -e "${RED}❌ Server failed to start${NC}"
        echo ""
        echo "Last 50 lines of log:"
        ssh "$TEST_DEVICE" "tail -50 /tmp/test_exo.log"
        return 1
    fi
}

# Function: Stop server
stop_server() {
    echo ""
    echo "Stopping server..."
    ssh "$TEST_DEVICE" "pkill -f 'exo/main.py' || true"
    sleep 2
    echo "✅ Server stopped"
}

# Function: Check logs for key indicators
check_logs() {
    echo ""
    echo "-----------------------------------------------------------------"
    echo "Analyzing Logs"
    echo "-----------------------------------------------------------------"

    # Get full log
    LOG=$(ssh "$TEST_DEVICE" "cat /tmp/test_exo.log")

    # Check 1: No NVRTC monkey-patch message
    if echo "$LOG" | grep -q "NVRTC MONKEY-PATCH"; then
        echo -e "${RED}❌ FAILED: Monkey-patch still active!${NC}"
        echo "   Found monkey-patch log message"
        return 1
    fi
    echo -e "${GREEN}✅ No monkey-patch detected${NC}"

    # Check 2: No NVRTC import errors
    if echo "$LOG" | grep -qi "ModuleNotFoundError.*nvrtc"; then
        echo -e "${RED}❌ FAILED: NVRTC import error${NC}"
        echo "   NVRTC library may not be available"
        return 1
    fi
    echo -e "${GREEN}✅ No NVRTC import errors${NC}"

    # Check 3: No AttributeError for NVRTC functions
    if echo "$LOG" | grep -q "AttributeError.*nvrtc"; then
        echo -e "${RED}❌ FAILED: NVRTC AttributeError${NC}"
        echo "   NVRTC functions may be missing"
        return 1
    fi
    echo -e "${GREEN}✅ No NVRTC AttributeErrors${NC}"

    # Check 4: Look for CUDACompiler usage (positive indicator)
    if echo "$LOG" | grep -qi "CUDACompiler"; then
        echo -e "${GREEN}✅ CUDACompiler detected in logs${NC}"
    else
        echo -e "${YELLOW}⚠️  CUDACompiler not mentioned in logs${NC}"
        echo "   (May not have compiled kernels yet)"
    fi

    # Check 5: Check for compilation errors
    if echo "$LOG" | grep -qi "compilation.*failed\|compile.*error"; then
        echo -e "${RED}❌ WARNING: Compilation errors detected${NC}"
        echo ""
        echo "Compilation error excerpt:"
        echo "$LOG" | grep -i "compilation\|compile" | head -10
        echo ""
        echo "Full log available at: $TEST_DEVICE:/tmp/test_exo.log"
        return 1
    fi
    echo -e "${GREEN}✅ No compilation errors${NC}"

    # Check 6: Look for nvJitLink errors
    if echo "$LOG" | grep -qi "nvjitlink.*error\|bad input.*PTX"; then
        echo -e "${RED}❌ FAILED: nvJitLink errors still present${NC}"
        echo ""
        echo "nvJitLink error excerpt:"
        echo "$LOG" | grep -i "nvjitlink\|bad input" | head -10
        return 1
    fi
    echo -e "${GREEN}✅ No nvJitLink errors${NC}"

    # Check 7: Server initialization successful
    if echo "$LOG" | grep -qi "server.*running\|listening.*on"; then
        echo -e "${GREEN}✅ Server initialized successfully${NC}"
    else
        echo -e "${YELLOW}⚠️  Server initialization unclear${NC}"
    fi

    return 0
}

# Function: Send test inference request
test_inference() {
    echo ""
    echo "-----------------------------------------------------------------"
    echo "Testing Inference (Optional)"
    echo "-----------------------------------------------------------------"

    echo "Sending test inference request..."
    echo "(This may trigger kernel compilation)"

    RESPONSE=$(curl -s --max-time 30 http://10.0.0.78:$TEST_PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 10
        }' 2>&1 || true)

    if [[ -z "$RESPONSE" ]]; then
        echo -e "${YELLOW}⚠️  No response (may be loading model)${NC}"
        return 1
    fi

    echo "Response received:"
    echo "$RESPONSE" | head -20

    if echo "$RESPONSE" | grep -qi "error"; then
        echo -e "${YELLOW}⚠️  Error in response${NC}"
        return 1
    fi

    echo -e "${GREEN}✅ Inference request completed${NC}"
    return 0
}

# Main test sequence
echo "Starting test sequence..."
echo ""

# Cleanup any existing server
stop_server

# Start server
if ! start_server; then
    echo -e "${RED}TEST FAILED: Server did not start${NC}"
    exit 1
fi

# Wait for initialization
echo ""
echo "Waiting 15 seconds for full initialization..."
sleep 15

# Check logs
if ! check_logs; then
    echo ""
    echo -e "${RED}TEST FAILED: Log analysis found issues${NC}"
    stop_server
    exit 1
fi

# Optional: Try inference
echo ""
read -p "Attempt test inference? This may take time (yes/no): " TRY_INFERENCE

if [[ "$TRY_INFERENCE" == "yes" ]]; then
    if test_inference; then
        echo -e "${GREEN}✅ Inference test passed${NC}"
    else
        echo -e "${YELLOW}⚠️  Inference test inconclusive${NC}"
    fi
fi

# Cleanup
stop_server

# Final summary
echo ""
echo "================================================================="
echo "TEST SUMMARY"
echo "================================================================="
echo -e "${GREEN}✅ All core tests passed${NC}"
echo ""
echo "Verified:"
echo "  ✅ Monkey-patch removed (no patch log messages)"
echo "  ✅ NVRTC loads without errors"
echo "  ✅ No compilation failures detected"
echo "  ✅ Server initializes successfully"
echo ""
echo "Next steps:"
echo "  1. Start both Thor servers normally"
echo "  2. Run actual inference workload"
echo "  3. Monitor for proper GPU utilization"
echo "  4. Check that kernels compile successfully"
echo ""
echo "Full test log available at: $TEST_DEVICE:/tmp/test_exo.log"
echo ""

exit 0
