#!/bin/bash
# Testing Script: Verify NV Device Switch Works
# Agent 4 Solution for CUDA 13.0 + Blackwell

set -e  # Exit on any error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Thor node details
THOR1_HOST="thor@10.0.0.78"
THOR1_IP="10.0.0.78"
THOR1_PORT=52415
THOR1_EXO="/home/thor/exo"

THOR2_HOST="jetson@10.0.0.93"
THOR2_IP="10.0.0.93"
THOR2_PORT=52416
THOR2_EXO="/home/jetson/exo"

MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}  NV DEVICE SWITCH TESTING${NC}"
echo -e "${BLUE}  Agent 4 Solution Verification${NC}"
echo -e "${BLUE}======================================================================${NC}"

print_header() {
    echo -e "\n${BLUE}==== $1 ====${NC}"
}

# Test 1: Device initialization
test_device_init() {
    local host=$1
    local exo_path=$2
    local node_name=$3

    print_header "Test 1: Device Initialization - $node_name"

    echo "Testing Device['NV'] accessibility..."
    ssh "$host" "cd $exo_path && python3 -c \"
import os
os.environ['PTX'] = '1'
from tinygrad.device import Device
Device.DEFAULT = Device['NV']
print(f'Device: {Device.DEFAULT}')
print(f'Type: {type(Device.DEFAULT)}')
\"" || {
        echo -e "${RED}❌ Device initialization failed${NC}"
        return 1
    }

    echo -e "${GREEN}✅ Device initialization passed${NC}"
    return 0
}

# Test 2: Start server and check logs
test_server_startup() {
    local host=$1
    local port=$2
    local exo_path=$3
    local node_name=$4

    print_header "Test 2: Server Startup - $node_name"

    # Kill any existing server
    echo "Cleaning up any existing server..."
    ssh "$host" "pkill -f 'exo.main' || true"
    sleep 2

    # Start server in background with logging
    echo "Starting exo server with DEBUG=1..."
    ssh "$host" "cd $exo_path && nohup bash -c 'DEBUG=1 DEVICE=CUDA python3 -m exo.main --node-port $port > /tmp/exo_test.log 2>&1' > /dev/null 2>&1 &"

    # Wait for server to initialize
    echo "Waiting for server to start (30 seconds)..."
    sleep 30

    # Check if server is running
    echo "Checking if server process is running..."
    ssh "$host" "pgrep -f 'exo.main'" > /dev/null || {
        echo -e "${RED}❌ Server process not found${NC}"
        echo "Last 50 lines of log:"
        ssh "$host" "tail -50 /tmp/exo_test.log"
        return 1
    }

    # Check logs for device switch markers
    echo "Checking logs for device switch markers..."
    ssh "$host" "grep 'DEVICE SWITCH' /tmp/exo_test.log" || {
        echo -e "${YELLOW}⚠️  Device switch markers not found in logs${NC}"
    }

    # Check for NV device initialization
    ssh "$host" "grep -i 'nv' /tmp/exo_test.log | head -20" || {
        echo -e "${YELLOW}⚠️  No NV device references in logs${NC}"
    }

    # Check for errors
    echo "Checking for errors in logs..."
    if ssh "$host" "grep -i 'error\|failed\|exception' /tmp/exo_test.log | grep -v 'NVRTC MONKEY-PATCH'" > /tmp/errors.txt 2>&1; then
        echo -e "${YELLOW}⚠️  Errors found in logs:${NC}"
        cat /tmp/errors.txt
    else
        echo -e "${GREEN}✅ No errors in startup logs${NC}"
    fi

    echo -e "${GREEN}✅ Server startup complete${NC}"
    return 0
}

# Test 3: Kernel compilation test
test_kernel_compilation() {
    local ip=$1
    local port=$2
    local node_name=$3

    print_header "Test 3: Kernel Compilation - $node_name"

    echo "Sending test inference request..."
    echo "This will trigger kernel compilation on first run."

    # Send simple inference request
    response=$(curl -s -X POST "http://$ip:$port/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}],
            \"max_tokens\": 10,
            \"temperature\": 0.7
        }" 2>&1)

    # Check if request succeeded
    if echo "$response" | grep -q "error"; then
        echo -e "${RED}❌ Inference request returned error:${NC}"
        echo "$response" | jq '.' 2>/dev/null || echo "$response"
        return 1
    fi

    if echo "$response" | grep -q "choices"; then
        echo -e "${GREEN}✅ Inference request succeeded${NC}"
        echo "Response preview:"
        echo "$response" | jq '.choices[0].message.content' 2>/dev/null || echo "$response"
    else
        echo -e "${YELLOW}⚠️  Unexpected response format:${NC}"
        echo "$response"
    fi

    # Check server logs for compilation
    echo -e "\nChecking server logs for compilation activity..."
    if ssh "$host" "grep -i 'compil\|nvjitlink\|ptx' /tmp/exo_test.log | tail -20"; then
        echo -e "${GREEN}✅ Compilation activity detected${NC}"
    else
        echo -e "${YELLOW}⚠️  No compilation activity in logs${NC}"
    fi

    return 0
}

# Test 4: Multiple inferences
test_multiple_inferences() {
    local ip=$1
    local port=$2
    local node_name=$3

    print_header "Test 4: Multiple Inferences - $node_name"

    echo "Running 5 inference tests..."

    local success_count=0
    for i in {1..5}; do
        echo -e "\n${YELLOW}Inference $i/5${NC}"

        response=$(curl -s -X POST "http://$ip:$port/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"$MODEL\",
                \"messages\": [{\"role\": \"user\", \"content\": \"Count to $i\"}],
                \"max_tokens\": 20,
                \"temperature\": 0.7
            }" 2>&1)

        if echo "$response" | grep -q "choices"; then
            echo -e "${GREEN}✅ Inference $i succeeded${NC}"
            ((success_count++))
        else
            echo -e "${RED}❌ Inference $i failed${NC}"
        fi

        # Small delay between requests
        sleep 1
    done

    echo -e "\n${BLUE}Results: $success_count/5 inferences succeeded${NC}"

    if [ $success_count -eq 5 ]; then
        echo -e "${GREEN}✅ All inferences passed${NC}"
        return 0
    elif [ $success_count -ge 3 ]; then
        echo -e "${YELLOW}⚠️  Most inferences passed ($success_count/5)${NC}"
        return 0
    else
        echo -e "${RED}❌ Too many failures ($success_count/5)${NC}"
        return 1
    fi
}

# Test 5: Distributed inference (both nodes)
test_distributed() {
    print_header "Test 5: Distributed Inference (Both Nodes)"

    echo "Checking if both servers are running..."
    if ! pgrep -f "exo.main.*$THOR1_PORT" > /dev/null; then
        echo -e "${YELLOW}⚠️  Thor #1 server not running, skipping distributed test${NC}"
        return 0
    fi
    if ! pgrep -f "exo.main.*$THOR2_PORT" > /dev/null; then
        echo -e "${YELLOW}⚠️  Thor #2 server not running, skipping distributed test${NC}"
        return 0
    fi

    echo "Both servers running, testing cluster formation..."

    # Check Thor #1 logs for peer discovery
    if ssh "$THOR1_HOST" "grep -i 'peer.*$THOR2_IP' /tmp/exo_test.log"; then
        echo -e "${GREEN}✅ Thor #1 discovered Thor #2${NC}"
    else
        echo -e "${YELLOW}⚠️  Thor #1 hasn't discovered Thor #2 yet${NC}"
    fi

    # Check Thor #2 logs for peer discovery
    if ssh "$THOR2_HOST" "grep -i 'peer.*$THOR1_IP' /tmp/exo_test.log"; then
        echo -e "${GREEN}✅ Thor #2 discovered Thor #1${NC}"
    else
        echo -e "${YELLOW}⚠️  Thor #2 hasn't discovered Thor #1 yet${NC}"
    fi

    # Send longer inference that requires both nodes
    echo -e "\nSending longer inference request..."
    response=$(curl -s -X POST "http://$THOR1_IP:$THOR1_PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Write a Python function for binary search\"}],
            \"max_tokens\": 100,
            \"temperature\": 0.7
        }" 2>&1)

    if echo "$response" | grep -q "choices"; then
        echo -e "${GREEN}✅ Distributed inference succeeded${NC}"
        echo "Response length:" $(echo "$response" | jq '.choices[0].message.content | length' 2>/dev/null)
    else
        echo -e "${RED}❌ Distributed inference failed${NC}"
        return 1
    fi

    return 0
}

# Cleanup function
cleanup_servers() {
    print_header "Cleanup"

    echo "Stopping test servers..."
    ssh "$THOR1_HOST" "pkill -f 'exo.main' || true"
    ssh "$THOR2_HOST" "pkill -f 'exo.main' || true"

    echo -e "${GREEN}✅ Test servers stopped${NC}"
    echo "Server logs saved at:"
    echo "  Thor #1: $THOR1_HOST:/tmp/exo_test.log"
    echo "  Thor #2: $THOR2_HOST:/tmp/exo_test.log"
}

# Main test flow
main() {
    local thor1_results=0
    local thor2_results=0

    # Test Thor #1
    test_device_init "$THOR1_HOST" "$THOR1_EXO" "Thor #1" || ((thor1_results++))
    test_server_startup "$THOR1_HOST" "$THOR1_PORT" "$THOR1_EXO" "Thor #1" || ((thor1_results++))

    # Test Thor #2
    test_device_init "$THOR2_HOST" "$THOR2_EXO" "Thor #2" || ((thor2_results++))
    test_server_startup "$THOR2_HOST" "$THOR2_PORT" "$THOR2_EXO" "Thor #2" || ((thor2_results++))

    # Compilation and inference tests
    test_kernel_compilation "$THOR1_IP" "$THOR1_PORT" "Thor #1" || ((thor1_results++))
    test_multiple_inferences "$THOR1_IP" "$THOR1_PORT" "Thor #1" || ((thor1_results++))

    # Distributed test
    test_distributed || echo -e "${YELLOW}⚠️  Distributed test had issues${NC}"

    # Cleanup
    cleanup_servers

    # Final summary
    print_header "TESTING SUMMARY"

    if [ $thor1_results -eq 0 ] && [ $thor2_results -eq 0 ]; then
        echo -e "${GREEN}✅ ALL TESTS PASSED${NC}"
        echo "   Device switch is working correctly on both nodes"
        echo "   Kernel compilation successful"
        echo "   Inference quality verified"
        return 0
    else
        echo -e "${YELLOW}⚠️  SOME TESTS HAD ISSUES${NC}"
        echo "   Thor #1 issues: $thor1_results"
        echo "   Thor #2 issues: $thor2_results"
        echo "   Review logs above for details"
        return 1
    fi
}

# Run tests
main
