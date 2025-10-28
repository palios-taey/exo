#!/bin/bash
# Phase 5: End-to-End Inference Test
# Tests complete distributed inference across Thor cluster

set -e

echo "============================================================"
echo "Phase 5: End-to-End Inference Test"
echo "============================================================"

# Configuration
THOR1="jetson@10.0.0.8"
THOR2="thor@10.0.0.78"
PORT1=52415
PORT2=52416
MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"

# Test parameters
PROMPT="Write a Python function to calculate fibonacci numbers"
MAX_TOKENS=100
TEMPERATURE=0.7

echo ""
echo "=== TEST 1: Check Exo Servers Running ==="
echo ""

echo "Checking Thor #1..."
ssh $THOR1 'pgrep -f "python3.*exo/main.py" && echo "✅ Exo running" || echo "❌ Exo not running"'

echo "Checking Thor #2..."
ssh $THOR2 'pgrep -f "python3.*exo/main.py" && echo "✅ Exo running" || echo "❌ Exo not running"'

echo ""
echo "=== TEST 2: Check API Endpoints ==="
echo ""

echo "Testing Thor #1 API..."
curl -s -m 5 http://10.0.0.8:$PORT1/health && echo " ✅" || echo " ❌"

echo "Testing Thor #2 API..."
curl -s -m 5 http://10.0.0.78:$PORT2/health && echo " ✅" || echo " ❌"

echo ""
echo "=== TEST 3: Send Inference Request ==="
echo ""

# Send request to Thor #2 (primary)
echo "Sending inference request to Thor #2..."

RESPONSE=$(curl -s -X POST http://10.0.0.78:$PORT2/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"$PROMPT\"}
    ],
    \"max_tokens\": $MAX_TOKENS,
    \"temperature\": $TEMPERATURE
  }")

echo "Response:"
echo "$RESPONSE" | python3 -m json.tool

# Check for errors
if echo "$RESPONSE" | grep -q "error\|Error\|ERROR"; then
  echo "❌ Inference request failed"
  exit 1
else
  echo "✅ Inference request succeeded"
fi

echo ""
echo "=== TEST 4: Check Distributed Computation ==="
echo ""

echo "Checking Thor #1 logs for coordination..."
ssh $THOR1 'tail -50 /tmp/thor_exo.log | grep -i "shard\|coordinate\|peer" | tail -10' || echo "No coordination logs"

echo ""
echo "Checking Thor #2 logs for coordination..."
ssh $THOR2 'tail -50 /tmp/thor_exo.log | grep -i "shard\|coordinate\|peer" | tail -10' || echo "No coordination logs"

echo ""
echo "=== TEST 5: Performance Metrics ==="
echo ""

# Extract timing from response (if available)
LATENCY=$(echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'usage' in data:
        print(f\"Tokens: {data['usage'].get('total_tokens', 'N/A')}\")
    if 'latency' in data:
        print(f\"Latency: {data['latency']}s\")
except:
    pass
")

echo "$LATENCY"

# Check GPU utilization on both devices
echo ""
echo "Thor #1 GPU utilization:"
ssh $THOR1 'nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader' || echo "N/A (nvidia-smi issue)"

echo ""
echo "Thor #2 GPU utilization:"
ssh $THOR2 'nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader' || echo "N/A (nvidia-smi issue)"

echo ""
echo "============================================================"
echo "PHASE 5: END-TO-END TEST COMPLETE"
echo "============================================================"
echo ""
echo "✅ Success criteria:"
echo "  - Both servers running"
echo "  - API endpoints responding"
echo "  - Inference request completes"
echo "  - Distributed coordination visible in logs"
echo "  - GPU utilization >70% on active device"
echo ""
echo "⚠️  Review logs and metrics above to verify all criteria met"
