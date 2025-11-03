#!/bin/bash
# Thor #1 (Blackwell sm_110) distributed inference startup
# Edison Cycle 3 - CUDA=1 environment fix

# CRITICAL: Set CUDA environment BEFORE importing tinygrad
export CUDA=1
export DEVICE=CUDA

# Node configuration
NODE_ID="thor1"
NODE_PORT=50051
NODE_HOST="0.0.0.0"
CHATGPT_PORT=52415

# Manual discovery - specify peers
DISCOVERY_MODULE="manual"
MIRA_PEER="10.0.0.163:50051"
THOR2_PEER="10.0.0.78:50051"

# Discovery config file path (shared across all nodes)
DISCOVERY_CONFIG="/tmp/exo_discovery_3nodes.json"

echo "=========================================="
echo "EXO DISTRIBUTED INFERENCE - THOR #1 NODE"
echo "=========================================="
echo "Node ID: $NODE_ID"
echo "Host: $NODE_HOST"
echo "Port: $NODE_PORT"
echo "ChatGPT API: http://10.0.0.93:$CHATGPT_PORT"
echo "Discovery: $DISCOVERY_MODULE"
echo "Config: $DISCOVERY_CONFIG"
echo "Environment: CUDA=$CUDA, DEVICE=$DEVICE"
echo "=========================================="

echo "Starting Thor #1 node..."

# Start exo with manual discovery
python3 -m exo.main \
  --node-id "$NODE_ID" \
  --node-host "$NODE_HOST" \
  --node-port "$NODE_PORT" \
  --chatgpt-api-port "$CHATGPT_PORT" \
  --inference-engine tinygrad \
  --discovery-module "$DISCOVERY_MODULE" \
  --discovery-config-path "$DISCOVERY_CONFIG" \
  --wait-for-peers 2
