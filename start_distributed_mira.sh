#!/bin/bash
# Mira (RTX 4090) distributed inference startup
# Edison Cycle 3 - CUDA=1 environment fix

# CRITICAL: Set CUDA environment BEFORE importing tinygrad
export CUDA=1
export DEVICE=CUDA

# Node configuration
NODE_ID="mira"
NODE_PORT=50051
NODE_HOST="0.0.0.0"
CHATGPT_PORT=52415

# Manual discovery - specify Thor peers
DISCOVERY_MODULE="manual"
THOR1_PEER="10.0.0.93:50051"
THOR2_PEER="10.0.0.78:50051"

# Discovery config file path
DISCOVERY_CONFIG="/tmp/exo_discovery_mira.json"

echo "=========================================="
echo "EXO DISTRIBUTED INFERENCE - MIRA NODE"
echo "=========================================="
echo "Node ID: $NODE_ID"
echo "Host: $NODE_HOST"
echo "Port: $NODE_PORT"
echo "ChatGPT API: http://10.0.0.163:$CHATGPT_PORT"
echo "Discovery: $DISCOVERY_MODULE"
echo "Peers: $THOR1_PEER, $THOR2_PEER"
echo "Environment: CUDA=$CUDA, DEVICE=$DEVICE"
echo "=========================================="

# Create manual discovery config
cat > $DISCOVERY_CONFIG << EOF
{
  "peers": [
    {"host": "10.0.0.93", "port": 50051, "device_capabilities": {"model": "Thor", "memory": "128GB"}},
    {"host": "10.0.0.78", "port": 50051, "device_capabilities": {"model": "Thor", "memory": "128GB"}}
  ]
}
EOF

echo "Starting Mira node..."

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
