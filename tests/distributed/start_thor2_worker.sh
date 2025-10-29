#!/bin/bash
#
# Start Exo Server on Thor #2 (Worker Node)
# Device: Jetson Thor at 10.0.0.78
#

THOR2_IP="10.0.0.78"
THOR2_USER="thor"
THOR2_PASSWORD="papaDons1001s$"
THOR2_PORT="52415"
EXO_PATH="/home/thor/exo"
LOG_FILE="/tmp/exo_thor2_worker.log"

# Default model (can override with argument)
MODEL="${1:-llama-3.2-1b}"

echo "=========================================="
echo "  Starting Exo on Thor #2 (Worker Node)"
echo "=========================================="
echo "  IP: $THOR2_IP"
echo "  Port: $THOR2_PORT"
echo "  Model: $MODEL"
echo "  Log: $LOG_FILE"
echo "=========================================="

# Stop any existing processes
echo "Stopping any existing exo processes..."
sshpass -p "$THOR2_PASSWORD" ssh -o StrictHostKeyChecking=no "${THOR2_USER}@${THOR2_IP}" \
    "pkill -9 -f 'exo/main.py' || true"

sleep 2

# Start server
echo "Starting exo server..."

START_COMMAND="cd ${EXO_PATH} && \
DEVICE=CUDA DEBUG=3 nohup python3 exo/main.py \
--inference-engine tinygrad \
--default-model ${MODEL} \
--chatgpt-api-port ${THOR2_PORT} \
--node-port ${THOR2_PORT} \
--node-host 0.0.0.0 \
--disable-tui \
> ${LOG_FILE} 2>&1 & \
echo \$!"

PID=$(sshpass -p "$THOR2_PASSWORD" ssh -o StrictHostKeyChecking=no "${THOR2_USER}@${THOR2_IP}" "$START_COMMAND")

if [ -n "$PID" ]; then
    echo "✓ Server started with PID: $PID"
    echo "  API endpoint: http://${THOR2_IP}:${THOR2_PORT}"
    echo "  Log location: ${THOR2_IP}:${LOG_FILE}"
    echo ""
    echo "Waiting 10s for server to initialize..."
    sleep 10

    # Check if still running
    STILL_RUNNING=$(sshpass -p "$THOR2_PASSWORD" ssh -o StrictHostKeyChecking=no "${THOR2_USER}@${THOR2_IP}" \
        "ps -p $PID > /dev/null && echo 'yes' || echo 'no'")

    if [ "$STILL_RUNNING" = "yes" ]; then
        echo "✓ Server is running"
        echo ""
        echo "To monitor logs:"
        echo "  ssh ${THOR2_USER}@${THOR2_IP} 'tail -f ${LOG_FILE}'"
        echo ""
        echo "To stop server:"
        echo "  ssh ${THOR2_USER}@${THOR2_IP} 'kill ${PID}'"
        exit 0
    else
        echo "✗ Server process died"
        echo ""
        echo "Last 30 lines of log:"
        sshpass -p "$THOR2_PASSWORD" ssh -o StrictHostKeyChecking=no "${THOR2_USER}@${THOR2_IP}" \
            "tail -30 ${LOG_FILE}"
        exit 1
    fi
else
    echo "✗ Failed to start server"
    exit 1
fi
