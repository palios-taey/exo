#!/bin/bash
#
# Monitor Both Thor Devices
# Shows GPU usage, network traffic, and log tails in real-time
#

THOR1_IP="10.0.0.93"
THOR1_USER="jetson"
THOR1_PASSWORD="papaDons1001s$"
THOR1_LOG="/tmp/exo_thor1_head.log"

THOR2_IP="10.0.0.78"
THOR2_USER="thor"
THOR2_PASSWORD="papaDons1001s$"
THOR2_LOG="/tmp/exo_thor2_worker.log"

# Interval in seconds
INTERVAL="${1:-5}"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

ssh_execute() {
    local host=$1
    local user=$2
    local password=$3
    local command=$4

    sshpass -p "$password" ssh -o StrictHostKeyChecking=no "${user}@${host}" "$command" 2>/dev/null
}

monitor_gpu() {
    local host=$1
    local user=$2
    local password=$3
    local device_name=$4

    local gpu_info=$(ssh_execute "$host" "$user" "$password" \
        "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader")

    echo -e "${GREEN}$device_name GPU:${NC}"
    echo "  $gpu_info"
}

monitor_network() {
    local host=$1
    local user=$2
    local password=$3
    local device_name=$4

    # Get network connections related to exo
    local connections=$(ssh_execute "$host" "$user" "$password" \
        "netstat -tn | grep ESTABLISHED | grep -E ':(5678|52415|52416)' | wc -l")

    echo -e "${GREEN}$device_name Network:${NC}"
    echo "  Active connections: $connections"
}

monitor_process() {
    local host=$1
    local user=$2
    local password=$3
    local device_name=$4

    # Check if exo process is running
    local pid=$(ssh_execute "$host" "$user" "$password" \
        "pgrep -f 'exo/main.py' || echo 'none'")

    echo -e "${GREEN}$device_name Process:${NC}"
    if [ "$pid" != "none" ]; then
        echo -e "  ${GREEN}✓${NC} Running (PID: $pid)"

        # Get CPU and memory usage
        local process_info=$(ssh_execute "$host" "$user" "$password" \
            "ps -p $pid -o %cpu,%mem,etime --no-headers")
        echo "  $process_info"
    else
        echo -e "  ${YELLOW}✗ Not running${NC}"
    fi
}

show_log_tail() {
    local host=$1
    local user=$2
    local password=$3
    local log_file=$4
    local device_name=$5
    local lines=$6

    echo -e "${GREEN}$device_name Log (last $lines lines):${NC}"
    ssh_execute "$host" "$user" "$password" "tail -$lines $log_file 2>/dev/null || echo 'Log not found'"
    echo ""
}

main() {
    # Check if we should show logs or just stats
    MODE="${2:-stats}"  # stats or logs

    if [ "$MODE" = "logs" ]; then
        # Show log tails once
        clear
        print_header "Thor Device Logs"
        date
        echo ""

        show_log_tail "$THOR1_IP" "$THOR1_USER" "$THOR1_PASSWORD" "$THOR1_LOG" "Thor #1" 20
        show_log_tail "$THOR2_IP" "$THOR2_USER" "$THOR2_PASSWORD" "$THOR2_LOG" "Thor #2" 20

        echo "To continuously monitor, use:"
        echo "  ssh ${THOR1_USER}@${THOR1_IP} 'tail -f ${THOR1_LOG}'"
        echo "  ssh ${THOR2_USER}@${THOR2_IP} 'tail -f ${THOR2_LOG}'"

    else
        # Continuous monitoring mode
        echo "Monitoring Thor devices every ${INTERVAL}s (Ctrl+C to stop)"
        echo ""

        while true; do
            clear
            print_header "Thor Distributed Monitoring"
            date
            echo ""

            # Thor #1
            print_header "Thor #1 (Head) - ${THOR1_IP}"
            monitor_process "$THOR1_IP" "$THOR1_USER" "$THOR1_PASSWORD" "Thor #1"
            echo ""
            monitor_gpu "$THOR1_IP" "$THOR1_USER" "$THOR1_PASSWORD" "Thor #1"
            echo ""
            monitor_network "$THOR1_IP" "$THOR1_USER" "$THOR1_PASSWORD" "Thor #1"
            echo ""

            # Thor #2
            print_header "Thor #2 (Worker) - ${THOR2_IP}"
            monitor_process "$THOR2_IP" "$THOR2_USER" "$THOR2_PASSWORD" "Thor #2"
            echo ""
            monitor_gpu "$THOR2_IP" "$THOR2_USER" "$THOR2_PASSWORD" "Thor #2"
            echo ""
            monitor_network "$THOR2_IP" "$THOR2_USER" "$THOR2_PASSWORD" "Thor #2"
            echo ""

            echo "Press Ctrl+C to stop monitoring"
            echo "To view logs, run: $0 $INTERVAL logs"

            sleep "$INTERVAL"
        done
    fi
}

main
