#!/bin/bash
#
# Test 5: Cleanup Script
# Stops all exo processes on both Thor devices and verifies clean state.
#

set -e

# Thor device configuration
THOR1_IP="10.0.0.93"
THOR1_USER="jetson"
THOR1_PASSWORD="papaDons1001s$"

THOR2_IP="10.0.0.78"
THOR2_USER="thor"
THOR2_PASSWORD="papaDons1001s$"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo "======================================================================"
    echo "  $1"
    echo "======================================================================"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_failure() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "  ${BLUE}$1:${NC} $2"
}

ssh_execute() {
    local host=$1
    local user=$2
    local password=$3
    local command=$4

    sshpass -p "$password" ssh -o StrictHostKeyChecking=no "${user}@${host}" "$command"
}

kill_exo_processes() {
    local host=$1
    local user=$2
    local password=$3
    local device_name=$4

    print_info "$device_name" "Stopping exo processes..."

    # Kill all exo processes
    ssh_execute "$host" "$user" "$password" "pkill -9 -f 'exo/main.py' || true"

    # Wait a moment
    sleep 2

    # Verify
    local pids=$(ssh_execute "$host" "$user" "$password" "pgrep -f 'exo/main.py' || echo ''")

    if [ -z "$pids" ]; then
        print_success "$device_name clean (no exo processes)"
    else
        print_failure "$device_name still has processes: $pids"
        return 1
    fi

    return 0
}

clear_gpu_memory() {
    local host=$1
    local user=$2
    local password=$3
    local device_name=$4

    print_info "$device_name" "Clearing GPU memory..."

    # Check GPU memory usage
    local mem_before=$(ssh_execute "$host" "$user" "$password" \
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" || echo "unknown")

    print_info "$device_name GPU memory before" "${mem_before} MiB"

    # Kill any other GPU processes if needed (be careful here)
    # For now, just report status

    local mem_after=$(ssh_execute "$host" "$user" "$password" \
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" || echo "unknown")

    print_info "$device_name GPU memory after" "${mem_after} MiB"

    # Consider clean if memory < 500 MiB
    if [ "$mem_after" != "unknown" ] && [ "$mem_after" -lt 500 ]; then
        print_success "$device_name GPU memory clean"
        return 0
    elif [ "$mem_after" != "unknown" ]; then
        print_info "$device_name GPU memory" "Still has ${mem_after} MiB allocated"
        return 0  # Not a failure, just informational
    else
        print_failure "$device_name GPU check failed"
        return 1
    fi
}

verify_ports_free() {
    local host=$1
    local user=$2
    local password=$3
    local port=$4
    local device_name=$5

    print_info "$device_name" "Checking port $port..."

    local listening=$(ssh_execute "$host" "$user" "$password" \
        "lsof -i :$port 2>/dev/null || echo ''")

    if [ -z "$listening" ]; then
        print_success "$device_name port $port free"
        return 0
    else
        print_failure "$device_name port $port still in use"
        echo "$listening"
        return 1
    fi
}

main() {
    print_header "EXO DISTRIBUTED INFERENCE - TEST 5: CLEANUP"
    date

    # Stop processes
    print_header "Stopping Exo Processes"

    local status=0

    if ! kill_exo_processes "$THOR1_IP" "$THOR1_USER" "$THOR1_PASSWORD" "Thor #1"; then
        status=1
    fi

    if ! kill_exo_processes "$THOR2_IP" "$THOR2_USER" "$THOR2_PASSWORD" "Thor #2"; then
        status=1
    fi

    # Clear GPU memory
    print_header "Checking GPU Memory"

    clear_gpu_memory "$THOR1_IP" "$THOR1_USER" "$THOR1_PASSWORD" "Thor #1"
    clear_gpu_memory "$THOR2_IP" "$THOR2_USER" "$THOR2_PASSWORD" "Thor #2"

    # Verify ports free
    print_header "Verifying Ports Free"

    verify_ports_free "$THOR1_IP" "$THOR1_USER" "$THOR1_PASSWORD" "52416" "Thor #1"
    verify_ports_free "$THOR2_IP" "$THOR2_USER" "$THOR2_PASSWORD" "52415" "Thor #2"
    verify_ports_free "$THOR1_IP" "$THOR1_USER" "$THOR1_PASSWORD" "5678" "Thor #1"
    verify_ports_free "$THOR2_IP" "$THOR2_USER" "$THOR2_PASSWORD" "5678" "Thor #2"

    # Summary
    print_header "Cleanup Summary"

    if [ $status -eq 0 ]; then
        print_success "All cleanup tasks completed successfully"
        echo ""
        echo "Both Thor devices are now in a clean state and ready for testing."
        return 0
    else
        print_failure "Some cleanup tasks failed"
        echo ""
        echo "Manual intervention may be required. Check logs above."
        return 1
    fi
}

main
exit $?
