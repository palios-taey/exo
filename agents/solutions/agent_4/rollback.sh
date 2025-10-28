#!/bin/bash
# Rollback Script: Restore Pre-NV-Switch State
# Agent 4 Solution Rollback

set -e  # Exit on any error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Thor node details
THOR1_HOST="thor@10.0.0.78"
THOR1_EXO="/home/thor/exo"

THOR2_HOST="jetson@10.0.0.93"
THOR2_EXO="/home/jetson/exo"

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}  NV DEVICE SWITCH ROLLBACK${NC}"
echo -e "${BLUE}  Restoring Pre-Switch State${NC}"
echo -e "${BLUE}======================================================================${NC}"

print_header() {
    echo -e "\n${BLUE}==== $1 ====${NC}"
}

# Function: Rollback a single node
rollback_node() {
    local host=$1
    local exo_path=$2
    local node_name=$3

    print_header "Rolling back $node_name ($host)"

    # Check if backup exists
    echo "Checking for backup file..."
    if ! ssh "$host" "[ -f $exo_path/exo/inference/tinygrad/inference.py.pre_nv_switch ]"; then
        echo -e "${RED}❌ Backup file not found${NC}"
        echo "   Expected: $exo_path/exo/inference/tinygrad/inference.py.pre_nv_switch"
        echo "   Cannot rollback without backup"
        return 1
    fi
    echo -e "${GREEN}✅ Backup file found${NC}"

    # Stop any running server
    echo "Stopping exo server..."
    ssh "$host" "pkill -f 'exo.main' || true"
    sleep 2

    # Restore backup
    echo "Restoring pre-switch version..."
    ssh "$host" "cd $exo_path && cp exo/inference/tinygrad/inference.py.pre_nv_switch exo/inference/tinygrad/inference.py" || {
        echo -e "${RED}❌ Failed to restore backup${NC}"
        return 1
    }
    echo -e "${GREEN}✅ Backup restored${NC}"

    # Verify rollback
    echo "Verifying rollback..."
    if ssh "$host" "grep -q 'DEVICE SWITCH' $exo_path/exo/inference/tinygrad/inference.py"; then
        echo -e "${RED}❌ Rollback verification failed - new code still present${NC}"
        return 1
    fi
    echo -e "${GREEN}✅ Rollback verified - device switch code removed${NC}"

    # Check what device logic is now present
    echo "Checking restored device logic..."
    ssh "$host" "grep -A 2 'Device.DEFAULT' $exo_path/exo/inference/tinygrad/inference.py | head -5"

    echo -e "${GREEN}✅ $node_name rollback complete${NC}"
    return 0
}

# Main rollback flow
main() {
    print_header "Pre-Rollback Checks"

    echo -e "${YELLOW}WARNING: This will restore the pre-NV-switch code${NC}"
    echo "  - Device['NV'] changes will be removed"
    echo "  - PTX=1 setting will be removed"
    echo "  - System will revert to Device['CUDA'] (which may have compilation issues)"
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " confirm

    if [ "$confirm" != "yes" ]; then
        echo -e "${YELLOW}Rollback cancelled${NC}"
        exit 0
    fi

    # Rollback Thor #1
    rollback_node "$THOR1_HOST" "$THOR1_EXO" "Thor #1" || {
        echo -e "${RED}❌ Thor #1 rollback failed${NC}"
        exit 1
    }

    # Rollback Thor #2
    rollback_node "$THOR2_HOST" "$THOR2_EXO" "Thor #2" || {
        echo -e "${RED}❌ Thor #2 rollback failed${NC}"
        exit 1
    }

    print_header "Rollback Summary"

    echo -e "${GREEN}✅ Rollback complete on both nodes${NC}"
    echo ""
    echo "System restored to pre-NV-switch state:"
    echo "  - Device: CUDA (not NV)"
    echo "  - PTX: Not forced to 1"
    echo "  - Compiler: Will attempt CUDACompiler (may fail on CUDA 13.0)"
    echo ""
    echo "Note: Original NVRTC monkey-patch may still be present"
    echo ""
    echo "To re-apply device switch:"
    echo "  ./install.sh"
    echo ""
    echo "Backup files preserved at:"
    echo "  Thor #1: $THOR1_EXO/exo/inference/tinygrad/inference.py.pre_nv_switch"
    echo "  Thor #2: $THOR2_EXO/exo/inference/tinygrad/inference.py.pre_nv_switch"
}

# Run rollback
main
