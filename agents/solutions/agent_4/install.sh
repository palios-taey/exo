#!/bin/bash
# Installation Script: Deploy NV Device Switch to Thor Nodes
# Agent 4 Solution for CUDA 13.0 + Blackwell

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Thor node details
THOR1_HOST="thor@10.0.0.78"
THOR1_PORT=52415
THOR1_EXO="/home/thor/exo"

THOR2_HOST="jetson@10.0.0.93"
THOR2_PORT=52416
THOR2_EXO="/home/jetson/exo"

# Password (from CLAUDE.md)
PASSWORD="papaDons1001s$"

# Solution directory
SOLUTION_DIR="/home/mira/exo/agents/solutions/agent_4"

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}  NV DEVICE SWITCH DEPLOYMENT${NC}"
echo -e "${BLUE}  Agent 4 Solution for CUDA 13.0 + Blackwell${NC}"
echo -e "${BLUE}======================================================================${NC}"

# Function: Print section header
print_header() {
    echo -e "\n${BLUE}==== $1 ====${NC}"
}

# Function: Deploy to a node
deploy_to_node() {
    local host=$1
    local exo_path=$2
    local node_name=$3

    print_header "Deploying to $node_name ($host)"

    # 1. Backup current inference.py
    echo -e "${YELLOW}[1/5]${NC} Creating backup..."
    ssh "$host" "cd $exo_path && cp exo/inference/tinygrad/inference.py exo/inference/tinygrad/inference.py.pre_nv_switch" || {
        echo -e "${RED}❌ Backup failed${NC}"
        return 1
    }
    echo -e "${GREEN}✅ Backup created${NC}"

    # 2. Copy patch file to node
    echo -e "${YELLOW}[2/5]${NC} Copying patch file..."
    scp "$SOLUTION_DIR/device_switch.patch" "$host:/tmp/" || {
        echo -e "${RED}❌ Failed to copy patch${NC}"
        return 1
    }
    echo -e "${GREEN}✅ Patch copied${NC}"

    # 3. Apply patch
    echo -e "${YELLOW}[3/5]${NC} Applying patch..."
    ssh "$host" "cd $exo_path && patch -p1 < /tmp/device_switch.patch" || {
        echo -e "${RED}❌ Patch failed to apply${NC}"
        echo -e "${YELLOW}Rolling back...${NC}"
        ssh "$host" "cd $exo_path && cp exo/inference/tinygrad/inference.py.pre_nv_switch exo/inference/tinygrad/inference.py"
        return 1
    }
    echo -e "${GREEN}✅ Patch applied${NC}"

    # 4. Verify patch applied
    echo -e "${YELLOW}[4/5]${NC} Verifying patch..."
    ssh "$host" "grep -q 'DEVICE SWITCH' $exo_path/exo/inference/tinygrad/inference.py" || {
        echo -e "${RED}❌ Verification failed - patch markers not found${NC}"
        return 1
    }
    ssh "$host" "grep -q 'Device\[\"NV\"\]' $exo_path/exo/inference/tinygrad/inference.py" || {
        echo -e "${RED}❌ Verification failed - Device['NV'] not found${NC}"
        return 1
    }
    echo -e "${GREEN}✅ Patch verified${NC}"

    # 5. Copy verification script
    echo -e "${YELLOW}[5/5]${NC} Copying verification script..."
    scp "$SOLUTION_DIR/verification.py" "$host:/tmp/" || {
        echo -e "${RED}❌ Failed to copy verification script${NC}"
        return 1
    }
    ssh "$host" "chmod +x /tmp/verification.py"
    echo -e "${GREEN}✅ Verification script ready${NC}"

    echo -e "${GREEN}✅ $node_name deployment complete${NC}"
    return 0
}

# Function: Run verification on a node
verify_node() {
    local host=$1
    local node_name=$2

    print_header "Verifying $node_name ($host)"

    echo "Running verification script..."
    ssh "$host" "cd /tmp && PTX=1 DEVICE=CUDA python3 verification.py" || {
        echo -e "${RED}❌ Verification failed${NC}"
        return 1
    }

    echo -e "${GREEN}✅ $node_name verification passed${NC}"
    return 0
}

# Main deployment flow
main() {
    print_header "Pre-Deployment Checks"

    # Check solution files exist
    echo "Checking solution files..."
    for file in device_switch.patch verification.py; do
        if [ ! -f "$SOLUTION_DIR/$file" ]; then
            echo -e "${RED}❌ Missing file: $file${NC}"
            exit 1
        fi
    done
    echo -e "${GREEN}✅ All solution files present${NC}"

    # Check SSH connectivity
    echo -e "\nChecking SSH connectivity..."
    for host in "$THOR1_HOST" "$THOR2_HOST"; do
        ssh -o ConnectTimeout=5 "$host" "echo OK" > /dev/null 2>&1 || {
            echo -e "${RED}❌ Cannot connect to $host${NC}"
            echo "   Make sure SSH keys are configured or use password: $PASSWORD"
            exit 1
        }
    done
    echo -e "${GREEN}✅ SSH connectivity verified${NC}"

    # Deploy to Thor #1
    deploy_to_node "$THOR1_HOST" "$THOR1_EXO" "Thor #1" || {
        echo -e "${RED}❌ Thor #1 deployment failed${NC}"
        exit 1
    }

    # Deploy to Thor #2
    deploy_to_node "$THOR2_HOST" "$THOR2_EXO" "Thor #2" || {
        echo -e "${RED}❌ Thor #2 deployment failed${NC}"
        exit 1
    }

    print_header "Post-Deployment Verification"

    # Verify Thor #1
    verify_node "$THOR1_HOST" "Thor #1" || {
        echo -e "${YELLOW}⚠️  Thor #1 verification had issues${NC}"
    }

    # Verify Thor #2
    verify_node "$THOR2_HOST" "Thor #2" || {
        echo -e "${YELLOW}⚠️  Thor #2 verification had issues${NC}"
    }

    print_header "Deployment Summary"

    echo -e "${GREEN}✅ Device switch deployed to both Thor nodes${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run test.sh to test kernel compilation"
    echo "  2. Start exo servers on both nodes"
    echo "  3. Test inference with client"
    echo ""
    echo "Rollback if needed:"
    echo "  ./rollback.sh"
    echo ""
    echo "Manual server start (for testing):"
    echo "  ssh $THOR1_HOST 'cd $THOR1_EXO && DEBUG=1 DEVICE=CUDA python3 -m exo.main --node-port $THOR1_PORT'"
    echo "  ssh $THOR2_HOST 'cd $THOR2_EXO && DEBUG=1 DEVICE=CUDA python3 -m exo.main --node-port $THOR2_PORT'"
}

# Run main deployment
main
