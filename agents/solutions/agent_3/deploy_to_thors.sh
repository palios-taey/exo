#!/bin/bash
#
# Deployment Script for Thor Devices
#
# Deploys NVCCCompiler solution to both Jetson Thor devices
#
# Author: Solution Agent 3
# Date: 2025-10-23

set -e  # Exit on error

echo "=================================================="
echo "Thor Device Deployment Script"
echo "=================================================="
echo ""

# Configuration
SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THOR1_HOST="thor@10.0.0.78"
THOR2_HOST="jetson@10.0.0.93"
SSH_PASSWORD="papaDons1001s$"
REMOTE_SOLUTION_DIR="/home/\$(whoami)/exo/agents/solutions/agent_3"

echo "[Configuration]"
echo "  Solution directory: $SOLUTION_DIR"
echo "  Thor #1: $THOR1_HOST"
echo "  Thor #2: $THOR2_HOST"
echo "  Remote directory: $REMOTE_SOLUTION_DIR"
echo ""

# Verify we have all files
echo "[1/7] Verifying local files..."

REQUIRED_FILES=(
    "README.md"
    "nvcc_compiler.py"
    "integration_patch.py"
    "install.sh"
    "test.sh"
    "rollback.sh"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$SOLUTION_DIR/$file" ]; then
        echo "  ✗ Missing required file: $file"
        exit 1
    fi
done

echo "  ✓ All required files present"

# Function to deploy to one device
deploy_to_device() {
    local host="$1"
    local device_name="$2"

    echo ""
    echo "[$device_name] Deploying to $host..."

    # Test SSH connection
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$host" "echo test" > /dev/null 2>&1; then
        echo "  ✗ Cannot connect to $host"
        echo "  Ensure SSH key is set up or run with: sshpass -p '$SSH_PASSWORD' ssh $host"
        return 1
    fi

    echo "  ✓ SSH connection successful"

    # Create remote directory
    ssh "$host" "mkdir -p $REMOTE_SOLUTION_DIR" || {
        echo "  ✗ Failed to create remote directory"
        return 1
    }
    echo "  ✓ Remote directory created"

    # Copy all files
    scp -r "$SOLUTION_DIR"/* "$host:$REMOTE_SOLUTION_DIR/" > /dev/null 2>&1 || {
        echo "  ✗ Failed to copy files"
        return 1
    }
    echo "  ✓ Files copied"

    # Make scripts executable
    ssh "$host" "chmod +x $REMOTE_SOLUTION_DIR/*.sh" || {
        echo "  ✗ Failed to make scripts executable"
        return 1
    }
    echo "  ✓ Scripts made executable"

    echo "  ✓ Deployment to $device_name complete"
    return 0
}

# Deploy to both devices
echo ""
echo "[2/7] Deploying to Thor #1..."
if deploy_to_device "$THOR1_HOST" "Thor #1"; then
    THOR1_DEPLOYED=true
else
    THOR1_DEPLOYED=false
    echo "  ⚠ Thor #1 deployment failed"
fi

echo ""
echo "[3/7] Deploying to Thor #2..."
if deploy_to_device "$THOR2_HOST" "Thor #2"; then
    THOR2_DEPLOYED=true
else
    THOR2_DEPLOYED=false
    echo "  ⚠ Thor #2 deployment failed"
fi

# Verify nvcc on devices
echo ""
echo "[4/7] Verifying nvcc on Thor devices..."

if [ "$THOR1_DEPLOYED" = true ]; then
    echo "  [Thor #1]"
    if ssh "$THOR1_HOST" "which nvcc" > /dev/null 2>&1; then
        NVCC_VERSION=$(ssh "$THOR1_HOST" "nvcc --version | grep 'release' | sed 's/.*release //' | cut -d',' -f1")
        echo "    ✓ nvcc found: version $NVCC_VERSION"
    else
        echo "    ✗ nvcc not found - installation will fail"
    fi
fi

if [ "$THOR2_DEPLOYED" = true ]; then
    echo "  [Thor #2]"
    if ssh "$THOR2_HOST" "which nvcc" > /dev/null 2>&1; then
        NVCC_VERSION=$(ssh "$THOR2_HOST" "nvcc --version | grep 'release' | sed 's/.*release //' | cut -d',' -f1")
        echo "    ✓ nvcc found: version $NVCC_VERSION"
    else
        echo "    ✗ nvcc not found - installation will fail"
    fi
fi

# Ask if user wants to install immediately
echo ""
echo "[5/7] Installation options..."
echo ""
echo "Files deployed successfully. Next steps:"
echo ""
echo "Option A: Install immediately on both devices (RECOMMENDED)"
echo "Option B: Install manually later"
echo ""
read -p "Install now? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "[6/7] Installing on Thor devices..."

    if [ "$THOR1_DEPLOYED" = true ]; then
        echo "  [Thor #1] Running install.sh..."
        if ssh "$THOR1_HOST" "cd $REMOTE_SOLUTION_DIR && bash install.sh"; then
            echo "    ✓ Installation successful"
        else
            echo "    ✗ Installation failed - check output above"
        fi
    fi

    if [ "$THOR2_DEPLOYED" = true ]; then
        echo "  [Thor #2] Running install.sh..."
        if ssh "$THOR2_HOST" "cd $REMOTE_SOLUTION_DIR && bash install.sh"; then
            echo "    ✓ Installation successful"
        else
            echo "    ✗ Installation failed - check output above"
        fi
    fi

    # Run tests
    echo ""
    echo "[7/7] Running tests on Thor devices..."

    if [ "$THOR1_DEPLOYED" = true ]; then
        echo "  [Thor #1] Running test.sh..."
        if ssh "$THOR1_HOST" "cd $REMOTE_SOLUTION_DIR && bash test.sh"; then
            echo "    ✓ Tests passed"
        else
            echo "    ⚠ Tests failed - review output above"
        fi
    fi

    if [ "$THOR2_DEPLOYED" = true ]; then
        echo "  [Thor #2] Running test.sh..."
        if ssh "$THOR2_HOST" "cd $REMOTE_SOLUTION_DIR && bash test.sh"; then
            echo "    ✓ Tests passed"
        else
            echo "    ⚠ Tests failed - review output above"
        fi
    fi

else
    echo ""
    echo "[6/7] Skipping automatic installation"
    echo ""
    echo "To install manually:"
    echo "  ssh $THOR1_HOST 'cd $REMOTE_SOLUTION_DIR && bash install.sh'"
    echo "  ssh $THOR2_HOST 'cd $REMOTE_SOLUTION_DIR && bash install.sh'"
    echo ""
    echo "To test after installation:"
    echo "  ssh $THOR1_HOST 'cd $REMOTE_SOLUTION_DIR && bash test.sh'"
    echo "  ssh $THOR2_HOST 'cd $REMOTE_SOLUTION_DIR && bash test.sh'"
    echo ""
fi

echo ""
echo "=================================================="
echo "Deployment Summary"
echo "=================================================="
echo ""

if [ "$THOR1_DEPLOYED" = true ] && [ "$THOR2_DEPLOYED" = true ]; then
    echo "✓ Both Thor devices deployed successfully"
elif [ "$THOR1_DEPLOYED" = true ]; then
    echo "⚠ Thor #1 deployed, Thor #2 failed"
elif [ "$THOR2_DEPLOYED" = true ]; then
    echo "⚠ Thor #2 deployed, Thor #1 failed"
else
    echo "✗ Both deployments failed"
    exit 1
fi

echo ""
echo "Files deployed to:"
echo "  Thor #1: $THOR1_HOST:$REMOTE_SOLUTION_DIR"
echo "  Thor #2: $THOR2_HOST:$REMOTE_SOLUTION_DIR"
echo ""
echo "Next steps:"
echo "  1. Verify installation worked (check logs above)"
echo "  2. Restart exo servers on both nodes"
echo "  3. Test distributed inference"
echo ""
