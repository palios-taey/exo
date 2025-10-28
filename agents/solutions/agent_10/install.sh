#!/bin/bash
# Install Configuration-Based Compiler Selector
# Solution Agent 10

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Installing Configuration-Based Compiler Selector${NC}"
echo "=================================================="

# Parse arguments
TARGET="mira"
HOST=""
USER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            TARGET="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --user)
            USER="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Determine installation directory
if [ "$TARGET" = "mira" ]; then
    INSTALL_DIR="/home/mira/exo"
elif [ "$TARGET" = "thor" ]; then
    HOST="${HOST:-10.0.0.78}"
    USER="${USER:-thor}"
    INSTALL_DIR="/home/thor/exo"
elif [ "$TARGET" = "jetson" ]; then
    HOST="${HOST:-10.0.0.93}"
    USER="${USER:-jetson}"
    INSTALL_DIR="/home/jetson/exo"
else
    echo -e "${RED}Unknown target: $TARGET${NC}"
    exit 1
fi

echo "Target: $TARGET"
echo "Install directory: $INSTALL_DIR"

# Installation steps
install_local() {
    echo -e "\n${YELLOW}Installing on local machine (Mira)...${NC}"

    # Create directory structure
    mkdir -p "$INSTALL_DIR/compiler_selector"
    mkdir -p "$INSTALL_DIR/compiler_selector/compilers"

    # Copy files
    cp compiler_config.yaml "$INSTALL_DIR/"
    cp compiler_selector.py "$INSTALL_DIR/compiler_selector/"
    cp compilers/*.py "$INSTALL_DIR/compiler_selector/compilers/"

    echo -e "${GREEN}✓ Files copied to $INSTALL_DIR${NC}"

    # Backup original inference.py
    if [ -f "$INSTALL_DIR/exo/inference/tinygrad/inference.py" ]; then
        cp "$INSTALL_DIR/exo/inference/tinygrad/inference.py" \
           "$INSTALL_DIR/exo/inference/tinygrad/inference.py.backup"
        echo -e "${GREEN}✓ Backed up inference.py${NC}"
    fi

    # Add compiler selector import to inference.py
    PATCH_LINE="# COMPILER SELECTOR: Import before tinygrad"
    if ! grep -q "$PATCH_LINE" "$INSTALL_DIR/exo/inference/tinygrad/inference.py"; then
        # Insert at line 4 (after existing imports)
        sed -i '4i\\n# COMPILER SELECTOR: Import before tinygrad\nimport sys\nsys.path.insert(0, "/home/mira/exo/compiler_selector")\nfrom compiler_selector import patch_tinygrad\npatch_tinygrad()\n' \
            "$INSTALL_DIR/exo/inference/tinygrad/inference.py"
        echo -e "${GREEN}✓ Patched inference.py${NC}"
    else
        echo -e "${YELLOW}! inference.py already patched${NC}"
    fi

    echo -e "\n${GREEN}Installation complete on Mira!${NC}"
}

install_remote() {
    echo -e "\n${YELLOW}Installing on remote machine: $USER@$HOST...${NC}"

    # Create temp directory
    TEMP_DIR=$(mktemp -d)
    mkdir -p "$TEMP_DIR/compiler_selector/compilers"

    # Copy files to temp
    cp compiler_config.yaml "$TEMP_DIR/"
    cp compiler_selector.py "$TEMP_DIR/compiler_selector/"
    cp compilers/*.py "$TEMP_DIR/compiler_selector/compilers/"

    # SCP files to remote
    scp -r "$TEMP_DIR"/* "$USER@$HOST:$INSTALL_DIR/"

    # Backup and patch inference.py remotely
    ssh "$USER@$HOST" "bash -s" << 'REMOTE_SCRIPT'
        INSTALL_DIR="$1"

        # Backup
        if [ -f "$INSTALL_DIR/exo/inference/tinygrad/inference.py" ]; then
            cp "$INSTALL_DIR/exo/inference/tinygrad/inference.py" \
               "$INSTALL_DIR/exo/inference/tinygrad/inference.py.backup"
            echo "✓ Backed up inference.py"
        fi

        # Patch
        PATCH_LINE="# COMPILER SELECTOR: Import before tinygrad"
        if ! grep -q "$PATCH_LINE" "$INSTALL_DIR/exo/inference/tinygrad/inference.py"; then
            sed -i '4i\\n# COMPILER SELECTOR: Import before tinygrad\nimport sys\nsys.path.insert(0, "'$INSTALL_DIR'/compiler_selector")\nfrom compiler_selector import patch_tinygrad\npatch_tinygrad()\n' \
                "$INSTALL_DIR/exo/inference/tinygrad/inference.py"
            echo "✓ Patched inference.py"
        else
            echo "! inference.py already patched"
        fi
REMOTE_SCRIPT

    # Cleanup temp
    rm -rf "$TEMP_DIR"

    echo -e "\n${GREEN}Installation complete on $USER@$HOST!${NC}"
}

# Execute installation
if [ "$TARGET" = "mira" ]; then
    install_local
else
    install_remote
fi

echo ""
echo "Next steps:"
echo "1. Review compiler_config.yaml and adjust if needed"
echo "2. Run ./test.sh to verify installation"
echo "3. Use exo normally - compiler selector activates automatically"
echo ""
echo "To rollback: ./rollback.sh"
