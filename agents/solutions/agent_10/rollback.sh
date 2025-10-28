#!/bin/bash
# Rollback Configuration-Based Compiler Selector
# Solution Agent 10

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Rolling back Configuration-Based Compiler Selector${NC}"
echo "===================================================="

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

rollback_local() {
    echo -e "\n${YELLOW}Rolling back on local machine (Mira)...${NC}"

    # Restore original inference.py if backup exists
    if [ -f "$INSTALL_DIR/exo/inference/tinygrad/inference.py.backup" ]; then
        cp "$INSTALL_DIR/exo/inference/tinygrad/inference.py.backup" \
           "$INSTALL_DIR/exo/inference/tinygrad/inference.py"
        echo -e "${GREEN}✓ Restored original inference.py${NC}"
    else
        echo -e "${YELLOW}! No backup found - removing compiler selector imports manually${NC}"
        # Remove compiler selector lines from inference.py
        sed -i '/# COMPILER SELECTOR: Import before tinygrad/,/patch_tinygrad()/d' \
            "$INSTALL_DIR/exo/inference/tinygrad/inference.py"
        echo -e "${GREEN}✓ Removed compiler selector from inference.py${NC}"
    fi

    # Remove compiler selector directory
    if [ -d "$INSTALL_DIR/compiler_selector" ]; then
        rm -rf "$INSTALL_DIR/compiler_selector"
        echo -e "${GREEN}✓ Removed compiler_selector directory${NC}"
    fi

    # Remove config file
    if [ -f "$INSTALL_DIR/compiler_config.yaml" ]; then
        rm "$INSTALL_DIR/compiler_config.yaml"
        echo -e "${GREEN}✓ Removed compiler_config.yaml${NC}"
    fi

    echo -e "\n${GREEN}Rollback complete on Mira!${NC}"
}

rollback_remote() {
    echo -e "\n${YELLOW}Rolling back on remote machine: $USER@$HOST...${NC}"

    ssh "$USER@$HOST" "bash -s" << 'REMOTE_SCRIPT'
        INSTALL_DIR="$1"

        # Restore backup
        if [ -f "$INSTALL_DIR/exo/inference/tinygrad/inference.py.backup" ]; then
            cp "$INSTALL_DIR/exo/inference/tinygrad/inference.py.backup" \
               "$INSTALL_DIR/exo/inference/tinygrad/inference.py"
            echo "✓ Restored original inference.py"
        else
            echo "! No backup found - removing compiler selector imports manually"
            sed -i '/# COMPILER SELECTOR: Import before tinygrad/,/patch_tinygrad()/d' \
                "$INSTALL_DIR/exo/inference/tinygrad/inference.py"
            echo "✓ Removed compiler selector from inference.py"
        fi

        # Remove files
        rm -rf "$INSTALL_DIR/compiler_selector"
        rm -f "$INSTALL_DIR/compiler_config.yaml"
        echo "✓ Removed compiler selector files"
REMOTE_SCRIPT

    echo -e "\n${GREEN}Rollback complete on $USER@$HOST!${NC}"
}

# Confirm rollback
echo -e "\n${RED}WARNING: This will remove the compiler selector and restore original files.${NC}"
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Rollback cancelled"
    exit 1
fi

# Execute rollback
if [ "$TARGET" = "mira" ]; then
    rollback_local
else
    rollback_remote
fi

echo ""
echo "Rollback complete. Exo will now use original compilation path."
echo "To reinstall: ./install.sh"
