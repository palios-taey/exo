#!/bin/bash
# Rollback script - restores original inference.py

set -e

echo "======================================================================"
echo "Rollback NVPTXCompilerV2 Installation"
echo "======================================================================"
echo

# Detect platform
if [ -f /home/thor/exo/exo/inference/tinygrad/inference.py ]; then
    EXO_ROOT="/home/thor/exo"
elif [ -f /home/jetson/exo/exo/inference/tinygrad/inference.py ]; then
    EXO_ROOT="/home/jetson/exo"
elif [ -f /home/mira/exo/exo/inference/tinygrad/inference.py ]; then
    EXO_ROOT="/home/mira/exo"
else
    echo "ERROR: Could not find exo installation"
    exit 1
fi

INFERENCE_PY="$EXO_ROOT/exo/inference/tinygrad/inference.py"

# Find most recent backup
BACKUP=$(ls -t "${INFERENCE_PY}.backup_"* 2>/dev/null | head -1)

if [ -z "$BACKUP" ]; then
    echo "ERROR: No backup found"
    echo "Cannot rollback without backup file"
    exit 1
fi

echo "Found backup: $(basename $BACKUP)"
echo
read -p "Restore from this backup? [y/N] " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Restoring..."
    cp "$BACKUP" "$INFERENCE_PY"
    echo "✓ Rollback complete"
    echo
    echo "Original inference.py restored from:"
    echo "  $BACKUP"
    echo
    echo "Restart exo servers to apply changes."
else
    echo "Rollback cancelled"
    exit 1
fi
