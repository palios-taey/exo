#!/bin/bash
#
# Rollback Script for NVCCCompiler Patch
#
# Restores original inference.py from backup
# Removes all traces of the NVCCCompiler integration
#
# Author: Solution Agent 3
# Date: 2025-10-23

set -e  # Exit on error

echo "=================================================="
echo "NVCCCompiler Rollback Script"
echo "=================================================="
echo ""

# Configuration
EXO_ROOT="${EXO_ROOT:-/home/$(whoami)/exo}"
INFERENCE_FILE="$EXO_ROOT/exo/inference/tinygrad/inference.py"
BACKUP_SUFFIX=".before_nvcc_patch"
BACKUP_FILE="${INFERENCE_FILE}${BACKUP_SUFFIX}"

echo "[1/4] Checking backup availability..."

if [ ! -f "$BACKUP_FILE" ]; then
    echo "ERROR: Backup file not found: $BACKUP_FILE"
    echo ""
    echo "Cannot rollback without backup. Possible reasons:"
    echo "  1. Patch was never installed"
    echo "  2. Backup was deleted"
    echo "  3. Running from wrong directory/machine"
    echo ""
    echo "Current inference.py: $INFERENCE_FILE"
    echo ""
    exit 1
fi

echo "  ✓ Backup found: $BACKUP_FILE"

# Show what will be restored
BACKUP_SIZE=$(stat -f%z "$BACKUP_FILE" 2>/dev/null || stat -c%s "$BACKUP_FILE" 2>/dev/null)
CURRENT_SIZE=$(stat -f%z "$INFERENCE_FILE" 2>/dev/null || stat -c%s "$INFERENCE_FILE" 2>/dev/null)

echo "  Backup size: $BACKUP_SIZE bytes"
echo "  Current size: $CURRENT_SIZE bytes"

# Check if patch is actually applied
if grep -q "integration_patch" "$INFERENCE_FILE"; then
    echo "  ✓ Patch detected in current inference.py"
else
    echo "  ⚠ Patch not detected in current inference.py"
    echo "  The file may have already been rolled back or never patched"
    read -p "Continue with rollback anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Rollback cancelled"
        exit 0
    fi
fi

echo ""
echo "[2/4] Confirming rollback..."

echo ""
echo "This will restore inference.py to its state before NVCCCompiler patch"
echo ""
echo "  From: $INFERENCE_FILE (patched)"
echo "  To:   $BACKUP_FILE (original)"
echo ""
read -p "Are you sure you want to rollback? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Rollback cancelled"
    exit 0
fi

echo ""
echo "[3/4] Restoring original inference.py..."

# Create backup of current (patched) state in case we need to revert
ROLLBACK_BACKUP="${INFERENCE_FILE}.before_rollback"
cp "$INFERENCE_FILE" "$ROLLBACK_BACKUP"
echo "  ✓ Current state backed up to: $ROLLBACK_BACKUP"

# Restore original
cp "$BACKUP_FILE" "$INFERENCE_FILE"
echo "  ✓ Original inference.py restored"

# Verify restoration
if python3 -m py_compile "$INFERENCE_FILE"; then
    echo "  ✓ Restored file has valid Python syntax"
else
    echo "  ❌ Restored file has syntax errors!"
    echo "  Restoring from rollback backup..."
    cp "$ROLLBACK_BACKUP" "$INFERENCE_FILE"
    echo "  ✗ Rollback failed - restored to patched state"
    exit 1
fi

# Check patch is removed
if grep -q "integration_patch" "$INFERENCE_FILE"; then
    echo "  ⚠ WARNING: integration_patch still found in file"
    echo "  Backup may not be correct or file was modified after patch"
else
    echo "  ✓ Patch successfully removed"
fi

echo ""
echo "[4/4] Cleaning up..."

# Keep the .before_nvcc_patch backup for potential re-installation
echo "  Keeping original backup: $BACKUP_FILE"

# Keep the .before_rollback backup in case we need to undo rollback
echo "  Keeping rollback backup: $ROLLBACK_BACKUP"

echo ""
echo "=================================================="
echo "Rollback Complete!"
echo "=================================================="
echo ""
echo "Summary:"
echo "  - Original inference.py restored from backup"
echo "  - NVCCCompiler patch removed"
echo "  - Backups preserved for reference"
echo ""
echo "Backups kept:"
echo "  - Original (before patch): $BACKUP_FILE"
echo "  - Patched (before rollback): $ROLLBACK_BACKUP"
echo ""
echo "Next steps:"
echo "  1. Restart exo servers if running"
echo "  2. Verify original behavior restored"
echo ""
echo "If you need to re-apply the patch:"
echo "  bash install.sh"
echo ""
echo "If something went wrong:"
echo "  - Patched state still available at: $ROLLBACK_BACKUP"
echo "  - Can restore: cp $ROLLBACK_BACKUP $INFERENCE_FILE"
echo ""
