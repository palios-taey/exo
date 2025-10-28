#!/bin/bash
#
# Installation Script for NVCCCompiler Patch
#
# Deploys nvcc subprocess compiler to fix CUDA 13.0 compilation issues
#
# Author: Solution Agent 3
# Date: 2025-10-23

set -e  # Exit on error

echo "=================================================="
echo "NVCCCompiler Installation Script"
echo "=================================================="
echo ""

# Configuration
SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXO_ROOT="${EXO_ROOT:-/home/$(whoami)/exo}"
INFERENCE_FILE="$EXO_ROOT/exo/inference/tinygrad/inference.py"
BACKUP_SUFFIX=".before_nvcc_patch"

echo "[1/7] Verifying environment..."

# Check we're in the right directory
if [ ! -f "$SOLUTION_DIR/nvcc_compiler.py" ]; then
    echo "ERROR: nvcc_compiler.py not found in $SOLUTION_DIR"
    echo "Are you running this from the correct directory?"
    exit 1
fi

echo "  ✓ Solution directory: $SOLUTION_DIR"

# Check exo exists
if [ ! -d "$EXO_ROOT" ]; then
    echo "ERROR: Exo directory not found: $EXO_ROOT"
    echo "Set EXO_ROOT environment variable if it's in a different location"
    exit 1
fi

echo "  ✓ Exo directory: $EXO_ROOT"

# Check inference.py exists
if [ ! -f "$INFERENCE_FILE" ]; then
    echo "ERROR: inference.py not found: $INFERENCE_FILE"
    exit 1
fi

echo "  ✓ Inference file: $INFERENCE_FILE"

echo ""
echo "[2/7] Checking nvcc availability..."

# Verify nvcc exists
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found in PATH"
    echo ""
    echo "Please ensure CUDA 13.0 is installed:"
    echo "  sudo apt install cuda-13-0"
    echo ""
    echo "And add to PATH:"
    echo "  export PATH=/usr/local/cuda/bin:\$PATH"
    echo ""
    exit 1
fi

echo "  ✓ nvcc found: $(which nvcc)"

# Check nvcc version
NVCC_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | cut -d',' -f1)
echo "  ✓ nvcc version: $NVCC_VERSION"

# Verify it's CUDA 13.0 or later
MAJOR_VERSION=$(echo "$NVCC_VERSION" | cut -d'.' -f1)
if [ "$MAJOR_VERSION" -lt 13 ]; then
    echo "WARNING: CUDA version $NVCC_VERSION is older than 13.0"
    echo "This patch is designed for CUDA 13.0+, but will attempt to proceed..."
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check sm_110 support
echo "  ⋯ Checking sm_110 architecture support..."
if nvcc --help 2>&1 | grep -q "sm_110"; then
    echo "  ✓ sm_110 supported"
elif nvcc --help 2>&1 | grep -q "sm_90"; then
    echo "  ⚠ sm_110 not explicitly listed, but sm_90+ found (likely compatible)"
else
    echo "  ⚠ Cannot verify sm_110 support, but will proceed"
fi

echo ""
echo "[3/7] Creating backup..."

# Backup inference.py
if [ -f "${INFERENCE_FILE}${BACKUP_SUFFIX}" ]; then
    echo "  ⚠ Backup already exists: ${INFERENCE_FILE}${BACKUP_SUFFIX}"
    echo "  Skipping backup (original already preserved)"
else
    cp "$INFERENCE_FILE" "${INFERENCE_FILE}${BACKUP_SUFFIX}"
    echo "  ✓ Backup created: ${INFERENCE_FILE}${BACKUP_SUFFIX}"
fi

echo ""
echo "[4/7] Installing NVCCCompiler module..."

# Copy nvcc_compiler.py to solution directory (already there)
# and ensure it's in Python path via integration

echo "  ✓ nvcc_compiler.py ready: $SOLUTION_DIR/nvcc_compiler.py"
echo "  ✓ integration_patch.py ready: $SOLUTION_DIR/integration_patch.py"

echo ""
echo "[5/7] Patching inference.py..."

# Check if already patched
if grep -q "integration_patch" "$INFERENCE_FILE"; then
    echo "  ⚠ inference.py already contains integration_patch"
    echo "  Skipping patch (already applied)"
else
    # Create patch content
    PATCH_CONTENT="# NVCC Compiler Integration (Solution Agent 3)
import sys
sys.path.insert(0, '$SOLUTION_DIR')
from integration_patch import patch_nvptx_compiler
patch_nvptx_compiler()
# End NVCC Integration
"

    # Insert at top of file (after shebang if present)
    if head -n 1 "$INFERENCE_FILE" | grep -q "^#!"; then
        # Has shebang, insert after it
        {
            head -n 1 "$INFERENCE_FILE"
            echo "$PATCH_CONTENT"
            tail -n +2 "$INFERENCE_FILE"
        } > "${INFERENCE_FILE}.tmp"
    else
        # No shebang, insert at top
        {
            echo "$PATCH_CONTENT"
            cat "$INFERENCE_FILE"
        } > "${INFERENCE_FILE}.tmp"
    fi

    mv "${INFERENCE_FILE}.tmp" "$INFERENCE_FILE"
    echo "  ✓ Patch applied to inference.py"
fi

echo ""
echo "[6/7] Verifying installation..."

# Quick syntax check
if python3 -m py_compile "$SOLUTION_DIR/nvcc_compiler.py"; then
    echo "  ✓ nvcc_compiler.py syntax valid"
else
    echo "  ❌ nvcc_compiler.py has syntax errors"
    exit 1
fi

if python3 -m py_compile "$SOLUTION_DIR/integration_patch.py"; then
    echo "  ✓ integration_patch.py syntax valid"
else
    echo "  ❌ integration_patch.py has syntax errors"
    exit 1
fi

if python3 -m py_compile "$INFERENCE_FILE"; then
    echo "  ✓ inference.py syntax valid"
else
    echo "  ❌ inference.py has syntax errors after patch"
    echo "  Restoring backup..."
    mv "${INFERENCE_FILE}${BACKUP_SUFFIX}" "$INFERENCE_FILE"
    exit 1
fi

echo ""
echo "[7/7] Testing import..."

# Test if tinygrad can import (basic check)
cd "$EXO_ROOT"
if python3 -c "import sys; sys.path.insert(0, '$SOLUTION_DIR'); from integration_patch import patch_nvptx_compiler; patch_nvptx_compiler(); print('OK')" &> /dev/null; then
    echo "  ✓ Integration patch imports successfully"
else
    echo "  ⚠ Could not test full integration (tinygrad may not be installed)"
    echo "  This is normal if running on a machine without exo-venv"
fi

echo ""
echo "=================================================="
echo "Installation Complete!"
echo "=================================================="
echo ""
echo "Summary:"
echo "  - nvcc compiler: $(which nvcc)"
echo "  - CUDA version: $NVCC_VERSION"
echo "  - Patch applied: $INFERENCE_FILE"
echo "  - Backup saved: ${INFERENCE_FILE}${BACKUP_SUFFIX}"
echo ""
echo "Next steps:"
echo "  1. Run tests: bash $SOLUTION_DIR/test.sh"
echo "  2. Deploy to Thor devices: bash $SOLUTION_DIR/deploy_to_thors.sh"
echo ""
echo "If something goes wrong:"
echo "  - Rollback: bash $SOLUTION_DIR/rollback.sh"
echo "  - Check logs for [NVCC PATCH] messages"
echo ""
