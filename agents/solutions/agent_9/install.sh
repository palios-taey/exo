#!/bin/bash
# Installation script for NVPTXCompilerV2
# Deploys compiler fix to exo codebase

set -e  # Exit on error

echo "======================================================================"
echo "Installing NVPTXCompilerV2 for CUDA 13.0 + Blackwell"
echo "======================================================================"
echo

# Detect platform
if [ -f /home/thor/exo/exo/inference/tinygrad/inference.py ]; then
    EXO_ROOT="/home/thor/exo"
    USER="thor"
elif [ -f /home/jetson/exo/exo/inference/tinygrad/inference.py ]; then
    EXO_ROOT="/home/jetson/exo"
    USER="jetson"
elif [ -f /home/mira/exo/exo/inference/tinygrad/inference.py ]; then
    EXO_ROOT="/home/mira/exo"
    USER="mira"
else
    echo "ERROR: Could not find exo installation"
    exit 1
fi

echo "Detected exo at: $EXO_ROOT"
echo "User: $USER"
echo

# Create solution directory in exo
SOLUTION_DIR="$EXO_ROOT/agents/solutions/agent_9"
echo "[1/4] Creating solution directory: $SOLUTION_DIR"
mkdir -p "$SOLUTION_DIR"

# Copy compiler implementation
echo "[2/4] Copying nvptx_compiler_v2.py..."
cp "$(dirname $0)/nvptx_compiler_v2.py" "$SOLUTION_DIR/"
echo "✓ Copied compiler"

# Backup original inference.py
INFERENCE_PY="$EXO_ROOT/exo/inference/tinygrad/inference.py"
BACKUP_PY="${INFERENCE_PY}.backup_$(date +%s)"
echo "[3/4] Backing up inference.py to $(basename $BACKUP_PY)..."
cp "$INFERENCE_PY" "$BACKUP_PY"
echo "✓ Backup created"

# Add patch to inference.py
echo "[4/4] Patching inference.py..."

# Find insertion point (after existing NVRTC patch, around line 20)
# We'll add our patch after the _apply_nvrtc_patch() function

PATCH_CODE="
# ============================================================================
# CUDA 13.0 COMPILER FIX: NVPTXCompilerV2
# ============================================================================
# Problem: tinygrad's NVPTXCompiler expects PTX assembly but receives CUDA C
# Solution: Use NVPTXCompilerV2 that properly compiles CUDA C → PTX → CUBIN
# Agent: Solution Agent 9 (2025-10-23)

import sys
sys.path.insert(0, '$SOLUTION_DIR')

try:
    from nvptx_compiler_v2 import NVPTXCompilerV2
    import tinygrad.runtime.support.compiler_cuda as cuda_compiler
    cuda_compiler.NVPTXCompiler = NVPTXCompilerV2
    print('[NVPTX FIX] Using NVPTXCompilerV2 for CUDA 13.0', file=sys.stderr)
except Exception as e:
    print(f'[NVPTX FIX] WARNING: Failed to apply fix: {e}', file=sys.stderr)

"

# Check if patch already applied
if grep -q "NVPTXCompilerV2" "$INFERENCE_PY"; then
    echo "⚠ Patch already applied, skipping..."
else
    # Add patch after line 20 (after _apply_nvrtc_patch)
    LINE_NUM=20
    sed -i "${LINE_NUM}a\\${PATCH_CODE}" "$INFERENCE_PY"
    echo "✓ Patch applied to inference.py"
fi

echo
echo "======================================================================"
echo "✓ Installation complete!"
echo "======================================================================"
echo
echo "Backup saved to: $BACKUP_PY"
echo "Solution installed at: $SOLUTION_DIR"
echo
echo "To verify installation:"
echo "  cd $SOLUTION_DIR && python3 unit_tests.py"
echo
echo "To test with tinygrad:"
echo "  cd $SOLUTION_DIR && python3 integration_test.py"
echo
echo "To rollback if needed:"
echo "  bash rollback.sh"
echo
