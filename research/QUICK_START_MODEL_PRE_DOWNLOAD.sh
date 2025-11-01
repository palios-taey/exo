#!/bin/bash
# QUICK START: Pre-Download Llama 3.1 70B FP8 for Exo Distributed Inference
# Run on both Thor #1 (10.0.0.93) and Thor #2 (10.0.0.78)
#
# PURPOSE: Eliminate 16-minute model download delay during first inference
# EXPECTED TIME: 11-22 minutes per node (one-time cost)
# EXPECTED SIZE: ~65GB per node
#
# Based on research: /home/mira/exo/research/RESEARCHER_EDISON_MODEL_DOWNLOAD_OPTIMIZATION.md

set -e  # Exit on error

echo "=================================================="
echo "Llama 3.1 70B FP8 Pre-Download for Exo"
echo "=================================================="

# Configuration
MODEL="neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8"
CACHE_DIR="$HOME/.cache/exo/downloads"
MODEL_DIR="$CACHE_DIR/Meta-Llama-3.1-70B-Instruct-FP8"

# Step 0: Check disk space
echo ""
echo "[1/6] Checking disk space..."
AVAILABLE_GB=$(df -BG "$HOME" | awk 'NR==2 {print $4}' | sed 's/G//')
REQUIRED_GB=70

if [ "$AVAILABLE_GB" -lt "$REQUIRED_GB" ]; then
    echo "ERROR: Insufficient disk space!"
    echo "  Available: ${AVAILABLE_GB}GB"
    echo "  Required: ${REQUIRED_GB}GB"
    exit 1
fi

echo "  Available: ${AVAILABLE_GB}GB"
echo "  Required: ${REQUIRED_GB}GB"
echo "  ✓ Sufficient space"

# Step 1: Verify HuggingFace CLI installed
echo ""
echo "[2/6] Verifying HuggingFace CLI..."
if ! command -v huggingface-cli &> /dev/null; then
    echo "  huggingface-cli not found. Installing..."
    pip install -U "huggingface_hub[cli]"
else
    echo "  ✓ huggingface-cli found"
fi

# Step 2: Login to HuggingFace (interactive if token not set)
echo ""
echo "[3/6] Logging in to HuggingFace..."
if [ -z "$HF_TOKEN" ]; then
    echo "  HF_TOKEN not set. Please login interactively:"
    huggingface-cli login
else
    echo "  Using HF_TOKEN from environment"
    huggingface-cli login --token "$HF_TOKEN"
fi

# Step 3: Create cache directory
echo ""
echo "[4/6] Creating cache directory..."
mkdir -p "$CACHE_DIR"
echo "  ✓ Created: $CACHE_DIR"

# Step 4: Download model
echo ""
echo "[5/6] Downloading model..."
echo "  Model: $MODEL"
echo "  Destination: $MODEL_DIR"
echo "  Expected size: ~65GB"
echo "  Expected time: 11-22 minutes (depending on network speed)"
echo ""
echo "  Starting download..."

HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
  "$MODEL" \
  --local-dir "$MODEL_DIR" \
  --local-dir-use-symlinks False

# Step 5: Verify download
echo ""
echo "[6/6] Verifying download..."
if [ -d "$MODEL_DIR" ]; then
    DOWNLOAD_SIZE=$(du -sh "$MODEL_DIR" | cut -f1)
    FILE_COUNT=$(find "$MODEL_DIR" -name "*.safetensors" | wc -l)

    echo "  ✓ Model directory exists"
    echo "  Downloaded size: $DOWNLOAD_SIZE"
    echo "  Safetensors files: $FILE_COUNT"

    if [ "$FILE_COUNT" -gt 0 ]; then
        echo ""
        echo "  Sample files:"
        find "$MODEL_DIR" -name "*.safetensors" | head -5
    else
        echo "  WARNING: No safetensors files found!"
        exit 1
    fi
else
    echo "  ERROR: Model directory not found!"
    exit 1
fi

# Success!
echo ""
echo "=================================================="
echo "✓ PRE-DOWNLOAD COMPLETE!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Repeat on other Thor node (if not done yet)"
echo "  2. Run exo with pre-downloaded model:"
echo ""
echo "     export EXO_HOME=$HOME/.cache/exo"
echo "     export HF_HOME=$HOME/.cache/huggingface"
echo "     python3 exo/main.py"
echo ""
echo "Expected results:"
echo "  - No model download during inference"
echo "  - TTFT (Time To First Token) < 2 seconds"
echo "  - Inference speed: 2-5 tokens/s for 70B distributed"
echo ""
echo "=================================================="
