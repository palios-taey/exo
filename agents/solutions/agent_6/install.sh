#!/bin/bash
#
# install.sh - Deploy Enhanced PTXRenderer Solution
#
# This script:
# 1. Sets PTX=1 in inference.py (CRITICAL)
# 2. Deploys enhanced PTXCompiler (OPTIONAL)
# 3. Copies to both Thor nodes
# 4. Creates backup of original files
#
# Usage:
#   bash install.sh [--basic|--enhanced]
#
#   --basic: Only set PTX=1 (minimal fix)
#   --enhanced: Set PTX=1 + deploy enhanced compiler (full solution)
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Directories
EXO_ROOT="/home/mira/exo"
SOLUTION_DIR="/home/mira/exo/agents/solutions/agent_6"
PATCH_DIR="/home/mira/exo/tinygrad_cuda13_patch"
INFERENCE_FILE="${EXO_ROOT}/exo/inference/tinygrad/inference.py"

# Thor nodes
THOR1="thor@10.0.0.78"
THOR2="jetson@10.0.0.93"

# Parse arguments
MODE="${1:---basic}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Enhanced PTXRenderer Deployment${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Mode: ${MODE}"
echo "Solution Directory: ${SOLUTION_DIR}"
echo "Target File: ${INFERENCE_FILE}"
echo ""

# Check if files exist
if [ ! -f "${INFERENCE_FILE}" ]; then
    echo -e "${RED}ERROR: ${INFERENCE_FILE} not found${NC}"
    exit 1
fi

if [ ! -f "${SOLUTION_DIR}/ptx_renderer_enhanced.py" ]; then
    echo -e "${RED}ERROR: ${SOLUTION_DIR}/ptx_renderer_enhanced.py not found${NC}"
    exit 1
fi

# Step 1: Backup original file
echo -e "${YELLOW}Step 1: Backing up original file...${NC}"
BACKUP_FILE="${INFERENCE_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
cp "${INFERENCE_FILE}" "${BACKUP_FILE}"
echo "Backup created: ${BACKUP_FILE}"
echo ""

# Step 2: Add PTX=1 to inference.py
echo -e "${YELLOW}Step 2: Adding PTX=1 to inference.py...${NC}"

# Check if PTX=1 already exists
if grep -q "os.environ\['PTX'\] = '1'" "${INFERENCE_FILE}"; then
    echo "PTX=1 already set in ${INFERENCE_FILE}"
else
    # Create modified file with PTX=1 at the top
    cat > "${INFERENCE_FILE}.tmp" << 'EOF'
# CRITICAL: Force PTX compilation path for CUDA 13.0 / Blackwell
# Must be set BEFORE any tinygrad imports
import os
os.environ['PTX'] = '1'

EOF

    # Append original file content
    cat "${INFERENCE_FILE}" >> "${INFERENCE_FILE}.tmp"

    # Replace original with modified
    mv "${INFERENCE_FILE}.tmp" "${INFERENCE_FILE}"

    echo -e "${GREEN}✅ PTX=1 added to inference.py${NC}"
fi
echo ""

# Step 3: Deploy enhanced compiler (if --enhanced mode)
if [ "${MODE}" == "--enhanced" ]; then
    echo -e "${YELLOW}Step 3: Deploying enhanced PTX compiler...${NC}"

    # Create patch directory
    mkdir -p "${PATCH_DIR}"

    # Copy enhanced compiler
    cp "${SOLUTION_DIR}/ptx_renderer_enhanced.py" "${PATCH_DIR}/"
    echo "Enhanced compiler copied to ${PATCH_DIR}"

    # Add import to inference.py (after PTX=1)
    if grep -q "from ptx_renderer_enhanced import apply_enhancements" "${INFERENCE_FILE}"; then
        echo "Enhanced compiler already imported in inference.py"
    else
        # Find line with "from tinygrad" (first import)
        LINE_NUM=$(grep -n "from tinygrad" "${INFERENCE_FILE}" | head -1 | cut -d: -f1)

        if [ -z "${LINE_NUM}" ]; then
            echo -e "${RED}ERROR: Could not find tinygrad import line${NC}"
            exit 1
        fi

        # Insert enhanced compiler import before tinygrad imports
        sed -i "${LINE_NUM}i\\
# Import and apply enhanced PTX compiler\\
import sys\\
sys.path.insert(0, '${PATCH_DIR}')\\
from ptx_renderer_enhanced import apply_enhancements\\
apply_enhancements()\\
" "${INFERENCE_FILE}"

        echo -e "${GREEN}✅ Enhanced compiler imported and applied${NC}"
    fi
else
    echo -e "${YELLOW}Step 3: Skipping enhanced compiler (basic mode)${NC}"
fi
echo ""

# Step 4: Verify changes
echo -e "${YELLOW}Step 4: Verifying changes...${NC}"

echo "Checking for PTX=1:"
if grep -q "os.environ\['PTX'\] = '1'" "${INFERENCE_FILE}"; then
    echo -e "${GREEN}✅ PTX=1 found${NC}"
else
    echo -e "${RED}❌ PTX=1 NOT found${NC}"
    exit 1
fi

if [ "${MODE}" == "--enhanced" ]; then
    echo "Checking for enhanced compiler import:"
    if grep -q "from ptx_renderer_enhanced import apply_enhancements" "${INFERENCE_FILE}"; then
        echo -e "${GREEN}✅ Enhanced compiler import found${NC}"
    else
        echo -e "${RED}❌ Enhanced compiler import NOT found${NC}"
        exit 1
    fi
fi
echo ""

# Step 5: Deploy to Thor nodes
echo -e "${YELLOW}Step 5: Deploying to Thor nodes...${NC}"

# Deploy inference.py to Thor #1
echo "Deploying to Thor #1 (${THOR1})..."
scp "${INFERENCE_FILE}" "${THOR1}:${EXO_ROOT}/exo/inference/tinygrad/inference.py"
echo -e "${GREEN}✅ Deployed to Thor #1${NC}"

# Deploy inference.py to Thor #2
echo "Deploying to Thor #2 (${THOR2})..."
scp "${INFERENCE_FILE}" "${THOR2}:${EXO_ROOT}/exo/inference/tinygrad/inference.py"
echo -e "${GREEN}✅ Deployed to Thor #2${NC}"

# Deploy enhanced compiler if --enhanced mode
if [ "${MODE}" == "--enhanced" ]; then
    echo "Deploying enhanced compiler to Thor #1..."
    ssh "${THOR1}" "mkdir -p ${PATCH_DIR}"
    scp "${PATCH_DIR}/ptx_renderer_enhanced.py" "${THOR1}:${PATCH_DIR}/"
    echo -e "${GREEN}✅ Enhanced compiler deployed to Thor #1${NC}"

    echo "Deploying enhanced compiler to Thor #2..."
    ssh "${THOR2}" "mkdir -p ${PATCH_DIR}"
    scp "${PATCH_DIR}/ptx_renderer_enhanced.py" "${THOR2}:${PATCH_DIR}/"
    echo -e "${GREEN}✅ Enhanced compiler deployed to Thor #2${NC}"
fi
echo ""

# Step 6: Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Changes applied:"
echo "  - PTX=1 set in inference.py (${INFERENCE_FILE})"
if [ "${MODE}" == "--enhanced" ]; then
    echo "  - Enhanced PTX compiler deployed (${PATCH_DIR})"
fi
echo ""
echo "Backup saved: ${BACKUP_FILE}"
echo ""
echo "Next steps:"
echo "  1. Restart exo servers on both Thor nodes"
echo "  2. Run test.sh to verify compilation works"
echo "  3. Run benchmark.sh to measure performance"
echo ""
echo -e "${YELLOW}To rollback: bash rollback.sh${NC}"
echo ""
