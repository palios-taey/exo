#!/bin/bash
#
# rollback.sh - Rollback Enhanced PTXRenderer Changes
#
# This script:
# 1. Restores original inference.py from backup
# 2. Removes enhanced compiler files
# 3. Deploys rollback to Thor nodes
#
# Usage:
#   bash rollback.sh
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Directories
EXO_ROOT="/home/mira/exo"
PATCH_DIR="/home/mira/exo/tinygrad_cuda13_patch"
INFERENCE_FILE="${EXO_ROOT}/exo/inference/tinygrad/inference.py"

# Thor nodes
THOR1="thor@10.0.0.78"
THOR2="jetson@10.0.0.93"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}PTXRenderer Solution Rollback${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Step 1: Find most recent backup
echo -e "${YELLOW}Step 1: Finding backup file...${NC}"

BACKUP_FILE=$(ls -t "${INFERENCE_FILE}.backup."* 2>/dev/null | head -1)

if [ -z "${BACKUP_FILE}" ]; then
    echo -e "${RED}ERROR: No backup file found${NC}"
    echo "Looked for: ${INFERENCE_FILE}.backup.*"
    exit 1
fi

echo "Found backup: ${BACKUP_FILE}"
echo ""

# Step 2: Confirm rollback
echo -e "${YELLOW}Step 2: Confirming rollback...${NC}"
echo "This will:"
echo "  - Restore ${INFERENCE_FILE} from ${BACKUP_FILE}"
echo "  - Remove enhanced compiler from ${PATCH_DIR}"
echo "  - Deploy rollback to both Thor nodes"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Rollback cancelled"
    exit 0
fi

# Step 3: Restore original file
echo -e "${YELLOW}Step 3: Restoring original inference.py...${NC}"

# Create backup of modified file (in case we need to re-apply)
MODIFIED_BACKUP="${INFERENCE_FILE}.modified.$(date +%Y%m%d_%H%M%S)"
cp "${INFERENCE_FILE}" "${MODIFIED_BACKUP}"
echo "Modified file saved to: ${MODIFIED_BACKUP}"

# Restore original
cp "${BACKUP_FILE}" "${INFERENCE_FILE}"
echo -e "${GREEN}✅ Original file restored${NC}"
echo ""

# Step 4: Remove enhanced compiler
echo -e "${YELLOW}Step 4: Removing enhanced compiler...${NC}"

if [ -f "${PATCH_DIR}/ptx_renderer_enhanced.py" ]; then
    rm "${PATCH_DIR}/ptx_renderer_enhanced.py"
    echo "Removed: ${PATCH_DIR}/ptx_renderer_enhanced.py"
fi

# Remove patch directory if empty
if [ -d "${PATCH_DIR}" ] && [ -z "$(ls -A ${PATCH_DIR})" ]; then
    rmdir "${PATCH_DIR}"
    echo "Removed empty directory: ${PATCH_DIR}"
fi

echo -e "${GREEN}✅ Enhanced compiler removed${NC}"
echo ""

# Step 5: Deploy to Thor nodes
echo -e "${YELLOW}Step 5: Deploying rollback to Thor nodes...${NC}"

# Deploy to Thor #1
echo "Deploying to Thor #1 (${THOR1})..."
scp "${INFERENCE_FILE}" "${THOR1}:${EXO_ROOT}/exo/inference/tinygrad/inference.py"

# Remove enhanced compiler from Thor #1
ssh "${THOR1}" "rm -f ${PATCH_DIR}/ptx_renderer_enhanced.py 2>/dev/null || true"
ssh "${THOR1}" "rmdir ${PATCH_DIR} 2>/dev/null || true"

echo -e "${GREEN}✅ Rolled back on Thor #1${NC}"

# Deploy to Thor #2
echo "Deploying to Thor #2 (${THOR2})..."
scp "${INFERENCE_FILE}" "${THOR2}:${EXO_ROOT}/exo/inference/tinygrad/inference.py"

# Remove enhanced compiler from Thor #2
ssh "${THOR2}" "rm -f ${PATCH_DIR}/ptx_renderer_enhanced.py 2>/dev/null || true"
ssh "${THOR2}" "rmdir ${PATCH_DIR} 2>/dev/null || true"

echo -e "${GREEN}✅ Rolled back on Thor #2${NC}"
echo ""

# Step 6: Verify rollback
echo -e "${YELLOW}Step 6: Verifying rollback...${NC}"

# Check that PTX=1 is NOT in restored file
if grep -q "os.environ\['PTX'\] = '1'" "${INFERENCE_FILE}"; then
    echo -e "${RED}⚠️  WARNING: PTX=1 still found in ${INFERENCE_FILE}${NC}"
    echo "Backup may have been from after modification."
    echo "Check ${MODIFIED_BACKUP} if you need the modified version."
else
    echo -e "${GREEN}✅ PTX=1 removed from inference.py${NC}"
fi

# Check that enhanced compiler is gone
if [ -f "${PATCH_DIR}/ptx_renderer_enhanced.py" ]; then
    echo -e "${RED}⚠️  WARNING: Enhanced compiler still exists${NC}"
else
    echo -e "${GREEN}✅ Enhanced compiler removed${NC}"
fi

echo ""

# Step 7: Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Rollback Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Changes reverted:"
echo "  - Original inference.py restored from: ${BACKUP_FILE}"
echo "  - Enhanced compiler removed"
echo "  - Rollback deployed to both Thor nodes"
echo ""
echo "Backups saved:"
echo "  - Original backup: ${BACKUP_FILE}"
echo "  - Modified version: ${MODIFIED_BACKUP}"
echo ""
echo "Next steps:"
echo "  1. Restart exo servers on both Thor nodes"
echo "  2. System is now in original state (pre-enhancement)"
echo ""
echo -e "${YELLOW}To re-apply enhancement: bash install.sh${NC}"
echo ""
