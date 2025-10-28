#!/bin/bash
# Installation Script: Remove NVRTC Monkey-Patch
# ===============================================
# Applies fix to remove monkey-patch from all 3 devices

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIX_SCRIPT="$SCRIPT_DIR/apply_fix.py"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================================="
echo "NVRTC Monkey-Patch Removal - Installation Script"
echo "================================================================="

# Check fix script exists
if [[ ! -f "$FIX_SCRIPT" ]]; then
    echo -e "${RED}ERROR: Fix script not found: $FIX_SCRIPT${NC}"
    exit 1
fi

# Function: Apply fix to single device
apply_fix() {
    local device_name="$1"
    local device_host="$2"
    local device_user="$3"
    local exo_path="$4"

    echo ""
    echo "-----------------------------------------------------------------"
    echo "Processing: $device_name ($device_user@$device_host)"
    echo "-----------------------------------------------------------------"

    # Check if device is reachable
    if ! ssh -o ConnectTimeout=5 "$device_user@$device_host" "echo 'Connection OK'" > /dev/null 2>&1; then
        echo -e "${RED}ERROR: Cannot connect to $device_name${NC}"
        echo "Skipping this device..."
        return 1
    fi

    echo "✅ Connection verified"

    # Copy fix script to device
    echo "Copying fix script to device..."
    scp "$FIX_SCRIPT" "$device_user@$device_host:/tmp/apply_fix.py" > /dev/null 2>&1

    if [[ $? -ne 0 ]]; then
        echo -e "${RED}ERROR: Could not copy fix script to $device_name${NC}"
        return 1
    fi

    echo "✅ Fix script copied to device"

    # Run fix script
    echo "Running fix script..."
    ssh "$device_user@$device_host" "cd $exo_path && python3 /tmp/apply_fix.py exo/inference/tinygrad/inference.py"

    if [[ $? -ne 0 ]]; then
        echo -e "${RED}ERROR: Fix script failed on $device_name${NC}"
        return 1
    fi

    echo "✅ Fix script completed successfully"

    # Verify monkey-patch is gone
    echo "Verifying fix application..."
    MONKEYPATCH_PRESENT=$(ssh "$device_user@$device_host" "grep -c 'NVRTC MONKEY-PATCH' $exo_path/exo/inference/tinygrad/inference.py || true")

    if [[ "$MONKEYPATCH_PRESENT" -ne "0" ]]; then
        echo -e "${RED}ERROR: Monkey-patch still present after fix!${NC}"
        return 1
    fi

    echo "✅ Verification passed: Monkey-patch successfully removed"

    # Cleanup temp file
    ssh "$device_user@$device_host" "rm /tmp/apply_fix.py"

    echo -e "${GREEN}SUCCESS: $device_name fixed successfully${NC}"
    return 0
}

# Main installation sequence
echo ""
echo "This script will:"
echo "  1. Copy fix script to each device"
echo "  2. Run script to remove NVRTC monkey-patch"
echo "  3. Create backup (inference.py.nvrtc_backup)"
echo "  4. Verify fix was applied correctly"
echo ""
echo -e "${YELLOW}WARNING: This modifies vendor code in exo framework${NC}"
echo ""
read -p "Continue? (yes/no): " CONFIRM

if [[ "$CONFIRM" != "yes" ]]; then
    echo "Installation cancelled."
    exit 0
fi

# Device configurations
DEVICES=(
    "Mira|10.0.0.163|mira|/home/mira/exo"
    "Thor|10.0.0.78|thor|/home/thor/exo"
    "Jetson|10.0.0.93|jetson|/home/jetson/exo"
)

SUCCESS_COUNT=0
FAIL_COUNT=0

# Apply to each device
for device_config in "${DEVICES[@]}"; do
    IFS='|' read -r name host user path <<< "$device_config"

    if apply_fix "$name" "$host" "$user" "$path"; then
        ((SUCCESS_COUNT++))
    else
        ((FAIL_COUNT++))
    fi
done

# Summary
echo ""
echo "================================================================="
echo "INSTALLATION SUMMARY"
echo "================================================================="
echo "Devices fixed successfully: $SUCCESS_COUNT"
echo "Devices failed: $FAIL_COUNT"

if [[ $FAIL_COUNT -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}✅ ALL DEVICES FIXED SUCCESSFULLY${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Restart exo servers on all devices"
    echo "  2. Run ./test.sh to verify compilation works"
    echo "  3. Monitor logs for NVRTC usage (should use CUDACompiler)"
    echo ""
    echo "Backup files created:"
    echo "  - inference.py.nvrtc_backup (on each device)"
    echo ""
    echo "To rollback: ./rollback.sh"
else
    echo ""
    echo -e "${YELLOW}⚠️  SOME DEVICES FAILED${NC}"
    echo ""
    echo "Manual intervention may be required for failed devices."
    echo "Check error messages above for details."
fi

exit $FAIL_COUNT
