#!/bin/bash
# Rollback Script: Restore NVRTC Monkey-Patch
# ===========================================
# Restores monkey-patch from backup if removal causes issues

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================================="
echo "NVRTC Monkey-Patch Rollback"
echo "================================================================="
echo ""
echo -e "${YELLOW}WARNING: This will restore the NVRTC monkey-patch${NC}"
echo ""
echo "Use this if:"
echo "  - Removing the patch caused NEW errors"
echo "  - NVRTC is not actually available on your system"
echo "  - You need to revert to known working state"
echo ""
read -p "Continue with rollback? (yes/no): " CONFIRM

if [[ "$CONFIRM" != "yes" ]]; then
    echo "Rollback cancelled."
    exit 0
fi

# Function: Rollback single device
rollback_device() {
    local device_name="$1"
    local device_host="$2"
    local device_user="$3"
    local exo_path="$4"

    echo ""
    echo "-----------------------------------------------------------------"
    echo "Rolling back: $device_name ($device_user@$device_host)"
    echo "-----------------------------------------------------------------"

    # Check if device is reachable
    if ! ssh -o ConnectTimeout=5 "$device_user@$device_host" "echo 'Connection OK'" > /dev/null 2>&1; then
        echo -e "${RED}ERROR: Cannot connect to $device_name${NC}"
        echo "Skipping this device..."
        return 1
    fi

    echo "✅ Connection verified"

    # Check if backup exists
    BACKUP_EXISTS=$(ssh "$device_user@$device_host" "test -f $exo_path/exo/inference/tinygrad/inference.py.nvrtc_backup && echo 'yes' || echo 'no'")

    if [[ "$BACKUP_EXISTS" != "yes" ]]; then
        echo -e "${RED}ERROR: Backup file not found on $device_name${NC}"
        echo "   Expected: $exo_path/exo/inference/tinygrad/inference.py.nvrtc_backup"
        echo "   Cannot rollback without backup!"
        return 1
    fi

    echo "✅ Backup file found"

    # Restore from backup
    echo "Restoring inference.py from backup..."
    ssh "$device_user@$device_host" "cp $exo_path/exo/inference/tinygrad/inference.py.nvrtc_backup $exo_path/exo/inference/tinygrad/inference.py"

    if [[ $? -ne 0 ]]; then
        echo -e "${RED}ERROR: Restore failed on $device_name${NC}"
        return 1
    fi

    echo "✅ File restored from backup"

    # Verify monkey-patch is present
    echo "Verifying monkey-patch restoration..."
    MONKEYPATCH_PRESENT=$(ssh "$device_user@$device_host" "grep -c 'NVRTC MONKEY-PATCH' $exo_path/exo/inference/tinygrad/inference.py || true")

    if [[ "$MONKEYPATCH_PRESENT" -eq "0" ]]; then
        echo -e "${RED}ERROR: Monkey-patch NOT found after restoration!${NC}"
        echo "   Backup may be corrupted"
        return 1
    fi

    echo "✅ Verification passed: Monkey-patch restored"
    echo -e "${GREEN}SUCCESS: $device_name rolled back successfully${NC}"
    return 0
}

# Device configurations
DEVICES=(
    "Mira|10.0.0.163|mira|/home/mira/exo"
    "Thor|10.0.0.78|thor|/home/thor/exo"
    "Jetson|10.0.0.93|jetson|/home/jetson/exo"
)

SUCCESS_COUNT=0
FAIL_COUNT=0

# Rollback each device
for device_config in "${DEVICES[@]}"; do
    IFS='|' read -r name host user path <<< "$device_config"

    if rollback_device "$name" "$host" "$user" "$path"; then
        ((SUCCESS_COUNT++))
    else
        ((FAIL_COUNT++))
    fi
done

# Summary
echo ""
echo "================================================================="
echo "ROLLBACK SUMMARY"
echo "================================================================="
echo "Devices rolled back successfully: $SUCCESS_COUNT"
echo "Devices failed: $FAIL_COUNT"

if [[ $FAIL_COUNT -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}✅ ALL DEVICES ROLLED BACK SUCCESSFULLY${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Restart exo servers on all devices"
    echo "  2. Monkey-patch is now active again"
    echo "  3. Document what issue caused the rollback"
    echo "  4. Investigate root cause before trying removal again"
    echo ""
    echo "To try removal again: ./install.sh"
else
    echo ""
    echo -e "${YELLOW}⚠️  SOME DEVICES FAILED${NC}"
    echo ""
    echo "Manual intervention may be required for failed devices."
    echo "Check error messages above for details."
fi

exit $FAIL_COUNT
