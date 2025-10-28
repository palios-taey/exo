#!/bin/bash
#
# Rollback script for NVRTC Proper Usage Fix (Solution Agent 2)
#
# This script restores the original files with NVRTC monkey-patch
# from backup files created during installation.
#
# Usage:
#   ./rollback.sh [--backup-date YYYYMMDD_HHMMSS]
#
# If no backup date specified, uses most recent backup on each node.
#
# Author: Solution Agent 2
# Date: 2025-10-23

set -e

# Configuration
NODES=(
    "mira@10.0.0.163:/home/mira/exo"
    "thor@10.0.0.78:/home/thor/exo"
    "jetson@10.0.0.93:/home/jetson/exo"
)

TARGET_FILE="exo/inference/tinygrad/inference.py"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
BACKUP_DATE=""
if [[ "$1" == "--backup-date" ]]; then
    BACKUP_DATE="$2"
fi

log_warning "============================================"
log_warning "  ROLLBACK: Restore NVRTC Monkey-Patch"
log_warning "============================================"
echo

log_warning "This will restore the original files with NVRTC monkey-patch."
log_warning "The NVRTC proper usage fix will be undone."
echo

read -p "Are you sure you want to rollback? (yes/no): " confirm
if [[ "$confirm" != "yes" ]]; then
    log_info "Rollback cancelled."
    exit 0
fi

echo

# Rollback on each node
for node_info in "${NODES[@]}"; do
    node="${node_info%%:*}"
    base_path="${node_info##*:}"
    full_path="$base_path/$TARGET_FILE"

    log_info "Processing $node..."

    # Find backup file
    if [[ -n "$BACKUP_DATE" ]]; then
        # Use specified backup
        backup_path="${full_path}.nvrtc_patch_backup_${BACKUP_DATE}"
    else
        # Find most recent backup
        backup_path=$(ssh "$node" "ls -t '${full_path}'.nvrtc_patch_backup_* 2>/dev/null | head -1 || echo ''")

        if [[ -z "$backup_path" ]]; then
            log_error "No backup found on $node"
            log_info "Available backups:"
            ssh "$node" "ls -la '${full_path}'.nvrtc_patch_backup_* 2>/dev/null || echo '  (none)'"
            continue
        fi
    fi

    # Verify backup exists
    if ! ssh "$node" "[[ -f '$backup_path' ]]"; then
        log_error "Backup file not found: $node:$backup_path"
        continue
    fi

    log_info "Found backup: $backup_path"

    # Create backup of current (fixed) file
    current_backup="${full_path}.pre_rollback_$(date +%Y%m%d_%H%M%S)"
    ssh "$node" "cp '$full_path' '$current_backup'" || {
        log_error "Failed to backup current file on $node"
        continue
    }
    log_success "Current file backed up: $current_backup"

    # Restore from backup
    ssh "$node" "cp '$backup_path' '$full_path'" || {
        log_error "Failed to restore backup on $node"
        # Try to restore current backup
        ssh "$node" "cp '$current_backup' '$full_path'"
        log_warning "Restored current file (rollback failed)"
        continue
    }

    log_success "Backup restored on $node"

    # Verify monkey-patch is present
    if ssh "$node" "grep -q '_apply_nvrtc_patch' '$full_path'"; then
        log_success "✅ Verification passed: Monkey-patch restored on $node"
    else
        log_error "❌ Verification failed: Monkey-patch not found on $node"
        log_warning "Manual inspection may be required"
    fi

    echo
done

echo
log_success "============================================"
log_success "  ROLLBACK COMPLETE"
log_success "============================================"
echo

log_info "Rollback applied to all nodes."
echo

log_warning "IMPORTANT: Servers need to be restarted for changes to take effect."
echo

log_info "Next steps:"
echo "  1. Stop all exo servers"
echo "     ssh thor@10.0.0.78 'pkill -f exo.main'"
echo "     ssh jetson@10.0.0.93 'pkill -f exo.main'"
echo
echo "  2. Restart servers"
echo "     ssh thor@10.0.0.78 'cd ~/exo && python3 -m exo.main --node-port 52415 > /tmp/thor_exo.log 2>&1 &'"
echo "     ssh jetson@10.0.0.93 'cd ~/exo && python3 -m exo.main --node-port 52416 > /tmp/jetson_exo.log 2>&1 &'"
echo
echo "  3. Verify monkey-patch is active"
echo "     ssh thor@10.0.0.78 'grep \"NVRTC MONKEY-PATCH\" /tmp/thor_exo.log'"
echo

log_warning "Files with NVRTC proper usage fix have been backed up with .pre_rollback suffix"
log_info "You can re-apply the fix later by running:"
echo "  ./install.sh"
echo

log_success "Done!"
