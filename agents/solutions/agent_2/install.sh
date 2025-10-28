#!/bin/bash
#
# Installation script for NVRTC Proper Usage Fix (Solution Agent 2)
#
# This script:
# 1. Creates backups on all nodes
# 2. Removes NVRTC monkey-patch from inference.py
# 3. Deploys fixed version to all nodes
# 4. Verifies changes
#
# Usage:
#   ./install.sh [--dry-run]
#
# Author: Solution Agent 2
# Date: 2025-10-23

set -e  # Exit on error

# Configuration
NODES=(
    "mira@10.0.0.163:/home/mira/exo"
    "thor@10.0.0.78:/home/thor/exo"
    "jetson@10.0.0.93:/home/jetson/exo"
)

TARGET_FILE="exo/inference/tinygrad/inference.py"
BACKUP_SUFFIX=".nvrtc_patch_backup_$(date +%Y%m%d_%H%M%S)"
SOLUTION_DIR="/home/mira/exo/agents/solutions/agent_2"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo -e "${YELLOW}[DRY RUN MODE] No actual changes will be made${NC}"
    echo
fi

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

# Step 0: Pre-flight checks
log_info "Starting pre-flight checks..."

# Check if patch file exists
if [[ ! -f "$SOLUTION_DIR/remove_monkeypatch.patch" ]]; then
    log_error "Patch file not found: $SOLUTION_DIR/remove_monkeypatch.patch"
    exit 1
fi
log_success "Patch file found"

# Check if we can reach all nodes
for node_info in "${NODES[@]}"; do
    node="${node_info%%:*}"
    if ! ssh -o ConnectTimeout=5 "$node" "echo 'Connected'" &>/dev/null; then
        log_error "Cannot reach node: $node"
        exit 1
    fi
    log_success "Node reachable: $node"
done

echo

# Step 1: Create backups on all nodes
log_info "Step 1: Creating backups on all nodes..."

for node_info in "${NODES[@]}"; do
    node="${node_info%%:*}"
    base_path="${node_info##*:}"
    full_path="$base_path/$TARGET_FILE"
    backup_path="$full_path$BACKUP_SUFFIX"

    log_info "Creating backup on $node: $backup_path"

    if [[ "$DRY_RUN" == false ]]; then
        ssh "$node" "cp '$full_path' '$backup_path'" || {
            log_error "Failed to create backup on $node"
            exit 1
        }

        # Verify backup exists
        if ssh "$node" "[[ -f '$backup_path' ]]"; then
            log_success "Backup created: $node:$backup_path"
        else
            log_error "Backup verification failed on $node"
            exit 1
        fi
    else
        log_info "[DRY RUN] Would create: $node:$backup_path"
    fi
done

echo

# Step 2: Apply patch to local Mira copy
log_info "Step 2: Applying patch to local Mira copy..."

if [[ "$DRY_RUN" == false ]]; then
    cd /home/mira/exo/

    # Apply patch
    if patch -p1 < "$SOLUTION_DIR/remove_monkeypatch.patch"; then
        log_success "Patch applied successfully to local copy"
    else
        log_error "Failed to apply patch"
        log_info "Attempting to restore backups..."
        # Restore local backup
        cp "${TARGET_FILE}${BACKUP_SUFFIX}" "$TARGET_FILE"
        log_warning "Local backup restored. Manual intervention may be required."
        exit 1
    fi

    # Verify patch applied correctly
    if ! grep -q "NVRTC MONKEY-PATCH" "$TARGET_FILE"; then
        log_success "Verification passed: Monkey-patch removed from local copy"
    else
        log_error "Verification failed: Monkey-patch still present"
        exit 1
    fi
else
    log_info "[DRY RUN] Would apply patch: $SOLUTION_DIR/remove_monkeypatch.patch"
fi

echo

# Step 3: Deploy to remote nodes (Thor, Jetson)
log_info "Step 3: Deploying to remote nodes..."

for node_info in "${NODES[@]}"; do
    node="${node_info%%:*}"
    base_path="${node_info##*:}"
    full_path="$base_path/$TARGET_FILE"

    # Skip Mira (already patched locally)
    if [[ "$node" == "mira@10.0.0.163" ]]; then
        log_info "Skipping mira (already patched locally)"
        continue
    fi

    log_info "Deploying to $node..."

    if [[ "$DRY_RUN" == false ]]; then
        # Copy patched file
        if scp "/home/mira/exo/$TARGET_FILE" "$node:$full_path"; then
            log_success "Deployed to $node"
        else
            log_error "Failed to deploy to $node"
            log_warning "Manual deployment may be required"
            exit 1
        fi

        # Verify on remote node
        if ssh "$node" "! grep -q 'NVRTC MONKEY-PATCH' '$full_path'"; then
            log_success "Verification passed on $node: Monkey-patch removed"
        else
            log_error "Verification failed on $node: Monkey-patch still present"
            exit 1
        fi
    else
        log_info "[DRY RUN] Would deploy to: $node:$full_path"
    fi
done

echo

# Step 4: Final verification
log_info "Step 4: Final verification on all nodes..."

for node_info in "${NODES[@]}"; do
    node="${node_info%%:*}"
    base_path="${node_info##*:}"
    full_path="$base_path/$TARGET_FILE"

    log_info "Verifying $node..."

    if [[ "$DRY_RUN" == false ]]; then
        # Check monkey-patch removed
        if ssh "$node" "! grep -q '_apply_nvrtc_patch' '$full_path'"; then
            log_success "✅ $node: _apply_nvrtc_patch function removed"
        else
            log_error "❌ $node: _apply_nvrtc_patch still present"
            exit 1
        fi

        # Check file starts correctly
        first_line=$(ssh "$node" "head -1 '$full_path'")
        if [[ "$first_line" == "from pathlib import Path" ]]; then
            log_success "✅ $node: File starts with correct import"
        else
            log_error "❌ $node: Unexpected first line: $first_line"
            exit 1
        fi

        # Check backup exists
        backup_path="$full_path$BACKUP_SUFFIX"
        if ssh "$node" "[[ -f '$backup_path' ]]"; then
            log_success "✅ $node: Backup exists at $backup_path"
        else
            log_warning "⚠️  $node: Backup not found (may have been moved)"
        fi
    else
        log_info "[DRY RUN] Would verify: $node:$full_path"
    fi
done

echo

# Step 5: Summary and next steps
log_success "============================================"
log_success "  INSTALLATION COMPLETE"
log_success "============================================"
echo

if [[ "$DRY_RUN" == false ]]; then
    echo "Changes applied to all nodes:"
    for node_info in "${NODES[@]}"; do
        node="${node_info%%:*}"
        echo "  - $node"
    done
    echo

    echo "Backup files created (restore with rollback.sh if needed):"
    for node_info in "${NODES[@]}"; do
        node="${node_info%%:*}"
        base_path="${node_info##*:}"
        echo "  - $node:$base_path/$TARGET_FILE$BACKUP_SUFFIX"
    done
    echo

    log_info "Next steps:"
    echo "  1. Test server startup on each node"
    echo "     ssh thor@10.0.0.78 'cd /home/thor/exo && python3 -m exo.main --node-port 52415'"
    echo "     ssh jetson@10.0.0.93 'cd /home/jetson/exo && python3 -m exo.main --node-port 52416'"
    echo
    echo "  2. Watch logs for NVRTC usage"
    echo "     ssh thor@10.0.0.78 'tail -f /tmp/thor_exo.log | grep -i nvrtc'"
    echo
    echo "  3. Test inference request"
    echo "     ./test.sh"
    echo
    echo "  4. If issues occur, rollback with:"
    echo "     ./rollback.sh"
    echo
else
    log_info "DRY RUN completed. No changes were made."
    log_info "Run without --dry-run to apply changes:"
    echo "  ./install.sh"
    echo
fi

log_success "Done!"
