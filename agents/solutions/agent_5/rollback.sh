#!/bin/bash
# Hybrid Compiler Rollback Script
# Clean removal of hybrid compilation fix

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
BACKUP_DIR="/home/mira/exo/agents/solutions/agent_5/backups"
PASSWORD="papaDons1001s$"

declare -A NODES
NODES[mira]="mira@10.0.0.163"
NODES[thor1]="thor@10.0.0.78"
NODES[thor2]="jetson@10.0.0.93"

log_info() {
    echo -e "${BLUE}[ROLLBACK]${NC} $1"
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

# List available backups
list_backups() {
    local node=$1

    log_info "Available backups for $node:"

    if [ ! -d "$BACKUP_DIR" ]; then
        log_warning "No backup directory found"
        return 1
    fi

    local backups=$(ls -d ${BACKUP_DIR}/${node}_* 2>/dev/null)

    if [ -z "$backups" ]; then
        log_warning "No backups found for $node"
        return 1
    fi

    for backup in $backups; do
        local timestamp=$(basename "$backup" | sed "s/${node}_//")
        echo "  - $timestamp"
    done

    return 0
}

# Rollback one node
rollback_node() {
    local node=$1
    local timestamp=$2
    local ssh_target="${NODES[$node]}"

    log_info "Rolling back $node to backup $timestamp..."

    # Determine paths
    case $node in
        mira) REMOTE_PATH="/home/mira/exo" ;;
        thor1) REMOTE_PATH="/home/thor/exo" ;;
        thor2) REMOTE_PATH="/home/jetson/exo" ;;
    esac

    local backup_path="${BACKUP_DIR}/${node}_${timestamp}"

    if [ ! -d "$backup_path" ]; then
        log_error "Backup not found: $backup_path"
        return 1
    fi

    # Restore inference.py if backup exists
    if [ -f "$backup_path/inference.py.backup" ]; then
        log_info "Restoring inference.py..."
        sshpass -p "$PASSWORD" scp \
            "$backup_path/inference.py.backup" \
            "${ssh_target}:${REMOTE_PATH}/exo/inference/tinygrad/inference.py"
        log_success "inference.py restored"
    else
        log_warning "No inference.py backup found"
    fi

    # Restore tinygrad_helpers.py if backup exists
    if [ -f "$backup_path/tinygrad_helpers.py.backup" ]; then
        log_info "Restoring tinygrad_helpers.py..."
        sshpass -p "$PASSWORD" scp \
            "$backup_path/tinygrad_helpers.py.backup" \
            "${ssh_target}:${REMOTE_PATH}/exo/inference/tinygrad/tinygrad_helpers.py"
        log_success "tinygrad_helpers.py restored"
    else
        log_warning "No tinygrad_helpers.py backup found"
    fi

    # Remove hybrid compiler patch directory
    log_info "Removing hybrid compiler patch..."
    sshpass -p "$PASSWORD" ssh "$ssh_target" \
        "rm -rf ${REMOTE_PATH}/tinygrad_cuda13_patch"
    log_success "Hybrid compiler patch removed"

    log_success "Rollback complete for $node!"
}

# Clean rollback (remove patch without restoring backup)
clean_rollback_node() {
    local node=$1
    local ssh_target="${NODES[$node]}"

    log_info "Performing clean rollback on $node..."

    # Determine paths
    case $node in
        mira) REMOTE_PATH="/home/mira/exo" ;;
        thor1) REMOTE_PATH="/home/thor/exo" ;;
        thor2) REMOTE_PATH="/home/jetson/exo" ;;
    esac

    # Remove hybrid compiler patch directory
    log_info "Removing hybrid compiler patch..."
    sshpass -p "$PASSWORD" ssh "$ssh_target" \
        "rm -rf ${REMOTE_PATH}/tinygrad_cuda13_patch"

    # Remove patch from inference.py
    log_info "Removing patch from inference.py..."

    cat > /tmp/unpatch_inference_${node}.py <<'EOF'
# Read original file
with open('inference.py', 'r') as f:
    content = f.read()

# Check if patched
if 'HybridCUDACompiler' not in content:
    print("[UNPATCH] Not patched, nothing to do")
    exit(0)

# Remove patch block
lines = content.split('\n')
new_lines = []
skip = False

for line in lines:
    if '========== HYBRID CUDA COMPILER PATCH ==========' in line:
        skip = True
    elif '========== END HYBRID CUDA COMPILER PATCH ==========' in line:
        skip = False
        continue

    if not skip:
        new_lines.append(line)

# Write unpatched file
with open('inference.py', 'w') as f:
    f.write('\n'.join(new_lines))

print("[UNPATCH] Successfully removed patch")
EOF

    # Copy unpatch script
    sshpass -p "$PASSWORD" scp \
        /tmp/unpatch_inference_${node}.py \
        "${ssh_target}:${REMOTE_PATH}/exo/inference/tinygrad/"

    # Execute unpatch
    sshpass -p "$PASSWORD" ssh "$ssh_target" \
        "cd ${REMOTE_PATH}/exo/inference/tinygrad && python3 unpatch_inference_${node}.py"

    # Clean up unpatch script
    sshpass -p "$PASSWORD" ssh "$ssh_target" \
        "rm ${REMOTE_PATH}/exo/inference/tinygrad/unpatch_inference_${node}.py"

    rm /tmp/unpatch_inference_${node}.py

    log_success "Clean rollback complete for $node!"
}

# Verify rollback
verify_rollback() {
    local node=$1
    local ssh_target="${NODES[$node]}"

    log_info "Verifying rollback on $node..."

    case $node in
        mira) REMOTE_PATH="/home/mira/exo" ;;
        thor1) REMOTE_PATH="/home/thor/exo" ;;
        thor2) REMOTE_PATH="/home/jetson/exo" ;;
    esac

    # Check if patch directory removed
    local patch_check=$(sshpass -p "$PASSWORD" ssh "$ssh_target" \
        "[ -d ${REMOTE_PATH}/tinygrad_cuda13_patch ] && echo 'EXISTS' || echo 'REMOVED'")

    if [ "$patch_check" == "REMOVED" ]; then
        log_success "Patch directory removed"
    else
        log_error "Patch directory still exists"
        return 1
    fi

    # Check if inference.py is clean
    local inference_check=$(sshpass -p "$PASSWORD" ssh "$ssh_target" \
        "grep -q 'HybridCUDACompiler' ${REMOTE_PATH}/exo/inference/tinygrad/inference.py && echo 'PATCHED' || echo 'CLEAN'")

    if [ "$inference_check" == "CLEAN" ]; then
        log_success "inference.py is clean"
    else
        log_warning "inference.py still contains patch references"
    fi

    log_success "Verification complete for $node"
}

# Main execution
TARGET_NODE=""
TIMESTAMP=""
CLEAN_MODE=false
VERIFY_ONLY=false
LIST_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            TARGET_NODE="$2"
            shift 2
            ;;
        --timestamp)
            TIMESTAMP="$2"
            shift 2
            ;;
        --latest)
            # Will use latest backup automatically
            shift
            ;;
        --clean)
            CLEAN_MODE=true
            shift
            ;;
        --verify)
            VERIFY_ONLY=true
            shift
            ;;
        --list)
            LIST_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --target NODE       Target node (mira, thor1, thor2, or all)"
            echo "  --timestamp TIME    Specific backup timestamp to restore"
            echo "  --latest            Use latest backup (default)"
            echo "  --clean             Remove patch without restoring backup"
            echo "  --verify            Verify rollback without performing it"
            echo "  --list              List available backups"
            echo "  --help              Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --target thor1 --latest"
            echo "  $0 --target all --clean"
            echo "  $0 --list --target mira"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate
if [ -z "$TARGET_NODE" ] && [ "$LIST_ONLY" == "false" ]; then
    log_error "No target specified"
    log_info "Use --target NODE or --help for usage"
    exit 1
fi

# List backups
if [ "$LIST_ONLY" == "true" ]; then
    if [ "$TARGET_NODE" == "all" ]; then
        for node in "${!NODES[@]}"; do
            list_backups "$node"
            echo ""
        done
    else
        list_backups "$TARGET_NODE"
    fi
    exit 0
fi

# Verify only
if [ "$VERIFY_ONLY" == "true" ]; then
    if [ "$TARGET_NODE" == "all" ]; then
        for node in "${!NODES[@]}"; do
            verify_rollback "$node"
        done
    else
        verify_rollback "$TARGET_NODE"
    fi
    exit 0
fi

# Perform rollback
log_info "Hybrid Compiler Rollback"
log_info "Target: $TARGET_NODE"
log_info "Clean mode: $CLEAN_MODE"
echo ""

if [ "$TARGET_NODE" == "all" ]; then
    for node in "${!NODES[@]}"; do
        if [ "$CLEAN_MODE" == "true" ]; then
            clean_rollback_node "$node"
        else
            # Find latest backup if timestamp not specified
            if [ -z "$TIMESTAMP" ]; then
                TIMESTAMP=$(ls -d ${BACKUP_DIR}/${node}_* 2>/dev/null | sort -r | head -1 | xargs basename | sed "s/${node}_//")
            fi

            if [ -z "$TIMESTAMP" ]; then
                log_error "No backup found for $node"
                continue
            fi

            rollback_node "$node" "$TIMESTAMP"
        fi

        # Verify
        verify_rollback "$node"
        echo ""
    done
else
    if [ "$CLEAN_MODE" == "true" ]; then
        clean_rollback_node "$TARGET_NODE"
    else
        # Find latest backup if timestamp not specified
        if [ -z "$TIMESTAMP" ]; then
            TIMESTAMP=$(ls -d ${BACKUP_DIR}/${TARGET_NODE}_* 2>/dev/null | sort -r | head -1 | xargs basename | sed "s/${TARGET_NODE}_//")
        fi

        if [ -z "$TIMESTAMP" ]; then
            log_error "No backup found for $TARGET_NODE"
            log_info "Use --clean for clean rollback without restoring backup"
            exit 1
        fi

        rollback_node "$TARGET_NODE" "$TIMESTAMP"
    fi

    # Verify
    verify_rollback "$TARGET_NODE"
fi

log_success "Rollback complete!"
