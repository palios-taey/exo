#!/bin/bash
# Hybrid Compiler Installation Script
# Deploys hybrid compilation fix to exo nodes

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SOLUTION_DIR="/home/mira/exo/agents/solutions/agent_5"
BACKUP_DIR="/home/mira/exo/agents/solutions/agent_5/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Node configurations
declare -A NODES
NODES[mira]="mira@10.0.0.163"
NODES[thor1]="thor@10.0.0.78"
NODES[thor2]="jetson@10.0.0.93"

# Passwords (from CLAUDE.md)
PASSWORD="papaDons1001s$"

# Functions
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

# Backup function
backup_files() {
    local node=$1
    local ssh_target=$2

    log_info "Backing up files on $node..."

    # Create backup directory
    mkdir -p "$BACKUP_DIR/${node}_${TIMESTAMP}"

    # Determine remote path based on node
    if [ "$node" == "mira" ]; then
        REMOTE_EXO_PATH="/home/mira/exo"
    elif [ "$node" == "thor1" ]; then
        REMOTE_EXO_PATH="/home/thor/exo"
    elif [ "$node" == "thor2" ]; then
        REMOTE_EXO_PATH="/home/jetson/exo"
    fi

    # Backup inference.py
    sshpass -p "$PASSWORD" scp \
        "${ssh_target}:${REMOTE_EXO_PATH}/exo/inference/tinygrad/inference.py" \
        "$BACKUP_DIR/${node}_${TIMESTAMP}/inference.py.backup" || {
        log_warning "Could not backup inference.py (may not exist yet)"
    }

    # Backup tinygrad_helpers.py if exists
    sshpass -p "$PASSWORD" scp \
        "${ssh_target}:${REMOTE_EXO_PATH}/exo/inference/tinygrad/tinygrad_helpers.py" \
        "$BACKUP_DIR/${node}_${TIMESTAMP}/tinygrad_helpers.py.backup" || {
        log_warning "Could not backup tinygrad_helpers.py (may not exist yet)"
    }

    log_success "Backup complete: $BACKUP_DIR/${node}_${TIMESTAMP}/"
}

# Deploy hybrid compiler
deploy_hybrid_compiler() {
    local node=$1
    local ssh_target=$2

    log_info "Deploying hybrid compiler to $node..."

    # Determine remote path
    if [ "$node" == "mira" ]; then
        REMOTE_EXO_PATH="/home/mira/exo"
        REMOTE_USER="mira"
    elif [ "$node" == "thor1" ]; then
        REMOTE_EXO_PATH="/home/thor/exo"
        REMOTE_USER="thor"
    elif [ "$node" == "thor2" ]; then
        REMOTE_EXO_PATH="/home/jetson/exo"
        REMOTE_USER="jetson"
    fi

    # Create tinygrad_cuda13_patch directory
    sshpass -p "$PASSWORD" ssh "$ssh_target" \
        "mkdir -p ${REMOTE_EXO_PATH}/tinygrad_cuda13_patch"

    # Copy hybrid_compiler.py
    sshpass -p "$PASSWORD" scp \
        "$SOLUTION_DIR/hybrid_compiler.py" \
        "${ssh_target}:${REMOTE_EXO_PATH}/tinygrad_cuda13_patch/"

    # Copy capability_detection.py
    sshpass -p "$PASSWORD" scp \
        "$SOLUTION_DIR/capability_detection.py" \
        "${ssh_target}:${REMOTE_EXO_PATH}/tinygrad_cuda13_patch/"

    log_success "Hybrid compiler deployed to $node"
}

# Patch inference.py
patch_inference() {
    local node=$1
    local ssh_target=$2

    log_info "Patching inference.py on $node..."

    # Determine remote path
    if [ "$node" == "mira" ]; then
        REMOTE_EXO_PATH="/home/mira/exo"
    elif [ "$node" == "thor1" ]; then
        REMOTE_EXO_PATH="/home/thor/exo"
    elif [ "$node" == "thor2" ]; then
        REMOTE_EXO_PATH="/home/jetson/exo"
    fi

    # Create patch script
    cat > /tmp/patch_inference_${node}.py <<'EOF'
import sys

# Read original file
with open('inference.py', 'r') as f:
    content = f.read()

# Check if already patched
if 'HybridCUDACompiler' in content:
    print("[PATCH] Already patched, skipping")
    sys.exit(0)

# Find insertion point (after imports, before class definitions)
lines = content.split('\n')
insert_index = 0

for i, line in enumerate(lines):
    if line.startswith('from tinygrad'):
        insert_index = i + 1

if insert_index == 0:
    print("[PATCH] ERROR: Could not find insertion point")
    sys.exit(1)

# Insert hybrid compiler patch
patch_lines = [
    "",
    "# ========== HYBRID CUDA COMPILER PATCH ==========",
    "# Auto-inserted by Solution Agent 5",
    "# Date: 2025-10-23",
    "",
    "import sys",
    "import os",
    "",
    "# Add hybrid compiler to path",
    "sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../tinygrad_cuda13_patch'))",
    "",
    "# Import and monkey-patch tinygrad's CUDA compiler",
    "try:",
    "    from hybrid_compiler import HybridCUDACompiler",
    "    ",
    "    # Detect architecture",
    "    from capability_detection import CUDACapability",
    "    detector = CUDACapability(verbose=False)",
    "    caps = detector.detect_all()",
    "    ",
    "    arch = caps.get('compute_capability', {}).get('arch', 'sm_110')",
    "    ",
    "    # Patch NVPTXCompiler with hybrid compiler",
    "    import tinygrad.runtime.support.compiler_cuda as cuda_compiler",
    "    ",
    "    class PatchedNVPTXCompiler(HybridCUDACompiler):",
    "        def __init__(self, arch: str):",
    "            super().__init__(arch, verbose=True, log_file='/tmp/hybrid_compiler.log')",
    "            ",
    "    cuda_compiler.NVPTXCompiler = PatchedNVPTXCompiler",
    "    ",
    "    print('[HYBRID PATCH] Successfully patched NVPTXCompiler', file=sys.stderr)",
    "    print(f'[HYBRID PATCH] Target architecture: {arch}', file=sys.stderr)",
    "    ",
    "except Exception as e:",
    "    print(f'[HYBRID PATCH] WARNING: Failed to patch: {e}', file=sys.stderr)",
    "    print('[HYBRID PATCH] Falling back to standard compilation', file=sys.stderr)",
    "",
    "# ========== END HYBRID CUDA COMPILER PATCH ==========",
    ""
]

# Insert patch
lines = lines[:insert_index] + patch_lines + lines[insert_index:]

# Write patched file
with open('inference.py', 'w') as f:
    f.write('\n'.join(lines))

print("[PATCH] Successfully patched inference.py")
EOF

    # Copy patch script to node
    sshpass -p "$PASSWORD" scp \
        /tmp/patch_inference_${node}.py \
        "${ssh_target}:${REMOTE_EXO_PATH}/exo/inference/tinygrad/"

    # Execute patch
    sshpass -p "$PASSWORD" ssh "$ssh_target" \
        "cd ${REMOTE_EXO_PATH}/exo/inference/tinygrad && python3 patch_inference_${node}.py"

    # Clean up patch script
    sshpass -p "$PASSWORD" ssh "$ssh_target" \
        "rm ${REMOTE_EXO_PATH}/exo/inference/tinygrad/patch_inference_${node}.py"

    rm /tmp/patch_inference_${node}.py

    log_success "inference.py patched on $node"
}

# Test deployment
test_deployment() {
    local node=$1
    local ssh_target=$2

    log_info "Testing deployment on $node..."

    # Determine remote path
    if [ "$node" == "mira" ]; then
        REMOTE_EXO_PATH="/home/mira/exo"
    elif [ "$node" == "thor1" ]; then
        REMOTE_EXO_PATH="/home/thor/exo"
    elif [ "$node" == "thor2" ]; then
        REMOTE_EXO_PATH="/home/jetson/exo"
    fi

    # Test capability detection
    sshpass -p "$PASSWORD" ssh "$ssh_target" \
        "cd ${REMOTE_EXO_PATH}/tinygrad_cuda13_patch && python3 capability_detection.py --quiet"

    if [ $? -eq 0 ]; then
        log_success "Capability detection works on $node"
    else
        log_error "Capability detection failed on $node"
        return 1
    fi

    # Test hybrid compiler import
    sshpass -p "$PASSWORD" ssh "$ssh_target" \
        "cd ${REMOTE_EXO_PATH}/tinygrad_cuda13_patch && python3 -c 'from hybrid_compiler import HybridCUDACompiler; print(\"Import OK\")'"

    if [ $? -eq 0 ]; then
        log_success "Hybrid compiler import works on $node"
    else
        log_error "Hybrid compiler import failed on $node"
        return 1
    fi

    log_success "All tests passed on $node"
}

# Main installation function
install_node() {
    local node=$1

    if [ -z "${NODES[$node]}" ]; then
        log_error "Unknown node: $node"
        log_info "Available nodes: ${!NODES[@]}"
        exit 1
    fi

    local ssh_target="${NODES[$node]}"

    log_info "Installing hybrid compiler on $node ($ssh_target)..."

    # Check if sshpass is installed
    if ! command -v sshpass &> /dev/null; then
        log_error "sshpass is required but not installed"
        log_info "Install with: sudo apt-get install sshpass"
        exit 1
    fi

    # Backup
    backup_files "$node" "$ssh_target"

    # Deploy
    deploy_hybrid_compiler "$node" "$ssh_target"

    # Patch
    patch_inference "$node" "$ssh_target"

    # Test
    if [ "$TEST_AFTER" == "true" ]; then
        test_deployment "$node" "$ssh_target"
    fi

    log_success "Installation complete on $node!"
}

# Parse arguments
BACKUP_ONLY=false
TEST_AFTER=false
TARGET_NODE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --backup-only)
            BACKUP_ONLY=true
            shift
            ;;
        --test-after)
            TEST_AFTER=true
            shift
            ;;
        --target)
            TARGET_NODE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --backup-only    Only backup files, don't deploy"
            echo "  --test-after     Run tests after deployment"
            echo "  --target NODE    Target node (mira, thor1, thor2, or all)"
            echo "  --help           Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --target mira --test-after"
            echo "  $0 --target all"
            echo "  $0 --backup-only --target thor1"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            log_info "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Default to all nodes if none specified
if [ -z "$TARGET_NODE" ]; then
    TARGET_NODE="all"
fi

# Main execution
log_info "Hybrid Compiler Installation Script"
log_info "Target: $TARGET_NODE"
log_info "Backup only: $BACKUP_ONLY"
log_info "Test after: $TEST_AFTER"
echo ""

if [ "$TARGET_NODE" == "all" ]; then
    for node in "${!NODES[@]}"; do
        install_node "$node"
        echo ""
    done
else
    install_node "$TARGET_NODE"
fi

log_success "All installations complete!"
