#!/bin/bash
# rollback.sh - Restore original files before architecture detection fixes
# Agent 7 Solution: Blackwell sm_110 Detection

set -e

echo "==================================================================="
echo "Architecture Detection Fix Rollback"
echo "==================================================================="
echo

# Node configurations
declare -A NODES=(
    ["mira"]="mira@10.0.0.163:/home/mira/exo"
    ["thor"]="thor@10.0.0.78:/home/thor/exo"
    ["jetson"]="jetson@10.0.0.93:/home/jetson/exo"
)

echo "This will restore the original files from backups."
echo
echo "WARNING: This will undo all architecture detection fixes."
echo
read -p "Continue with rollback? (yes/no): " confirm

if [[ "$confirm" != "yes" ]]; then
    echo "Rollback cancelled."
    exit 0
fi

echo

rollback_node() {
    local node_name=$1
    local node_ssh=$(echo "${NODES[$node_name]}" | cut -d: -f1)
    local node_path=$(echo "${NODES[$node_name]}" | cut -d: -f2)

    echo "-------------------------------------------------------------------"
    echo "Rolling back $node_name ($node_ssh)"
    echo "-------------------------------------------------------------------"

    # Find most recent backup
    echo "[1/4] Finding most recent backup..."
    local backup_dir=$(ssh "$node_ssh" "ls -td $node_path/backups/arch_fix_* 2>/dev/null | head -1" || echo "")

    if [[ -z "$backup_dir" ]]; then
        echo "⚠️  No backup found for $node_name, skipping..."
        echo
        return
    fi

    echo "   Found: $backup_dir"

    # Restore files
    echo "[2/4] Restoring ops_nv.py..."
    ssh "$node_ssh" "cp $backup_dir/ops_nv.py $node_path/exo-venv/lib/python3.12/site-packages/tinygrad/runtime/ops_nv.py 2>/dev/null || echo '   (file not in backup)'"

    echo "[3/4] Restoring compiler_cuda.py..."
    ssh "$node_ssh" "cp $backup_dir/compiler_cuda.py $node_path/exo-venv/lib/python3.12/site-packages/tinygrad/runtime/support/compiler_cuda.py 2>/dev/null || echo '   (file not in backup)'"

    echo "[4/4] Restoring inference.py..."
    ssh "$node_ssh" "cp $backup_dir/inference.py $node_path/exo/inference/tinygrad/inference.py 2>/dev/null || echo '   (file not in backup)'"

    # Clear bytecode cache
    echo "[5/5] Clearing Python bytecode cache..."
    ssh "$node_ssh" "find $node_path/exo-venv/lib/python3.12/site-packages/tinygrad -name '*.pyc' -delete 2>/dev/null || true"
    ssh "$node_ssh" "find $node_path/exo-venv/lib/python3.12/site-packages/tinygrad -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true"

    echo "✅ Rollback complete for $node_name"
    echo
}

# Rollback each node
for node in "${!NODES[@]}"; do
    rollback_node "$node"
done

echo "==================================================================="
echo "Rollback Complete!"
echo "==================================================================="
echo
echo "All nodes have been restored to pre-fix state."
echo
echo "Architecture detection will now show:"
echo "  • sm_version 0xa04 → sm_120 (incorrect)"
echo "  • PTX version 7.8 (too old for Blackwell)"
echo "  • PTX=0 default (uses CUDACompiler/NVRTC - broken)"
echo
echo "To re-apply fixes, run: ./install.sh"
echo
