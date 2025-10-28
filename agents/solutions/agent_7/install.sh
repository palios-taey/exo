#!/bin/bash
# install.sh - Deploy architecture detection fixes
# Agent 7 Solution: Blackwell sm_110 Detection

set -e  # Exit on error

SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Solution directory: $SOLUTION_DIR"
echo

# Node configurations
declare -A NODES=(
    ["mira"]="mira@10.0.0.163:/home/mira/exo"
    ["thor"]="thor@10.0.0.78:/home/thor/exo"
    ["jetson"]="jetson@10.0.0.93:/home/jetson/exo"
)

# Files to patch (relative to node's exo directory)
TINYGRAD_OPS_NV="exo-venv/lib/python3.12/site-packages/tinygrad/runtime/ops_nv.py"
TINYGRAD_COMPILER="exo-venv/lib/python3.12/site-packages/tinygrad/runtime/support/compiler_cuda.py"
EXO_INFERENCE="exo/inference/tinygrad/inference.py"

echo "==================================================================="
echo "Architecture Detection Fix Deployment"
echo "==================================================================="
echo
echo "This will apply 3 fixes:"
echo "  1. ops_nv.py: Map sm_version 0xa04 → sm_110 (Blackwell CC 11.0)"
echo "  2. compiler_cuda.py: Use PTX 8.5 for sm_110+ (ISA 9.0)"
echo "  3. inference.py: Set PTX=1 before imports (force NVPTXCompiler)"
echo
read -p "Continue? (yes/no): " confirm

if [[ "$confirm" != "yes" ]]; then
    echo "Deployment cancelled."
    exit 0
fi

echo

# Function to deploy to a single node
deploy_to_node() {
    local node_name=$1
    local node_ssh=$(echo "${NODES[$node_name]}" | cut -d: -f1)
    local node_path=$(echo "${NODES[$node_name]}" | cut -d: -f2)

    echo "-------------------------------------------------------------------"
    echo "Deploying to $node_name ($node_ssh)"
    echo "-------------------------------------------------------------------"

    # Create backup directory
    echo "[1/6] Creating backup directory..."
    ssh "$node_ssh" "mkdir -p $node_path/backups/arch_fix_$(date +%Y%m%d_%H%M%S)"
    local backup_dir="$node_path/backups/arch_fix_$(date +%Y%m%d_%H%M%S)"

    # Backup original files
    echo "[2/6] Backing up original files..."
    ssh "$node_ssh" "cp $node_path/$TINYGRAD_OPS_NV $backup_dir/ 2>/dev/null || true"
    ssh "$node_ssh" "cp $node_path/$TINYGRAD_COMPILER $backup_dir/ 2>/dev/null || true"
    ssh "$node_ssh" "cp $node_path/$EXO_INFERENCE $backup_dir/ 2>/dev/null || true"

    # Apply patches

    echo "[3/6] Patching ops_nv.py (architecture detection)..."
    # Create temporary patched file locally
    cat > /tmp/ops_nv_patch.py << 'EOF'
# Architecture detection fix for Blackwell
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, 'r') as f:
    content = f.read()

# Find and replace the architecture detection line
old_line = '    # FIXME: no idea how to convert this for blackwells\n    self.arch: str = "sm_120" if self.sm_version==0xa04 else f"sm_{(self.sm_version>>8)&0xff}{(val>>4) if (val:=self.sm_version&0xff) > 0xf else val}"'

new_code = '''    # Blackwell architecture detection (CC 11.0) - Fixed for Jetson Thor
    # Hardware reports sm_version 0xa04, which is CC 11.0 (not 10.4 or 12.0)
    # Verified: nvidia-smi shows compute capability 11.0 on Jetson Thor devices
    if self.sm_version == 0xa04:
      # Jetson Thor Blackwell: sm_version 0xa04 → CC 11.0 → sm_110
      self.arch = "sm_110"
    elif (self.sm_version & 0xf00) == 0xa00:
      # Blackwell family (0xa00-0xaff): Hardware major 10.x → CC 11.x
      # Handle future Blackwell variants: 0xa00→sm_110, 0xa01→sm_111, etc.
      minor = self.sm_version & 0xff
      self.arch = f"sm_11{minor}"
    else:
      # Previous architecture conversion (unchanged for non-Blackwell)
      self.arch = f"sm_{(self.sm_version>>8)&0xff}{(val>>4) if (val:=self.sm_version&0xff) > 0xf else val}"'''

content = content.replace(old_line, new_code)

with open(output_file, 'w') as f:
    f.write(content)
EOF

    # Copy patcher to node, run it, copy result back
    scp /tmp/ops_nv_patch.py "$node_ssh:/tmp/"
    ssh "$node_ssh" "python3 /tmp/ops_nv_patch.py $node_path/$TINYGRAD_OPS_NV /tmp/ops_nv_patched.py"
    ssh "$node_ssh" "cp /tmp/ops_nv_patched.py $node_path/$TINYGRAD_OPS_NV"

    echo "[4/6] Patching compiler_cuda.py (PTX version)..."
    cat > /tmp/compiler_cuda_patch.py << 'EOF'
# PTX version fix for Blackwell
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, 'r') as f:
    content = f.read()

# Find and replace PTX version selection
old_line = '  def compile(self, src:str) -> bytes: return src.replace("TARGET", self.arch).replace("VERSION", "7.8" if self.arch >= "sm_89" else "7.5").encode()'

new_code = '''  def compile(self, src:str) -> bytes:
    # PTX version selection based on architecture requirements
    # Blackwell (sm_110+) requires PTX ISA 9.0, needs version 8.5+
    if self.arch >= "sm_110":
      ptx_version = "8.5"  # Blackwell CC 11.0+ (PTX ISA 9.0)
    elif self.arch >= "sm_89":
      ptx_version = "7.8"  # Hopper CC 9.0
    else:
      ptx_version = "7.5"  # Earlier architectures
    return src.replace("TARGET", self.arch).replace("VERSION", ptx_version).encode()'''

content = content.replace(old_line, new_code)

with open(output_file, 'w') as f:
    f.write(content)
EOF

    scp /tmp/compiler_cuda_patch.py "$node_ssh:/tmp/"
    ssh "$node_ssh" "python3 /tmp/compiler_cuda_patch.py $node_path/$TINYGRAD_COMPILER /tmp/compiler_cuda_patched.py"
    ssh "$node_ssh" "cp /tmp/compiler_cuda_patched.py $node_path/$TINYGRAD_COMPILER"

    echo "[5/6] Patching inference.py (PTX=1 environment)..."
    cat > /tmp/inference_patch.py << 'EOF'
# PTX=1 environment variable fix
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, 'r') as f:
    lines = f.readlines()

# Find insertion point (after "import os")
insert_idx = -1
for i, line in enumerate(lines):
    if line.strip() == "import os":
        insert_idx = i + 1
        break

if insert_idx == -1:
    print("ERROR: Could not find 'import os' line", file=sys.stderr)
    sys.exit(1)

# Insert PTX=1 code
ptx_code = '''
# CRITICAL: Force PTX=1 for CUDA 13.0 Blackwell compilation
# This must be set BEFORE any tinygrad imports to ensure proper compiler selection
# PTX=0 (default): CUDARenderer → CUDA C → CUDACompiler → NVRTC (removed in CUDA 13.0) ❌
# PTX=1 (correct):  PTXRenderer → PTX asm → NVPTXCompiler → nvJitLink (CUDA 13.0) ✅
# See: Agent 3 research (NVPTX Architecture) and Agent 8 (Debugging Forensics)
# nvJitLink requires actual PTX assembly, not CUDA C source code
os.environ['PTX'] = '1'

'''

lines.insert(insert_idx, ptx_code)

with open(output_file, 'w') as f:
    f.writelines(lines)
EOF

    scp /tmp/inference_patch.py "$node_ssh:/tmp/"
    ssh "$node_ssh" "python3 /tmp/inference_patch.py $node_path/$EXO_INFERENCE /tmp/inference_patched.py"
    ssh "$node_ssh" "cp /tmp/inference_patched.py $node_path/$EXO_INFERENCE"

    # Clear Python bytecode cache
    echo "[6/6] Clearing Python bytecode cache..."
    ssh "$node_ssh" "find $node_path/exo-venv/lib/python3.12/site-packages/tinygrad -name '*.pyc' -delete 2>/dev/null || true"
    ssh "$node_ssh" "find $node_path/exo-venv/lib/python3.12/site-packages/tinygrad -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true"

    echo "✅ Deployment to $node_name complete!"
    echo "   Backup: $backup_dir"
    echo
}

# Deploy to each node
for node in "${!NODES[@]}"; do
    deploy_to_node "$node"
done

# Cleanup temp files
rm -f /tmp/ops_nv_patch.py /tmp/compiler_cuda_patch.py /tmp/inference_patch.py

echo "==================================================================="
echo "Deployment Complete!"
echo "==================================================================="
echo
echo "Next steps:"
echo "  1. Run verify tests: ssh NODE 'python3 verify_detection.py'"
echo "  2. Restart exo servers on both Thor nodes"
echo "  3. Check logs for 'arch: sm_110' and 'PTX version: 8.5'"
echo "  4. Monitor compilation success in exo logs"
echo
echo "To rollback, use: ./rollback.sh"
echo
