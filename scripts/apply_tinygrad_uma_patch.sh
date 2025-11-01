#!/usr/bin/env bash
# Apply tinygrad UMA optimization for Jetson Thor sm_110
# Fixes distributed inference hang on unified memory architecture
#
# Usage: ./apply_tinygrad_uma_patch.sh [thor1|thor2|both]
#
# This patches tinygrad's CUDA allocator to use synchronous copy
# on integrated GPUs (Jetson Thor) instead of async pinned memory pattern

set -euo pipefail

TARGET="${1:-both}"

apply_patch() {
    local HOST="$1"
    local USER="$2"

    echo "Applying UMA patch to $HOST..."

    ssh "${USER}@${HOST}" bash <<'REMOTE_SCRIPT'
set -euo pipefail

# Locate tinygrad ops_cuda.py
TINYGRAD_PATH=$(python3 -c "import tinygrad.runtime.ops_cuda as m; print(m.__file__)")
echo "Found tinygrad at: $TINYGRAD_PATH"

# Backup original
BACKUP_PATH="${TINYGRAD_PATH}.backup_$(date +%Y%m%d_%H%M%S)"
cp "$TINYGRAD_PATH" "$BACKUP_PATH"
echo "Backup created: $BACKUP_PATH"

# Create patch script
cat > /tmp/patch_tinygrad_uma.py <<'PYTHON_SCRIPT'
#!/usr/bin/env python3
import sys

with open(sys.argv[1], 'r') as f:
    content = f.read()

# Check if already patched
if 'AI Native Fix' in content or 'is_integrated' in content:
    print("Already patched - skipping")
    sys.exit(0)

# Find _copyin method and replace
old_copyin = '''  def _copyin(self, dest, src:memoryview):
    check(cuda.cuCtxSetCurrent(self.dev.context))
    host_mem = self.alloc(len(src), BufferSpec(host=True))
    self.dev.pending_copyin.append((host_mem, len(src), BufferSpec(host=True)))
    ctypes.memmove(host_mem, mv_address(src), len(src))
    check(cuda.cuMemcpyHtoDAsync_v2(dest, host_mem, len(src), None))'''

new_copyin = '''  def _copyin(self, dest, src:memoryview):
    check(cuda.cuCtxSetCurrent(self.dev.context))

    # AI Native Fix: On integrated GPUs (Jetson Thor sm_110), use synchronous copy
    # Unified memory architecture makes async pinned memory pattern inefficient
    if self.dev.is_integrated:
        # Use heap-allocated temp buffer (not pinned memory) to handle read-only source
        temp_buf = (ctypes.c_char * len(src)).from_buffer_copy(src)
        check(cuda.cuMemcpyHtoD_v2(dest, ctypes.addressof(temp_buf), len(src)))
        return

    # Original async path for discrete GPUs
    host_mem = self.alloc(len(src), BufferSpec(host=True))
    self.dev.pending_copyin.append((host_mem, len(src), BufferSpec(host=True)))
    ctypes.memmove(host_mem, mv_address(src), len(src))
    check(cuda.cuMemcpyHtoDAsync_v2(dest, host_mem, len(src), None))'''

if old_copyin not in content:
    print("ERROR: _copyin method not found or has unexpected format")
    sys.exit(1)

content = content.replace(old_copyin, new_copyin)

# Add is_integrated property to CUDADevice class (after __init__)
# Find the line after "self.allocator = CUDAAllocator(self)"
insertion_point = content.find('self.allocator = CUDAAllocator(self)')
if insertion_point == -1:
    print("ERROR: Could not find CUDAAllocator init")
    sys.exit(1)

# Find next method definition after allocator init
next_method = content.find('\n  def ', insertion_point)
if next_method == -1:
    print("ERROR: Could not find next method after allocator")
    sys.exit(1)

is_integrated_property = '''
  # AI Native: Detect integrated GPU for unified memory optimization
  integrated = ctypes.c_int()
  check(cuda.cuDeviceGetAttribute(ctypes.byref(integrated), 64, device_id))
  self.is_integrated = (integrated.value != 0)
'''

content = content[:next_method] + is_integrated_property + content[next_method:]

with open(sys.argv[1], 'w') as f:
    f.write(content)

print("Patch applied successfully")
PYTHON_SCRIPT

# Apply patch
python3 /tmp/patch_tinygrad_uma.py "$TINYGRAD_PATH"

# Verify patch
if grep -q "is_integrated" "$TINYGRAD_PATH"; then
    echo "✅ Patch verified on $(hostname)"
else
    echo "❌ Patch verification failed"
    exit 1
fi

rm /tmp/patch_tinygrad_uma.py
REMOTE_SCRIPT
}

case "$TARGET" in
    thor1)
        apply_patch "10.0.0.93" "jetson"
        ;;
    thor2)
        apply_patch "10.0.0.78" "thor"
        ;;
    both)
        apply_patch "10.0.0.93" "jetson"
        apply_patch "10.0.0.78" "thor"
        ;;
    *)
        echo "Usage: $0 [thor1|thor2|both]"
        exit 1
        ;;
esac

echo ""
echo "✅ UMA patch deployment complete"
echo ""
echo "Next steps:"
echo "1. Restart exo servers on patched nodes"
echo "2. Test distributed inference"
echo "3. Monitor performance (expect ~0.2-1 tokens/s initially)"
