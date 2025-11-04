# Fix Attempt 1: Git Sync + Discovery Fix

## Date
2025-11-04 20:20 UTC

## Fixes Applied
- [x] Git synced to commit 5b06048 (bfloat16 fixes)
- [x] Removed --node-host argument
- [x] Added --wait-for-peers 1
- [FAILED] Thor #2 unavailable (SSH timeout)

## Git Synchronization

### Mira State
```
5b06048 fix: Complete bfloat16 compatibility via safetensors framework change
e820caf Fix bfloat16 handling in safetensors loading
d6b0eed fix: Implement lazy safetensors loading for distributed inference
```

### Thor #1 State (SYNCED)
```
5b06048 fix: Complete bfloat16 compatibility via safetensors framework change
e820caf Fix bfloat16 handling in safetensors loading
d6b0eed fix: Implement lazy safetensors loading for distributed inference
```

### Thor #2 State (FAILED - SSH TIMEOUT)
- SSH connection timed out after 10 seconds
- Network: Pingable (0.2ms latency)
- SSH: Unresponsive
- Status: HARDWARE/SSH ISSUE

## Test Results

### Cluster Formation
**Status**: BLOCKED - Cannot form 2-node cluster with Thor #2 down

**Thor #1 Launch**:
```bash
cd /home/jetson/exo-clean && python3 -m exo.main \
  --node-port 52415 \
  --chatgpt-api-port 8080 \
  --wait-for-peers 1
```

**Observed Behavior**:
1. Process started (PID 5426)
2. Port 52415 opened (gRPC discovery)
3. Memory usage climbed to 69GB (loading full 70B model)
4. Port 8080 never opened (API not ready)
5. Process appears HUNG waiting for peer

**Root Cause Analysis**:
The `--wait-for-peers` implementation in `exo/networking/udp/udp_discovery.py` has NO TIMEOUT:

```python
async def discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
    if wait_for_peers > 0:
        while len(self.known_peers) < wait_for_peers:
            if DEBUG_DISCOVERY >= 2:
                print(f"Current peers: {len(self.known_peers)}/{wait_for_peers}. Waiting for more peers...")
            await asyncio.sleep(0.1)
    return [peer_handle for peer_handle, _, _, _ in self.known_peers.values()]
```

**Problem**: Infinite loop with no timeout. If peer never appears, node hangs forever.

### Memory Usage
**Thor #1**: 69GB / 122GB used
- **Expected (if sharding works)**: ~60-70GB (half of 70B model)
- **Actual**: 69GB (full model loaded)
- **Assessment**: LOADED FULL MODEL - sharding NOT working

**Why Full Model?**:
Even though `--wait-for-peers` was specified, the node appears to have either:
1. Proceeded after internal timeout (not visible in code)
2. Started loading model BEFORE peer discovery completed
3. Ignored peer count when partitioning model

### Inference Test
**Status**: NOT ATTEMPTED
- Reason: API endpoint (port 8080) never became available
- Process appeared hung waiting for peer

## Infrastructure Issues Discovered

### Thor #2 SSH Failure
```
Connection to 10.0.0.78 port 22 timed out
```
- **Network**: Working (ping successful, 0.2ms)
- **SSH**: Not responding
- **Impact**: Cannot test 2-node distributed inference
- **Action Required**: Physical access or reboot needed

### Code Issue: No Timeout in --wait-for-peers
**File**: `exo/networking/udp/udp_discovery.py`
**Line**: `async def discover_peers(self, wait_for_peers: int = 0)`
**Problem**: Infinite loop if peer count never reached
**Fix Needed**: Add timeout parameter with default (e.g., 60 seconds)

## Overall Status
**FAILED** - Cannot test distributed inference

## Blockers
1. **Critical**: Thor #2 SSH unresponsive - cannot form 2-node cluster
2. **Critical**: --wait-for-peers has no timeout - Thor #1 hung indefinitely
3. **Unresolved**: Why does single node load full 70B model (69GB) instead of sharding?

## Next Steps

### Immediate (Thor #2 Recovery)
1. Physical reboot of Thor #2 at 10.0.0.78
2. Verify SSH service running post-reboot
3. Sync git to commit 5b06048

### Code Fix (--wait-for-peers timeout)
**Option A**: Add timeout to discover_peers
```python
async def discover_peers(self, wait_for_peers: int = 0, timeout: int = 60) -> List[PeerHandle]:
    if wait_for_peers > 0:
        start_time = time.time()
        while len(self.known_peers) < wait_for_peers:
            if time.time() - start_time > timeout:
                print(f"Warning: Timeout waiting for {wait_for_peers} peers. Found {len(self.known_peers)}. Proceeding...")
                break
            await asyncio.sleep(0.1)
    return [peer_handle for peer_handle, _, _, _ in self.known_peers.values()]
```

**Option B**: Remove --wait-for-peers, rely on natural discovery
- Let both nodes start independently
- Discovery happens via UDP broadcast
- May take 30-60 seconds to form cluster

### Investigation (Single Node Loading Full Model)
1. Why does single node with --wait-for-peers=1 load full 70B?
2. Check if model partitioning happens BEFORE or AFTER peer discovery
3. Review topology collection logic

## Errors Encountered

### Thor #2 SSH Timeout
```
ssh: connect to host 10.0.0.78 port 22: Connection timed out
```

### Thor #1 Hung Process
- No explicit error
- Process running but unresponsive
- No log output (Python buffering or waiting silently)
- Port 8080 never opened
- Memory usage: 69GB (full model loaded)

## Test Environment
- **Mira**: 10.0.0.163 (Ubuntu 24.04)
- **Thor #1**: 10.0.0.93 (Jetson Thor, Blackwell sm_101, 122GB RAM)
- **Thor #2**: 10.0.0.78 (Jetson Thor, SSH DOWN)
- **Model**: llama-3.3-70b (~140GB)
- **Exo Branch**: thor-compatibility-2025-11-02 (commit 5b06048)

## Single Node Test Results (Without --wait-for-peers)

### Configuration
```bash
python3 -m exo.main --node-port 52415 --chatgpt-api-port 8080
```
(NO --wait-for-peers flag)

### Results
- **Process**: Started successfully (PID 5656)
- **Ports**:
  - 52415 (gRPC) - LISTENING
  - 8080 (API) - LISTENING ✓
  - 5678 (UDP discovery) - ACTIVE
- **Memory**: 69GB / 122GB (full 70B model loaded)
- **API Status**: Ready and responding
- **Inference**: FAILED with BFloat16 error

### Inference Test
```bash
curl -X POST http://10.0.0.93:8080/v1/chat/completions \
  -d '{"model": "llama-3.3-70b", "messages": [{"role": "user", "content": "Count from 1 to 3"}], "max_tokens": 20}'
```

**Response**:
```json
{"detail": "Error processing prompt (see logs with DEBUG>=2): Got unsupported ScalarType BFloat16"}
```
- **HTTP Status**: 500
- **Time**: 1.58s
- **Status**: FAILED

### Root Cause: INCOMPLETE BFLOAT16 FIX

**Error**:
```python
File "/home/jetson/exo-clean/exo/inference/tinygrad/tinygrad_helpers.py", line 60
    tensor_data = tensor_data.numpy()
TypeError: Got unsupported ScalarType BFloat16
```

**What Went Wrong**:
Commit e820caf changed from `framework="numpy"` to `framework="pt"` and added `.numpy()` conversion:

```python
with safe_open(fn, framework="pt") as f:
    tensor_data = f.get_tensor(k)
    if hasattr(tensor_data, 'numpy'):
        tensor_data = tensor_data.numpy()  # FAILS HERE!
```

**The Problem**:
- PyTorch loads bfloat16 tensors successfully ✓
- But PyTorch's `.numpy()` method does NOT support bfloat16! ✗
- NumPy has no native bfloat16 dtype
- Calling `.numpy()` on bfloat16 tensor raises: `Got unsupported ScalarType BFloat16`

**The Fix Needed**:
Convert bfloat16 → float32 BEFORE calling `.numpy()`:

```python
with safe_open(fn, framework="pt") as f:
    tensor_data = f.get_tensor(k)
    if hasattr(tensor_data, 'numpy'):
        # Handle bfloat16 by converting to float32 first
        if hasattr(tensor_data, 'dtype') and str(tensor_data.dtype) == 'torch.bfloat16':
            tensor_data = tensor_data.to(torch.float32)
        tensor_data = tensor_data.numpy()
```

**Impact**:
- Current code CANNOT load Llama 3.3 70B model
- All commits (e820caf, 5b06048) focused on research, not complete fix
- Distributed inference testing BLOCKED until this is resolved

## Recommendation
**DO NOT PROCEED** with further distributed testing until:
1. ✗ CRITICAL: Complete bfloat16 fix (convert to float32 before .numpy())
2. Thor #2 SSH is restored
3. --wait-for-peers timeout is implemented OR removed from launch commands
4. Root cause of single-node full model loading is understood

The --wait-for-peers flag, while theoretically correct, creates an operational hazard (infinite hang) and does NOT prevent full model loading even when specified.

## Critical Code Fix Required

**File**: `/home/jetson/exo-clean/exo/inference/tinygrad/tinygrad_helpers.py`
**Line**: ~60
**Current Code** (BROKEN):
```python
tensor_data = f.get_tensor(k)
if hasattr(tensor_data, 'numpy'):
    tensor_data = tensor_data.numpy()  # FAILS on bfloat16
```

**Fixed Code**:
```python
import torch

tensor_data = f.get_tensor(k)
if hasattr(tensor_data, 'numpy'):
    # Convert bfloat16 to float32 before numpy conversion
    if hasattr(tensor_data, 'dtype') and tensor_data.dtype == torch.bfloat16:
        tensor_data = tensor_data.to(torch.float32)
    tensor_data = tensor_data.numpy()
```
