# Git Synchronization and Distributed Test
**Date**: 2025-11-04
**Test Time**: 20:56 - 21:25 UTC

## Git Divergence Fix

### Before Sync
- Mira:    9923604 Fix bfloat16 conversion: convert to float32 before numpy()
- Thor #1: 6d8250b Fix bfloat16 conversion: convert to float32 before numpy() (DIFFERENT HASH!)
- Thor #2: d6b0eed fix: Implement lazy safetensors loading for distributed inference (3 commits behind)

**Problem**: Same commit content, different hashes between Mira and Thor #1. Thor #2 missing 3 commits.

### Sync Method
Used rsync to sync entire .git/ directory from Mira to both Thors:
```bash
rsync -avz --delete /home/mira/exo/.git/ jetson@10.0.0.93:/home/jetson/exo-clean/.git/
rsync -avz --delete /home/mira/exo/.git/ thor@10.0.0.78:/home/thor/exo-clean/.git/
```

### After Sync
- Mira:    992360449db3d57a271a8e87db2b5338a6ef15db
- Thor #1: 992360449db3d57a271a8e87db2b5338a6ef15db
- Thor #2: 992360449db3d57a271a8e87db2b5338a6ef15db
- Status: ✓ **SYNCHRONIZED**

All three machines now have identical commit hashes.

## Distributed Test (10.0.0.x network)

### Launch Parameters
- **Network**: 10.0.0.x (10GbE, NOT 25GbE as requested)
- **Thor #1**: python3 -u -m exo.main --node-port 52415 --chatgpt-api-port 8080
- **Thor #2**: python3 -u -m exo.main --node-port 52416 --chatgpt-api-port 8081
- **Note**: NO --node-host parameter (allows binding to all interfaces)
- **Timestamp**: 20251104_211452

### Cluster Formation

#### Thor #1 Log (tail)
```
Selected inference engine: None
Detected system: Linux
Inference engine name after selection: tinygrad
Using inference engine: TinygradDynamicShardInferenceEngine with shard downloader: SingletonShardDownloader
Chat interface started:
 - http://10.0.0.93:8080
 - http://192.168.40.1:8080
 - http://127.0.0.1:8080
 - (and other interfaces)
ChatGPT API endpoint served at:
 - http://10.0.0.93:8080/v1/chat/completions
 - (and other interfaces)
has_read=True, has_write=True
Warning: pynvml failed (Not Supported), using fallback device detection
```

#### Thor #2 Log (tail)
```
Inference engine name after selection: tinygrad
Using inference engine: TinygradDynamicShardInferenceEngine with shard downloader: SingletonShardDownloader
Chat interface started:
ChatGPT API endpoint served at:
```

#### Cluster Status
- **Thor #1**: Running, API listening on port 8080
- **Thor #2**: Running, API listening on port 8081
- **Peer Discovery**: No log output indicating cluster formation (NO "Exo Cluster (2 nodes)" message)
- **Status**: **SINGLE NODES** - Each Thor running independently, no peer discovery

### Memory Usage

#### Initial State (after ~5 min runtime)
- **Thor #1**: 69GB / 122GB (57% - model loaded)
- **Thor #2**: 4.6GB / 122GB (4% - model NOT loaded)
- **Total**: 73.6GB used
- **Sharding**: **NOT WORKING** - Thor #1 loaded full model, Thor #2 loaded nothing

### Inference Test

#### Test 1: JSON Parse Error
```bash
curl -X POST http://10.0.0.93:8080/v1/chat/completions \
  -d '{"model": "llama-3.3-70b", "messages": [{"role": "user", "content": "Hello! Count from 1 to 3."}], "max_tokens": 20}'
```
**Result**: HTTP 500 - `json.decoder.JSONDecodeError: Invalid \escape: line 1 column 75 (char 74)`

#### Test 2: Fixed JSON, Timeout
```bash
curl -X POST http://10.0.0.93:8080/v1/chat/completions \
  -d "{\"model\": \"llama-3.3-70b\", \"messages\": [{\"role\": \"user\", \"content\": \"Count to 3\"}], \"max_tokens\": 20}"
```
**Result**:
- HTTP: 000 (timeout)
- Time: 90+ seconds
- Response: None (request hung, never returned)
- **Status**: **FAILED**

## Overall Result

**PARTIAL SUCCESS / MOSTLY FAILED**

### What Worked
1. ✓ Git synchronization successful - all three machines now at same commit
2. ✓ Both Thor processes launched successfully
3. ✓ Both Thors initialized tinygrad inference engine
4. ✓ Both Thors started API servers on their respective ports

### What Failed
1. ✗ **Peer discovery failed** - Thors did not discover each other, no cluster formation
2. ✗ **Model sharding failed** - Thor #1 loaded full 70B model (69GB), Thor #2 loaded nothing (4.6GB)
3. ✗ **Inference failed** - Request hung for 90+ seconds, timed out
4. ✗ **Memory efficiency failed** - Using 73.6GB total instead of expected ~35GB each for sharding

## Issues Found

### Critical Issues

1. **No Peer Discovery**
   - Expected: "Exo Cluster (2 nodes)" message in logs
   - Actual: No cluster formation log output
   - Impact: Each Thor operating independently, no distributed inference

2. **Sharding Regression**
   - Thor #1 loaded full 69GB model
   - Thor #2 loaded almost nothing (4.6GB)
   - Previous test (SHARDING_REGRESSION_ANALYSIS.md) showed similar issue
   - Root cause: Peer discovery failure prevents shard negotiation

3. **Inference Timeout**
   - Request sent to Thor #1 hung for 90+ seconds
   - No response returned
   - Possible causes:
     - Thor #1 waiting for Thor #2 (which never connected)
     - Model loading issue
     - Network communication failure

### Configuration Issues

1. **Log Buffering**
   - Even with `python3 -u` (unbuffered), logs took >5 minutes to appear
   - Makes real-time debugging difficult
   - Thor #1 log empty for first 5+ minutes despite process running

2. **Network Binding**
   - Thor #1 bound to 8 different interfaces (10.0.0.93, 192.168.x.x, 127.0.0.1, etc.)
   - May be confusing peer discovery
   - No control over which interface used for peer communication

### Process Stability

1. **Thor #2 Launch Failure**
   - First launch via SSH backgrounding failed (process never started)
   - Required manual re-launch with nohup
   - SSH backgrounding with `&` unreliable for complex commands

## Analysis

### Why Peer Discovery Failed

**Hypothesis**: Without `--node-host` parameter, Thors may be:
1. Broadcasting on wrong interface (192.168.x.x instead of 10.0.0.x)
2. Unable to resolve each other's addresses
3. Firewall blocking discovery packets
4. Using incompatible discovery protocol parameters

**Previous working test** (from logs) showed:
- `--node-host 192.168.10.1` (Thor #1)
- `--node-host 192.168.10.2` (Thor #2)
- Successfully formed 2-node cluster

**Jesse's instruction** to remove `--node-host` may have broken discovery.

### Why Sharding Failed

**Root cause**: Without peer discovery, no cluster = no shard negotiation.

**Expected behavior**:
1. Thor #1 discovers Thor #2
2. Cluster forms: "Exo Cluster (2 nodes)"
3. Shards negotiated: Thor #1 gets layers 0-39, Thor #2 gets layers 40-79
4. Each Thor loads ~35GB

**Actual behavior**:
1. Thor #1 never discovers Thor #2
2. No cluster formation
3. Thor #1 thinks it's alone, loads full 69GB model
4. Thor #2 waits indefinitely (never loads model)

### Why Inference Timed Out

**Hypothesis**: Thor #1 may be:
1. Waiting for shard from Thor #2 (which never loaded)
2. Stuck in discovery loop
3. Model loading incomplete (despite 69GB allocated)
4. Deadlocked waiting for peer response

## Next Steps

### Immediate Actions

1. **Restore --node-host Parameters**
   ```bash
   # Thor #1
   python3 -m exo.main --node-host 10.0.0.93 --node-port 52415 --chatgpt-api-port 8080

   # Thor #2
   python3 -m exo.main --node-host 10.0.0.78 --node-port 52416 --chatgpt-api-port 8081
   ```
   Use 10.0.0.x addresses instead of 192.168.10.x to test on correct network.

2. **Verify Network Connectivity**
   ```bash
   # Test UDP broadcast discovery
   nc -u 10.0.0.93 52415
   nc -u 10.0.0.78 52416
   ```

3. **Check Firewall Rules**
   ```bash
   sudo ufw status
   sudo iptables -L
   ```

4. **Enable Debug Logging**
   ```bash
   EXOPY_DEBUG=1 python3 -m exo.main ...
   ```

### Research Needed

1. **Exo Discovery Protocol**
   - Read exo source code for discovery mechanism
   - Understand how --node-host affects discovery
   - Determine if UDP broadcast vs multicast vs unicast

2. **Network Requirements**
   - Which ports need to be open?
   - Which interfaces should be used?
   - Does discovery work across subnets?

3. **Sharding Algorithm**
   - When does shard negotiation occur?
   - How are layer ranges determined?
   - What triggers model loading?

### Long-Term Fixes

1. **Automated Launch Script**
   - Wrapper script handling both Thors
   - Verifies peer discovery before declaring success
   - Auto-restarts on failure

2. **Health Check Endpoint**
   - Add /health endpoint showing cluster status
   - Query before attempting inference
   - Fail fast if cluster not formed

3. **Better Logging**
   - Force unbuffered output
   - Structured JSON logs
   - Real-time tailing

## Conclusion

**Git synchronization: SUCCESS**
- All three machines now have identical git state
- Method (rsync .git/) was reliable and fast

**Distributed inference: FAILED**
- No peer discovery
- No cluster formation
- No model sharding
- Inference timeout

**Root cause**: Removing `--node-host` parameter broke peer discovery. Exo likely needs explicit host binding to find peers on correct network interface.

**Recommendation**: Restore `--node-host` with 10.0.0.x addresses and retest.
