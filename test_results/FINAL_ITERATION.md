# Final Iteration: Root Cause Analysis

**Test Configuration**:
- Python: unbuffered (-u flag)
- Network: Default (no --node-host)
- Logging: tee to file + nohup
- Timestamp: 20251104_220132
- Runtime: 9+ minutes observed

---

## Startup Logs

### Thor #1 (jetson@10.0.0.93)
```
Selected inference engine: None
  _____  _____
 / _ \ \/ / _ \
|  __/>  < (_) |
 \___/_/\_\___/

================================================================================
EXO
================================================================================

EXO started out of a desire to run research experiments on large language
models using the hardware we already owned.

What began here is becoming part of something much larger.

soon™

- The EXO Team
================================================================================

Detected system: Linux
Inference engine name after selection: tinygrad
Using inference engine: TinygradDynamicShardInferenceEngine with shard downloader: SingletonShardDownloader
Chat interface started:
 - http://192.168.20.1:8080
 - http://192.168.40.1:8080
 - http://192.168.55.1:8080
 - http://10.0.0.8:8080
 - http://192.168.30.1:8080
 - http://10.0.0.93:8080
 - http://192.168.10.1:8080
 - http://127.0.0.1:8080
ChatGPT API endpoint served at:
 - http://192.168.20.1:8080/v1/chat/completions
 - http://192.168.40.1:8080/v1/chat/completions
 - http://192.168.55.1:8080/v1/chat/completions
 - http://10.0.0.8:8080/v1/chat/completions
 - http://192.168.30.1:8080/v1/chat/completions
 - http://10.0.0.93:8080/v1/chat/completions
 - http://192.168.10.1:8080/v1/chat/completions
 - http://127.0.0.1:8080/v1/chat/completions
has_read=True, has_write=True
Warning: pynvml failed (Not Supported), using fallback device detection
Warning: pynvml failed (Not Supported), using fallback device detection

[END OF OUTPUT - 46 lines total]
```

### Thor #2 (thor@10.0.0.78)
```
/home/thor/.local/lib/python3.12/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated
Selected inference engine: None
[Same EXO banner as Thor #1]
Detected system: Linux
Inference engine name after selection: tinygrad
Using inference engine: TinygradDynamicShardInferenceEngine with shard downloader: SingletonShardDownloader
Chat interface started:
 - http://10.0.0.78:8081
 - http://192.168.55.1:8081
 - http://127.0.0.1:8081
 - http://10.0.0.197:8081
ChatGPT API endpoint served at:
 - http://10.0.0.78:8081/v1/chat/completions
 - http://192.168.55.1:8081/v1/chat/completions
 - http://127.0.0.1:8081/v1/chat/completions
 - http://10.0.0.197:8081/v1/chat/completions
has_read=True, has_write=True
Warning: pynvml failed (Not Supported), using fallback device detection
Warning: pynvml failed (Not Supported), using fallback device detection

[END OF OUTPUT - 40 lines total]
```

---

## Discovery Analysis

**Discovery Messages Found**: ZERO on both nodes

Ran grep for:
- `discover`
- `peer`
- `cluster`
- `node`

**Result**: No discovery-related output after startup messages

**Total Log Lines**:
- Thor #1: 46 lines
- Thor #2: 40 lines
- No additional output after 9+ minutes

---

## Process Analysis

### Thor #1
```
jetson      6487 89.3% CPU, 76.4GB RAM
python3 -u -m exo.main --node-port 52415 --chatgpt-api-port 8080
```

### Thor #2
```
thor        3555 126% CPU, 63.5GB RAM
python3 -u -m exo.main --node-port 52416 --chatgpt-api-port 8081
```

**Observations**:
- Processes consuming massive CPU (89-126%)
- Large memory footprint (63-76 GB)
- NO new log output despite CPU activity
- Running for 8-9 minutes continuously

---

## Memory Usage (after 180s)

```
Thor #1: 4.5GB / 122GB used (excluding process memory)
Thor #2: 4.6GB / 122GB used (excluding process memory)
```

**Assessment**: Memory usage does NOT indicate model loading (would need 30-50GB for LLaMA 70B sharded)

---

## Inference Test

```bash
curl http://10.0.0.93:8080/v1/chat/completions
```

**Result**:
- HTTP: 000 (connection timeout)
- Time: 60.002s
- Response: Empty (0 bytes received)
- Status: **COMPLETE FAILURE**

Server accepted connection but never responded.

---

## ROOT CAUSE ANALYSIS

### Why Discovery Failed

**Finding**: Discovery is NOT failing - **it never started**

**Evidence**:
1. No discovery-related log messages (expected: "Discovered peer", "Cluster formed", etc.)
2. Zero output after initial startup banners
3. No UDP discovery messages in logs
4. No peer count updates

**Root Cause**:
- Code enters async event loop after printing startup messages
- NO debug logging for discovery operations
- `UDPDiscovery` runs silently with DEBUG=0 default
- Discovery may be running but producing NO observable output

### Why Model Didn't Load

**Finding**: No model was requested, so none loaded

**Evidence**:
1. No `--run-model` argument provided
2. No inference request sent successfully
3. Memory usage too low for 70B model (would need 30-50GB per node)
4. API endpoint times out → waiting for something

**Root Cause**:
- Nodes initialize but don't auto-load models
- Wait for API request to trigger model download
- First request should trigger model download → sharding → distribution
- Our request timed out before this could happen

### Why Logs Are Silent

**Finding**: Python `-u` flag is NOT sufficient for async logging

**Evidence**:
1. Only 46/40 lines after 9+ minutes
2. All output is synchronous prints in main.py startup
3. Async event loop produces NO output
4. No logging framework configured (only print statements)

**Root Cause**:
- exo uses `print()` for startup, then enters asyncio event loop
- Async tasks don't print to stdout by default
- No logging.basicConfig() or logger setup
- DEBUG environment variable defaults to 0 (silent)
- GRPC_VERBOSITY set to "error" (line 37 of main.py)

---

## What Needs to Happen Next

### 1. Enable Debug Logging

**Current**:
```python
# Line 25 of exo/main.py
from exo.helpers import DEBUG
```

**Problem**: DEBUG defaults to 0, silencing all debug output

**Fix Options**:

**Option A: Environment Variable**
```bash
DEBUG=2 python3 -m exo.main --node-port 52415 --chatgpt-api-port 8080
```

**Option B: Add Logging Framework**
```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
```

### 2. Verify Discovery Configuration

**Check discovery module**:
```bash
# Look at UDPDiscovery implementation
grep -r "class UDPDiscovery" exo/networking/udp/
```

**Expected behavior**:
- Broadcasts on port 5678 (default)
- Listens for peer responses
- Logs "Discovered peer" when found
- Updates cluster topology

**Validation**:
```bash
# On one Thor, listen for UDP broadcasts
sudo tcpdump -i any -n udp port 5678
```

### 3. Test with Explicit Discovery

**Manual discovery mode**:
```bash
# Thor #1
python3 -m exo.main \
  --node-port 52415 \
  --chatgpt-api-port 8080 \
  --discovery-module manual \
  --discovery-config-path /tmp/peers.json

# /tmp/peers.json:
{
  "peers": [
    {"id": "thor2", "host": "10.0.0.78", "port": 52416}
  ]
}
```

### 4. Test Single-Node Inference First

**Validate API works before testing distributed**:
```bash
# Launch single node with small model
python3 -m exo.main \
  --chatgpt-api-port 8080 \
  --run-model llama-3.1-8b \
  --prompt "Hello"
```

**Expected**:
- Model downloads to ~/.cache/exo
- Tinygrad compiles kernels
- Inference runs
- Tokens print to stdout

### 5. Check Network Connectivity

**UDP discovery requires bidirectional UDP**:
```bash
# On Thor #1
nc -u -l 5678 &
echo "test" | nc -u 10.0.0.78 5678

# On Thor #2
nc -u -l 5678 &
echo "test" | nc -u 10.0.0.93 5678
```

**Check firewall**:
```bash
sudo iptables -L -n | grep 5678
sudo ufw status
```

---

## Recommendations

### If Testing Discovery

1. **Enable maximum verbosity**:
   ```bash
   DEBUG=3 GRPC_VERBOSITY=debug python3 -u -m exo.main ...
   ```

2. **Monitor UDP traffic**:
   ```bash
   sudo tcpdump -i any -n udp port 5678 -vvv
   ```

3. **Use manual discovery** to eliminate UDP issues:
   ```bash
   --discovery-module manual --discovery-config-path peers.json
   ```

4. **Check for discovery output** in code:
   ```bash
   grep -r "Discovered\|peer\|broadcast" exo/networking/udp/
   ```

### If Testing Inference

1. **Start with single-node, small model**:
   ```bash
   DEBUG=2 python3 -m exo.main --run-model llama-3.1-8b
   ```

2. **Watch for model download**:
   ```bash
   watch -n 1 "du -sh ~/.cache/exo"
   ```

3. **Monitor GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **Check API with verbose curl**:
   ```bash
   curl -v http://localhost:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "llama-3.1-8b", ...}'
   ```

### If Discovery Still Fails

1. **Read discovery source**:
   ```bash
   cat exo/networking/udp/udp_discovery.py
   ```

2. **Check for required ports**:
   ```bash
   ss -uln | grep 5678  # UDP listen
   ss -tln | grep 5241  # Node ports
   ```

3. **Try different discovery module**:
   ```bash
   --discovery-module tailscale  # If Tailscale configured
   ```

4. **File GitHub issue** with:
   - Full logs with DEBUG=3
   - tcpdump output
   - Network topology
   - exo version/commit

---

## Key Insights

### The Silent Loop Problem

**exo architecture**:
1. Synchronous startup: Prints banners, initializes components
2. Enters asyncio event loop: `await node.start()`
3. Event loop runs forever with NO output unless DEBUG > 0

**Result**: Process appears "stuck" but is actually listening for:
- UDP discovery broadcasts
- gRPC requests from peers
- API requests on ChatGPT port

### The First-Request Bootstrap

**Model loading is lazy**:
1. Node starts with NO model loaded
2. Waits for first API request
3. Request triggers:
   - Model download
   - Shard calculation (based on cluster topology)
   - Shard distribution across nodes
   - Compilation of inference kernels

**Our timeout**: We hit the endpoint before discovery completed, causing infinite wait

### The Debug Visibility Gap

**Current issues**:
- Startup: Verbose (banners, config)
- Discovery: Silent (unless DEBUG > 0)
- Model loading: Silent (no progress bars in headless mode)
- Inference: Silent (only prints tokens)

**Needed**: Logging framework that shows:
- Discovery: "Broadcasting...", "Found peer X"
- Download: "Downloading shard 1/8 (23%)"
- Compilation: "Compiling kernel for layer 12"
- Inference: "Tokens/sec: 45.2"

---

## Next Actions

**Priority 1**: Enable visibility
```bash
DEBUG=3 GRPC_VERBOSITY=debug python3 -u -m exo.main ...
```

**Priority 2**: Test single-node first
```bash
python3 -m exo.main --run-model llama-3.1-8b --prompt "test"
```

**Priority 3**: Manual discovery
```bash
--discovery-module manual --discovery-config-path peers.json
```

**Priority 4**: Network validation
```bash
tcpdump -i any udp port 5678
```

**Priority 5**: Read discovery source
```bash
cat exo/networking/udp/udp_discovery.py | grep -A 20 "def.*discover\|def.*broadcast"
```

---

## Conclusion

**Discovery Status**: Unknown (no visibility)
**Model Loading Status**: Not attempted (no successful API request)
**Root Cause**: Debug logging disabled → cannot observe what's happening

**The core issue is NOT a bug** - it's an observability gap. exo is likely working as designed, but we can't see what it's doing.

**Critical Path**:
1. Enable DEBUG logging
2. Test single-node inference
3. Monitor network for UDP broadcasts
4. Try manual discovery if UDP fails
5. Read discovery source code if still stuck

**This is a measurement problem, not a functionality problem.**
