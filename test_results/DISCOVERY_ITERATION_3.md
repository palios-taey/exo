# Discovery Iteration 3: 10GbE Network Testing
**Date**: 2025-11-04
**Test Duration**: 25+ minutes
**Timestamp**: 21:30 - 21:56 UTC

## Discovery Failure Analysis (Previous Test)

### What Interfaces Were Used (Sync Test)
From `/tmp/exo_thor1_sync_20251104_205639.log` and `/tmp/exo_thor2_sync_20251104_205639.log`:

**Thor #1 Bound Interfaces**:
- http://192.168.10.1:8080
- http://192.168.55.1:8080
- http://192.168.20.1:8080
- http://10.0.0.93:8080
- http://192.168.30.1:8080
- http://10.0.0.8:8080
- http://127.0.0.1:8080
- http://192.168.40.1:8080

**Thor #2 Bound Interfaces**:
- http://10.0.0.197:8081
- http://10.0.0.78:8081
- http://127.0.0.1:8081
- http://192.168.55.1:8081

### UDP Discovery Status
**CRITICAL FINDING**: NO discovery messages in either log
- No "Listening" or "Bound" messages for UDP
- No "Interface" or "broadcast" messages
- No "peer discovered" messages
- No "Exo Cluster (X nodes)" messages

### Root Cause of Sync Test Failure
**Processes were killed prematurely (SIGTERM) before cluster formation**

Both logs show:
```
Received exit signal SIGTERM...
Thank you for using exo.
Cancelling 5 outstanding tasks
```

**Conclusion**: The sync test failed NOT because of discovery issues, but because the test script killed processes too early. No time for UDP discovery to occur.

---

## Test 1: --node-host on 10.0.0.x

### Launch Configuration
```bash
# Thor #1
python3 -m exo.main --node-host 10.0.0.93 --node-port 52415 --chatgpt-api-port 8080

# Thor #2
python3 -m exo.main --node-host 10.0.0.78 --node-port 52416 --chatgpt-api-port 8081
```

**Launch Time**: 21:30:38 UTC (Thor #1), 21:33:XX UTC (Thor #2)
**Wait Time**: 120+ seconds
**Test Time**: 21:56 UTC (25+ minutes total)

### Cluster Formation
**Status**: NO CLUSTER FORMED

**Evidence**:
```bash
# Thor #1 log
grep -E '(Exo Cluster|peer|node discovered|partition)' /tmp/exo_thor1_10gbe_*.log
# No output

# Thor #2 log
grep -E '(Exo Cluster|peer|node discovered|partition)' /tmp/exo_thor2_10gbe_*.log
# No output
```

**Thor #1 Log**: EMPTY (0 bytes stdout output to file)
**Thor #2 Log**: Only 2 lines (pynvml deprecation warning)

### Memory Usage
**After 25+ minutes runtime**:

| Thor | Memory Used | Expected if Sharding | Expected if Single | Status |
|------|-------------|---------------------|-------------------|--------|
| #1   | 4.5 GB      | ~35 GB             | ~70 GB           | NO MODEL LOADED |
| #2   | 4.7 GB      | ~35 GB             | ~70 GB           | NO MODEL LOADED |
| **Total** | **9.2 GB** | **~70 GB**      | **~140 GB**      | **FAILED** |

**Analysis**: Only ~4.7GB per Thor = baseline Python + exo overhead. NO model weights loaded whatsoever.

### Process Status
Both processes RUNNING:
```
# Thor #1 (jetson@10.0.0.93)
jetson    6318  1.0%  0.5%  19024528  750644  Sl  21:30  0:16  python3 -m exo.main...

# Thor #2 (thor@10.0.0.78)
thor      3378  1.3%  0.7%  556985744  959404  Sl  21:33  0:19  python3 -m exo.main...
```

**CPU Usage**: 1.0-1.3% (very low - idle state)
**Runtime**: 16-19 minutes

### Inference Test
```bash
curl -X POST http://10.0.0.93:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3.3-70b", "messages": [{"role": "user", "content": "Say hello in 3 words"}], "max_tokens": 10}' \
  --max-time 60
```

**Result**:
- HTTP: 000 (connection timeout)
- Time: 60.002s (full timeout)
- Response: 0 bytes received
- Status: **FAILED - Server not responding**

---

## Root Cause Analysis

### Problem 1: Log Output Not Written
Thor #1 log file exists but is EMPTY (0 bytes)
Thor #2 log file has only 2 lines (stderr warning)

**Root Cause**: stdout/stderr buffering preventing log output
**Why**: Python buffers stdout by default. File redirection (`> file.log 2>&1`) doesn't force flush.

**Solution**:
```bash
python3 -u -m exo.main ...  # -u = unbuffered output
# OR
stdbuf -oL -eL python3 -m exo.main ...  # Line-buffered
```

### Problem 2: No Cluster Formation
After 25+ minutes, no peer discovery occurred.

**Possible Causes**:
1. **UDP Discovery Not Working**:
   - --node-host binds TCP listener but may not enable UDP broadcast
   - Firewall blocking UDP on port 52415/52416
   - Network interfaces not correctly bound for discovery

2. **Wrong Network Interface**:
   - 10.0.0.x is correct network
   - But exo may be discovering on wrong interface (192.168.x.x)
   - Need to verify which interface exo uses for discovery

3. **Discovery Mechanism Disabled**:
   - --node-host may disable auto-discovery
   - Might require manual peer specification

### Problem 3: Model Never Loaded
No memory increase = model download/loading never started.

**Root Cause**: Model loading triggered by inference request OR peer connection
**Why Failed**: No peer connection + inference request timeout = no trigger

---

## Overall Result
**STATUS**: FAILED

**Failures**:
- ❌ Cluster formation: NO
- ❌ Model sharding: NO
- ❌ Model loading: NO
- ❌ Inference: NO (timeout)
- ❌ Logs visible: NO (buffering issue)

**Successes**:
- ✅ Processes launched
- ✅ Processes still running after 25+ minutes
- ✅ --node-host binding accepted (no startup errors)
- ✅ Network connectivity confirmed (can SSH, curl attempted)

---

## Recommended Next Steps

### Immediate Actions
1. **Fix Log Visibility**:
   ```bash
   python3 -u -m exo.main --node-host 10.0.0.93 --node-port 52415 --chatgpt-api-port 8080
   ```

2. **Check Firewall**:
   ```bash
   # On both Thors
   sudo ufw status
   sudo iptables -L -n | grep 524
   ```

3. **Verify Discovery Mechanism**:
   ```bash
   # Check if exo uses UDP broadcast or multicast
   # Check if --node-host disables discovery
   netstat -ulnp | grep python  # Check UDP listeners
   ```

4. **Try Manual Peer Specification**:
   ```bash
   # If discovery is broken, try explicit peering
   # Check exo docs for --peer or --connect-to flags
   ```

### Investigation Required
1. **Read exo source code**:
   - How does --node-host affect discovery?
   - What UDP ports does discovery use?
   - Is there a --peer flag for manual connection?

2. **Network packet capture**:
   ```bash
   sudo tcpdump -i any port 52415 or port 52416 -n
   ```

3. **Check exo discovery config**:
   - Is there a config file for discovery?
   - Can we force specific discovery interface?

### Alternative Approaches
1. **Try WITHOUT --node-host**:
   - Let exo auto-detect network
   - See if default discovery works

2. **Try Manual Discovery Config** (if supported):
   ```json
   {
     "peers": {
       "thor1": {"addr": "10.0.0.93", "port": 52415},
       "thor2": {"addr": "10.0.0.78", "port": 52416}
     }
   }
   ```

3. **Check for Discovery Debug Flags**:
   ```bash
   python3 -m exo.main --help | grep -i discover
   python3 -m exo.main --help | grep -i peer
   ```

---

## Key Learnings

1. **Buffered Output Hides Problems**: Can't debug without logs. Always use `-u` or `stdbuf`.

2. **Process Running ≠ Working**: Both processes ran for 25+ minutes but did nothing useful.

3. **--node-host May Disable Discovery**: Explicit binding might conflict with UDP broadcast.

4. **No Model = No Memory**: Easy way to check if anything loaded: `free -h`

5. **Timeouts Are Expensive**: 60s timeout * 3 attempts = 3 minutes wasted per test.

---

## Next Iteration Focus
**PRIORITY**: Get visibility into what exo is actually doing

1. Enable unbuffered logging (`python3 -u`)
2. Add verbose/debug flags if available
3. Monitor network traffic during startup
4. Try without --node-host to see if discovery works at all
5. Read exo source to understand discovery mechanism

**SUCCESS CRITERIA**:
- See "Exo Cluster (2 nodes)" message
- See memory usage ~35GB per Thor (sharding)
- Get successful inference response <30s

---

**Test Conducted By**: Claude (Sonnet 4.5)
**Infrastructure**: 2x Jetson Thor (Blackwell sm_101, 128GB unified memory each)
**Network**: 4×25GbE per Thor, 43.6 Gbps aggregate stable
**Software**: exo (claude-cuda13-blackwell-patches branch)
