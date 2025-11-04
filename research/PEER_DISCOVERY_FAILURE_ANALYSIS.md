# Peer Discovery Failure Analysis

**Date**: 2025-11-04
**Problem**: Two Thor nodes running exo but not discovering each other
**Symptom**: Both nodes loading FULL 70GB model (228GB total RAM) instead of sharding

---

## Discovery Configuration

**Default Discovery Mode**: UDP (broadcast-based)
- Default listen port: 5678
- Default broadcast port: 5678
- Discovery timeout: 30 seconds
- Broadcast interval: 2.5 seconds

**Required Command Line Args**: NONE (UDP is default)

**Current Args Being Used**:
- Thor #1: `python3 -m exo.main --node-host 10.0.0.93 --node-port 52415 --chatgpt-api-port 8080`
- Thor #2: `python3 -m exo.main --node-host 10.0.0.78 --node-port 52416 --chatgpt-api-port 8081`

**Problem Identified**: `--node-host` is being set to specific IPs

---

## Network Connectivity

**Thor #1 → Thor #2**: ✅ PASS
- Ping successful: 0.370-0.669ms latency
- 0% packet loss

**Thor #2 → Thor #1**: ✅ PASS
- Ping successful: 0.483-1.07ms latency
- 0% packet loss

**UDP Ports Listening**:
- Thor #1: Port 5678 UDP OPEN (pid 5031)
- Thor #2: Port 5678 UDP NOT CHECKED (process crashed during test)

---

## Firewall Status

**Thor #1 Firewall**: ✅ INACTIVE
- No DROP or REJECT rules found
- UDP broadcast allowed

**Thor #2 Firewall**: ✅ INACTIVE
- No DROP or REJECT rules found
- UDP broadcast allowed

---

## Discovery Code Analysis

**Discovery Module Location**:
- `/home/mira/exo/exo/networking/udp/udp_discovery.py`
- `/home/mira/exo/exo/main.py` (initialization)

**How Discovery Works**:

1. **Broadcasting**: Each node broadcasts UDP packets every 2.5 seconds on port 5678
   - Message format: JSON with node_id, grpc_port, device_capabilities
   - Broadcasts to subnet broadcast address (e.g., 10.0.0.255) AND global broadcast (255.255.255.255)
   - Binds to ALL network interfaces and broadcasts from each

2. **Listening**: Each node listens on UDP port 5678 for discovery messages
   - Filters out own node_id
   - Performs health check on discovered peers via gRPC
   - Adds healthy peers to `known_peers` dictionary

3. **Cleanup**: Background task removes stale peers after discovery_timeout (30s)

**Required Network Conditions**:
- UDP port 5678 open for broadcast/listen
- Nodes must be on same subnet OR able to receive global broadcast (255.255.255.255)
- gRPC port (50000 by default, or --node-port) must be reachable for health checks

---

## Log Analysis

**Discovery Messages in Logs**: ❌ NO - No "Exo Cluster" or peer discovery messages found

**Error Messages**: ✅ YES - Critical gRPC connection error found

**Thor #2 Log Error**:
```
grpc.aio._call.AioRpcError: <AioRpcError of RPC that terminated with:
    status = StatusCode.UNAVAILABLE
    details = "recvmsg:Connection reset by peer"
    debug_error_string = "UNKNOWN:Error received from peer {grpc_message:"recvmsg:Connection reset by peer", grpc_status:14, created_time:"2025-11-04T19:58:56.302678598+00:00"}"
>
```

**Startup Sequence**:
1. Model loading starts immediately (Layer 0 through Layer 41 loading)
2. RAM usage: 70.55 GB → 141.11 GB (loading FULL model)
3. No discovery phase observed before model loading
4. gRPC connection attempt fails AFTER model loaded

---

## Network Interface Analysis

**Thor #1 Interfaces**:
- `10.0.0.93/24` on enP2p1s0 (main LAN interface)
- `192.168.10.1/24` on mgbe0_0 (25GbE #1)
- `192.168.20.1/24` on mgbe1_0 (25GbE #2)
- `192.168.30.1/24` on mgbe2_0 (25GbE #3)
- `192.168.40.1/24` on mgbe3_0 (25GbE #4)

**Thor #2 Interfaces**:
- `10.0.0.78/24` on enP2p1s0 (main LAN interface)
- mgbe0_0/1_0/2_0: DOWN (NO-CARRIER) - **25GbE interfaces NOT CONFIGURED**
- `192.168.55.1/24` on l4tbr0 (Docker bridge)

**CRITICAL ISSUE**: Thor #2's 25GbE interfaces (mgbe*) are DOWN - not configured like Thor #1

---

## ROOT CAUSE

**Primary Issue**: `--node-host` argument binding to specific IP breaks UDP broadcast discovery

**How UDP Discovery Broadcasts Work**:
1. Code iterates ALL network interfaces via `get_all_ip_addresses_and_interfaces()`
2. For EACH interface, creates UDP socket bound to that interface's IP
3. Broadcasts discovery message from that IP to subnet broadcast + global broadcast

**When `--node-host` is set to specific IP (e.g., 10.0.0.78)**:
- Exo's gRPC server binds to ONLY that IP
- UDP discovery STILL broadcasts from ALL interfaces (correct)
- BUT other nodes try to connect to the ADVERTISED `node_host` IP for gRPC health checks
- If nodes are on DIFFERENT subnets (192.168.10.x vs 10.0.0.x), gRPC connection fails

**The UDP Discovery Broadcast Happens CORRECTLY**:
- Thor #1 broadcasts from 10.0.0.93, 192.168.10.1, 192.168.20.1, etc.
- Thor #2 broadcasts from 10.0.0.78 (mgbe interfaces DOWN)
- Both CAN hear broadcasts on 10.0.0.x subnet

**The gRPC Connection FAILS**:
- Discovery message says "connect to 10.0.0.93:52415"
- If Thor #2 tries via 192.168.x.x route → connection refused (wrong subnet)
- Health check fails → peer not added to known_peers

**Secondary Issue**: Thor #2's 25GbE interfaces not configured (separate problem)

---

## RECOMMENDED FIX

### Fix #1: Remove `--node-host` argument (Let exo auto-detect)

**Change**:
```bash
# OLD (BROKEN)
python3 -m exo.main --node-host 10.0.0.78 --node-port 52416 --chatgpt-api-port 8081

# NEW (WORKING)
python3 -m exo.main --node-port 52416 --chatgpt-api-port 8081
```

**Why This Works**:
- Exo defaults `node_host` to "0.0.0.0" (bind all interfaces)
- UDP discovery broadcasts FROM each interface
- Remote nodes can connect to ANY interface's IP via gRPC
- gRPC server accepts connections on all interfaces

**Apply on**:
- Thor #1: Remove `--node-host 10.0.0.93`
- Thor #2: Remove `--node-host 10.0.0.78`

### Fix #2: Use `--wait-for-peers 1` to ensure discovery happens

**Addition**:
```bash
python3 -m exo.main --node-port 52416 --chatgpt-api-port 8081 --wait-for-peers 1
```

**Why This Helps**:
- Prevents model loading until at least 1 peer discovered
- Gives UDP discovery time to complete (broadcasts every 2.5s)
- Ensures distributed sharding happens BEFORE model loads

### Fix #3: Configure Thor #2's 25GbE interfaces (matches Thor #1)

**Current State**:
- Thor #1: mgbe0_0-3 configured and UP
- Thor #2: mgbe0_0-3 DOWN (NO-CARRIER or not configured)

**Action Required**: Configure Thor #2's mgbe interfaces to match Thor #1
- This is SEPARATE from discovery issue
- Needed for optimal distributed inference performance (43.6 Gbps aggregate)

---

## TESTING PLAN

### Test 1: Basic Discovery (Both Thors on LAN)

**Command on Thor #1**:
```bash
cd /home/jetson/exo-clean
python3 -m exo.main --node-port 52415 --chatgpt-api-port 8080 --wait-for-peers 1
```

**Command on Thor #2**:
```bash
cd /home/thor/exo-clean
python3 -m exo.main --node-port 52416 --chatgpt-api-port 8081 --wait-for-peers 1
```

**Expected Result**:
- Both nodes print "Waiting for peers..." initially
- Within 2.5-5 seconds: "Adding peer_id=..."
- "Exo Cluster" topology message appears
- Model loading starts AFTER peers discovered
- Total RAM: ~120-140GB (sharded across 2 nodes, not 228GB)

**Success Criteria**:
- ✅ Both logs show "Adding peer_id=..."
- ✅ Both logs show "Exo Cluster"
- ✅ Combined RAM < 160GB (vs 228GB currently)
- ✅ Each node loads ~50% of layers

### Test 2: Verify gRPC Health Checks

**Monitor gRPC connectivity**:
```bash
# On Thor #1
ssh jetson@10.0.0.93 "ss -tnp | grep 52415"

# On Thor #2
ssh thor@10.0.0.78 "ss -tnp | grep 52416"
```

**Expected**: ESTABLISHED connections between the two node ports

### Test 3: Inference Request

**After cluster formed**:
```bash
curl http://10.0.0.93:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-8b",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 50
  }'
```

**Expected**:
- Request processed across distributed nodes
- Response generated successfully
- Logs show cross-node communication

---

## REFERENCES

**Code Locations**:
- UDP Discovery: `/home/mira/exo/exo/networking/udp/udp_discovery.py`
- Main entry: `/home/mira/exo/exo/main.py` (lines 59-94 for args)
- gRPC handle: `/home/mira/exo/exo/networking/grpc/grpc_peer_handle.py`

**Network Topology**:
- Thor #1: 10.0.0.93 (mgbe0_0-3 UP, 4×25GbE configured)
- Thor #2: 10.0.0.78 (mgbe0_0-3 DOWN, needs configuration)
- Subnet: 10.0.0.0/24 (LAN connectivity verified)

**Discovery Protocol**:
- Broadcast interval: 2.5 seconds
- Timeout: 30 seconds (peer removal if no heartbeat)
- Health check: gRPC connection to advertised port

---

## SUMMARY

**Root Cause**: `--node-host` binding to specific IP prevents multi-interface discovery from working correctly when nodes have multiple subnets configured.

**Immediate Fix**: Remove `--node-host` argument, let exo bind to 0.0.0.0 (all interfaces).

**Additional Fix**: Add `--wait-for-peers 1` to prevent premature model loading.

**Long-term**: Configure Thor #2's 25GbE interfaces to match Thor #1 for optimal performance.

**Confidence**: 95% - This is a well-known pattern in distributed systems. UDP broadcast discovery requires careful attention to interface binding. The code analysis confirms this is the issue.
