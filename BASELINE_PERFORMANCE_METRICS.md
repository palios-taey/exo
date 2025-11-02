# Baseline Performance Metrics - Distributed Inference
**Date**: 2025-11-02
**Hardware**: 2× Jetson Thor (sm_101 Blackwell, 128GB unified memory each)
**Network**: 4×25GbE + 10GbE management
**Model**: llama-3.1-8b
**Framework**: exo + tinygrad

---

## Executive Summary

**Status**: CRITICAL ISSUES BLOCKING BASELINE - Inference non-functional
**Primary Blocker**: Tensor reshaping error in tinygrad model code
**Network Performance**: Mixed (10GbE working, 25GbE degraded)
**Recommendation**: Fix model tensor issues before BYOK optimization

---

## 1. Network Performance

### 10.0.0.x Management Network (Working)
- **Latency**: 0.545ms average (min: 0.489ms, max: 0.565ms, mdev: 0.022ms)
- **Packet Loss**: 0% (10 packets transmitted/received)
- **Bandwidth (iperf3, 4 parallel streams)**: 940 Mbits/sec sustained
  - Per-stream: ~230-240 Mbits/sec
  - Retransmissions: 35 over 10 seconds (0.35%)
  - **Assessment**: Stable, performing at expected gigabit speeds

### 192.168.x.x 25GbE Network (DEGRADED)
- **Test**: mgbe0_0 (192.168.10.1 → 192.168.10.2)
- **Bandwidth**: 1.68 Mbits/sec (CRITICAL: 99.99% degradation from expected)
- **Expected**: ~20-25 Gbps
- **Actual**: 0.00168 Gbps
- **Retransmissions**: 20 over 5 seconds (4 per second)
- **Assessment**: FAILED - Network configuration issue or hardware problem
- **Impact**: Cannot use high-speed interconnect for distributed inference

**Network Bottleneck Analysis**:
- 10GbE network = 940 Mbps ≈ 117.5 MB/sec
- For TP=2 with llama-3.1-8b (4096 hidden size, fp16):
  - Per-token activation transfer: ~4096 × 2 bytes = 8 KB
  - At 117.5 MB/sec: Can support 14,700 tokens/sec theoretically
  - **Conclusion**: Network NOT the bottleneck for inference (if functional)

---

## 2. Cold Start Performance

### Server Initialization
- **Thor #1 Startup**: Successfully started (PID 137792)
- **Thor #2 Startup**: Successfully started (PID 8472)
- **Wait Time**: 60 seconds configured
- **Total Cold Start Time**: N/A (inference failed)

### Peer Discovery
- **Status**: Unable to verify peer discovery
- **Logs**: No "discovered" or "peer connected" messages found
- **Issue**: Logs contain only tensor errors, no network discovery events

### First Token Generation
- **Status**: FAILED
- **Request Timeout**: 120 seconds
- **Actual Duration**: 121.54 seconds (timed out)
- **Response**: No tokens generated
- **Error**: `ValueError: size mismatched, can't reshape self.shape=(1, 1, 4096, 4096) -> new_shape=(1, 1, 32, 128)`

**Cold Start Bottleneck**: Cannot measure - model code blocking execution

---

## 3. Inference Performance

### Token Generation (BLOCKED)
- **Test**: "Hi" prompt with max_tokens=5
- **Status**: FAILED - No tokens generated
- **Time**: >120 seconds timeout
- **Tokens/Second**: 0 (error state)

### Root Cause Analysis
**Error Pattern**:
```
ValueError: size mismatched, can't reshape self.shape=(1, 1, 4096, 4096) -> new_shape=(1, 1, 32, 128)
Location: exo/inference/tinygrad/models/llama.py, line 76
Context: xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
```

**Analysis**:
- Shape (1, 1, 4096, 4096) has 16,777,216 elements
- Shape (1, 1, 32, 128) has 4,096 elements
- Mismatch: 4096× difference (impossible reshape)
- **Hypothesis**: Incorrect tensor shape from previous operation
- **Impact**: Inference completely blocked, cannot proceed

**What Should Happen**:
- For llama-3.1-8b: n_heads=32, head_dim=128
- Query projection: (batch, seq, 4096) → (batch, seq, 4096) [stays same]
- Reshape to heads: (batch, seq, 4096) → (batch, seq, 32, 128)
- **Expected input shape**: (1, 1, 4096) NOT (1, 1, 4096, 4096)

**Likely Cause**: Weight matrix shape error or incorrect matmul operation creating 4096×4096 instead of keeping dimension 4096.

---

## 4. Resource Utilization

### Thor #1 (jetson@10.0.0.93)
- **Total Memory**: 122 GiB
- **Used**: 30 GiB (24.6%)
- **Free**: 3.0 GiB
- **Buffer/Cache**: 90 GiB
- **Available**: 92 GiB
- **Swap**: 0 (disabled)
- **Load Average**: 0.03, 0.04, 0.01 (very low)
- **Uptime**: 3 days, 10:27 hours

### Thor #2 (thor@10.0.0.78)
- **Total Memory**: 122 GiB
- **Used**: 62 GiB (50.8%)
- **Free**: 1.6 GiB
- **Buffer/Cache**: 59 GiB
- **Available**: 60 GiB
- **Swap**: 63 GiB (56 MiB used, 0.09%)
- **Load Average**: 0.02, 0.04, 0.05 (very low)
- **Uptime**: 19:45 hours

### GPU Utilization
- **nvidia-smi Query**: Returned "0%" utilization (both devices)
- **Memory Stats**: N/A (unified memory architecture)
- **Note**: Jetson Thor uses integrated GPU with unified memory, no separate VRAM tracking
- **Assessment**: GPU idle due to inference failure

**Resource Bottleneck Analysis**:
- **CPU Load**: Near zero (not compute-bound)
- **Memory**: Abundant free space (not memory-bound)
- **GPU**: Idle (cannot test - inference blocked)
- **Conclusion**: Resources available, bottleneck is MODEL CODE ERROR

---

## 5. Comparison to Expected Performance

### Theoretical Performance (llama-3.1-8b on 2× Thor)
- **Model Size**: ~16 GB (fp16)
- **TP=2 Split**: ~8 GB per device (fits easily in 128 GB)
- **Expected Tokens/Sec**: 30-50 tokens/sec (distributed)
- **Expected Latency**: 20-35ms per token
- **Expected TFLOPS**: ~150-200 TFLOPS combined (Blackwell sm_101)

### Actual Performance
- **Tokens/Sec**: 0 (blocked by error)
- **Latency**: ∞ (timeout)
- **TFLOPS**: 0 (not executing)
- **Gap**: 100% failure

### Network vs Compute Trade-off Analysis
**If inference were working**:
- Network overhead (10GbE): ~0.07ms per token (8KB / 117.5 MB/s)
- Expected compute per token: ~20-35ms
- **Network would be 0.2-0.35% of total latency** (negligible)
- BBR or BYOK network optimizations would provide <1% improvement
- **Conclusion**: Even with degraded network, compute should dominate

---

## 6. Bottleneck Identification

### Primary Bottleneck: MODEL CODE ERROR
**Severity**: CRITICAL - Complete failure to generate tokens
**Location**: `exo/inference/tinygrad/models/llama.py` line 76
**Error**: Tensor shape mismatch in attention head reshape
**Impact**: 100% performance loss, cannot baseline anything

### Secondary Issue: 25GbE Network Degradation
**Severity**: HIGH - 99.99% bandwidth loss
**Impact**: Blocks high-speed distributed inference path
**Workaround**: 10GbE network functional for testing
**Priority**: Fix after model code resolved

### Resource Availability: EXCELLENT
**CPU**: Idle (0.03-0.05 load average)
**Memory**: 92 GB available (Thor #1), 60 GB available (Thor #2)
**GPU**: Idle (cannot engage due to code error)
**Assessment**: Hardware ready, software blocking

---

## 7. Root Cause Hypothesis

### Tensor Shape Error Chain
1. **Expected Flow**:
   - Input: (1, 1, 4096) - [batch, sequence, hidden]
   - QKV projection: (1, 1, 4096) @ (4096, 4096) → (1, 1, 4096)
   - Reshape to heads: (1, 1, 4096) → (1, 1, 32, 128)

2. **Actual Flow (WRONG)**:
   - Something produces: (1, 1, 4096, 4096) - INCORRECT extra dimension
   - Reshape attempt: (1, 1, 4096, 4096) → (1, 1, 32, 128) - FAILS (element count mismatch)

3. **Likely Causes**:
   - Weight matrix not properly transposed
   - Matmul broadcasting creating extra dimension
   - Distributed shard boundaries causing shape corruption
   - Tinygrad-specific reshape semantics difference

### Investigation Needed
- [ ] Check weight loading for layer 0 attention
- [ ] Verify QKV projection shapes match expected (4096, 4096) weights
- [ ] Test single-device inference (eliminate TP=2 as variable)
- [ ] Add shape debugging before line 76 reshape
- [ ] Compare to working tinygrad llama implementation

---

## 8. Recommendations

### Immediate Actions (Priority Order)

1. **Fix Model Code** (CRITICAL)
   - Debug tensor shapes in `llama.py` attention mechanism
   - Add shape assertions before reshape operations
   - Test single-device inference first (eliminate distribution complexity)
   - Compare with reference tinygrad llama implementation

2. **Network Diagnosis** (HIGH)
   - Investigate 25GbE mgbe interfaces
   - Check NetworkManager configuration for 192.168.x.x networks
   - Verify cable connections and switch configuration
   - Test with iperf3 on different port combinations

3. **Baseline Establishment** (BLOCKED until #1 complete)
   - Once inference works, measure:
     - Cold start (server init + model load + first token)
     - Warm inference (sustained tokens/sec)
     - Peer discovery time
     - Per-token latency distribution
     - GPU utilization during inference

4. **BYOK Decision** (DEFERRED)
   - Current state: Cannot measure userspace performance
   - BBR/kernel optimization ROI: <1% (network not bottleneck)
   - **Recommendation**: Defer BYOK until exo fully functional
   - **Estimated Timeline**: Fix model code → 1-2 weeks testing → BYOK

### Testing Protocol (When Unblocked)

```bash
# 1. Single device test (eliminate TP=2)
ssh jetson@10.0.0.93 'cd /home/jetson/exo && DEVICE=CUDA python3 exo/main.py --listen-port 52415'
curl -X POST http://10.0.0.93:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3.1-8b", "messages": [{"role": "user", "content": "Count 1 to 10"}], "max_tokens": 20}'

# 2. Distributed test (if #1 works)
# Start both Thors, test peer discovery, measure performance

# 3. Sustained load test
# 100 requests, measure avg/p50/p95/p99 latency
```

---

## 9. Metrics Summary Table

| Metric | Target | Measured | Status | Gap |
|--------|--------|----------|--------|-----|
| **Network Latency** | <1ms | 0.545ms | ✅ PASS | 0% |
| **10GbE Bandwidth** | ~940 Mbps | 940 Mbps | ✅ PASS | 0% |
| **25GbE Bandwidth** | ~20 Gbps | 0.00168 Gbps | ❌ FAIL | 99.99% |
| **Cold Start** | <60s | N/A | ❌ BLOCKED | N/A |
| **First Token** | <5s | >120s | ❌ FAIL | 2400%+ |
| **Tokens/Sec** | 30-50 | 0 | ❌ FAIL | 100% |
| **Latency/Token** | 20-35ms | ∞ | ❌ FAIL | ∞ |
| **GPU Utilization** | 60-80% | 0% | ❌ FAIL | 100% |
| **Memory Available** | >50GB | 92GB / 60GB | ✅ PASS | +84% / +20% |
| **CPU Load** | <50% | <5% | ✅ PASS | 0% |

**Overall Status**: 4/10 metrics passing (40%), 6/10 failing/blocked (60%)

---

## 10. Conclusion

**Current State**: Distributed inference is completely non-functional due to model code errors. Network performance is mixed (10GbE working, 25GbE degraded), but network is NOT the bottleneck—the tensor reshaping error is.

**Critical Path**:
1. Fix `llama.py` attention tensor shapes (CRITICAL)
2. Establish working single-device inference (HIGH)
3. Debug 25GbE network degradation (HIGH)
4. Measure baseline performance (BLOCKED)
5. Evaluate BYOK optimization ROI (DEFERRED)

**BYOK Decision**: Even if 25GbE worked perfectly (20 Gbps = 2.5 GB/sec), network overhead would be 0.003ms per token (8KB / 2.5 GB/s), compared to 20-35ms compute time. Network is 0.01% of latency budget. **BYOK kernel optimization targeting network will provide <0.1% performance improvement.** Focus on fixing model code first.

**Estimated Timeline**:
- Model code fix: 1-3 days (depends on root cause complexity)
- Network diagnosis: 1 day (parallel work)
- Baseline establishment: 1 day (after fixes)
- BYOK evaluation: 1-2 weeks (after stable baseline)

**Next Steps**: Create detailed issue report for model tensor error, add debug logging to track tensor shapes through attention mechanism, test single-device inference to isolate TP=2 as variable.

---

**Generated**: 2025-11-02 04:10 UTC
**Machine**: mira@10.0.0.163
**Duration**: ~12 minutes (including wait times and timeouts)
