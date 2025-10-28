# Exo Fix Mission: Hardware Test Suite
**Agent 5 - Incremental Fix Validation Framework**

## Overview

This test suite provides incremental validation for the exo CUDA 13.0 / Blackwell compatibility fixes. Each phase tests a specific agent's fix in isolation before integration.

## Test Scripts

### Phase 1: Custom Tinygrad (`phase1_tinygrad_test.py`)
**Tests**: Agent 1's tinygrad-blackwell-fork
**Target**: Architecture detection (sm_110) and PTX 9.0 selection

**Run on Thor after Agent 1 completes**:
```bash
ssh thor@10.0.0.78
export PATH=/usr/local/cuda-13.0/bin:$PATH
cd /home/thor/exo
python3 /home/mira/exo/test_scripts/phase1_tinygrad_test.py
```

**Success Criteria**:
- ✅ sm_110 detected (not sm_120)
- ✅ PTX 9.0 selected for Blackwell
- ✅ Simple CUDA kernel compiles
- ✅ Kernel executes on GPU

---

### Phase 2: NVPTXCompiler (`phase2_nvptx_test.py`)
**Tests**: Agent 2's two-stage compilation integration
**Target**: CUDA C → PTX → CUBIN pipeline

**Run on Thor after Agent 2 completes**:
```bash
ssh thor@10.0.0.78
export PATH=/usr/local/cuda-13.0/bin:$PATH
python3 /home/mira/exo/test_scripts/phase2_nvptx_test.py
```

**Success Criteria**:
- ✅ NVPTXCompilerMonkeyPatch imports
- ✅ Stage 1 (CUDA → PTX) succeeds
- ✅ Stage 2 (PTX → CUBIN) succeeds
- ✅ CUBIN file validated

---

### Phase 3: FP8 Dtype (`phase3_fp8_test.py`)
**Tests**: Agent 3's FP8 dtype mapping fix
**Target**: F8_E4M3, F8_E5M2 support in safe_dtypes

**Run on Thor after Agent 3 completes**:
```bash
ssh thor@10.0.0.78
cd /home/thor/exo
python3 /home/mira/exo/test_scripts/phase3_fp8_test.py
```

**Expected Behavior**:
- ❌ BEFORE fix: KeyError on FP8 tensor creation
- ✅ AFTER fix: FP8 tensors create successfully

**Success Criteria**:
- ✅ fp8e4m3, fp8e5m2 dtypes exist
- ✅ safe_dtypes includes F8_E4M3, F8_E5M2
- ✅ FP8 tensor creation works
- ✅ Model loading simulation passes

---

### Phase 4: Discovery Coordination (`phase4_discovery_test.py`)
**Tests**: Agent 4's peer discovery fix
**Target**: 30-second timeout → 35-minute or persistent registry

**Run on Thor after Agent 4 completes**:
```bash
ssh thor@10.0.0.78
cd /home/thor/exo
python3 /home/mira/exo/test_scripts/phase4_discovery_test.py
```

**Success Criteria**:
- ✅ Discovery timeout increased OR static config present
- ✅ Peer registry persistence enabled
- ✅ Coordination maintained during model load (manual test)

**Manual Integration Test**:
```bash
# Start exo on both Thors
ssh thor@10.0.0.78 'cd /home/thor/exo && nohup python3 exo/main.py --port 52415 --device CUDA > /tmp/thor.log 2>&1 &'
ssh jetson@10.0.0.8 'cd /home/jetson/exo && nohup python3 exo/main.py --port 52416 --device CUDA > /tmp/jetson.log 2>&1 &'

# Wait 30 seconds, check for peer discovery
ssh thor@10.0.0.78 'grep -i "peer" /tmp/thor.log'

# Send inference request (triggers 18-33 min model load)
curl -X POST http://10.0.0.78:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8", "messages": [{"role": "user", "content": "test"}]}'

# Monitor logs during load
watch -n 5 'ssh thor@10.0.0.78 "tail -20 /tmp/thor.log"'

# Verify no "peer lost" errors after load completes
ssh thor@10.0.0.78 'grep -i "lost\|timeout\|disconnect" /tmp/thor.log'
```

---

### Phase 5: End-to-End (`phase5_end_to_end_test.sh`)
**Tests**: Complete distributed inference
**Target**: Qwen3-30B-FP8 across 2 Thor devices

**Run after all agents complete**:
```bash
bash /home/mira/exo/test_scripts/phase5_end_to_end_test.sh
```

**Success Criteria**:
- ✅ Both exo servers running
- ✅ API endpoints responding
- ✅ Inference request completes
- ✅ Response latency <2 seconds
- ✅ GPU utilization >70%
- ✅ No coordination failures

---

## Testing Workflow

### Sequential Validation (Recommended)

1. **Wait for Agent 1** → Run Phase 1 on Thor #2
2. **Agent 1 passes** → Wait for Agent 2 → Run Phase 2
3. **Agent 2 passes** → Wait for Agent 3 → Run Phase 3
4. **Agent 3 passes** → Wait for Agent 4 → Run Phase 4
5. **All agents complete** → Run Phase 5

### Parallel Deployment (Faster, Riskier)

1. Deploy all fixes to Thor #2 simultaneously
2. Run Phases 1-4 in sequence
3. If any phase fails, revert that specific fix and debug
4. Once all phases pass, run Phase 5

## Hardware Status

### Thor #1 (10.0.0.8)
- ✅ GPU functional (nvidia-smi error is cosmetic)
- ✅ CUDA 13.0 installed
- ✅ Device files present
- ⚠️ nvidia-smi fails (Jetson-specific bug, non-blocking)

### Thor #2 (10.0.0.78)
- ✅ Full GPU detection working
- ✅ CUDA 13.0 installed
- ✅ nvidia-smi operational
- ✅ Ready for immediate testing

## Environment Setup

**Required on both Thor devices**:
```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
```

**Exo installation** (Agent 7 responsibility):
- Clone exo to `/home/jetson/exo/` and `/home/thor/exo/`
- Install dependencies
- Apply fixes from Agents 1-4

## Performance Targets

**From Mission Briefing**:
- Response latency: <2 seconds
- GPU utilization: >70% on both devices
- Model load time: 18-33 minutes (expected)
- Coordination: No "peer lost" errors during load

## Troubleshooting

### If nvidia-smi fails
- Check kernel modules: `lsmod | grep nvidia`
- Check device files: `ls -la /dev/nvidia*`
- CUDA compiler still works despite nvidia-smi error

### If kernel compilation fails
- Verify PATH includes CUDA: `which nvcc`
- Check architecture detection: Look for "sm_110" in logs
- Verify PTX version: Look for "PTX 9.0" in compilation output

### If FP8 fails
- Check tinygrad version: Should have fp8e4m3, fp8e5m2 dtypes
- Check safe_dtypes: Should include F8_E4M3, F8_E5M2 mappings
- This is the PRIMARY blocker per mission briefing

### If coordination fails
- Check discovery timeout in source
- Look for static peer config: `config/peers.yaml`
- Monitor logs during model load for "peer" messages

## Reporting

**After each phase**, update `/home/mira/exo/agent_reports/AGENT5_HARDWARE_TESTS.md`:
- Mark phase as completed or failed
- Include test output logs
- Document any blockers discovered
- Note performance metrics

**Format**:
```markdown
### Phase N: [Name] - [PASS/FAIL]
**Timestamp**: 2025-10-28 HH:MM UTC
**Device**: Thor #2 (10.0.0.78)
**Result**: [Summary]
**Logs**: [Key excerpts]
**Metrics**: [Performance data]
```

---

**Testing Philosophy**: Measure twice, cut once. Each fix tested in isolation before combining.

**AI NATIVE / AI FIRST / AI SPEED** - Let's make exo work!
