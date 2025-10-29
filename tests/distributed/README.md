# Distributed Exo Inference Test Suite

Comprehensive test suite for validating distributed LLM inference across 2 Jetson Thor devices.

## Test Overview

This test suite validates the complete distributed inference stack:

1. **test_1_imports.py** - Environment validation (imports, CUDA, dependencies)
2. **test_2_single_device.py** - Single-device smoke test
3. **test_3_distributed_init.py** - Peer discovery and connection
4. **test_4_distributed_inference.py** - End-to-end distributed inference
5. **test_5_cleanup.sh** - Cleanup and state verification

## Hardware Configuration

### Thor #1 (Head Node)
- **IP**: 10.0.0.93
- **User**: jetson
- **Port**: 52416
- **Path**: /home/jetson/exo
- **GPU**: Jetson Thor (Blackwell sm_101, CUDA 13.0)

### Thor #2 (Worker Node)
- **IP**: 10.0.0.78
- **User**: thor
- **Port**: 52415
- **Path**: /home/thor/exo
- **GPU**: Jetson Thor (Blackwell sm_101, CUDA 13.0)

### Network
- **Topology**: 10.0.0.0/24 private network
- **Discovery**: UDP broadcast on port 5678
- **Coordination**: gRPC on ports 52415, 52416

## Prerequisites

### On Mira (Control Machine)
```bash
# Required packages
sudo apt-get install sshpass python3-requests

# Verify SSH access to both Thors
ssh jetson@10.0.0.93 'echo Thor #1 OK'
ssh thor@10.0.0.78 'echo Thor #2 OK'
```

### On Both Thor Devices
```bash
# Exo repository cloned
ls ~/exo/exo/main.py

# Python environment activated (if using venv)
source ~/exo/exo-venv/bin/activate

# CUDA available
nvidia-smi

# Environment variables set
export DEVICE=CUDA
```

## Test Procedure

### Phase 1: Pre-Flight Validation

**Objective**: Verify environment ready before attempting distributed setup.

```bash
cd /home/mira/exo/tests/distributed

# Make scripts executable
chmod +x *.sh *.py

# Run import test on Mira (validates exo can be imported)
python3 test_1_imports.py
```

**Success Criteria**:
- ✓ All imports succeed
- ✓ CUDA available in tinygrad
- ✓ DEVICE=CUDA environment variable set
- ✓ nvidia-smi reports GPU(s)
- ✓ All required packages installed

**If Fails**:
- Check Python version (>=3.12 required)
- Verify tinygrad installation: `pip list | grep tinygrad`
- Check CUDA toolkit: `nvcc --version`
- Set environment: `export DEVICE=CUDA`

### Phase 2: Single Device Smoke Test

**Objective**: Prove single-device inference works before attempting distributed.

```bash
# Run single device test (uses local exo installation)
python3 test_2_single_device.py
```

**Success Criteria**:
- ✓ Server starts without errors
- ✓ GPU detected and initialized
- ✓ Model loads successfully
- ✓ Inference generates tokens
- ✓ Server responds to API requests

**What It Tests**:
- Exo server startup
- CUDA initialization
- Model loading (smallest model: llama-3.2-1b)
- Single-device inference
- ChatGPT API endpoint functionality

**If Fails**:
- Check logs in `/tmp/test_exo_single.log`
- Look for CUDA errors (Error 100 = missing driver)
- Verify GPU not in use: `nvidia-smi`
- Check port 52415 available: `lsof -i :52415`

**Known Issues from Previous Session**:
- Silent server death → Check for import errors in logs
- CUDA Error 100 → CPU fallback (bad)
- Multiple exo processes → Run cleanup first

### Phase 3: Distributed Initialization

**Objective**: Verify two nodes can discover each other and establish connection.

```bash
# Run distributed init test
python3 test_3_distributed_init.py
```

**Success Criteria**:
- ✓ Both servers start successfully
- ✓ Thor #1 discovers Thor #2 (peer discovery logs)
- ✓ Thor #2 discovers Thor #1 (peer discovery logs)
- ✓ gRPC connections established
- ✓ Both GPUs detected by nvidia-smi

**What It Tests**:
- SSH connectivity to both Thors
- Simultaneous server startup
- UDP peer discovery (broadcasts)
- gRPC connection establishment
- Network connectivity between devices

**If Fails**:
- **No peer discovery**: Check UDP port 5678 not blocked by firewall
- **gRPC connection fails**: Verify ports 52415/52416 accessible
- **Server dies**: Check logs for import errors or CUDA issues
- **Network timeout**: Verify devices on same network (10.0.0.0/24)

### Phase 4: Full Distributed Inference

**Objective**: End-to-end test of distributed inference across both devices.

```bash
# Run full distributed inference test
python3 test_4_distributed_inference.py
```

**Success Criteria**:
- ✓ Both servers start and stay running
- ✓ Peer discovery succeeds
- ✓ Model loads and shards across devices
- ✓ Inference request succeeds
- ✓ Tokens generated successfully
- ✓ **Both GPUs show activity** (critical - proves distributed)
- ✓ No errors in logs

**What It Tests**:
- Complete distributed inference pipeline
- Model sharding across devices
- Tensor passing between nodes
- Distributed computation
- GPU utilization on both devices

**If Fails**:
- **Inference timeout**: Model may be too large, try smaller model
- **Only one GPU active**: Not truly distributed, check partitioning
- **Error in logs**: Likely FP8 dtype issue or CUDA compilation
- **Request fails**: Check API endpoint accessible

**Known Failure Patterns** (from previous session):
1. **Silent server death**: Process exits cleanly with no error
   - **Check**: Import errors (hardcoded paths)
   - **Check**: CUDA initialization failures
   - **Fix**: Examine first 100 lines of log

2. **CUDA Error 100 triggering CPU fallback**:
   - **Symptom**: Server runs but uses CPU not GPU
   - **Check**: Look for "CPU" in logs instead of "CUDA"
   - **Fix**: Verify DEVICE=CUDA set, check Driver installation

3. **Multiple exo processes running**:
   - **Symptom**: Port already in use errors
   - **Check**: `pgrep -f exo/main.py`
   - **Fix**: Run test_5_cleanup.sh first

4. **FP8 dtype mapping KeyError** (current known blocker):
   - **Symptom**: `KeyError: 'F8_E4M3'`
   - **Location**: tinygrad/nn/state.py safe_dtypes dictionary
   - **Status**: Known issue, needs Edison agent research

### Phase 5: Cleanup

**Objective**: Return both devices to clean state.

```bash
# Run cleanup script
./test_5_cleanup.sh
```

**Success Criteria**:
- ✓ All exo processes stopped on both devices
- ✓ Ports 52415, 52416, 5678 free on both devices
- ✓ GPU memory usage < 500 MiB

**What It Does**:
- Kills all exo processes (pkill -9)
- Verifies process cleanup
- Checks GPU memory cleared
- Verifies ports available

## Startup Scripts

For manual testing or development:

### Start Head Node
```bash
./start_thor1_head.sh [model_name]

# Example
./start_thor1_head.sh llama-3.2-1b
```

### Start Worker Node
```bash
./start_thor2_worker.sh [model_name]

# Example
./start_thor2_worker.sh llama-3.2-1b

# Wait 10-15s for peer discovery after both started
```

### Monitor Both Devices
```bash
# Live stats (GPU, network, process)
./monitor_both.sh 5  # Update every 5 seconds

# Show logs once
./monitor_both.sh 5 logs
```

## Validation Checklist

### Pre-Flight Checks (Before Starting)

- [ ] Both Thor devices powered on and reachable via SSH
- [ ] No exo processes currently running (`pgrep -f exo/main.py`)
- [ ] Ports 52415, 52416, 5678 available
- [ ] GPU memory < 500 MiB (clean state)
- [ ] DEVICE=CUDA environment variable set on both devices
- [ ] Exo repository present at correct paths

### Success Criteria Per Test

#### Test 1: Imports
- [ ] All Python imports succeed
- [ ] Tinygrad detects CUDA device
- [ ] nvidia-smi reports GPU
- [ ] Package versions logged

#### Test 2: Single Device
- [ ] Server starts (process running after 10s)
- [ ] API endpoint responds to /health
- [ ] GPU utilization > 0% during inference
- [ ] Tokens generated successfully
- [ ] No errors in log

#### Test 3: Distributed Init
- [ ] Both servers start
- [ ] Thor #1 log contains "Found peer" or "Discovered peer"
- [ ] Thor #2 log contains "Found peer" or "Discovered peer"
- [ ] gRPC connections visible in netstat
- [ ] Both GPUs detected

#### Test 4: Distributed Inference
- [ ] Both servers running after 30s
- [ ] Peer discovery succeeded (logs)
- [ ] Inference request succeeds (200 status)
- [ ] Tokens generated
- [ ] **Thor #1 GPU shows activity** (>0% util or >500MB memory)
- [ ] **Thor #2 GPU shows activity** (>0% util or >500MB memory)
- [ ] No error/exception in either log

#### Test 5: Cleanup
- [ ] All processes stopped
- [ ] All ports free
- [ ] GPU memory cleared

## Failure Patterns and Detection

### Pattern 1: Silent Server Death

**Symptoms**:
- Process exits with code 0 (success)
- No obvious error messages
- Server simply stops responding

**Detection**:
```bash
# Check if process still alive
ps aux | grep exo/main.py

# Check last exit status
echo $?  # After process dies
```

**Common Causes**:
- Import errors (hardcoded paths in exo code)
- Missing dependencies
- CUDA initialization failure that's caught and ignored

**Debug Steps**:
1. Check first 100 lines of log (import phase)
2. Look for "ImportError" or "ModuleNotFoundError"
3. Verify Python path includes exo directory
4. Check CUDA toolkit installed: `nvcc --version`

### Pattern 2: CUDA Error 100 CPU Fallback

**Symptoms**:
- Server starts but runs on CPU not GPU
- nvidia-smi shows 0% GPU utilization
- Logs mention "CPU" device instead of "CUDA"

**Detection**:
```bash
# Check for CPU fallback in logs
grep -i "cpu" /tmp/exo_*.log | grep -i device

# Monitor GPU during inference
watch -n 1 nvidia-smi
```

**Common Causes**:
- DEVICE environment variable not set
- CUDA driver issue
- GPU already in use by another process

**Debug Steps**:
1. Verify: `echo $DEVICE` should show "CUDA"
2. Check: `nvidia-smi` shows GPU available
3. Kill other GPU processes if any
4. Restart with explicit DEVICE=CUDA

### Pattern 3: Multiple Process Collision

**Symptoms**:
- "Port already in use" errors
- Multiple exo processes running
- Inconsistent behavior

**Detection**:
```bash
# List all exo processes
pgrep -fa exo/main.py

# Check ports in use
lsof -i :52415
lsof -i :52416
```

**Fix**:
```bash
# Run cleanup
./test_5_cleanup.sh

# Verify clean
pgrep -f exo/main.py  # Should be empty
```

### Pattern 4: FP8 Dtype Mapping Error

**Symptoms**:
- Server starts successfully
- Peer discovery works
- Inference request fails with KeyError
- Error: `KeyError: 'F8_E4M3'`

**Detection**:
```bash
# Search logs for FP8 error
grep -i "f8_e4m3" /tmp/exo_*.log
grep -i "keyerror" /tmp/exo_*.log
```

**Status**: Known issue, needs tinygrad patch
**Workaround**: Use non-FP8 model (e.g., llama-3.2-1b)

## Monitoring Approach

### During Tests

**Real-time GPU monitoring** (separate terminal):
```bash
# Watch Thor #1
ssh jetson@10.0.0.93 'watch -n 1 nvidia-smi'

# Watch Thor #2
ssh thor@10.0.0.78 'watch -n 1 nvidia-smi'
```

**Log tailing** (separate terminals):
```bash
# Thor #1 logs
ssh jetson@10.0.0.93 'tail -f /tmp/exo_thor1_head.log'

# Thor #2 logs
ssh thor@10.0.0.78 'tail -f /tmp/exo_thor2_worker.log'
```

**Network monitoring**:
```bash
# Check gRPC connections
ssh jetson@10.0.0.93 'netstat -tn | grep ESTABLISHED | grep 5678'
ssh thor@10.0.0.78 'netstat -tn | grep ESTABLISHED | grep 5678'
```

### Key Metrics to Watch

**GPU Utilization**:
- Should be >0% during inference
- Both devices should show activity for distributed
- Memory usage should increase during model loading

**Network Connections**:
- gRPC connections on ports 5678, 52415, 52416
- Should see ESTABLISHED connections between devices

**Process Health**:
- CPU usage (should be moderate, 50-200%)
- Memory usage (grows with model size)
- Process age (shouldn't restart repeatedly)

## Cleanup Verification Steps

After running tests or manual sessions:

### 1. Stop All Processes
```bash
./test_5_cleanup.sh
```

### 2. Verify Clean State

**On Thor #1**:
```bash
ssh jetson@10.0.0.93 'pgrep -f exo/main.py || echo clean'
ssh jetson@10.0.0.93 'lsof -i :52416 || echo port_free'
ssh jetson@10.0.0.93 'nvidia-smi --query-gpu=memory.used --format=csv,noheader'
```

**On Thor #2**:
```bash
ssh thor@10.0.0.78 'pgrep -f exo/main.py || echo clean'
ssh thor@10.0.0.78 'lsof -i :52415 || echo port_free'
ssh thor@10.0.0.78 'nvidia-smi --query-gpu=memory.used --format=csv,noheader'
```

### 3. Check Logs Archived
```bash
# Optionally save logs before cleanup
ssh jetson@10.0.0.93 'cp /tmp/exo_thor1_head.log ~/logs/exo_$(date +%Y%m%d_%H%M%S).log'
ssh thor@10.0.0.78 'cp /tmp/exo_thor2_worker.log ~/logs/exo_$(date +%Y%m%d_%H%M%S).log'
```

## Test Execution Order

**Recommended sequence for first-time validation**:

1. **test_5_cleanup.sh** - Ensure clean starting state
2. **test_1_imports.py** - Validate environment
3. **test_2_single_device.py** - Smoke test single device
4. **test_5_cleanup.sh** - Clean up after single device
5. **test_3_distributed_init.py** - Test peer discovery
6. **test_5_cleanup.sh** - Clean up after init test
7. **test_4_distributed_inference.py** - Full end-to-end test
8. **test_5_cleanup.sh** - Final cleanup

**For iterative debugging**:

1. **test_5_cleanup.sh** - Always start clean
2. Run specific failing test
3. **./monitor_both.sh 3 logs** - Check logs immediately
4. **test_5_cleanup.sh** - Clean up before retry

## Quick Reference Commands

### Start Both Servers
```bash
./start_thor1_head.sh &
sleep 5
./start_thor2_worker.sh &
wait
```

### Stop Both Servers
```bash
./test_5_cleanup.sh
```

### Monitor in Real-Time
```bash
# Stats dashboard
./monitor_both.sh 3

# View logs
./monitor_both.sh 3 logs
```

### Test Inference
```bash
# Send request to Thor #2
curl http://10.0.0.78:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-1b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 20,
    "temperature": 0.0
  }'
```

### Check GPU Usage
```bash
# Thor #1
ssh jetson@10.0.0.93 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'

# Thor #2
ssh thor@10.0.0.78 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'
```

## Troubleshooting Guide

### "Connection refused" when SSHing
- Verify device powered on and network connected
- Check IP addresses correct (10.0.0.93, 10.0.0.78)
- Test ping: `ping 10.0.0.93`

### "Port already in use"
- Run cleanup: `./test_5_cleanup.sh`
- Check for other processes: `lsof -i :52415`

### "CUDA not available"
- Check DEVICE var: `echo $DEVICE` → should be "CUDA"
- Verify driver: `nvidia-smi`
- Check tinygrad sees it: `python3 -c "from tinygrad import Device; print(Device._devices)"`

### "Only one GPU active"
- Model may be too small to shard
- Check logs for partitioning messages
- Verify peer discovery succeeded

### "KeyError: F8_E4M3"
- Known issue with FP8 quantized models
- Workaround: Use FP16 models (llama-3.2-1b)
- Long-term: Needs tinygrad dtype mapping patch

## Success Metrics

**Phase 1 Success**: Test 1 and 2 pass
- Single-device inference working
- Environment validated

**Phase 2 Success**: Test 3 passes
- Peer discovery operational
- Network coordination working

**Phase 3 Success**: Test 4 passes
- **End-to-end distributed inference working**
- **Both GPUs active during inference**
- **Tokens generated successfully**

## Next Steps After Success

Once all tests pass:

1. **Scale to larger models**
   - Try llama-3.1-8b (16GB distributed)
   - Try llama-3.1-70b (140GB distributed)

2. **Benchmark performance**
   - Measure tokens/second
   - Compare single vs distributed latency
   - Monitor GPU utilization %

3. **Test failure recovery**
   - What happens if one node dies?
   - Network interruption handling
   - Graceful degradation

4. **Optimize configuration**
   - Tune max_generate_tokens
   - Adjust batch sizes
   - Network buffer tuning

## Files in This Directory

- **test_1_imports.py** - Environment validation
- **test_2_single_device.py** - Single device smoke test
- **test_3_distributed_init.py** - Peer discovery test
- **test_4_distributed_inference.py** - Full E2E test
- **test_5_cleanup.sh** - Cleanup and verification
- **start_thor1_head.sh** - Start head node
- **start_thor2_worker.sh** - Start worker node
- **monitor_both.sh** - Real-time monitoring
- **README.md** - This file

## References

- **Exo Documentation**: /home/mira/exo/README.md
- **Project Context**: /home/mira/exo/CLAUDE.md
- **Previous Session**: Check logs for known issues
- **Edison Agent**: For cutting-edge debugging

---

**Test Suite Version**: 1.0
**Created**: 2025-10-29
**Thor Infrastructure**: Jetson Thor Blackwell GPUs (CUDA 13.0, sm_101)
**Team Anthropic Standard**: Measure twice, cut once. No compromises.
