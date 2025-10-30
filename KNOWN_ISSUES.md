# Known Issues - Exo Distributed LLM on Jetson Thor

## Critical Issues

### 1. Distributed Inference Timeout (UNRESOLVED)
**Severity**: Critical - Blocks production use
**Status**: Under Investigation
**First Observed**: 2025-10-30
**Devices Affected**: Both Thor #1 (10.0.0.93) and Thor #2 (10.0.0.78)

#### Description
All distributed inference attempts timeout after 120-180 seconds, regardless of model size. The timeout occurs during the inference execution phase, not during model loading or peer discovery.

#### Symptoms
- Peer discovery completes successfully (2 nodes connected)
- Model weights load without errors
- All patches load correctly (PTX, NVPTX, Device.DEFAULT, FP8)
- NVRTC libraries accessible
- API servers bind to ports correctly
- **Inference hangs/times out after 2-3 minutes**
- Process may exit prematurely or hang indefinitely

#### Reproduction Steps
1. Start exo servers on both Thor devices:
   ```bash
   # Thor #1
   bash /tmp/start_exo_thor1_53k.sh

   # Thor #2
   bash /tmp/start_exo_thor2_53k.sh
   ```

2. Wait for peer discovery to complete (~30 seconds)

3. Send inference request via API:
   ```bash
   curl -X POST http://10.0.0.93:53000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "qwen-3-coder-30b-a3b",
       "messages": [{"role": "user", "content": "Hello"}],
       "temperature": 0.7
     }'
   ```

4. Observe timeout after 120-180 seconds

#### Tested Configurations
- ✓ qwen-3-coder-30b-a3b (30B parameters) - **TIMEOUT**
- ✓ llama-3.2-1b (1B parameters) - **TIMEOUT**
- Both configurations show identical timeout behavior

#### What's Working
- ✓ Network infrastructure: 51.0 Gbps aggregate bandwidth (4×25GbE)
- ✓ CUDA 13.0 runtime compilation (NVRTC) configured correctly
- ✓ FP16 support headers installed (cuda_fp16.h)
- ✓ Tinygrad patches active (PTX, NVPTX, Device.DEFAULT, FP8)
- ✓ Peer discovery via UDP broadcast
- ✓ Model weights synchronized (57GB qwen-3-coder-30b-a3b)
- ✓ API endpoints responding to /v1/models requests

#### What's NOT Working
- ✗ Actual inference execution
- ✗ Distributed CUDA kernel execution
- ✗ Inter-device tensor communication

#### Possible Root Causes (Under Investigation)

1. **NVRTC First-Run Compilation Bottleneck**
   - Tinygrad compiles CUDA kernels at runtime via NVRTC
   - First compilation may take excessive time on Blackwell (sm_110)
   - Hypothesis: Compilation timeout triggers process exit
   - **Test**: Check if second inference attempt (using cached kernels) succeeds

2. **Distributed Coordination Deadlock**
   - Exo uses UDP for peer discovery, TCP for data transfer
   - Possible deadlock in distributed tensor sharding logic
   - Hypothesis: Devices waiting on each other indefinitely
   - **Test**: Single-device inference (no distribution) to isolate coordination vs execution

3. **Tinygrad Compatibility with Blackwell/CUDA 13.0**
   - Jetson Thor uses Blackwell GPU (sm_110, compute capability 10.1)
   - Tinygrad v0.11.0 may have incomplete Blackwell support
   - Hypothesis: Unsupported CUDA operation hanging
   - **Test**: Run tinygrad test suite on Thor devices
   - **Test**: Check if CPU inference works (DEVICE=CPU)

4. **Memory/Resource Constraints**
   - Thor devices have limited memory vs datacenter GPUs
   - Hypothesis: Memory allocation failure triggering hang
   - **Test**: Monitor GPU memory during inference (nvidia-smi)
   - **Test**: Try smaller batch size / sequence length

5. **Network Bandwidth Saturation**
   - 30B model requires substantial inter-device communication
   - Hypothesis: Network saturation causing timeouts
   - **Test**: Monitor network utilization during inference (iftop)
   - **Test**: Reduce model size to test correlation

#### Investigation Steps (Next)

1. **Enable DEBUG=2 logging**:
   ```bash
   DEVICE=CUDA DEBUG=2 python3 -m exo.main --inference-engine tinygrad ...
   ```
   Capture detailed logs during timeout

2. **Test single-device inference**:
   ```bash
   # On Thor #1 only, no peer
   python3 -m exo.main --inference-engine tinygrad --default-model llama-3.2-1b
   ```
   Isolate distributed coordination from inference execution

3. **Test CPU inference**:
   ```bash
   DEVICE=CPU python3 -m exo.main --inference-engine tinygrad --default-model llama-3.2-1b
   ```
   Isolate CUDA/GPU from inference logic

4. **Monitor resource utilization**:
   ```bash
   # Terminal 1: GPU monitoring
   watch -n 1 nvidia-smi

   # Terminal 2: Network monitoring
   iftop -i eth1

   # Terminal 3: Process monitoring
   top -p $(pgrep -f exo.main)
   ```

5. **Check NVRTC compilation cache**:
   ```bash
   ls -lh ~/.cache/tinygrad/
   # If cache exists, check if second inference faster
   ```

6. **Run tinygrad test suite**:
   ```bash
   cd ~/tinygrad
   python3 -m pytest test/ -k cuda -v
   ```
   Verify basic CUDA functionality

#### Workarounds
None available. System not functional for inference.

#### Impact
- **Production Readiness**: Blocked
- **Evaluation**: Cannot test model quality
- **Development**: Cannot iterate on patches/optimizations

#### Related Issues
- See `SYSTEM_REQUIREMENTS.md` for confirmed working components
- See git log for patch history

#### Last Updated
2025-10-30

---

## Resolved Issues

### 1. NVRTC Library Not Found (RESOLVED)
**Severity**: Critical
**Status**: RESOLVED
**Resolved Date**: 2025-10-30

#### Description
Tinygrad couldn't locate NVRTC libraries even though the `cuda-nvrtc-13-0` package was available via apt.

#### Root Cause
- Library installed at `/usr/local/cuda-13.0/targets/sbsa-linux/lib/`
- Path not included in ldconfig database
- `ctypes.util.find_library('nvrtc')` returned None

#### Solution
1. Install package:
   ```bash
   sudo apt-get install cuda-nvrtc-13-0
   ```

2. Configure ldconfig:
   ```bash
   sudo bash -c "echo \"/usr/local/cuda-13.0/targets/sbsa-linux/lib\" > /etc/ld.so.conf.d/cuda-13-0-nvrtc.conf"
   sudo ldconfig
   ```

3. Verify:
   ```bash
   python3 -c "from ctypes.util import find_library; print(find_library('nvrtc'))"
   # Output: libnvrtc.so.13
   ```

#### Prevention
Document ldconfig configuration in `SYSTEM_REQUIREMENTS.md` and deployment scripts.

---

### 2. Missing CUDA FP16 Headers (RESOLVED)
**Severity**: High
**Status**: RESOLVED
**Resolved Date**: 2025-10-30

#### Description
NVRTC compilation failed with "cannot open source file cuda_fp16.h"

#### Root Cause
`cuda-cudart-dev-13-0` package not installed

#### Solution
```bash
sudo apt-get install cuda-cudart-dev-13-0
```

#### Prevention
Add to `requirements.txt` system packages section.

---

### 3. Port Binding Conflicts (RESOLVED)
**Severity**: Low
**Status**: RESOLVED
**Resolved Date**: 2025-10-30

#### Description
"Address already in use" errors when restarting exo servers

#### Root Cause
Old processes holding ports from previous sessions

#### Solution
Kill old processes before restart:
```bash
lsof -i:<port> -t | xargs -r kill -9
```

#### Prevention
Startup scripts now include port cleanup before launch.

---

## Feature Requests / Enhancements

### 1. Automatic NVRTC Cache Warming
Precompile common CUDA kernels during initialization to avoid first-run compilation delays.

### 2. Improved Timeout Handling
Add configurable timeouts with progress indicators for long-running operations.

### 3. Health Check Endpoints
Add `/health` and `/ready` endpoints to API for monitoring deployment status.

### 4. Graceful Degradation
Fall back to single-device inference if peer discovery fails.

---

## Reporting New Issues

When reporting issues, please include:
1. Device information (Thor #1 / #2, IP address)
2. Exact command used to reproduce
3. Full log output (`/tmp/exo_thor*.log`)
4. Output of verification commands from `SYSTEM_REQUIREMENTS.md`
5. Network configuration (`ip addr`, `ss -tlnp`)
6. GPU status (`nvidia-smi`)
