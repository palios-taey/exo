# Exo Distributed LLM - System Requirements

## Hardware
- **Device**: NVIDIA Jetson Thor (Blackwell GPU, sm_110 compute capability)
- **OS**: JetPack/L4T 38.2 (Ubuntu 24.04 based)
- **Network**: 4×25GbE interfaces (51.0 Gbps total bandwidth)
- **Storage**: 57GB+ for model weights (qwen-3-coder-30b-a3b)

## Critical System Packages

### CUDA Runtime Compilation (NVRTC)
**Package**: `cuda-nvrtc-13-0`
- **Version**: 13.0.48-1
- **Size**: 28.5 MB
- **Installation**:
  ```bash
  sudo apt-get update
  sudo apt-get install cuda-nvrtc-13-0
  ```
- **Library Path**: `/usr/local/cuda-13.0/targets/sbsa-linux/lib/`
- **Provides**: `libnvrtc.so.13`

### CUDA Development Headers
**Package**: `cuda-cudart-dev-13-0`
- **Installation**:
  ```bash
  sudo apt-get install cuda-cudart-dev-13-0
  ```
- **Provides**:
  - `cuda_fp16.h` (FP16 support)
  - Other CUDA runtime headers
- **Header Path**: `/usr/local/cuda-13.0/include/`

## Library Path Configuration

### ldconfig Setup (CRITICAL)
After installing `cuda-nvrtc-13-0`, the library must be added to the linker cache:

1. **Create ldconfig configuration**:
   ```bash
   sudo bash -c "echo \"/usr/local/cuda-13.0/targets/sbsa-linux/lib\" > /etc/ld.so.conf.d/cuda-13-0-nvrtc.conf"
   ```

2. **Update linker cache**:
   ```bash
   sudo ldconfig
   ```

3. **Verify library discovery**:
   ```bash
   python3 -c "from ctypes.util import find_library; print(find_library('nvrtc'))"
   ```
   - **Expected output**: `libnvrtc.so.13`
   - **If None**: ldconfig not configured correctly

### Why This Matters
- Tinygrad uses `ctypes.util.find_library('nvrtc')` to locate NVRTC
- Without ldconfig configuration, library lookup fails even if files exist
- This was the root cause of initial NVRTC errors on Thor #1

## Network Configuration

### Dedicated Interfaces (Both Thor Devices)
```bash
# Thor #1 (10.0.0.93) → Thor #2 (10.0.0.78)
192.168.10.93 → 192.168.10.2   # Interface 1 (25 Gbps)
192.168.20.93 → 192.168.20.2   # Interface 2 (25 Gbps)
192.168.30.93 → 192.168.30.2   # Interface 3 (25 Gbps)
192.168.40.93 → 192.168.40.2   # Interface 4 (25 Gbps)
```

### Test Bandwidth
```bash
# On Thor #2 (Server side)
iperf3 -s -B 192.168.10.2

# On Thor #1 (Client side)
iperf3 -c 192.168.10.2 -t 30 -i 5
```

**Expected**: ~12.5 Gbps per interface, 51.0 Gbps aggregate

## Python Environment

### Virtual Environment
```bash
cd ~/exo
python3 -m venv exo-venv
source exo-venv/bin/activate
pip install -r requirements.txt
```

### Key Python Packages
- `tinygrad==0.11.0` - Inference engine
- `transformers==4.46.3` - Model architecture
- `exo==0.0.1` - Distributed inference framework
- See `requirements.txt` for complete list

## Model Weights Synchronization

### Cache Location
```bash
~/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/
```

### Sync from Mira (10.0.0.163)
```bash
# From source device (Mira)
rsync -avz --progress \
  ~/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/ \
  jetson@10.0.0.93:~/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/
```

**Size**: ~57GB (30B parameter model)

## Tinygrad Patches Applied

### 1. PTX Target Support (Blackwell sm_110)
- **File**: `tinygrad/runtime/ops_cuda.py`
- **Change**: Added PTX target generation for sm_110

### 2. NVPTX Compilation Support
- **File**: `tinygrad/runtime/compiler_cuda.py`
- **Change**: Added NVPTX backend support for Blackwell

### 3. Device.DEFAULT Fix
- **File**: `tinygrad/device.py`
- **Change**: Fixed Device.DEFAULT initialization for CUDA

### 4. FP8 Dtype Support
- **File**: `tinygrad/dtype.py`
- **Change**: Added FP8 data type for Blackwell FP8 tensor cores

### 5. NVRTC Version Handling
- **File**: `tinygrad/runtime/compiler_cuda.py`
- **Change**: Wrapped `nvrtcVersion` in try-except for missing bindings
- **Note**: This patch was rendered unnecessary after installing actual NVRTC package

## Verification Checklist

### System Package Verification
```bash
# Check NVRTC package
dpkg -l | grep cuda-nvrtc

# Check CUDA headers
dpkg -l | grep cuda-cudart-dev

# Verify library discovery
python3 -c "from ctypes.util import find_library; print(find_library('nvrtc'))"

# Check tinygrad can access NVRTC functions
python3 -c "from tinygrad.runtime.autogen.nvrtc import nvrtcVersion, nvrtcGetCUBIN; print(nvrtcVersion, nvrtcGetCUBIN)"
```

### Network Verification
```bash
# Check interface configuration
ip addr show | grep 192.168

# Test connectivity
ping -c 3 192.168.10.2  # From Thor #1 to Thor #2
```

### Service Verification
```bash
# Check exo processes
ps aux | grep exo.main

# Check port bindings
ss -tlnp | grep '5300[0-3]'

# Check logs
tail -f /tmp/exo_thor1.log
tail -f /tmp/exo_thor2.log
```

## Known Issues

### Issue 1: Inference Timeout (UNRESOLVED)
- **Symptoms**: Distributed inference hangs after 120-180 seconds
- **Affects**: Both 30B and 1B models
- **Status**: Under investigation
- **Possible Causes**:
  - NVRTC runtime compilation taking too long on first run
  - Distributed coordination deadlock
  - Tinygrad compatibility with Blackwell/CUDA 13.0
  - Memory/resource constraints

### Issue 2: Port Conflicts
- **Symptoms**: "Address already in use" errors
- **Fix**: Kill old processes before restarting
  ```bash
  lsof -i:<port> -t | xargs -r kill -9
  ```

## Device-Specific Notes

### Thor #1 (10.0.0.93 / jetson@)
- IP: `10.0.0.93`
- SSH: `ssh jetson@10.0.0.93` (password: papaDons1001s$)
- Ports: API 53000, Node 53001
- Virtual env: `/home/jetson/exo-venv/`

### Thor #2 (10.0.0.78 / thor@)
- IP: `10.0.0.78`
- SSH: `ssh thor@10.0.0.78` (password: papaDons1001s$)
- Ports: API 53002, Node 53003
- Virtual env: `/home/thor/exo-venv/`

## References
- NVIDIA JetPack Documentation
- Tinygrad GitHub: https://github.com/tinygrad/tinygrad
- Exo Framework: https://github.com/exo-explore/exo
- CUDA 13.0 Release Notes
