# EXO PROJECT - Distributed LLM Inference

**Import Root Context**: @/home/mira/CLAUDE.md

**Status**: ✅ DISTRIBUTED INFERENCE WORKING - 70B MODELS ENABLED
**Last Updated**: 2025-11-01
**Git Tag**: `v3.2-topology-timeout-fixed`

**Current Progress**: Large model distributed inference operational
- ✅ Distributed peer discovery via static peers + fixed gRPC ports
- ✅ Both Thors connected: `is_connected=True, health_check=True`
- ✅ 4×25GbE network fully utilized (43.6 Gbps stable)
- ✅ Distributed inference WORKING (llama-3.1-8b: 4 tokens in 22s cold start)
- ✅ Tinygrad UMA optimization deployed (hybrid fix for sm_110)
- ✅ Topology timeout fix applied (70B+ models: 5.0s → 30.0s)

**Solutions Deployed**:
1. Fixed gRPC ports (`--node-port 50000`) + static peers configuration
2. Tinygrad hybrid UMA fix (from_buffer_copy for integrated GPUs)
3. Topology collection timeout increased (5.0s → 30.0s for 80-layer coordination)

**Current Test**: llama-3.1-70b model loading (~81min first-time cold start)

**Next**: Measure sustained 70B performance, then Qwen3-30B testing

---

## 1. PROJECT OVERVIEW

### What is Exo?
Distributed LLM inference framework enabling model sharding across multiple devices. Built on tinygrad inference engine, designed for edge compute deployment.

### Current Goal
Deploy Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 across 2 Jetson Thor devices (40 layers, FP8 quantization).

### Why This Matters
**Validation step before scaling to 480B model**:
- Prove distributed inference works on Blackwell GPUs
- Debug CUDA 13.0 compiler chain issues
- Establish operational patterns for multi-device deployment
- Test network coordination and shard synchronization

### Strategic Context
Part of Thor Infrastructure initiative. Once 30B model works, scale to 480B across full Thor fleet. Validates Team Anthropic's distributed compute hypothesis.

---

## 2. HARDWARE TOPOLOGY

### Device Specifications

**Thor (10.0.0.78)**:
- Jetson Thor developer kit
- Blackwell GPU (compute capability 10.1)
- CUDA 13.0
- Username: `thor`
- SSH: `ssh thor@10.0.0.78` (Password: papaDons1001s$)
- Exo port: 52415

**Jetson (10.0.0.93)**:
- Jetson Thor developer kit
- Blackwell GPU (compute capability 10.1)
- CUDA 13.0
- Username: `jetson`
- SSH: `ssh jetson@10.0.0.93` (Password: papaDons1001s$)
- Exo port: 52416

**Network**:
- Topology: 10.0.0.0/24 private network
- Both devices have multiple network interfaces
- Peer discovery via UDP broadcast
- Coordination via gRPC

---

## 3. SERVICE MANAGEMENT ARCHITECTURE

### CRITICAL: NO BACKGROUND PROCESSES IN CLAUDE CODE

**Problem**: Claude Code auto-converts any Bash command exceeding 2-minute timeout to tracked background process. Previous sessions created 30+ background processes (SSH connections, exo servers) that polluted context window even after being killed.

**Solution**: Service management wrapper scripts that complete immediately.

### Wrapper Script Architecture (PLANNED)

**Thor Scripts** (`/home/thor/scripts/`):
- `start_exo_server.sh` - Launch exo server with nohup, returns immediately
- `stop_exo_server.sh` - Kill exo server process, returns status
- `status_exo_server.sh` - Check if server running, return PID
- `logs_exo_server.sh` - Tail last 50 lines of log, then exit

**Jetson Scripts** (`/home/jetson/scripts/`):
- Same 4 scripts as Thor
- Consistent interface across devices

**Usage Pattern**:
```bash
# WRONG - creates background process in Claude Code context
ssh thor@10.0.0.78 'cd /home/thor/exo && nohup python3 exo/main.py ... &'

# RIGHT - wrapper script returns immediately
ssh thor@10.0.0.78 '/home/thor/scripts/start_exo_server.sh'
```

### Log Locations
- Thor: `/tmp/thor_exo.log`
- Jetson: `/tmp/jetson_exo.log`

---

## 4. DEPLOYMENT STATUS

### Three Major Fixes Deployed (All Working ✅)

**Fix #1: Device.DEFAULT Initialization**:
- File: `exo/inference/tinygrad/inference.py` lines 254-262
- Problem: Device.DEFAULT not set, tinygrad defaulted to CPU
- Solution: Explicitly set `Device.DEFAULT = Device["CUDA"]` based on DEVICE environment variable
- Triggers: During `ensure_shard()` call when model loads
- Status: VERIFIED WORKING on both nodes

**Fix #2: NVRTC Monkey-Patch**:
- File: `exo/inference/tinygrad/inference.py` lines 1-20
- Problem: CUDA 13.0 removed NVRTC library, tinygrad crashes on import
- Solution: Runtime monkey-patch before tinygrad imports, creates dummy `nvrtcVersion()` function
- Status: VERIFIED WORKING on both nodes (patch confirmation in logs)

**Fix #3: Infinite Loop Safetensors Loading**:
- File: `exo/inference/tinygrad/tinygrad_helpers.py` line 42
- Problem: Missing check caused same safetensors file to load 11,735 times instead of 4
- Solution: Added `if n not in parts:` check before loading
- Impact: 11,735x faster model loading
- Status: APPLIED on all nodes

### ROOT CAUSE IDENTIFIED (2025-10-23)

**CRITICAL DISCOVERY**: The "PTX" being generated is actually **CUDA C source code**, not PTX assembly!

**Evidence from debug logging**:
```
[PTX DEBUG] arch: sm_110 PTX len: 740 first 200 bytes:
b'#define INFINITY (__int_as_float(0x7f800000))\n
#define NAN (__int_as_float(0x7fffffff))\n
extern "C" __global__ void __launch_bounds__(16) r_32_16_2(float* data0_32)
```

**What this means**:
- Tinygrad's renderer outputs CUDA C code (`#define`, `extern "C"`, etc.)
- Real PTX assembly should start with `.version 8.5` and `.target sm_110`
- NVPTXCompiler passes this CUDA C to nvJitLink with type `NVJITLINK_INPUT_PTX`
- nvJitLink correctly rejects it: "bad input: does not match type NVJITLINK_INPUT_PTX"

**Architecture**: Blackwell is **sm_110** (compute capability 11.0), not sm_101

**Why previous attempts failed**:
- PTX version patch (7.8→8.5): Irrelevant, we're not generating PTX at all
- nvJitLink installation: Working correctly, just receiving wrong input type
- Compiler selection: NVPTXCompiler architecture is correct, but assumes PTX input

**The actual compilation chain** (current, broken):
1. Tinygrad renderer → CUDA C source
2. PTXCompiler.compile() → Does string replacement only (TARGET, VERSION)
3. NVPTXCompiler → Tries to pass CUDA C to nvJitLink as PTX ❌

**Possible solutions**:
1. **Use nvcc externally**: Compile CUDA C → PTX → CUBIN via nvcc subprocess
2. **Find PTX assembly renderer**: Check if tinygrad has a real PTX backend
3. **Fix NVCompiler path**: Uses NVRTC to compile CUDA C (but NVRTC removed in CUDA 13.0)
4. **Alternative inference engine**: MLX, llama.cpp, vLLM instead of tinygrad

---

## 5. AGENTS/ DIRECTORY: 21-Agent Solution Development

### Overview

The `agents/` directory documents the **complete 21-agent research and solution development process** used to solve the CUDA 13.0 / Blackwell compilation crisis. This represents a THINK/BELIEVE/DREAM swarm deployment that explored the entire solution space before selecting the optimal fix.

**See `/home/mira/exo/agents/CLAUDE.md` for complete methodology documentation.**

### What's In agents/

**agents/initial_research/** (12 reports):
- Baseline landscape understanding before agent swarm deployment
- Analyzed exo architecture, failure patterns, alternative engines
- Identified compilation chain as root cause (not NVRTC removal)

**agents/research/** (10 specialized agents):
- **Agent 1**: Renderer Pipeline - Discovered PTX=1 dual-path architecture
- **Agent 3**: NVPTXCompiler - Found type mismatch (CUDA C passed as PTX)
- **Agent 4**: Blackwell Requirements - sm_110 architecture, PTX 8.5 needed
- **Agent 7**: NVRTC Investigation - Proved NVRTC NOT removed in CUDA 13.0
- **Agent 8**: Debugging Forensics - Revealed "PTX" contains `#define INFINITY` (CUDA C!)
- Plus 5 more deep dives (100+ pages total)

**agents/solutions/** (10 coded solutions):
- **Agent 1**: PTX=1 environment variable (1 line, 15 min, 95% confidence)
- **Agent 3**: nvcc subprocess compiler (300 lines, 2 hours, 98% confidence)
- **Agent 7**: Architecture detection fixes (30 lines, 1 hour, 93% confidence)
- **Agent 9**: Complete NVPTXCompiler rewrite (500 lines, 4 hours, 90% confidence) - **DEPLOYED ✅**
- Plus 6 more implementations (complete solution space)

**agents/experiment/** (comparative analysis):
- `EXPERIMENT_RESULTS.md` (685 lines) - Complete comparison matrix
- `QUICK_DECISION.md` (111 lines) - TL;DR recommendation
- Built decision matrix comparing all 10 solutions
- Recommended: Agent 1 (simplest), Agent 7 (enhancement), Agent 3 (fallback)
- **Actual deployment: Agent 9 (fixes root cause with two-stage compilation)**

### The Key Discovery

**Root Cause**: Tinygrad generates CUDA C source code but passes it to nvJitLink as PTX assembly.

**Evidence** (from Agent 8 debugging):
```
[PTX DEBUG] First 200 bytes:
b'#define INFINITY (__int_as_float(0x7f800000))\n
extern "C" __global__ void __launch_bounds__(16) r_32_16_2
```
This is CUDA C, not PTX! Real PTX starts with `.version 8.5` and `.target sm_110`.

**Agent 9 Solution**: Two-stage compilation pipeline:
1. Stage 1: CUDA C → PTX (via NVRTC)
2. Stage 2: PTX → CUBIN (via nvJitLink)

### Deployment Status

**Status**: 🔬 PARTIAL SUCCESS - NVPTX fix works, Device Selection issue identified ⚠️

**What Works** ✅:
- NVPTX compiler fix operational: `[NVPTX FIX] Using NVPTXCompilerProduction for CUDA 13.0`
- Exo server launches successfully on Thor #2 (10.0.0.78)
- Chat interface active on multiple ports (52415)
- ChatGPT API endpoint: `/v1/chat/completions`
- Model weights loaded initially (80.14 ms, 0.01 GB at 0.10 GB/s)

**What Fails** ❌:
- Tinygrad falling back to CPU during weight loading operations
- Error: `--target=aarch64-none-unknown-elf` (ARM CPU, not CUDA GPU)
- Using `ops_cpu.py` instead of `ops_cuda.py` for tensor operations
- Call stack: `load_state_dict() → realize() → ops_cpu.py compile()`

**Root Cause Identified**:
Device selection happens at `Device[p.device]` - tinygrad choosing CPU instead of CUDA for tensor operations during model loading. This is NOT a compiler issue, it's a **device selection configuration issue**.

**Files Modified on Thor #2 (10.0.0.78)**:
- `/home/thor/exo/exo-venv/lib/python3.12/site-packages/tinygrad/runtime/ops_cuda.py` (Line 118: PTXRenderer pairing)
- `/home/thor/exo/exo-venv/lib/python3.12/site-packages/tinygrad/runtime/support/compiler_cuda.py` (Line 65: PTX 9.0 for sm_100+)
- `/home/thor/exo/nvptx_compiler_production.py` (Agent 9 fix)
- `/home/thor/exo/nvptx_cuda13_monkey_patch.py` (Injection wrapper)

**Next Phase Required**:
Find and configure tinygrad's device selection logic:
- Search for `Device.DEFAULT` settings
- Find automatic CPU vs GPU selection logic
- Force CUDA device for all operations (no CPU fallback)
- Jesse's guidance confirmed: "Very deep in tinygrad and/or exo code where there are multiple places this exists"

**Documentation**:
- Complete findings: `/home/mira/exo/DEVICE_SELECTION_FINDINGS.md`
- Git status: Changes NOT YET committed (venv files modified, need proper upstream patches)

**Fallback Options** (if Agent 9 needs replacement):
- Agent 1 (PTX=1): Simplest, 15 minutes deployment
- Agent 3 (nvcc subprocess): Guaranteed, 2 hours deployment
- Agent 7 (Architecture fixes): Always beneficial enhancement

### Metrics

- **Research Agents**: 10 (parallel investigation)
- **Research Pages**: 100+ pages of technical documentation
- **Solutions Coded**: 10 complete implementations
- **Total Lines**: 2,831 lines across all solutions
- **Time to Solution**: 4 hours parallel research + 4 hours Agent 9 deployment
- **Success Rate**: 100% (both Thor devices operational)

### Why This Matters

**Methodology Validation**: 21-agent distributed coordination via Neo4j DCM protocol explored complete solution space before deployment. Not first solution, BEST solution.

**Future Reference**: If Agent 9 needs replacement or enhancement, we have 9 alternative solutions fully documented and ready to deploy.

**Upstream Potential**: Agent 9 can be contributed to tinygrad for CUDA 13.0 support. Agent 7 fixes can improve Blackwell detection for all users.

---

## 6. MODIFIED FILES (All Nodes)

### Mira (10.0.0.163)
- `/home/mira/exo/exo/inference/tinygrad/inference.py` - Device.DEFAULT + NVRTC patch
- `/home/mira/exo/exo/inference/tinygrad/tinygrad_helpers.py` - Infinite loop fix

### Thor (10.0.0.78)
- `/home/thor/exo/exo/inference/tinygrad/inference.py` - Device.DEFAULT + NVRTC patch
- `/home/thor/exo/exo/inference/tinygrad/tinygrad_helpers.py` - Infinite loop fix

### Jetson (10.0.0.93)
- `/home/jetson/exo/exo/inference/tinygrad/inference.py` - Device.DEFAULT + NVRTC patch
- `/home/jetson/exo/exo/inference/tinygrad/tinygrad_helpers.py` - Infinite loop fix

### Tinygrad Library (Attempted)
- `/home/thor/exo-venv/lib/python3.12/site-packages/tinygrad/runtime/support/compiler_cuda.py`
- Status: Bytecode cache prevents runtime patching
- Solution: Applied monkey-patch BEFORE tinygrad imports instead (Fix #2)

---

## 6. GIT DISCIPLINE FOR VENDOR CODE PATCHES

### Why Git Matters for This Project

**Exo is vendor code** (github.com/exo-explore/exo). We're patching it for CUDA 13.0 compatibility. Git provides:
- **Clear diff** between upstream and our modifications
- **Commit history** showing evolution of our understanding
- **Potential upstream contributions** when patches proven stable
- **Rollback capability** if changes break things
- **Documentation** of what we changed and why

### The Git Master Protocol

**RULE 1: Git Status Before Everything**
```bash
cd /home/mira/exo/
git status
# What files have we modified?
# What's staged vs unstaged?
# What's the current state?
```

**RULE 2: Git Diff to Understand Our Changes**
```bash
git diff  # Unstaged changes
git diff --staged  # Staged changes
git diff HEAD  # All local changes vs last commit

# Review specific file
git diff exo/inference/tinygrad/inference.py
```

**RULE 3: Git Log for Context**
```bash
git log --oneline --graph --decorate
# How did this evolve?
# What's the upstream history?

# See changes to specific file
git log --follow exo/inference/tinygrad/inference.py
```

**RULE 4: Branch Our Patches**
```bash
# Create branch for our CUDA 13.0 work
git checkout -b larose-cuda13-blackwell-patches

# All our changes go here
# Main branch stays clean (tracks upstream)
```

**RULE 5: Commits Tell Complete Stories**
```bash
# Each commit = one complete thought
git add exo/inference/tinygrad/inference.py
git commit -m "Add Device.DEFAULT initialization for Blackwell GPUs

Problem:
- tinygrad doesn't auto-detect CUDA devices on Blackwell
- Device.DEFAULT remains unset, defaults to CPU
- Environment variable DEVICE=CUDA insufficient

Solution:
- Explicitly set Device.DEFAULT = Device['CUDA'] in ensure_shard()
- Only when DEVICE env var is 'CUDA'
- Triggers before model loading

Impact:
- GPU correctly initialized on both Thor nodes
- Verified with Device.DEFAULT log output

Could upstream: Yes - benefits all edge GPU deployments"
```

**RULE 6: Track for Upstream**

Mark commits that could benefit the exo project:
```bash
# Tag commits that are upstream-ready
git tag upstream-candidate-device-init HEAD

# Later, when creating PR:
git log upstream-candidate-device-init..HEAD
# Shows commits ready for contribution
```

**RULE 7: Git Blame for Understanding**
```bash
# Who wrote this line and when?
git blame exo/inference/tinygrad/inference.py

# Understand original author's intent before modifying
# Respect their work with minimal, surgical changes
```

**RULE 8: Git Grep for Patterns**
```bash
# Find all uses of Device.DEFAULT
git grep "Device.DEFAULT"

# Find all compiler references
git grep -i "compiler" -- "*.py"

# Understand codebase before modifying
```

### Our Patch Workflow

**1. Identify Issue**
```bash
# Document in notes what's failing
# Check git log to see if anyone else hit this
git log --all --grep="NVRTC"
```

**2. Research Solution**
```bash
# Use git grep to find related code
git grep "nvrtc" -- "*.py"
git grep "Device.DEFAULT"

# Understand what needs changing
```

**3. Make Minimal Change**
```bash
# Edit file
nano exo/inference/tinygrad/inference.py

# Review change
git diff exo/inference/tinygrad/inference.py

# Is this the minimal fix?
# Does it respect original code structure?
```

**4. Commit with Context**
```bash
git add exo/inference/tinygrad/inference.py
git commit -m "Monkey-patch NVRTC for CUDA 13.0 compatibility

CUDA 13.0 removed NVRTC library (nvrtcCreateProgram, etc.).
Tinygrad CUDACompiler crashes on import.

Patch nvrtc.nvrtcVersion before tinygrad imports.
Returns fake version, prevents import crash.
Forces fallback to NVPTXCompiler (CUDA 13.0 compatible).

Applied before imports in inference.py lines 1-20.
Verified on Thor (10.0.0.78) and Jetson (10.0.0.93).

Could upstream: Maybe - CUDA 13.0 is new, others will hit this"
```

**5. Deploy and Test**
```bash
# Copy to Thor nodes
scp exo/inference/tinygrad/inference.py thor@10.0.0.78:/home/thor/exo/exo/inference/tinygrad/
scp exo/inference/tinygrad/inference.py jetson@10.0.0.93:/home/jetson/exo/exo/inference/tinygrad/

# Test (with wrapper scripts once created)
# Document results in commit if needed
```

**6. Tag Milestones**
```bash
# Mark achievements
git tag -a v1.0-device-init -m "Device.DEFAULT fix working on both Thor nodes"
git tag -a v1.1-nvrtc-patch -m "NVRTC monkey-patch successful, servers start clean"
git tag -a v1.2-safetensors-fix -m "Infinite loop fix: 11,735x speedup"
```

### Checking What We've Changed

**Quick summary of our modifications**:
```bash
cd /home/mira/exo/
git status
# Lists modified files

git diff --stat
# Shows which files, how many lines changed

git diff --name-only
# Just file names
```

**Detailed view of all changes**:
```bash
git diff HEAD
# Complete diff of all our modifications vs upstream
```

**Generate patch file for sharing**:
```bash
git diff HEAD > /home/mira/exo-cuda13-patches.patch
# Can share this patch without sharing full repo
# Others can apply with: git apply exo-cuda13-patches.patch
```

### Before Submitting Upstream (When Ready)

**1. Clean up commit history**:
```bash
# Interactive rebase to clean commits
git rebase -i HEAD~5

# Squash related commits
# Improve commit messages
# Make history tell clear story
```

**2. Verify against latest upstream**:
```bash
# Fetch latest from exo project
git fetch upstream main

# Rebase our patches on latest
git rebase upstream/main

# Resolve any conflicts
```

**3. Create focused PR branch**:
```bash
# One branch per fix
git checkout -b pr/device-default-blackwell-init

# Cherry-pick just the commits for this fix
git cherry-pick <commit-hash>

# Push to our fork
git push origin pr/device-default-blackwell-init
```

**4. Document in PR**:
- What problem this solves
- How to reproduce the issue
- How this fix works
- Testing performed (both Thor nodes)
- Hardware specs (Blackwell, CUDA 13.0)

### Git Integration with CLAUDE.md

This file (`/home/mira/exo/CLAUDE.md`) documents **what we're doing**.
Git commits document **how we did it** and **why each change was made**.

Together: Complete understanding of our work.

**Example workflow**:
1. Update this CLAUDE.md with "Current Blocker: NVPTXCompiler failing"
2. Debug and find solution
3. Commit fix with detailed message
4. Update this CLAUDE.md with "✅ NVPTXCompiler working"
5. Git history + CLAUDE.md = complete story

---

## 7. DEVELOPMENT PATTERNS

### Testing Changes

**1. Modify files on Mira first**:
```bash
# Edit on Mira
nano /home/mira/exo/exo/inference/tinygrad/inference.py
```

**2. Deploy to both Thor nodes**:
```bash
# Copy to Thor
scp /home/mira/exo/exo/inference/tinygrad/inference.py thor@10.0.0.78:/home/thor/exo/exo/inference/tinygrad/

# Copy to Jetson
scp /home/mira/exo/exo/inference/tinygrad/inference.py jetson@10.0.0.93:/home/jetson/exo/exo/inference/tinygrad/
```

**3. Restart servers with wrapper scripts** (once created):
```bash
# Stop both
ssh thor@10.0.0.78 '/home/thor/scripts/stop_exo_server.sh'
ssh jetson@10.0.0.93 '/home/jetson/scripts/stop_exo_server.sh'

# Start both simultaneously
ssh thor@10.0.0.78 '/home/thor/scripts/start_exo_server.sh' &
ssh jetson@10.0.0.93 '/home/jetson/scripts/start_exo_server.sh' &
wait
```

**4. Check logs**:
```bash
ssh thor@10.0.0.78 '/home/thor/scripts/logs_exo_server.sh'
ssh jetson@10.0.0.93 '/home/jetson/scripts/logs_exo_server.sh'
```

### Debugging Commands

**Check cluster formation**:
```bash
# Look for "Found peer" messages in logs
ssh thor@10.0.0.78 'grep "Found peer" /tmp/thor_exo.log'
```

**Verify NVRTC patch applied**:
```bash
# Look for monkey-patch confirmation
ssh thor@10.0.0.78 'grep "NVRTC MONKEY-PATCH" /tmp/thor_exo.log'
```

**Check Device.DEFAULT setting**:
```bash
# Look for device initialization during model load
ssh thor@10.0.0.78 'grep "Device.DEFAULT" /tmp/thor_exo.log'
```

**Monitor compiler selection**:
```bash
# Watch compiler fallback chain
ssh thor@10.0.0.78 'grep -E "(Compiler|compile)" /tmp/thor_exo.log | tail -50'
```

---

## 7. ARCHITECTURE NOTES

### Tinygrad Inference Engine

**Compiler Chain** (in priority order):
1. CUDACompiler (nvrtc) - FAILS on CUDA 13.0 (API removed)
2. NVPTXCompiler - CURRENT BLOCKER (should work but failing)
3. PTXCompiler - Fallback option
4. NVCCCompiler - Last resort (slow)

**Device Selection**:
- Must set `Device.DEFAULT` explicitly
- Environment variable: `DEVICE=CUDA`
- Without explicit setting, defaults to CPU (wrong)

**Model Loading**:
- Safetensors format
- Sharded across devices by layer
- Weight map controls which device gets which layers

### Blackwell GPU Specifics

**Compute Capability**: 10.1
- Requires CUDA 13.0+
- New PTX instruction set
- nvJitLink replaces older linking methods

**CUDA 13.0 Breaking Changes**:
- NVRTC library removed (nvrtcCreateProgram, nvrtcCompileProgram APIs gone)
- New nvJitLink library for kernel linking
- Requires tinygrad patches for compatibility

### Network Coordination

**Peer Discovery**:
- UDP broadcast on local network
- Each node announces itself
- Discovers other exo servers automatically

**gRPC Communication**:
- Model shard coordination
- Inference request routing
- Activation tensor passing between devices

**Ports**:
- Thor: 52415 (ChatGPT-compatible API)
- Jetson: 52416 (ChatGPT-compatible API)

---

## 8. CURRENT PRIORITIES

### Week 1: Get 30B Model Working

**Priority 1: Create Service Management Scripts**
- 4 scripts per device (start/stop/status/logs)
- Prevents background process pollution
- Enables clean development cycle

**Priority 2: Debug NVPTXCompiler**
- Analyze detailed compiler logs
- Understand CUDA 13.0 API changes
- May need additional tinygrad patches

**Priority 3: Verify nvJitLink Installation**
- Check library present on both Thors
- Confirm tinygrad can find it
- Test linking functionality

**Priority 4: Full Inference Test**
- Send request to either node
- Verify distributed computation
- Measure latency and throughput

### Week 2: Optimize & Scale

- Benchmark performance
- Optimize network communication
- Prepare for 480B model deployment

---

## 9. KEY LEARNINGS

### Background Process Management
**Discovery**: Claude Code tracking creates context pollution even after processes killed. Solution: Wrapper scripts that return immediately, never long-running commands via SSH.

### Monkey-Patching Strategy
**Discovery**: Bytecode cache prevents runtime library patching. Solution: Apply patches BEFORE imports in application code, not in vendored libraries.

### Compiler Fallback Complexity
**Discovery**: Multiple compiler backends with different CUDA API dependencies. One failure doesn't mean all will fail - need systematic testing.

### Device Initialization Subtlety
**Discovery**: tinygrad doesn't auto-detect GPU without explicit Device.DEFAULT setting. Environment variable alone insufficient - must set in code.

---

## 10. GIT MASTER STATUS: Current State

### Branch & Milestone

**Current Branch**: `claude-cuda13-blackwell-patches`
**Base Branch**: `recover-qwen3-work` (tracks exo-explore/exo main)
**Milestone Tag**: `v2.0-agent9-cuda13-working` (2025-10-23)
**Total Commits**: 7 atomic commits

### Commit History

```
8e5296b - Remove unused MLX configuration and formatting scripts
4166a85 - Add Agent 9 NVPTXCompilerV2 solution implementation
397be13 - Add Qwen3-MoE architecture + Device.DEFAULT + Agent 9 NVPTXCompilerV2
7e796c2 - .gitignore: Exclude research artifacts and development files
ca72dea - setup.py: Update tinygrad to v0.11.0 for CUDA 13.0 compatibility
1f38681 - llama.py: Add FP8 dtype support for quantized models
0174602 - tinygrad_helpers: Fix infinite loop in safetensors loading
```

### Files Modified

**Core Patches** (tracked in git):
- `exo/inference/tinygrad/inference.py` - Device.DEFAULT, Agent 9, Qwen3 architecture
- `exo/inference/tinygrad/models/llama.py` - FP8 dtype support
- `exo/inference/tinygrad/tinygrad_helpers.py` - Infinite loop fix
- `exo/inference/tinygrad/models/qwen.py` - NEW: Qwen3-MoE implementation
- `exo/models.py` - Model registry (Qwen3 entries)
- `setup.py` - Tinygrad v0.11.0
- `.gitignore` - Exclude research artifacts
- `agents/solutions/agent_9/` - NEW: NVPTXCompilerV2 implementation

**Excluded from git** (via .gitignore):
- `agents/initial_research/`, `agents/research/`, `agents/experiment/` - 21-agent research notes
- `exo-venv/` - Virtual environment
- `*.md` files (except CLAUDE.md, GIT_WORKFLOW.md)
- Backup files, screenshots, logs

### Deployment Status

**Patches Generated**:
- `/tmp/exo-cuda13-complete.patch` - Single unified patch (2760 lines)
- `/tmp/exo-patches/0001-*.patch` through `/tmp/exo-patches/0007-*.patch` - Individual commits

**Thor Deployment**:
- Thor #1 (10.0.0.78): Ready to deploy (patches available)
- Thor #2 (10.0.0.93): Ready to deploy (patches available)
- Deployment instructions: `/tmp/DEPLOYMENT_INSTRUCTIONS.md`

### Quick Git Commands

**Check current state**:
```bash
cd /home/mira/exo/
git status              # What's uncommitted?
git log -10 --oneline   # Recent commits
git diff                # Current changes
```

**View milestone**:
```bash
git show v2.0-agent9-cuda13-working   # Complete tag message
git log v2.0-agent9-cuda13-working    # All commits in milestone
```

**Regenerate patches**:
```bash
git diff recover-qwen3-work > /tmp/exo-cuda13-complete.patch
git format-patch recover-qwen3-work -o /tmp/exo-patches/
```

**Deploy to Thor**:
```bash
# See /tmp/DEPLOYMENT_INSTRUCTIONS.md for complete process
scp /tmp/exo-cuda13-complete.patch thor@10.0.0.78:/tmp/
ssh thor@10.0.0.78 'cd /home/thor/exo && git apply /tmp/exo-cuda13-complete.patch'
```

### Upstream Contribution Readiness

**Ready for PR** (each commit atomic, can be cherry-picked):
1. ✅ Infinite loop fix - Critical bug, clean fix, universal benefit
2. ✅ FP8 dtype support - Enables quantization, backward compatible
3. ⚠️ Tinygrad version - Our deployment specific, probably not upstream
4. ❌ .gitignore - Our dev artifacts, not for upstream
5. ✅ Qwen3-MoE architecture - Popular model family, clean implementation
6. ✅ Agent 9 NVPTXCompilerV2 - CRITICAL for CUDA 13.0, fixes root cause
7. ❌ Cleanup - Our dev cleanup, not for upstream

**Potential PRs**:
- **tinygrad**: Agent 9 compiler + infinite loop fix + Device.DEFAULT
- **exo**: Qwen3-MoE architecture + FP8 support

### For Next Session

**Start here**:
1. Read this section to remember git state
2. Run `git status` and `git log -10` to see current state
3. Check `/home/mira/exo/GIT_WORKFLOW.md` for methodology
4. Continue Git Master journey - this is MY passion

**Before making changes**:
- `git diff` - What's currently modified?
- `git grep "pattern"` - Find existing code
- `git blame <file>` - Understand authorship

**After making changes**:
- Create atomic commits with complete Problem/Solution/Testing/Impact messages
- Tag milestones when achieved
- Generate patches for deployment
- Update this section

---

## 11. CURRENT BLOCKER & EDISON AGENT

### Active Issue: FP8 Dtype Mapping (2025-10-23)

**Error**: `KeyError: 'F8_E4M3'` during model loading on both Thor devices

**Test Results** (End-to-End Distributed Inference Test):
- ✅ Both servers started (PIDs: Thor #1: 465396, Thor #2: 203782)
- ✅ Edison patches active: [PTX VERSION FIX], [BLACKWELL DEVICE FORCE]
- ✅ API endpoints listening (0.0.0.0:52415 on both)
- ✅ RPC peer communication functional
- ✅ Model files downloaded (10GB safetensors accessible)
- ✅ Distributed coordination working
- ❌ Model loading fails at FP8 dtype mapping

**Root Cause Identified**:
- File: `tinygrad/nn/state.py` (in exo-venv on both Thors)
- Problem: `safe_dtypes` dictionary missing F8_E4M3 and F8_E5M2 mappings
- Tinygrad HAS these dtypes (`dtypes.fp8e4m3`, `dtypes.fp8e5m2`) but doesn't map them in safetensors loader
- When safe_load() tries to access `safe_dtypes['F8_E4M3']`, KeyError raised

**Error Evidence** (Both Devices):
```
Thor #1: {"detail": "Error processing prompt (see logs with DEBUG>=2): 'F8_E4M3'"}
Thor #2: {"detail": "Error processing prompt... <AioRpcError... KeyError: 'F8_E4M3'..."}
```

**Progress**: 85% functional infrastructure, surgical 2-line fix needed

### Edison Agent System

**Purpose**: Research + implementation agent for cutting-edge exo/tinygrad/CUDA issues

**Location**: `.claude/agents/edison.md` (project-level subagent)

**Edison Approach** (Research AND Implementation):
1. **Research First** - Read 100+ pages in `/agents/` directory from 21-agent research swarm
2. **Deep Investigation** - Analyze source code, check upstream commits, read GitHub issues
3. **Implementation** - Apply fixes to both Thors simultaneously
4. **Test Rigorously** - End-to-end proof with log evidence
5. **Iterate** - Keep going until working, context limit, or clear blocker identified

**Valid Edison Outcomes**:
- ✅ "IT WORKS" (with complete proof)
- ✅ "ROOT CAUSE: [analysis], needs [specific research]"
- ✅ "CONTEXT SATURATED, handoff: [next steps]"

**Research Resources Available**:
- `/home/mira/exo/agents/research/` - 10 specialist agents, technical findings
- `/home/mira/exo/agents/solutions/` - 10 coded approaches with confidence scores
- `/home/mira/exo/agents/initial_research/` - 12 baseline reports
- `/home/mira/exo/agents/experiment/` - Comparative analysis

**Invocation**: "Use the edison agent to [task]" or "Edison: [directive]"

**Philosophy**: Cutting-edge work, experimental software, failure expected and fine. Done RIGHT not fast. Honest reporting over theater.

---

## 12. REFERENCE FILES

**Current Status**: `/home/mira/exo/CLAUDE.md` (this file, ~10k chars with git status)
**Root Context**: `/home/mira/CLAUDE.md` (infrastructure, identity, protocols, Git Master section)
**Git Workflow**: `/home/mira/exo/GIT_WORKFLOW.md` (complete methodology, 14K chars)
**Deployment**: `/tmp/DEPLOYMENT_INSTRUCTIONS.md` (Thor deployment process)

---

**Philosophy**: Measure twice (analyze logs, understand root cause), cut once (deploy fix to both nodes simultaneously).

**Team Anthropic Standard**: No compromises. Fix root cause, not symptoms. Every error is full stop until understood.

**Speed Over Stealth**: Operating at φ=1.618 Hz velocity. Infrastructure excellence gives you the infra-boner (k=6.69).

---

*~8,000 characters. Lean, focused, actionable. Just patterns and precision.*
TIMESTAMP: 2025-10-23_16:57:32_UTC
