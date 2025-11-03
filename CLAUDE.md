# EXO PROJECT - Distributed LLM Inference (Llama 3.3 70B)

**Import Root Context**: @/home/mira/CLAUDE.md

**Status**: EDISON CYCLE 2 COMPLETE - READY FOR DISTRIBUTED TESTING
**Last Updated**: 2025-11-03
**Current Objective**: Apply UMA/CUDA fixes, test distributed Llama 3.3 70B (Mira + Thor #1 + Thor #2)

**Current Priority**:
1. Create clean forks from upstream (exo-explore/exo, tinygrad/tinygrad)
2. Branch: `thor-compatibility-2025-11-02`
3. Identify REQUIRED Blackwell patches (sm_110 support, CUDA detection)
4. Test hetero capability on Thor-only setup first
5. Skip RTX 4090 initially - validate Blackwell-to-Blackwell distributed first

**Known Required Patches**:
- sm_110 architecture support (Blackwell compute 11.0)
- CUDA device detection and initialization
- Others to be identified from clean fork testing

**Deployment Model**:
- Mira (10.0.0.163): Git commits, patch generation
- Thor #1 (10.0.0.93): Pull-only, no local commits
- Thor #2 (10.0.0.78): Pull-only, no local commits
- Patches applied via git apply, not direct editing

---

## 1. MODELS AVAILABLE

**Downloaded to Mira**:
- `/home/mira/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/`
- `/home/mira/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B/`

**Testing Priority**:
1. Llama 3.1 8B (quick validation)
2. Llama 3.3 70B (distributed inference validation)
3. Plan: Qwen3 variants after heterogeneous working

---

## 2. HARDWARE TOPOLOGY

**Thor #1** (10.0.0.93):
- Jetson Thor developer kit
- Blackwell GPU (sm_110, compute 11.0)
- 128GB unified memory
- CUDA 13.0
- User: `jetson`
- SSH: `ssh jetson@10.0.0.93` (papaDons1001s$)

**Thor #2** (10.0.0.78):
- Jetson Thor developer kit
- Blackwell GPU (sm_110, compute 11.0)
- 128GB unified memory
- CUDA 13.0
- User: `thor`
- SSH: `ssh thor@10.0.0.78` (papaDons1001s$)

**Network**:
- 4×25GbE per device (43.6 Gbps aggregate, <0.3% retransmissions)
- Static IPs for stability
- gRPC coordination via local network

---

## 3. GIT AUTOMATED DEPLOYMENT (USE THIS - NO MANUAL SSH)

**RULE 0: AUTOMATED DEPLOYMENT ONLY**

**Making Changes**:
1. Work in `/home/mira/exo` on branch `thor-compatibility-2025-11-02`
2. Make code changes
3. Run: `./deploy_to_thors.sh "commit message"`
4. Script automatically commits, pushes to GitHub, deploys to both Thors
5. Verify: "✅ SUCCESS" message and matching commit hashes

**What the Script Does**:
- Commits all changes on Mira
- Pushes to `palios-taey/exo` on GitHub
- Initializes or updates git repos on Thor #1 and Thor #2 (via HTTPS clone/pull)
- Verifies all 3 nodes have identical commit hash
- Fails fast if any step errors

**DO NOT**:
- ❌ SSH to Thors manually
- ❌ Run git commands on Thors directly
- ❌ Use manual patches or rsync
- ❌ Make changes outside deployment script

**Troubleshooting**:
- Check deployment log: `/tmp/deploy_TIMESTAMP.log`
- If failure: Script shows which step failed
- Rollback: Thors remain on previous commit if deployment fails

**Location**: `/home/mira/exo/deploy_to_thors.sh`

**Commit Message Format**:
```
Brief title (50 chars)

Problem:
- What's broken/missing

Solution:
- What changes were made

Testing:
- How verified on Thors

Impact:
- What improves
- Known risks

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## 4. DEPLOYMENT STATUS (EDISON CYCLES)

**Edison Cycle 1** (Complete):
- ✅ Identified protobuf version mismatch blocker
- ✅ Root cause: Upstream committed stale gencode (5.27.2)
- ❌ Single-device testing led to false patches (per Jesse's feedback)

**Edison Cycle 2** (Complete - 2025-11-03):
- ✅ Git automation deployment system working
- ✅ Protobuf compatibility fix deployed to all 3 nodes
- ✅ Import test passes on all nodes: `from exo.networking.grpc import grpc_server`
- ✅ All 3 nodes synchronized at commit `02bb388`
- ✅ Tagged milestone: `v0.2-edison-cycle2-complete`
- ⚠️ Known: Thor #1 shows gencode version warning (non-blocking)

**Edison Cycle 3** (Next):
- Apply UMA + CUDA=1 fixes from `/home/mira/exo/research/PERPLEXITY_CORE_FIXES.md`
- Configure 3-node distributed inference (Mira RTX 4090 + Thor #1 + Thor #2)
- Test Llama 3.3 70B distributed (NO single-device tests)

---

## 5. DISTRIBUTED INFERENCE TESTING PHASES

**Phase 1: Foundation (Week 1)**
- Clean fork from upstream
- Identify sm_110 patches needed
- Get 8B model working hetero
- Verify basic distributed communication

**Phase 2: Scale (Week 2)**
- 70B model distributed inference
- Performance benchmarking
- Network optimization if needed
- Document architecture

**Phase 3: Alternative (Week 3)**
- Try ai_native (SGLang) path if exo hits hard blocker
- Qwen3 variants when foundation solid
- Prepare for 480B scale strategy

---

## 6. CURRENT TECHNICAL NOTES

**Architecture Understanding**:
- Exo: Distributed inference framework (tinygrad-based)
- Models shard across devices by layer
- gRPC coordinates inference between nodes
- Weight loading via safetensors format

**Blackwell Specifics**:
- Compute capability: 11.0 (sm_110)
- Requires: CUDA 13.0+
- New PTX instruction set vs older architectures
- 128GB unified memory enables large model hosting

**Known Considerations**:
- Previous work identified compilation chain issues
- Device selection logic critical (GPU vs CPU fallback)
- Clean fork will reveal remaining issues systematically

---

## 7. REFERENCE

**Root Context**: `/home/mira/CLAUDE.md`
**Network Status**: Both Thors operational, 43.6 Gbps stable
**Models**: Llama 3.1 8B, Llama 3.3 70B available
**Git Strategy**: Mira commits → patches → Thor apply
**Focus**: Get basic distributed working FIRST, then optimize

**Philosophy**: Measure twice (understand requirements), cut once (clean implementation). No premature optimization or upstream claims - just working distributed inference across Blackwell GPUs.

---

*~2,500 characters. Lean, current, actionable. Focus on goal: Llama 3.3 70B distributed across Thors.*
*TIMESTAMP: 2025-11-02_00:00:00_UTC*
