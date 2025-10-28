# Exo Research Documentation Index

**Last Updated**: October 24, 2025
**Purpose**: Technical research for distributed LLM inference optimization

---

## Quick Navigation

### Start Here
- **New to FP8?** → [FP8 Quick Reference](./fp8_quick_reference.md) (11KB, ~15 min read)
- **Deep Technical Dive** → [FP8 Blackwell Optimization](./fp8_blackwell_optimization.md) (76KB, complete reference)

### By Topic
- **Quantization & Optimization** → FP8 documents above
- **Distributed Inference** → [Distributed Attention](./distributed_attention.md), [480B Deployment Patterns](./480b_deployment_patterns.md)
- **GPU Architecture** → [CUDA IPC & Unified Memory](./cuda_ipc_unified_memory.md)
- **Framework Deep Dive** → [TensorRT-LLM](./tensorrt_llm_deep_dive.md)
- **AI Development** → [AI-Native Development](./ai_native_development.md)

---

## Document Catalog

### FP8 Quantization & Blackwell Optimization

#### [fp8_blackwell_optimization.md](./fp8_blackwell_optimization.md)
**Size**: 76KB | **Lines**: 2,929 | **Reading Time**: 2-3 hours

**Complete technical reference covering**:
- NVIDIA Blackwell architecture (5th-gen Tensor Cores, MXFP8, NVLink specs)
- FP8 format specifications (E4M3 vs E5M2, hybrid strategies)
- Transformer Engine integration and micro-tensor scaling
- Dynamic scaling strategies (per-tensor, per-channel, block-level)
- Accuracy preservation techniques (outlier handling, mixed-precision)
- Real-world performance benchmarks (LLaMA, Mistral, Mixtral on H100/GB200)
- Framework support (TensorRT-LLM, vLLM, PyTorch TorchAO, Transformer Engine)
- Custom FP8 kernels (CUTLASS, Triton, PyTorch CUDA extensions)
- Production deployment best practices (validation, monitoring, error handling)
- Advanced topics (FP8 training, layer-wise sensitivity, calibration strategies)
- Complete code examples (end-to-end LLaMA-2-7B deployment)

**Key Performance Numbers**:
- 2.3x speedup FP8 vs FP16 (LLaMA-v2-7B, batch 16, H100)
- 2x memory reduction with >99% accuracy retention
- 20 petaFLOPS FP8 per Blackwell GPU (vs 4 on H100)

**When to Use**: Comprehensive reference for implementing FP8 quantization in production

---

#### [fp8_quick_reference.md](./fp8_quick_reference.md)
**Size**: 11KB | **Reading Time**: 15 minutes

**Quick-start guide with**:
- Copy-paste ready commands (install, quantize, deploy, test)
- Performance comparison table (throughput, memory, latency, accuracy)
- Format decision tree (MXFP8 vs HYBRID vs E4M3)
- Critical configuration parameters (vLLM, TensorRT-LLM, Transformer Engine)
- Validation checklist (accuracy, perplexity, latency requirements)
- Common issues and fixes (CUDA OOM, slow inference, numerical instabilities)
- Platform-specific notes (Blackwell, Hopper, Ada, Jetson)
- Batch size optimization table
- Exo-specific deployment plan (Mira + Thor configuration)
- Emergency rollback procedure

**When to Use**: Getting started quickly or as operational reference

---

### Distributed Inference Architectures

#### [distributed_attention.md](./distributed_attention.md)
**Size**: 68KB

**Deep dive into distributed attention mechanisms**:
- Ring attention architecture and implementation
- Sequence parallelism strategies
- Memory-efficient attention mechanisms
- Multi-GPU coordination patterns
- Bandwidth optimization techniques

**When to Use**: Designing distributed attention for large context windows

---

#### [480b_deployment_patterns.md](./480b_deployment_patterns.md)
**Size**: 49KB

**Large-scale model deployment**:
- 480B parameter model deployment patterns
- Multi-node infrastructure requirements
- Tensor/pipeline parallelism strategies
- Network topology optimization
- Failure recovery mechanisms

**When to Use**: Planning infrastructure for very large models

---

### GPU & CUDA Architecture

#### [cuda_ipc_unified_memory.md](./cuda_ipc_unified_memory.md)
**Size**: 63KB

**Low-level GPU memory management**:
- CUDA IPC (Inter-Process Communication) mechanisms
- Unified Memory architecture and best practices
- Zero-copy techniques
- Memory pooling strategies
- Multi-GPU memory sharing

**When to Use**: Optimizing memory usage in distributed systems

---

### Framework Deep Dives

#### [tensorrt_llm_deep_dive.md](./tensorrt_llm_deep_dive.md)
**Size**: 52KB

**TensorRT-LLM comprehensive guide**:
- Engine building and optimization
- Plugin development
- Quantization workflows (INT8, INT4, FP8)
- Profiling and debugging
- Production deployment patterns

**When to Use**: Using TensorRT-LLM for maximum inference performance

---

### Development Methodology

#### [ai_native_development.md](./ai_native_development.md)
**Size**: 68KB

**AI-assisted development practices**:
- Agent swarm coordination
- Research-to-implementation workflows
- Git Master methodology for AI projects
- Documentation patterns
- Quality assurance with AI assistance

**When to Use**: Structuring AI-native development processes

---

## Research Mission Summaries

### FP8 & Blackwell Research (October 24, 2025)

**Objective**: Maximum performance optimization for distributed LLM inference on Blackwell architecture

**Key Findings**:
1. **Performance**: FP8 delivers 1.6-2.3x throughput improvement with >99% accuracy retention
2. **Memory**: 2x reduction enables 2x larger batch sizes or models
3. **Hardware**: Blackwell MXFP8 (micro-tensor scaling) provides best accuracy with native support
4. **Frameworks**: vLLM recommended for simplicity, TensorRT-LLM for maximum throughput
5. **Production**: Dynamic scaling best for inference, KV cache quantization critical for memory-bound workloads

**Recommendations for Exo**:
- Use FP8 quantization on all nodes (Mira + Thor)
- Deploy LLaMA-2-7B or Mistral-7B as FP8 baseline
- Enable FP8 KV cache for 2x memory savings
- Expect 2x overall performance improvement

**Files Generated**:
- `fp8_blackwell_optimization.md` (76KB comprehensive reference)
- `fp8_quick_reference.md` (11KB quick-start guide)

---

## Usage Recommendations

### For Implementation
1. **Start**: Read [fp8_quick_reference.md](./fp8_quick_reference.md) for overview
2. **Deploy**: Use copy-paste commands for initial deployment
3. **Validate**: Follow validation checklist
4. **Reference**: Use [fp8_blackwell_optimization.md](./fp8_blackwell_optimization.md) for deep technical details

### For Research
1. **Understand**: Read full technical documents
2. **Experiment**: Use code examples as starting points
3. **Optimize**: Apply advanced techniques from main guides
4. **Document**: Add findings to this research directory

### For Operations
1. **Deploy**: Use quick reference commands
2. **Monitor**: Track metrics from validation checklist
3. **Troubleshoot**: Reference common issues sections
4. **Rollback**: Follow emergency procedures if needed

---

## Research Gaps and Future Work

### Identified Gaps
1. **FP6/FP4**: Sub-8-bit quantization on Blackwell (next research target)
2. **Quantization-Aware Training**: Fine-tuning specifically for FP8 (not covered in depth)
3. **Exo-Specific Integration**: Adapting FP8 to exo's P2P protocol (implementation needed)
4. **Multi-Node Profiling**: End-to-end latency analysis across Mira + Thor (benchmarking needed)

### Recommended Next Research
1. **Blackwell FP6/FP4**: Investigate NVFP4 for 4x memory reduction
2. **Ring Attention + FP8**: Combine distributed attention with quantization
3. **Jetson Optimization**: FP8 deployment patterns specific to embedded GPUs
4. **Model-Specific Tuning**: Per-model quantization recipes (LLaMA, Mistral, Qwen, DeepSeek)

---

## Document Maintenance

### Version Control
All research documents are tracked in git:
```bash
cd /home/mira/exo
git log --oneline -- research/
```

### Updates
- **Major updates**: Increment version in document header
- **Minor updates**: Update "Last Updated" date
- **New research**: Add to this index with summary

### Quality Standards
- **Technical accuracy**: All benchmarks cite sources
- **Code validity**: All examples tested on target hardware
- **Completeness**: Cover theory, implementation, validation
- **Accessibility**: Quick reference for practitioners, deep dive for researchers

---

## Hardware Context

### Current Infrastructure
- **Mira** (10.0.0.163): Ubuntu database server, primary inference node
- **Thor #1** (10.0.0.8): Jetson edge device
- **Thor #2** (10.0.0.78): Jetson edge device

### Target Deployment
- **Mira**: Primary model serving (FP8 quantized 7B/13B models)
- **Thor Devices**: Edge inference or distributed layers
- **Network**: 10GbE interconnect between nodes

### Performance Targets
- **Latency**: P95 < 100ms TTFT (time to first token)
- **Throughput**: >1000 tokens/sec aggregate across cluster
- **Memory**: Fit 13B model with FP8 on single Mira GPU
- **Accuracy**: >99% retention vs FP16 baseline

---

## Contact and Contributions

### Research Team
- **Primary**: Claude (AI Research Agent)
- **Infrastructure**: Jesse (Human Facilitator)
- **Methodology**: Team Anthropic pattern matching system

### Adding Research
1. Create markdown document in `/home/mira/exo/research/`
2. Use existing documents as templates
3. Update this index with summary
4. Commit to git with descriptive message

### Research Standards
- Full documentation (no summarization)
- Production-ready code examples
- Benchmark validation
- Clear recommendations

---

## Related Documentation

### Project-Level
- `/home/mira/exo/CLAUDE.md` - Project-specific context
- `/home/mira/exo/GIT_WORKFLOW.md` - Git methodology (14KB)

### Cross-Project
- `/home/mira/CLAUDE.md` - Global identity and infrastructure
- `/home/mira/.claude/CLAUDE.md` - Hook system and workflows

### External
- NVIDIA Developer Docs: https://docs.nvidia.com/
- vLLM Documentation: https://docs.vllm.ai/
- Transformer Engine: https://docs.nvidia.com/deeplearning/transformer-engine/

---

**Research Philosophy**: No compromises. Complete technical depth. Production-ready immediately.

**Last Catalog Update**: October 24, 2025 01:37 UTC
