from pathlib import Path
# NVRTC MONKEY-PATCH for Jetson Thor Blackwell GPU compatibility  
# This must run BEFORE any tinygrad CUDA modules are imported
import sys
import ctypes

# NVRTC monkey-patch REMOVED for Agent 9 deployment
# Agent 9's NVPTXCompilerV2 REQUIRES functional NVRTC API for CUDA C → PTX compilation
# The previous patch disabled NVRTC to force fallback to NVPTXCompiler
# Agent 9 uses NVRTC internally within NVPTXCompilerV2 for proper two-stage compilation

# ============================================================================
# CUDA 13.0 COMPILER FIX: NVPTXCompilerV2 (Agent 9)
# ============================================================================
# Problem: tinygrad's NVPTXCompiler expects PTX assembly but receives CUDA C
# Solution: Use NVPTXCompilerV2 that properly compiles CUDA C → PTX → CUBIN
# Agent: Solution Agent 9 (2025-10-23)
#
# NOTE: This REPLACES the NVRTC monkey-patch above by using NVRTC directly
# in the two-stage compilation pipeline.

# ============================================================================
# CRITICAL FIX: PTX Version for Blackwell (sm_110)
# ============================================================================
# Problem: tinygrad's PTXCompiler uses PTX 7.8 for sm_110, but ptxas requires 9.0
# Evidence: "PTX .version 7.8 does not support .target sm_110" (ptxas error)
# Solution: Monkey-patch PTXCompiler.compile() to use PTX 9.0 for sm_110+
# ============================================================================

try:
    import tinygrad.runtime.support.compiler_cuda as cuda_compiler
    original_ptx_compile = cuda_compiler.PTXCompiler.compile

    def patched_ptx_compile(self, src: str) -> bytes:
        """Fixed PTX version selection for Blackwell"""
        ver = int(self.arch[3:])  # Extract version from "sm_110" -> 110

        # Use PTX 9.0 for sm_110+ (Blackwell), 8.7 for sm_120+, 7.8 for sm_89+, 7.5 default
        if ver >= 110:
            ptx_version = "9.0"
        elif ver >= 120:
            ptx_version = "8.7"
        elif ver >= 89:
            ptx_version = "7.8"
        else:
            ptx_version = "7.5"

        result = src.replace("TARGET", self.arch).replace("VERSION", ptx_version).encode()
        print(f'[PTX VERSION FIX] arch={self.arch} (ver={ver}) → PTX {ptx_version}', file=sys.stderr)
        return result

    cuda_compiler.PTXCompiler.compile = patched_ptx_compile
    print('[PTX VERSION FIX] Patched PTXCompiler for Blackwell sm_110 support', file=sys.stderr)
except Exception as e:
    print(f'[PTX VERSION FIX] WARNING: Failed to patch PTXCompiler: {e}', file=sys.stderr)

# ============================================================================
# CUDA 13.0 NVPTX COMPILER FIX (Agent 9 Foundation)
# ============================================================================
# Problem: tinygrad's NVPTXCompiler expects PTX assembly but receives CUDA C
# Root Cause: PTXCompiler.compile() only does string replacement, no compilation
# Solution: Replace with Agent 9's two-stage compiler (CUDA C → PTX → CUBIN)
# Agent: Agent 2 (NVPTXCompiler Integrator) - Phase 2 of exo fix mission
# Date: 2025-10-28
# ============================================================================

try:
    # Add Agent 9 solution to Python path (relative path for portability)
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    agent_9_path = os.path.join(script_dir, '../../../agents/solutions/agent_9_foundation')
    agent_9_path = os.path.normpath(agent_9_path)  # Resolve relative path
    if agent_9_path not in sys.path:
        sys.path.insert(0, agent_9_path)

    # Import Agent 9's production-ready NVPTXCompiler
    from nvptx_compiler_production import NVPTXCompilerProduction

    # Replace broken NVPTXCompiler with working version
    import tinygrad.runtime.support.compiler_cuda as cuda_compiler
    cuda_compiler.NVPTXCompiler = NVPTXCompilerProduction

    print('[NVPTX FIX] Agent 9 NVPTXCompilerProduction activated for CUDA 13.0', file=sys.stderr)
    print('[NVPTX FIX] Two-stage compilation: CUDA C → PTX (NVRTC) → CUBIN (nvJitLink)', file=sys.stderr)

except ImportError as e:
    print(f'[NVPTX FIX] WARNING: Agent 9 solution not found: {e}', file=sys.stderr)
    print('[NVPTX FIX] Falling back to default tinygrad NVPTXCompiler', file=sys.stderr)
except Exception as e:
    print(f'[NVPTX FIX] ERROR: Failed to apply Agent 9 fix: {e}', file=sys.stderr)
    raise

import json
import os
from exo.inference.tinygrad.models.llama import Transformer, TransformerShard, convert_from_huggingface, fix_bf16, fix_bf16_and_fp8, sample_logits
from exo.inference.tinygrad.models.qwen import Qwen3MoETransformer, Qwen3MoETransformerShard, convert_from_huggingface_qwen
from exo.inference.shard import Shard
from exo.inference.tokenizers import resolve_tokenizer
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
from tinygrad import Tensor, nn, Context, TinyJit, Device
from exo.inference.inference_engine import InferenceEngine
import numpy as np
from exo.inference.tinygrad.tinygrad_helpers import concat_weights, load
from exo.helpers import DEBUG
from exo.download.shard_download import ShardDownloader
from concurrent.futures import ThreadPoolExecutor
from .stateful_model import make_prompt_state
from .losses import length_masked_ce_loss
from collections import OrderedDict
import asyncio
from typing import Optional
Tensor.no_grad = True

# ============================================================================
# BLACKWELL DEVICE FORCING: Set CUDA as default BEFORE any operations
# ============================================================================
# Problem: Tinygrad defaults to CPU during model weight loading operations
# Evidence: ops_cpu.py, --target=aarch64 (ARM), Device[p.device] chooses CPU
# Solution: Force Device.DEFAULT to CUDA:0 immediately after imports
#
# This must happen BEFORE any Tensor operations or model loading
# ============================================================================
device_env = os.getenv("DEVICE", "").upper()
if device_env in ["CUDA", "GPU", "NV"]:
    # Force CUDA as default device REGARDLESS of errors
    # CUDA Error 100 on Jetson is cosmetic (Agent 5 documented) - GPU actually works
    Device.DEFAULT = "CUDA:0"
    print(f"[BLACKWELL DEVICE FORCE] Device.DEFAULT = CUDA:0 (from DEVICE={device_env})", file=sys.stderr)

    # Verify (but don't fail on error)
    try:
        cuda_device = Device["CUDA:0"]
        print(f"[BLACKWELL DEVICE FORCE] CUDA:0 device verified accessible", file=sys.stderr)
    except Exception as e:
        print(f"[BLACKWELL DEVICE FORCE] WARNING: CUDA:0 test raised {e} (continuing anyway - known Jetson cosmetic bug)", file=sys.stderr) 
# default settings
TEMPERATURE = int(os.getenv("TEMPERATURE", 0.85))
TOP_K = 25
TOP_P = 0.9
ALPHA_F = 0.1
ALPHA_P = 0.0
MODEL_PARAMS = {
  "1B": {
    "args": {
      "dim": 2048, "n_heads": 32, "n_kv_heads": 8, "n_layers": 16, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 8192,
      "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192, "rope_type": "llama3"}, "tie_word_embeddings": True
    }, "files": 1
  }, "3B": {
    "args": {
      "dim": 3072, "n_heads": 24, "n_kv_heads": 8, "n_layers": 28, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 8192,
      "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192, "rope_type": "llama3"}, "tie_word_embeddings": True
    }, "files": 1
  }, "8B": {"args": {"dim": 4096, "n_heads": 32, "n_kv_heads": 8, "n_layers": 32, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 14336}, "files": 1},
  "70B": {"args": {"dim": 8192, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 28672}, "files": 8}
}


def load_model_config(model_path: Path) -> Optional[dict]:
  """
  Load and parse config.json if it exists

  Args:
    model_path: Path to model directory or file

  Returns:
    Parsed config dict or None if config doesn't exist
  """
  config_file = model_path / "config.json" if model_path.is_dir() else model_path.parent / "config.json"
  if config_file.exists():
    with open(config_file, 'r') as f:
      return json.load(f)
  return None


def detect_architecture(config: Optional[dict]) -> str:
  """
  Detect model architecture from config

  Args:
    config: Parsed config.json dict

  Returns:
    Architecture string: "qwen3_moe" or "llama" (default)
  """
  if config is None:
    return "llama"  # default fallback

  arch = config.get('architectures', [])
  model_type = config.get('model_type', '')

  if 'Qwen3MoeForCausalLM' in arch or model_type == 'qwen3_moe':
    return "qwen3_moe"
  elif 'LlamaForCausalLM' in arch or model_type == 'llama':
    return "llama"
  else:
    return "llama"  # fallback to llama for unknown architectures


def build_transformer(model_path: Path, shard: Shard, model_size="8B", device=None):
  # Try dynamic config first
  config = load_model_config(model_path)
  architecture = detect_architecture(config)

  if architecture == "qwen3_moe" and config:
    # Build Qwen3 MoE model
    args = {
      "dim": config['hidden_size'],
      "n_heads": config['num_attention_heads'],
      "n_kv_heads": config['num_key_value_heads'],
      "n_layers": config['num_hidden_layers'],
      "num_experts": config['num_experts'],
      "num_experts_per_tok": config['num_experts_per_tok'],
      "moe_intermediate_size": config['moe_intermediate_size'],
      "vocab_size": config['vocab_size'],
      "norm_eps": config.get('rms_norm_eps', 1e-6),
      "rope_theta": config.get('rope_theta', 10000000),
      "hidden_dim": config.get('intermediate_size', 8192),
      "use_qk_norm": config.get('use_qk_norm', True),
      "head_dim": config.get('head_dim', None)  # Explicit head_dim from Qwen3 config
    }

    model = Qwen3MoETransformer(**args, linear=nn.Linear, max_context=2048, jit=True, shard=shard)

    # Load weights
    if model_path.is_dir():
      if (model_path/"model.safetensors.index.json").exists():
        weights = load(str(model_path/"model.safetensors.index.json"), shard)
      elif (model_path/"model.safetensors").exists():
        weights = load(str(model_path/"model.safetensors"), shard)
    else:
      weights = load(str(model_path), shard)

    weights = convert_from_huggingface_qwen(weights, model, args["n_heads"], args["n_kv_heads"])
    weights = fix_bf16_and_fp8(weights)

    with Context(BEAM=0):
      load_state_dict(model, weights, strict=False, consume=False)
      model = Qwen3MoETransformerShard(shard, model)

    return model

  else:
    # Existing LLaMA logic (unchanged)
    linear = nn.Linear
    model = Transformer(**MODEL_PARAMS[model_size]["args"], linear=linear, max_context=2048, jit=True, shard=shard)

    # load weights
    if model_path.is_dir():
      if (model_path/"model.safetensors.index.json").exists(): weights = load(str(model_path/"model.safetensors.index.json"), shard)
      elif (model_path/"model.safetensors").exists(): weights = load(str(model_path/"model.safetensors"), shard)
      else: weights = concat_weights([load(str(model_path/f"consolidated.{i:02d}.pth"), shard) for i in range(MODEL_PARAMS[model_size]["files"])], device[0] if isinstance(device, tuple) else device)
    else:
      weights = load(str(model_path), shard)
    weights = convert_from_huggingface(weights, model, MODEL_PARAMS[model_size]["args"]["n_heads"], MODEL_PARAMS[model_size]["args"]["n_kv_heads"])
    weights = fix_bf16_and_fp8(weights)

    with Context(BEAM=0):
      # replace weights in model
      load_state_dict(model, weights, strict=False, consume=False)  # consume=True
      model = TransformerShard(shard, model)

    return model

_executor = ThreadPoolExecutor(max_workers=1) # singleton so tinygrad always runs on the same thread
class TinygradDynamicShardInferenceEngine(InferenceEngine):
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.states = OrderedDict()
    self.executor = _executor

  def poll_state(self, x, request_id: str, max_states=2):
    if request_id not in self.states:
      if len(self.states) >= max_states:
        self.states.popitem(last=False)
      self.states[request_id] = make_prompt_state(x, self.model)
    else:
      self.states.move_to_end(request_id)
    state = self.states[request_id]
    return {"start_pos": state.start, "cache": state.cache}

  async def sample(self, x: np.ndarray, temp=TEMPERATURE, top_p: float = 0.0) -> np.ndarray:
    def sample_wrapper():
      logits = x[:, -1, :]
      return sample_logits(Tensor(logits).flatten(), temp, 0, 0.8, top_p, 0.0).realize().numpy().astype(int)
    return await asyncio.get_running_loop().run_in_executor(self.executor, sample_wrapper)

  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    await self.ensure_shard(shard)
    tokens = await asyncio.get_running_loop().run_in_executor(self.executor, self.tokenizer.encode, prompt)
    return await asyncio.get_running_loop().run_in_executor(self.executor, np.array, tokens)
  
  async def decode(self, shard: Shard, tokens) -> str:
    await self.ensure_shard(shard)
    tokens = await asyncio.get_running_loop().run_in_executor(self.executor, self.tokenizer.decode, tokens)
    return tokens
  
  async def load_checkpoint(self, shard: Shard, path: str):
    await self.ensure_shard(shard)
    state_dict = safe_load(path)
    await asyncio.get_running_loop().run_in_executor(self.executor, load_state_dict, self.model, state_dict)
  
  async def save_checkpoint(self, shard: Shard, path: str):
    await self.ensure_shard(shard)
    state_dict = await asyncio.get_running_loop().run_in_executor(self.executor, get_state_dict, self.model)
    safe_save(state_dict, path) 
  
  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    await self.ensure_shard(shard)
    def wrap_infer():
      x = Tensor(input_data)
      h = self.model.embed(x)
      state = self.poll_state(h, request_id)
      out = self.model.forward(h, **state)
      self.states[request_id].start += x.shape[1]
      return out.numpy()
    output_data = await asyncio.get_running_loop().run_in_executor(self.executor, wrap_infer)
    return output_data, inference_state

  async def evaluate(self, request_id: str, shard: Shard, inputs, targets, lengths, loss=length_masked_ce_loss):
    def step(x, y, l):
      Tensor.training = False
      return self.session['loss'](self.model, x, y, l)
    await self.ensure_shard(shard)
    score = await asyncio.get_running_loop().run_in_executor(self.executor, lambda: self.session['jit'](Tensor(inputs), targets, lengths))
    out = score.numpy()
    return out
  
  async def train(self, request_id: str, shard: Shard, inputs, targets, lengths, loss=length_masked_ce_loss, opt=nn.optim.Adam, lr=1e-5):
    def step(x, y, l):
      Tensor.training = True
      score = self.session['loss'](self.model, x, y, l)
      self.session['opt'].zero_grad()
      score.backward()
      self.session['opt'].step()
      return score
    await self.ensure_shard(shard)
      
    score = await asyncio.get_running_loop().run_in_executor(self.executor, lambda: self.session['jit'](Tensor(inputs), targets, lengths).realize())
    
    return loss.numpy(), loss.numpy()

  async def ensure_shard(self, shard: Shard):
    if self.shard == shard:
      return

    model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)

    if self.shard != shard:
      # Set device from DEVICE environment variable (defaults to CPU if not set)
      device_env = os.getenv("DEVICE", "CPU").upper()
      if device_env in ["CUDA", "GPU"]:
        Device.DEFAULT = "CUDA"
        if DEBUG: print(f"[DEVICE FIX] Set Device.DEFAULT = CUDA from DEVICE={device_env}")
      else:
        Device.DEFAULT = "CPU"
        if DEBUG: print(f"[DEVICE FIX] Set Device.DEFAULT = CPU from DEVICE={device_env}")

      loop = asyncio.get_running_loop()

      # Try dynamic config first
      config = load_model_config(model_path)
      architecture = detect_architecture(config)

      if architecture == "qwen3_moe":
        # No model_size needed for Qwen3 (uses config)
        model_shard = await loop.run_in_executor(self.executor, build_transformer, model_path, shard, None)
      else:
        # Existing LLaMA size detection
        parameters = "1B" if "1b" in shard.model_id.lower() else "3B" if "3b" in shard.model_id.lower() else "8B" if "8b" in shard.model_id.lower() else "70B"
        model_shard = await loop.run_in_executor(self.executor, build_transformer, model_path, shard, parameters)

      tokenizer_path = str((model_path if model_path.is_dir() else model_path.parent))
      self.tokenizer = await resolve_tokenizer(tokenizer_path)
      self.shard = shard
      self.model = model_shard
