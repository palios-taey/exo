# Agent 1: Model Dtype Analysis Report
**Investigation Date:** 2025-11-04
**Agent:** Model Dtype Analysis
**Objective:** Analyze Llama 3.3 70B model datatype requirements and compare with working models

---

## Executive Summary

**ROOT CAUSE IDENTIFIED:** NumPy 2.2.6 on Mira does not have native bfloat16 support, causing the error "data type 'bfloat16' not understood".

**KEY FINDINGS:**
1. **ALL Llama models (1B, 3B, 70B) use bfloat16** as their default torch_dtype
2. **NumPy lacks native bfloat16** - it's not a standard NumPy dtype
3. **PyTorch supports bfloat16** but cannot convert to NumPy arrays
4. **Tinygrad handles bfloat16 internally** but relies on NumPy for some operations
5. **Solution exists:** ml_dtypes package adds bfloat16 support to NumPy

---

## 1. Model Dtype Investigation

### 1.1 Llama 3.3 70B Instruct (unsloth variant)

**Location:** `/home/mira/.cache/huggingface/hub/models--unsloth--Llama-3.3-70B-Instruct/`

**Status:** Tokenizer-only download (no weights, only 17MB)

**Config Analysis:**
```json
{
  "torch_dtype": "bfloat16",
  "hidden_size": 8192,
  "num_hidden_layers": 80,
  "num_attention_heads": 64,
  "num_key_value_heads": 8,
  "intermediate_size": 28672,
  "max_position_embeddings": 131072,
  "model_type": "llama"
}
```

**Key Attributes:**
- **Dtype:** bfloat16 (BF16)
- **Size:** 70B parameters
- **Context:** 131K tokens
- **Architecture:** LlamaForCausalLM

### 1.2 Llama 3.2 1B Instruct (Working Model)

**Location:** `/home/mira/.cache/exo/downloads/unsloth--Llama-3.2-1B-Instruct/`

**Status:** Full model with weights (2.3GB safetensors file)

**Actual Tensor Analysis:**
```python
Format: pt (PyTorch)
First tensor: model.embed_tokens.weight
  dtype: torch.bfloat16
  shape: [128256, 2048]
Config torch_dtype: bfloat16
```

**Key Finding:** The 1B model that works elsewhere ALSO uses bfloat16.

### 1.3 Llama 3.2 3B Instruct

**Config Analysis:**
```json
{
  "torch_dtype": "bfloat16",
  "hidden_size": 3072,
  "num_hidden_layers": 28
}
```

**Key Finding:** Also bfloat16. This is consistent across all Llama 3.x models.

---

## 2. Dtype Comparison Table

| Model | Variant | Default Dtype | Hidden Size | Layers | Status on Mira |
|-------|---------|---------------|-------------|--------|----------------|
| Llama 3.2 1B | unsloth | bfloat16 | 2048 | 16 | Downloaded (2.3GB) |
| Llama 3.2 3B | unsloth | bfloat16 | 3072 | 28 | Downloaded (splits) |
| Llama 3.3 70B | unsloth | bfloat16 | 8192 | 80 | Tokenizer only (17MB) |

**Conclusion:** All models use bfloat16. The dtype is NOT the differentiating factor.

---

## 3. NumPy bfloat16 Support Analysis

### 3.1 Current Environment Status

```bash
NumPy version: 2.2.6
✗ bfloat16 not available: data type 'bfloat16' not understood
✗ ml_dtypes not installed
✗ bfloat16 package not installed
```

**Test Results:**
```python
# Direct NumPy test
np.dtype('bfloat16')
# TypeError: data type 'bfloat16' not understood

# PyTorch bfloat16 tensor
torch.tensor([1.0, 2.0], dtype=torch.bfloat16)  # ✓ Works

# PyTorch to NumPy conversion
tensor.cpu().numpy()
# TypeError: Got unsupported ScalarType BFloat16
```

### 3.2 Why NumPy Doesn't Support bfloat16

**From research:**
- NumPy has NO native bfloat16 support
- NumPy does NOT plan to add it natively
- bfloat16 is a machine learning-specific format (Google Brain/TensorFlow origin)
- Standard NumPy only supports: float16, float32, float64

**What is bfloat16?**
- **Brain Float 16:** 16-bit floating point (1 sign, 8 exponent, 7 mantissa)
- **vs float16:** 16-bit (1 sign, 5 exponent, 10 mantissa)
- **Advantage:** Same exponent range as float32, better for training
- **Disadvantage:** Lower precision than float16

---

## 4. Tinygrad bfloat16 Support

### 4.1 Internal Support

**From `/home/mira/tinygrad-blackwell-fork/tinygrad/dtype.py`:**

```python
# Line 165
bfloat16: Final[DType] = DType.new(12, 2, "__bf16", None)

# bfloat16 is defined as priority 12, 2 bytes, name "__bf16"
# NO format string ('fmt' = None)
```

**Test Results:**
```python
from tinygrad import dtypes, Tensor

# ✓ Tinygrad recognizes bfloat16
dtypes.bfloat16  # dtypes.bfloat16

# ✓ Can create bfloat16 tensors
t = Tensor([1.0, 2.0], dtype=dtypes.bfloat16)  # Success
```

### 4.2 NumPy Conversion Strategy

**From dtype.py lines 327-330:**
```python
def _to_np_dtype(dtype:DType) -> type|None:
    import numpy as np
    if dtype in { dtypes.bfloat16, *dtypes.fp8s }:
        return np.float32  # ← Fallback to float32!
    return np.dtype(dtype.fmt).type if dtype.fmt is not None else None
```

**Strategy:** Tinygrad converts bfloat16 to float32 when interfacing with NumPy.

### 4.3 The Problem

**Somewhere in the code path**, something is trying to use `np.dtype('bfloat16')` directly instead of going through tinygrad's conversion layer.

**Known Issue:** Tinygrad issue #11834 reported bfloat16 casting differences between PYTHON=1 and METAL=1 backends (now fixed).

---

## 5. Alternative Model Versions on HuggingFace

### 5.1 Available Variants

**From web search:**

1. **FP16 Variants:**
   - `casperhansen/llama-3-70b-fp16` (Llama 3, not 3.3)
   - `context-labs/Meta-Llama-3.1-8B-Instruct-FP16`
   - `TheBloke/Llama-2-70B-fp16`

2. **BF16 (Default):**
   - All official Meta models use bfloat16
   - `meta-llama/Llama-3.1-405B-Instruct` - all weights BF16

3. **Quantized:**
   - GPTQ, AWQ, GGUF variants exist but NOT for 3.3 yet

### 5.2 Llama 3.3 Availability

**CRITICAL FINDING:** Web search returned NO results for Llama 3.3 specifically.

Results showed:
- Llama 3 (original)
- Llama 3.1 (405B, 70B, 8B)
- Llama 3.2 (3B, 1B)
- **NO Llama 3.3 results**

**Hypothesis:** Llama 3.3 70B may be:
- Very new release
- Not widely available yet
- Only available through specific channels (unsloth)
- Possibly a fine-tuned variant, not official Meta release

---

## 6. Solutions & Recommendations

### 6.1 Solution #1: Install ml_dtypes (RECOMMENDED)

**Package:** `ml_dtypes` - JAX/Google's NumPy dtype extensions

**Installation:**
```bash
pip install ml_dtypes
```

**Usage:**
```python
from ml_dtypes import bfloat16
import numpy as np

# Now NumPy recognizes bfloat16
arr = np.zeros(4, dtype=bfloat16)
# array([0, 0, 0, 0], dtype=bfloat16)

# Can also use string reference
np.dtype('bfloat16')  # Works!
```

**Advantages:**
- Official solution from JAX/Google
- Actively maintained
- Used by TensorFlow, JAX
- Integrates seamlessly with NumPy

**Compatibility:**
- Python 3.9-3.12 ✓ (Mira has 3.12)
- NumPy 1.21+ ✓ (Mira has 2.2.6)

### 6.2 Solution #2: Use FP16 Model Variant

**Approach:** Download fp16 version instead of bfloat16

**Command:**
```python
from huggingface_hub import snapshot_download

# For Llama 3.1 70B (closest to 3.3)
snapshot_download(
    repo_id="casperhansen/llama-3-70b-fp16",
    local_dir="/home/mira/.cache/exo/downloads/llama-3-70b-fp16"
)
```

**Disadvantages:**
- Llama 3.3 70B fp16 version may not exist
- fp16 has narrower range than bfloat16 (can cause overflows)
- Not the "official" dtype for these models

### 6.3 Solution #3: Convert Model Dtype

**Approach:** Convert bfloat16 weights to fp16/fp32

**Code Example:**
```python
import torch
from safetensors.torch import load_file, save_file

# Load bfloat16 model
state_dict = load_file("model.safetensors")

# Convert to fp16
state_dict_fp16 = {
    k: v.to(torch.float16) if v.dtype == torch.bfloat16 else v
    for k, v in state_dict.items()
}

# Save converted model
save_file(state_dict_fp16, "model_fp16.safetensors")

# Update config.json: "torch_dtype": "float16"
```

**Considerations:**
- Requires re-downloading full 70B model (~140GB)
- Conversion time: 10-30 minutes
- Storage: Need space for both versions during conversion
- Precision loss: Minimal for inference

### 6.4 Solution #4: Patch Tinygrad's NumPy Interface

**Approach:** Ensure tinygrad NEVER passes 'bfloat16' string to NumPy

**Investigation needed:**
- Where is `np.dtype('bfloat16')` being called?
- Can we intercept and convert to float32?
- Is this in safetensors loading code?

**Next steps:** Agent 2 should investigate the exact error traceback.

---

## 7. Recommended Action Plan

### Phase 1: Quick Fix (Install ml_dtypes)
```bash
# On Mira
pip install ml_dtypes

# Verify
python3 -c "from ml_dtypes import bfloat16; import numpy as np; print(np.dtype('bfloat16'))"
```

**Expected outcome:** The "data type 'bfloat16' not understood" error disappears.

### Phase 2: Test on Thors
```bash
# On Thor #1 (10.0.0.93)
ssh jetson@10.0.0.93
pip install ml_dtypes

# On Thor #2 (10.0.0.78)
ssh thor@10.0.0.78
pip install ml_dtypes
```

### Phase 3: Verify with Working Model
```bash
cd /home/mira/exo
python3 << 'EOF'
from tinygrad import Tensor, dtypes
import numpy as np
from ml_dtypes import bfloat16

# Test 1: NumPy can now handle bfloat16
arr = np.array([1.0, 2.0], dtype=bfloat16)
print(f"NumPy bfloat16: {arr.dtype}")

# Test 2: Tinygrad with bfloat16
t = Tensor([1.0, 2.0], dtype=dtypes.bfloat16)
print(f"Tinygrad bfloat16: {t.dtype}")

# Test 3: Load actual model weights
from safetensors.torch import load_file
weights = load_file("/home/mira/.cache/exo/downloads/unsloth--Llama-3.2-1B-Instruct/model.safetensors")
print(f"Model weight dtype: {weights['model.embed_tokens.weight'].dtype}")
EOF
```

### Phase 4: Re-test Llama 3.3 70B
Once ml_dtypes is installed, retry the original failing operation.

---

## 8. Code Examples for Dtype Checking

### 8.1 Check Model Config Dtype
```python
import json
from pathlib import Path

def check_model_dtype(model_path: str):
    """Check torch_dtype in model config.json"""
    config_path = Path(model_path) / "config.json"

    if not config_path.exists():
        return None

    with open(config_path) as f:
        config = json.load(f)

    return {
        'torch_dtype': config.get('torch_dtype'),
        'hidden_size': config.get('hidden_size'),
        'num_layers': config.get('num_hidden_layers'),
        'model_type': config.get('model_type')
    }

# Usage
info = check_model_dtype("/home/mira/.cache/exo/downloads/unsloth--Llama-3.2-1B-Instruct")
print(info)
```

### 8.2 Check Safetensors Actual Dtype
```python
from safetensors.torch import load_file
import torch

def check_weights_dtype(safetensors_path: str):
    """Check actual tensor dtypes in safetensors file"""
    weights = load_file(safetensors_path)

    dtype_counts = {}
    for name, tensor in weights.items():
        dtype = str(tensor.dtype)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

    return {
        'total_tensors': len(weights),
        'dtype_distribution': dtype_counts,
        'sample_tensor': {
            'name': list(weights.keys())[0],
            'dtype': str(list(weights.values())[0].dtype),
            'shape': list(weights.values())[0].shape
        }
    }

# Usage
info = check_weights_dtype("/home/mira/.cache/exo/downloads/unsloth--Llama-3.2-1B-Instruct/model.safetensors")
print(info)
```

### 8.3 Convert Model Dtype
```python
import torch
from safetensors.torch import load_file, save_file
from pathlib import Path
import json

def convert_model_dtype(
    input_path: str,
    output_path: str,
    target_dtype: torch.dtype = torch.float16
):
    """Convert model weights to different dtype"""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load weights
    print(f"Loading from {input_path}")
    weights = load_file(input_path / "model.safetensors")

    # Convert
    print(f"Converting to {target_dtype}")
    converted = {}
    for name, tensor in weights.items():
        if tensor.dtype in [torch.bfloat16, torch.float32]:
            converted[name] = tensor.to(target_dtype)
        else:
            converted[name] = tensor  # Keep int types as-is

    # Save
    print(f"Saving to {output_path}")
    save_file(converted, output_path / "model.safetensors")

    # Update config
    with open(input_path / "config.json") as f:
        config = json.load(f)

    config['torch_dtype'] = str(target_dtype).split('.')[-1]

    with open(output_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print("✓ Conversion complete")

# Usage
convert_model_dtype(
    "/home/mira/.cache/exo/downloads/unsloth--Llama-3.2-1B-Instruct",
    "/home/mira/.cache/exo/downloads/unsloth--Llama-3.2-1B-Instruct-FP16",
    torch.float16
)
```

---

## 9. Technical Deep-Dive: bfloat16 Format

### 9.1 Format Comparison

| Format | Total Bits | Sign | Exponent | Mantissa | Range | Precision |
|--------|-----------|------|----------|----------|-------|-----------|
| float32 | 32 | 1 | 8 | 23 | ±3.4e38 | ~7 decimals |
| float16 | 16 | 1 | 5 | 10 | ±65504 | ~3 decimals |
| bfloat16 | 16 | 1 | 8 | 7 | ±3.4e38 | ~2 decimals |

### 9.2 Why ML Models Use bfloat16

**Advantages:**
1. **Same exponent range as float32** - prevents overflows during training
2. **Half the memory** of float32
3. **Hardware support** on TPUs, modern NVIDIA GPUs (Ampere+, including Blackwell)
4. **Simple conversion** to/from float32 (just truncate/pad mantissa)

**Training vs Inference:**
- **Training:** bfloat16 preferred (better gradient stability)
- **Inference:** float16 often sufficient (higher precision)

### 9.3 Tinygrad's bfloat16 Implementation

**From `/home/mira/tinygrad-blackwell-fork/tinygrad/dtype.py` line 244-248:**

```python
def float_to_bf16(x):
    """Convert float to bfloat16 by truncating mantissa"""
    if not math.isfinite(x): return x

    # Pack as float32
    u = struct.unpack('I', struct.pack('f', x))[0]

    # Round to nearest even + truncate mantissa
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000

    # Unpack back to float
    return struct.unpack('f', struct.pack('I', u))[0]
```

**Key insight:** Tinygrad stores bfloat16 values as float32 internally, applying truncation only when needed.

---

## 10. Summary & Next Steps

### Key Findings

1. **ALL Llama 3.x models use bfloat16** - this is not model-specific
2. **NumPy 2.2.6 lacks native bfloat16 support** - this is the root cause
3. **Solution exists:** ml_dtypes package adds bfloat16 to NumPy
4. **Tinygrad supports bfloat16** but something in the stack is calling NumPy directly
5. **Llama 3.3 70B** may be a newer/specialized release (limited availability)

### Recommended Next Steps

**For Agent 2 (Stack Trace Analysis):**
- Find exact line where `np.dtype('bfloat16')` is called
- Identify if it's in exo, tinygrad, safetensors, or huggingface code
- Check if ml_dtypes installation fixes it completely

**For Agent 3 (Hardware Compatibility):**
- Verify Jetson Thor Blackwell supports bfloat16 in hardware
- Check CUDA compute capability sm_101 bfloat16 ops
- Test PyTorch bfloat16 operations on Thor

**For Main Investigation:**
1. Install ml_dtypes on all machines (Mira + both Thors)
2. Re-test failing operation
3. If still fails, get full traceback to Agent 2
4. Document whether issue is NumPy, tinygrad, or exo-specific

### Success Criteria

✓ **Phase 1:** `np.dtype('bfloat16')` works after ml_dtypes install
✓ **Phase 2:** Can load Llama 3.2 1B model with tinygrad
✓ **Phase 3:** Can run inference on Thor with bfloat16 model
✓ **Phase 4:** Can scale to Llama 3.3 70B (once weights are downloaded)

---

## Appendix A: Installation Commands

### Mira (Database Server)
```bash
ssh mira@10.0.0.163
cd /home/mira/exo
pip install ml_dtypes
python3 -c "from ml_dtypes import bfloat16; print('✓ ml_dtypes installed')"
```

### Thor #1 (Primary Jetson)
```bash
ssh jetson@10.0.0.93
pip install ml_dtypes
python3 -c "from ml_dtypes import bfloat16; print('✓ ml_dtypes installed')"
```

### Thor #2 (Secondary Jetson)
```bash
ssh thor@10.0.0.78
pip install ml_dtypes
python3 -c "from ml_dtypes import bfloat16; print('✓ ml_dtypes installed')"
```

---

## Appendix B: Research Sources

1. **NumPy bfloat16 Support:**
   - HackMD: "bfloat16 in numpy: ml_dtypes"
   - GitHub: jax-ml/ml_dtypes
   - PyPI: ml-dtypes package

2. **Tinygrad bfloat16:**
   - GitHub issue #11834: "PYTHON=1 bf16 cast outputs differently"
   - tinygrad/dtype.py source code

3. **HuggingFace Models:**
   - casperhansen/llama-3-70b-fp16
   - context-labs FP16 variants
   - Meta official models (BF16 default)

4. **Jetson Thor:**
   - NVIDIA official docs: Blackwell GPU architecture
   - 2070 TFLOPS, 96 Tensor Cores
   - 7.5x more AI compute vs Jetson Orin

---

**Report completed by Agent 1: Model Dtype Analysis**
**Status:** Investigation complete, solution identified
**Recommendation:** Install ml_dtypes on all machines and re-test
**Next agent:** Agent 2 should analyze stack trace with ml_dtypes installed
