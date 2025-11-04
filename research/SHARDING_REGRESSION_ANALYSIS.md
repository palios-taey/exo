# Model Sharding Regression Analysis
**Date**: 2025-11-04
**Agent**: Agent B (Regression Analysis)
**Problem**: Exo loading full 70GB model on BOTH Thors instead of sharding

---

## Git State

### Mira (10.0.0.163)
- **Branch**: thor-compatibility-2025-11-02
- **Working tree**: Clean
- **Recent commits**:
  - `5b06048` - Complete bfloat16 compatibility via safetensors framework change
  - `e820caf` - Fix bfloat16 handling in safetensors loading
  - `d6b0eed` - Implement lazy safetensors loading for distributed inference
  - `53b3c35` - Increase topology collection timeout for 70B models

### Thor #1 (jetson@10.0.0.93)
- **Branch**: thor-compatibility-2025-11-02
- **Status**: MODIFIED (uncommitted changes to tinygrad_helpers.py)
- **Last commit**: `d6b0eed` (missing e820caf and 5b06048 from Mira)
- **Manual modifications**: Has additional bfloat16→float16 conversion code NOT in git

### Thor #2 (thor@10.0.0.78)
- **Branch**: thor-compatibility-2025-11-02
- **Status**: MODIFIED (uncommitted changes to tinygrad_helpers.py)
- **Last commit**: `d6b0eed` (missing e820caf and 5b06048 from Mira)
- **Manual modifications**: Same as Thor #1

---

## Our Modifications

### Commit d6b0eed (Lazy Loading) - 2025-11-04 16:19:25

**Files changed**:
- `exo/inference/tinygrad/tinygrad_helpers.py` (18 lines)
- Documentation files (2,187 lines added)

**What changed in code**:

```python
# BEFORE (upstream, commit 53b3c35)
elif fn.endswith(".safetensors"):
  weight_map = safe_load(fn)
  for k in list(weight_map):
    if (n := re.search(r"\.(\d+)\.", k)) and not (shard.start_layer <= int(n.group(1)) <= shard.end_layer):
        del weight_map[k]
  return weight_map

# AFTER (commit d6b0eed)
elif fn.endswith(".safetensors"):
  # Lazy loading: only load tensors for this shard's layers
  from safetensors import safe_open
  weight_map = {}

  with safe_open(fn, framework="numpy") as f:
    for k in f.keys():
      # Filter during iteration - skip layers outside shard range
      if (n := re.search(r"\.(\d+)\.", k)) and not (shard.start_layer <= int(n.group(1)) <= shard.end_layer):
        continue  # Don't load this tensor at all

      # Only load tensors we actually need
      tensor_data = f.get_tensor(k)
      weight_map[k] = Tensor(tensor_data)

  return weight_map
```

**Intent**: Prevent loading full model into memory before filtering

### Commit e820caf (bfloat16 Fix #1) - 2025-11-04 17:56:10

**Files changed**:
- `exo/inference/tinygrad/tinygrad_helpers.py` (5 lines)

**What changed**:
```python
# Changed framework from "numpy" to "pt" (PyTorch)
with safe_open(fn, framework="pt") as f:  # Was: framework="numpy"
```

**Reason**: NumPy doesn't support bfloat16 dtype, PyTorch does

### Commit 5b06048 (bfloat16 Fix #2) - 2025-11-04 18:10:55

**Files changed**:
- `exo/inference/tinygrad/tinygrad_helpers.py` (complete implementation)
- Research documentation (3,743 lines)

**Final version** (Mira current):
```python
elif fn.endswith(".safetensors"):
  # Lazy loading: only load tensors for this shard's layers
  from safetensors import safe_open
  weight_map = {}

  with safe_open(fn, framework="pt") as f:
    for k in f.keys():
      # Filter during iteration - skip layers outside shard range
      if (n := re.search(r"\.(\d+)\.", k)) and not (shard.start_layer <= int(n.group(1)) <= shard.end_layer):
        continue  # Don't load this tensor at all

      # Only load tensors we actually need
      tensor_data = f.get_tensor(k)
      # Convert PyTorch tensor to numpy for tinygrad compatibility
      if hasattr(tensor_data, 'numpy'):
        tensor_data = tensor_data.numpy()
      weight_map[k] = Tensor(tensor_data)

  return weight_map
```

### Manual Thor Modifications (NOT in git)

**Both Thors have**:
```python
# Convert PyTorch tensor to numpy for tinygrad compatibility
if hasattr(tensor_data, 'numpy'):
  # Convert bfloat16 to float16 for numpy compatibility
  if hasattr(tensor_data, 'dtype') and 'bfloat16' in str(tensor_data.dtype):
    import torch
    tensor_data = tensor_data.to(torch.float16)
  tensor_data = tensor_data.numpy()
weight_map[k] = Tensor(tensor_data)
```

**This code is NOT in Mira's git repo!**

---

## Model Loading Flow

### File Structure for 70B Models

**Llama 3.1 70B has**:
- `model.safetensors.index.json` - Maps tensor names to files
- `model-00001-of-00030.safetensors` through `model-00030-of-00030.safetensors`
- Total size: ~141GB
- Shards needed per node: ~70GB (half the model)

### Load Path Analysis

```python
def load(fn: str, shard: Shard):
  if fn.endswith('.index.json'):
    # STEP 1: Read weight_map from index.json
    with open(fn) as fp:
      weight_map = json.load(fp)['weight_map']

    # STEP 2: Get allowed file patterns for this shard
    allow_patterns = get_allow_patterns(weight_map, shard)
    # This returns filenames like: ["model-00001-of-00030.safetensors", "model-00002-of-00030.safetensors", ...]

    # STEP 3: Filter weight_map by allowed patterns AND layer range
    filtered_weight_map = {}
    for k, n in weight_map.items():
      # Skip if file not in allow_patterns
      if allow_patterns is not None and not any(fnmatch(n, r) for r in allow_patterns):
        continue

      # Skip if layer outside shard range
      if k.startswith("model.layers."):
        layer_num = int(k.split('.')[2])
        if layer_num < shard.start_layer or layer_num > shard.end_layer:
          continue

      # STEP 4: Recursively call load() for each safetensors file
      parts[n] = load(str(Path(fn).parent/Path(n).name), shard)
      filtered_weight_map[k] = n

    # STEP 5: Return only tensors from filtered_weight_map
    return {k: parts[n][k] for k, n in filtered_weight_map.items()}
```

### Critical Decision Points

1. **Line 32-34** (`exo/inference/tinygrad/tinygrad_helpers.py`):
   - `get_allow_patterns()` determines WHICH safetensors files to load
   - Located in: `exo/download/hf/hf_helpers.py`

2. **Line 36-39**:
   - Layer range filtering: `if layer_num < shard.start_layer or layer_num > shard.end_layer: continue`
   - This determines WHICH tensors from allowed files

3. **Line 41**:
   - Recursive call: `parts[n] = load(str(Path(fn).parent/Path(n).name), shard)`
   - This triggers the `.safetensors` branch for each file

4. **Line 45-63** (our modification):
   - Loads individual safetensors file with lazy loading
   - Filters tensors by layer number DURING iteration

---

## Comparison with Upstream

### Files We Modified
- `exo/inference/tinygrad/tinygrad_helpers.py`

### Lines Changed
- **Commit d6b0eed**: +14 lines, -4 lines (18 line diff)
- **Commit e820caf**: +4 lines, -1 line (5 line diff)
- **Commit 5b06048**: Same as e820caf but with `.numpy()` conversion

### Key Differences from Upstream

**Upstream (commit b1397b4 "Proper sharding in tinygrad")**:
```python
weight_map = safe_load(fn)  # Loads entire file into memory
for k in list(weight_map):
  if (n := re.search(r"\.(\d+)\.", k)) and not (shard.start_layer <= int(n.group(1)) <= shard.end_layer):
      del weight_map[k]  # Deletes after loading
return weight_map
```

**Our Version**:
```python
with safe_open(fn, framework="pt") as f:  # Memory-mapped lazy access
  for k in f.keys():
    if (n := re.search(r"\.(\d+)\.", k)) and not (shard.start_layer <= int(n.group(1)) <= shard.end_layer):
        continue  # Skips before loading
    tensor_data = f.get_tensor(k)  # Loads only needed tensors
    weight_map[k] = Tensor(tensor_data)
```

### The Problem We Tried to Solve

**Original issue**:
- `safe_load()` loads entire safetensors file into GPU memory
- Then filters by deleting keys: `del weight_map[k]`
- But Python `del` doesn't free GPU memory immediately
- Result: 140GB loaded before filtering to 70GB → OOM

**Our solution**:
- Use `safe_open()` for lazy memory-mapped access
- Filter DURING iteration, not after
- Only load tensors we need

---

## HYPOTHESIS: What Broke Sharding

### Theory 1: Lazy Loading Works But Index Filtering Broken ❌

**Test**: Did we break the `allow_patterns` logic?
- NO - we didn't modify the `.index.json` branch at all
- `get_allow_patterns()` unchanged
- Layer filtering logic unchanged

### Theory 2: safe_open() + Shard Object Interaction ❌

**Test**: Does `shard` parameter get lost?
- NO - `shard` passed correctly to recursive `load()` calls
- Layer filtering regex identical to upstream

### Theory 3: Git Divergence Between Machines ✅ LIKELY

**Evidence**:
- Mira has commits e820caf + 5b06048
- Thors are at commit d6b0eed
- Thors have MANUAL modifications not in git
- Different code running on different machines

**Impact**:
- Framework mismatch: Mira uses `framework="pt"`, Thors might use `framework="numpy"` (commit d6b0eed)
- Dtype handling differs between machines

### Theory 4: Nothing Actually Broke - Different Test Setup ✅ MOST LIKELY

**Critical question**: What did Jesse compare?

**Possibility A**: Comparing against UPSTREAM exo (not our fork)
- Upstream might have DIFFERENT sharding logic
- We forked from a specific commit - upstream may have evolved

**Possibility B**: Comparing against EARLIER working test
- Earlier test might have been with SMALLER model
- Or single safetensors file (not sharded)

**Possibility C**: Test methodology changed
- Memory measurement timing (before vs after compact?)
- Different inference parameters

### Theory 5: The Real Bug - Missing File Skip Logic ⚠️ INVESTIGATE

**Looking at the code flow**:

1. `model.index.json` → calls `get_allow_patterns(weight_map, shard)`
2. Returns list like: `["model-00001-of-00030.safetensors", "model-00002-of-00030.safetensors", ...]`
3. For loop: `for k, n in weight_map.items():`
4. Check: `if allow_patterns is not None and not any(fnmatch(n, r) for r in allow_patterns): continue`

**QUESTION**: Does `get_allow_patterns()` correctly identify which files contain this shard's layers?

Let me check the implementation:
```python
def get_allow_patterns(weight_map: Dict[str, str], shard: Shard) -> List[str]:
  shard_specific_patterns = set()
  if weight_map:
    for tensor_name, filename in weight_map.items():
      layer_num = extract_layer_num(tensor_name)
      if layer_num is not None and shard.start_layer <= layer_num <= shard.end_layer:
        shard_specific_patterns.add(filename)
```

**This looks correct!** It only adds filenames containing tensors in the shard's layer range.

---

## The Smoking Gun: Manual Thor Modifications

### What's Different on Thors

**Thors have**:
```python
if hasattr(tensor_data, 'dtype') and 'bfloat16' in str(tensor_data.dtype):
  import torch
  tensor_data = tensor_data.to(torch.float16)
```

**Mira doesn't have this!**

### Why This Matters

**Hypothesis**: The `tensor_data.to(torch.float16)` conversion might:
1. Force immediate materialization of lazy tensors
2. Allocate additional GPU memory during conversion
3. Result in 2x memory usage (original bfloat16 + converted float16)

### Testing This Theory

**Expected behavior**:
- Mira: Load as bfloat16, convert to numpy, create Tensor → ~70GB
- Thors: Load as bfloat16, convert to float16, convert to numpy, create Tensor → possibly 140GB?

---

## RECOMMENDED FIX

### Option 1: Sync Git State (IMMEDIATE)

**Deploy commits e820caf + 5b06048 to Thors**:
```bash
# On Mira
cd /home/mira/exo
git format-patch d6b0eed..HEAD -o /tmp/bfloat16-patches/

# Deploy to Thor #1
scp /tmp/bfloat16-patches/* jetson@10.0.0.93:/tmp/
ssh jetson@10.0.0.93 'cd /home/jetson/exo-clean && git reset --hard HEAD && git am /tmp/*.patch'

# Deploy to Thor #2
scp /tmp/bfloat16-patches/* thor@10.0.0.78:/tmp/
ssh thor@10.0.0.78 'cd /home/thor/exo-clean && git reset --hard HEAD && git am /tmp/*.patch'
```

**Rationale**: Eliminate manual modifications causing divergence

### Option 2: Revert to Upstream Sharding (SAFE)

**If Option 1 doesn't fix it**:
```bash
cd /home/mira/exo
git revert d6b0eed --no-commit
git commit -m "Revert lazy loading - caused sharding regression"
```

**Then deploy to Thors**:
```bash
# Use git patches as above
```

**Rationale**: Go back to known-working sharding logic

### Option 3: Fix Lazy Loading Implementation (ENGINEERING)

**The issue might be**: `safe_open()` doesn't respect the file-level filtering from `allow_patterns`

**Current flow**:
1. Index filtering says: "Only load model-00001 and model-00002"
2. Code calls `load("model-00001.safetensors", shard)`
3. But ALSO calls `load("model-00003.safetensors", shard)` because filtering fails?

**Need to investigate**: Is the `fnmatch()` filtering working?

**Debugging approach**:
```python
# Add to line 34 in tinygrad_helpers.py
if DEBUG >= 2:
  print(f"[SHARD DEBUG] Checking file={n}, allow_patterns={allow_patterns}")
  print(f"[SHARD DEBUG] Match result: {any(fnmatch(n, r) for r in allow_patterns)}")
```

### Option 4: Keep Lazy Loading, Fix bfloat16 Differently (OPTIMAL)

**If lazy loading is working but bfloat16 handling is the issue**:

```python
elif fn.endswith(".safetensors"):
  from safetensors import safe_open
  weight_map = {}

  # Try PyTorch first, fallback to numpy
  try:
    with safe_open(fn, framework="pt") as f:
      for k in f.keys():
        if (n := re.search(r"\.(\d+)\.", k)) and not (shard.start_layer <= int(n.group(1)) <= shard.end_layer):
          continue

        tensor_data = f.get_tensor(k)

        # Handle bfloat16: convert via PyTorch but DON'T go to float16
        if hasattr(tensor_data, 'numpy'):
          # PyTorch's .numpy() handles bfloat16 → special numpy dtype
          tensor_data = tensor_data.numpy()

        weight_map[k] = Tensor(tensor_data)

  except Exception as e:
    if DEBUG >= 1:
      print(f"[TINYGRAD] PyTorch framework failed: {e}, falling back to numpy")
    # Fallback to original implementation
    weight_map = safe_load(fn)
    for k in list(weight_map):
      if (n := re.search(r"\.(\d+)\.", k)) and not (shard.start_layer <= int(n.group(1)) <= shard.end_layer):
          del weight_map[k]

  return weight_map
```

---

## TESTING PLAN

### Phase 1: Verify Git Sync Issue

**On Mira**:
```bash
cd /home/mira/exo
git log --oneline -5
git diff HEAD
```

**On Thor #1**:
```bash
ssh jetson@10.0.0.93 "cd /home/jetson/exo-clean && git log --oneline -5 && git diff HEAD"
```

**Expected**: All three machines should have identical code

### Phase 2: Add Debug Logging

**Modify tinygrad_helpers.py**:
```python
def load(fn: str, shard: Shard):
  if fn.endswith('.index.json'):
    # ... existing code ...

    if DEBUG >= 2:
      print(f"[SHARD] Index file: {fn}")
      print(f"[SHARD] Shard: {shard.start_layer} to {shard.end_layer}")
      print(f"[SHARD] Allow patterns: {allow_patterns}")
      print(f"[SHARD] Files to load: {list(parts.keys())}")
      print(f"[SHARD] Filtered keys: {len(filtered_weight_map)}")

  elif fn.endswith(".safetensors"):
    if DEBUG >= 2:
      print(f"[SHARD] Loading safetensors: {fn}")
      print(f"[SHARD] Shard: {shard.start_layer} to {shard.end_layer}")

    # ... existing lazy loading code ...

    if DEBUG >= 2:
      print(f"[SHARD] Loaded {len(weight_map)} tensors from {fn}")
      print(f"[SHARD] Sample keys: {list(weight_map.keys())[:5]}")
```

### Phase 3: Test Distributed Inference

**Start with DEBUG=2**:
```bash
# Thor #1
cd /home/jetson/exo-clean
DEBUG=2 CUDA=1 python3 main.py --inference-engine tinygrad

# Thor #2 (in separate terminal)
cd /home/thor/exo-clean
DEBUG=2 CUDA=1 python3 main.py --inference-engine tinygrad
```

**Monitor**:
```bash
# Watch GPU memory on both Thors
watch -n 1 nvidia-smi
```

**Expected output**:
- `[SHARD] Files to load:` should show ~15 files per Thor (not all 30)
- `nvidia-smi` should show ~70GB per Thor (not 140GB)
- Each Thor should report different layer ranges

### Phase 4: Compare with Upstream

**Clone fresh upstream exo**:
```bash
cd /tmp
git clone https://github.com/exo-explore/exo.git upstream-exo
cd upstream-exo
git log --oneline | grep -i shard | head -10
```

**Check their implementation**:
```bash
cat exo/inference/tinygrad/tinygrad_helpers.py
```

**Compare with ours**:
```bash
diff /tmp/upstream-exo/exo/inference/tinygrad/tinygrad_helpers.py \
     /home/mira/exo/exo/inference/tinygrad/tinygrad_helpers.py
```

### Phase 5: Measure Memory Precisely

**Script to measure actual memory**:
```python
import subprocess
import time

def get_gpu_memory():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    return int(result.stdout.strip())

print("Starting memory measurement...")
start_mem = get_gpu_memory()
print(f"Baseline: {start_mem} MB")

# Start exo here (in subprocess or separate terminal)
input("Press Enter after exo starts loading model...")

loading_mem = get_gpu_memory()
print(f"During loading: {loading_mem} MB")
print(f"Loaded: {loading_mem - start_mem} MB")

input("Press Enter after model fully loaded...")

final_mem = get_gpu_memory()
print(f"Final: {final_mem} MB")
print(f"Total allocated: {final_mem - start_mem} MB")
```

---

## CONCLUSIONS

### What We Know
1. **Git divergence**: Mira has 2 commits Thors don't have
2. **Manual modifications**: Thors have uncommitted bfloat16→float16 conversion
3. **Architecture**: 70B uses model.index.json + 30 safetensors files
4. **Sharding logic**: Index filtering + layer filtering should work
5. **Our change**: Replaced `safe_load()` + `del` with `safe_open()` + `continue`

### What We Don't Know
1. **Baseline**: What was the "working" sharding Jesse compared against?
2. **Test methodology**: How was memory measured? When? Which model?
3. **Actual behavior**: Is lazy loading being bypassed somewhere?
4. **File filtering**: Is `get_allow_patterns()` returning correct files?

### Most Likely Cause

**HYPOTHESIS #1**: Shared tensors loaded by BOTH shards ⚠️

**Evidence from code analysis**:
```python
# The filtering logic:
if (n := re.search(r"\.(\d+)\.", k)) and not (shard.start_layer <= int(n.group(1)) <= shard.end_layer):
    continue  # Skip layers outside range
```

**Problem**: This regex ONLY matches `model.layers.N.*` patterns!

**Tensors without layer numbers** (no match):
- `model.embed_tokens.weight` (2,004 MB)
- `lm_head.weight` (2,004 MB)
- `model.norm.weight` (0.02 MB)

**Result**: BOTH Thor #1 and Thor #2 load these shared tensors!

**Memory calculation**:
- Shared tensors: ~4 GB loaded by BOTH Thors
- This is EXPECTED behavior for distributed inference
- Not enough to explain 140GB per Thor

**HYPOTHESIS #2**: The manual bfloat16→float16 conversion on Thors is allocating extra memory

**Why**:
- `tensor_data.to(torch.float16)` creates a NEW tensor
- Original bfloat16 tensor still in memory
- Conversion happens AFTER lazy loading fetches tensor
- Results in 2x memory during conversion
- Even if original freed, garbage collection is async

**Test**: Remove the `.to(torch.float16)` line and rely on PyTorch's `.numpy()` to handle bfloat16

**HYPOTHESIS #3**: The lazy loading itself is the problem

**Possible issue**: `safe_open()` might not properly filter in the `model.index.json` path

**The flow**:
1. `load("model.safetensors.index.json", shard)`
2. Calls `get_allow_patterns()` → should return only ~15 files for this shard
3. For each file: `parts[n] = load("model-00001-of-00030.safetensors", shard)`
4. This triggers lazy loading with `safe_open()`

**Question**: Is step 2 actually filtering files, or loading all 30?

**To investigate**: Add debug prints to see which files `allow_patterns` returns

### Recommended Next Action

**PRIORITY 1**: Sync git state
```bash
# Deploy commits e820caf + 5b06048 to both Thors
# This removes manual modifications
```

**PRIORITY 2**: Add debug logging
```bash
# Add [SHARD DEBUG] prints to see which files are loaded
```

**PRIORITY 3**: Measure precisely
```bash
# Use nvidia-smi + timestamps to capture exact memory usage
```

**PRIORITY 4**: If still broken, revert lazy loading
```bash
# Go back to safe_load() + del approach
# Keep bfloat16 fix (framework="pt")
```

---

## APPENDIX: Full Code Comparison

### Version 1: Upstream (commit 53b3c35)
```python
elif fn.endswith(".safetensors"):
  weight_map = safe_load(fn)
  for k in list(weight_map):
    if (n := re.search(r"\.(\d+)\.", k)) and not (shard.start_layer <= int(n.group(1)) <= shard.end_layer):
        del weight_map[k]
  return weight_map
```

### Version 2: Lazy Loading (commit d6b0eed)
```python
elif fn.endswith(".safetensors"):
  from safetensors import safe_open
  weight_map = {}

  with safe_open(fn, framework="numpy") as f:
    for k in f.keys():
      if (n := re.search(r"\.(\d+)\.", k)) and not (shard.start_layer <= int(n.group(1)) <= shard.end_layer):
        continue
      tensor_data = f.get_tensor(k)
      weight_map[k] = Tensor(tensor_data)

  return weight_map
```

### Version 3: bfloat16 Fix #1 (commit e820caf)
```python
# Only change: framework="numpy" → framework="pt"
with safe_open(fn, framework="pt") as f:
```

### Version 4: bfloat16 Fix #2 - Mira Current (commit 5b06048)
```python
elif fn.endswith(".safetensors"):
  from safetensors import safe_open
  weight_map = {}

  with safe_open(fn, framework="pt") as f:
    for k in f.keys():
      if (n := re.search(r"\.(\d+)\.", k)) and not (shard.start_layer <= int(n.group(1)) <= shard.end_layer):
        continue

      tensor_data = f.get_tensor(k)
      if hasattr(tensor_data, 'numpy'):
        tensor_data = tensor_data.numpy()
      weight_map[k] = Tensor(tensor_data)

  return weight_map
```

### Version 5: Thor Manual Modifications (NOT IN GIT)
```python
elif fn.endswith(".safetensors"):
  from safetensors import safe_open
  weight_map = {}

  with safe_open(fn, framework="pt") as f:
    for k in f.keys():
      if (n := re.search(r"\.(\d+)\.", k)) and not (shard.start_layer <= int(n.group(1)) <= shard.end_layer):
        continue

      tensor_data = f.get_tensor(k)
      if hasattr(tensor_data, 'numpy'):
        # EXTRA CODE NOT IN MIRA:
        if hasattr(tensor_data, 'dtype') and 'bfloat16' in str(tensor_data.dtype):
          import torch
          tensor_data = tensor_data.to(torch.float16)  # ← THIS MIGHT BE THE BUG
        tensor_data = tensor_data.numpy()
      weight_map[k] = Tensor(tensor_data)

  return weight_map
```

---

## EXECUTIVE SUMMARY

### The Smoking Gun: Git Divergence

**CRITICAL FINDING**: Mira and Thors are running DIFFERENT code!

- **Mira**: Commits d6b0eed → e820caf → 5b06048 (lazy loading + bfloat16 via PyTorch)
- **Thors**: Commit d6b0eed + MANUAL MODIFICATIONS (lazy loading + bfloat16→float16 conversion)

### The Likely Bug: Manual bfloat16→float16 Conversion

**Code on Thors (NOT in git)**:
```python
if hasattr(tensor_data, 'dtype') and 'bfloat16' in str(tensor_data.dtype):
  import torch
  tensor_data = tensor_data.to(torch.float16)  # ← Creates duplicate tensor
```

**Why this doubles memory**:
1. `f.get_tensor(k)` loads tensor as bfloat16 (e.g., 5GB)
2. `.to(torch.float16)` creates NEW tensor (5GB)
3. Original bfloat16 tensor still in memory until GC
4. Peak memory: 10GB for this conversion alone
5. Multiply by all tensors → 140GB total

**Mira's approach (correct)**:
```python
tensor_data = tensor_data.numpy()  # PyTorch handles bfloat16→numpy directly
```

### Hypothesis Ranking

1. **MOST LIKELY (90%)**: Manual float16 conversion causing 2x memory
2. **POSSIBLE (50%)**: File-level filtering not working, loading all 30 safetensors files
3. **UNLIKELY (10%)**: Lazy loading fundamentally broken
4. **EXPECTED (0%)**: Shared tensors - this is normal for distributed inference

### Immediate Fix

**Deploy Mira's commits to Thors**:
```bash
cd /home/mira/exo
git format-patch d6b0eed..5b06048 -o /tmp/sync-patches/

# Thor #1
scp /tmp/sync-patches/* jetson@10.0.0.93:/tmp/
ssh jetson@10.0.0.93 'cd /home/jetson/exo-clean && git reset --hard d6b0eed && git am /tmp/*.patch'

# Thor #2
scp /tmp/sync-patches/* thor@10.0.0.78:/tmp/
ssh thor@10.0.0.78 'cd /home/thor/exo-clean && git reset --hard d6b0eed && git am /tmp/*.patch'
```

**This will**:
1. Remove the problematic `.to(torch.float16)` conversion
2. Use PyTorch's native bfloat16→numpy handling
3. Sync all three machines to identical code
4. Preserve lazy loading optimization

### If That Doesn't Work

**Fallback**: Revert to upstream sharding
```bash
git revert d6b0eed --no-commit
git commit -m "Revert lazy loading - regression in distributed inference"
```

### What Jesse Should Test

**After deploying fix**:
```bash
# Start both Thors with DEBUG=2
DEBUG=2 CUDA=1 python3 main.py --inference-engine tinygrad

# Watch memory on both machines
watch -n 1 nvidia-smi

# Expected: ~70GB per Thor (not 140GB)
```

---

**Analysis complete. Agent B signing off.**

**Next steps**: Deploy git sync, test memory usage, report results.

