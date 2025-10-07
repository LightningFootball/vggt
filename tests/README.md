# Tests Overview

## tests/test_vram_optimized.py
- Purpose: Verifies that the VRAM-optimized aggregator (`AggregatorVramOptimized`) preserves numerical equivalence and indexing semantics with the baseline `Aggregator`, and remains compatible with downstream heads (`CameraHead`, `DPTHead`).

### How to Run
- Prerequisites: `pip install -r requirements.txt`
- Run all tests in this file:
  - `pytest -q tests/test_vram_optimized.py`
- Useful pytest options (parameters):
  - `-q` quiet output, `-s` show print/logs, `-x` stop on first failure
  - `-k <expr>` run a subset, e.g. `-k heads_numerical_equivalence`
  - Example: `pytest -q tests/test_vram_optimized.py -k test_selected_view_indexing_and_shape`

### Test Flow
1. Seed RNG (`torch.manual_seed`) and create a small synthetic batch: `B=1, S=3, H=W=64, patch_size=16`.
2. Instantiate baseline `Aggregator` and `AggregatorVramOptimized(selected_layer_idx=[1,3,5,7])`; align weights via `load_state_dict(..., strict=False)`.
3. Forward once to collect outputs:
   - Baseline returns a full list of per-layer features and `patch_start`.
   - Optimized returns a `LayerSelectView`, `patch_start`, and the kept indices.
4. Assertions for indexing semantics and shapes:
   - `LayerSelectView` length equals model depth; selected indices are sorted and counted correctly.
   - Access to non-selected layers raises `KeyError`; last layer (`-1`) matches baseline.
5. Downstream heads equivalence:
   - `CameraHead` run on both outputs; last-level pose encoding tensors are `allclose` within tolerances.
   - `DPTHead` run on both; predicted depth and confidence maps are `allclose` within tolerances.

### Expected Results
- All assertions pass on CPU within seconds; no GPU is required.
- Tolerances: approximately `rtol=1e-4..1e-5`, `atol=1e-5..1e-6` as specified in the tests.
- Common failure causes if code changes:
  - View indexing semantics changed (e.g., selected indices, error types) without updating tests.
  - Heads’ interfaces or tensor shapes modified.
  - Numerical differences introduced by algorithmic changes beyond set tolerances.

