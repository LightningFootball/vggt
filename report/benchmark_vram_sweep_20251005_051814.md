# Benchmark: VRAM Sweep (2025-10-05 05:18:14)

- Source CSV: `benchmark_vram_sweep_20251005_051814.csv`
- Rows: 10
- Device/config details: not embedded in CSV

## Summary

- Average speedup (opt vs full): 1.059×
- Best speedup: 1.517× (`small_2frames`, 112px)
- Slowest speedup: 0.988× (`medium_B2_S4`, 224px)
- Peak VRAM change: −1.72% on average (opt uses slightly more)
  - Range: −0.43% to −2.41% (negative = opt higher VRAM)
- Storage reduction: 83.33% across all runs (consistent)

Notes:
- “Mem” columns are bytes for peak VRAM; values convert to MB below.
- “Storage” likely refers to persistent buffer/storage usage; it drops by 5/6 uniformly.

## Observations

- Performance
  - Clear win on smaller configs (112×112). Up to ~1.52× faster.
  - At 224px and above, speedups hover ~1.0×, with small regressions in some cases.
- VRAM
  - The optimized mode increases peak VRAM by ~0.4–2.4%. This suggests the
    optimization targets buffer/storage footprint rather than peak allocator use.
- Storage
  - Uniform 83.33% reduction indicates a consistent strategy (e.g., reduced
    preallocation/feature cache size) working as intended.

## By Resolution

- 112px (n=3)
  - Avg speedup: 1.189×; full 20.46 ms → opt 17.13 ms
  - Avg peak VRAM: 139.18 MB → 140.08 MB (opt +0.9 MB)
- 224px (n=5)
  - Avg speedup: 1.005×; full 31.96 ms → opt 31.78 ms
  - Avg peak VRAM: 437.83 MB → 447.30 MB (opt +9.47 MB)
- 448px (n=1)
  - Speedup: 0.997×; 78.06 ms → 78.32 ms
  - Peak VRAM: 704.81 MB → 720.89 MB (opt +16.08 MB)
- 518px (n=1)
  - Speedup: 1.000×; 117.15 ms → 117.16 ms
  - Peak VRAM: 904.69 MB → 926.51 MB (opt +21.82 MB)

## Detailed Results

| name | batch | seq | img | full_ms | opt_ms | speedup | full_mem_MB | opt_mem_MB | mem_red_% | storage_red_% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small_2frames | 1 | 2 | 112 | 26.735 | 17.620 | 1.517 | 126.61 | 127.15 | -0.426 | 83.333 |
| small_4frames | 1 | 4 | 112 | 17.818 | 16.932 | 1.052 | 145.46 | 146.54 | -0.741 | 83.333 |
| medium_B1_S8 | 1 | 8 | 224 | 34.163 | 33.695 | 1.014 | 407.91 | 416.66 | -2.145 | 83.333 |
| medium_B2_S4 | 2 | 4 | 224 | 27.768 | 28.107 | 0.988 | 408.06 | 416.88 | -2.163 | 83.333 |
| medium_B4_S2 | 4 | 2 | 224 | 25.129 | 25.238 | 0.996 | 408.06 | 416.88 | -2.163 | 83.333 |
| res_112 | 1 | 4 | 112 | 16.820 | 16.844 | 0.999 | 145.46 | 146.54 | -0.741 | 83.333 |
| res_224 | 1 | 4 | 224 | 21.281 | 20.899 | 1.018 | 259.07 | 263.81 | -1.832 | 83.333 |
| res_448 | 1 | 4 | 448 | 78.061 | 78.317 | 0.997 | 704.81 | 720.89 | -2.281 | 83.333 |
| large_B2_S8 | 2 | 8 | 224 | 51.478 | 50.956 | 1.010 | 706.07 | 722.26 | -2.294 | 83.333 |
| large_res_518 | 1 | 4 | 518 | 117.149 | 117.164 | 1.000 | 904.69 | 926.51 | -2.412 | 83.333 |

## Takeaways

- If your goal is runtime speed, the optimized path helps for small inputs and
  short sequences; expect diminishing returns at higher resolutions.
- If your goal is peak VRAM reduction, the current optimized mode does not lower
  peak allocation; it slightly increases it (likely overhead for chunking or
  alternate kernels). However, it does greatly reduce persistent storage usage.

## Recommendations

- Clarify “storage” vs “peak VRAM” in benchmark docs and the script output.
- Include GPU model, driver, and PyTorch/CUDA versions in the CSV or filename.
- Consider additional optimizations (e.g., memory-efficient attention,
  xFormers/Flash-Attn) if available for large resolutions.
- Add frames/sec to the report for easier throughput comparison
  (frames/ms → fps).

