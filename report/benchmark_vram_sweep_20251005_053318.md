## Benchmark: VRAM Sweep (2025-10-05 05:33:18)

- Source CSV: `benchmark_vram_sweep_20251005_053318.csv`
- Rows: 17
- Device/config details: not embedded in CSV

### Summary

- Average speedup (opt vs full): 1.036×
- Best speedup: 1.520× (`small_2frames`, 112px)
- Slowest speedup: 0.975× (`res_448`, 448px)
- Peak VRAM change (opt vs full): +2.19% on average (opt lower is positive)
  - Range: −6.38% to +25.68%
- Storage reduction: 83.33% across all runs (uniform)

Notes:
- “Mem” columns are bytes for peak VRAM; values below are MB.
- “Storage” likely refers to persistent buffer/storage; reduced by 5/6 uniformly.

### Observations

- Performance
  - Clear wins on small configs (112×112), up to ~1.52× faster.
  - At ≥224px, speedups hover ~1.0×; small regressions possible.
- VRAM
  - For short sequences (2–32 frames), optimized path slightly increases peak VRAM.
  - For long sequences at 518px, optimized path reduces peak VRAM substantially
    (up to ~25.7%). Overall average flips positive due to these long runs.
- Storage
  - Uniform 83.33% reduction indicates a consistent strategy working as intended.

### By Resolution

- 112px (n=3)
  - Avg speedup: 1.199×; full 20.97 ms → opt 17.59 ms
  - Avg peak VRAM: 139.18 MB → 140.08 MB (opt +0.90 MB); avg mem_red: −0.636%
- 224px (n=4)
  - Avg speedup: 0.998×; full 27.60 ms → opt 27.58 ms
  - Avg peak VRAM: 370.77 MB → 378.56 MB (opt +7.79 MB); avg mem_red: −2.076%
- 448px (n=1)
  - Speedup: 0.975×; 80.43 ms → 82.51 ms
  - Peak VRAM: 704.81 MB → 720.89 MB (opt +16.08 MB); mem_red: −2.281%
- 518px (n=9)
  - Avg speedup: 1.005×; full 37,340.55 ms → opt 37,217.33 ms
  - Avg peak VRAM: 4,878.16 MB → 3,950.49 MB (opt −927.67 MB); avg mem_red: +5.526%

### Scaling at 518px by Total Frames

- 2 frames: mem_red −2.35% (opt higher VRAM)
- 4 frames: −2.41%
- 8 frames: −2.52%
- 16 frames: −4.29%
- 32 frames: −6.38% (worst)
- 64 frames: +18.88%
- 128 frames: +25.54%
- 256 frames: +25.68% (best)

### Detailed Results

| name | batch | seq | frames | img | full_ms | opt_ms | speedup | full_mem_MB | opt_mem_MB | mem_red_% | storage_red_% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small_2frames | 1 | 2 | 2 | 112 | 25.609 | 16.847 | 1.520 | 126.61 | 127.15 | -0.426 | 83.333 |
| small_4frames | 1 | 4 | 4 | 112 | 17.684 | 17.193 | 1.029 | 145.46 | 146.54 | -0.741 | 83.333 |
| medium_B1_S8 | 1 | 8 | 8 | 224 | 34.320 | 33.461 | 1.026 | 407.91 | 416.66 | -2.145 | 83.333 |
| medium_B2_S4 | 2 | 4 | 8 | 224 | 27.861 | 27.823 | 1.001 | 408.06 | 416.88 | -2.163 | 83.333 |
| medium_B4_S2 | 4 | 2 | 8 | 224 | 25.154 | 25.631 | 0.981 | 408.06 | 416.88 | -2.163 | 83.333 |
| res_112 | 1 | 4 | 4 | 112 | 19.621 | 18.732 | 1.047 | 145.46 | 146.54 | -0.741 | 83.333 |
| res_224 | 1 | 4 | 4 | 224 | 23.044 | 23.391 | 0.985 | 259.07 | 263.81 | -1.832 | 83.333 |
| res_448 | 1 | 4 | 4 | 448 | 80.427 | 82.513 | 0.975 | 704.81 | 720.89 | -2.281 | 83.333 |
| 1frames_large_res | 1 | 4 | 4 | 518 | 122.219 | 119.862 | 1.020 | 904.69 | 926.51 | -2.412 | 83.333 |
| 2frames_large_res | 1 | 2 | 2 | 518 | 52.983 | 51.573 | 1.027 | 508.44 | 520.38 | -2.350 | 83.333 |
| 4frames_large_res | 1 | 4 | 4 | 518 | 116.918 | 116.980 | 0.999 | 904.69 | 926.51 | -2.412 | 83.333 |
| 8frames_large_res | 1 | 8 | 8 | 518 | 337.921 | 338.411 | 0.999 | 1702.07 | 1744.97 | -2.521 | 83.333 |
| 16frames_large_res | 1 | 16 | 16 | 518 | 1152.276 | 1142.385 | 1.009 | 2000.33 | 2086.21 | -4.293 | 83.333 |
| 32frames_large_res | 1 | 32 | 32 | 518 | 4184.606 | 4211.201 | 0.994 | 2603.09 | 2769.25 | -6.383 | 83.333 |
| 64frames_large_res | 1 | 64 | 64 | 518 | 16110.049 | 16185.813 | 0.995 | 5099.16 | 4136.38 | 18.881 | 83.333 |
| 128frames_large_res | 1 | 128 | 128 | 518 | 63139.352 | 63086.383 | 1.001 | 10095.08 | 7516.83 | 25.540 | 83.333 |
| 256frames_large_res | 1 | 256 | 256 | 518 | 250848.637 | 249703.369 | 1.005 | 20085.88 | 14927.38 | 25.682 | 83.333 |

### Takeaways

- Runtime: Optimized path primarily helps at smaller inputs; benefits diminish at higher resolutions.
- Peak VRAM: For long sequences at 518px, optimized mode delivers large peak VRAM savings (up to ~25–26%).
- Storage: Massive, consistent 83% reduction across all runs.

### Recommendations

- Document the crossover behavior: short seqs see slight VRAM increase; long seqs benefit.
- Include GPU model/driver and PyTorch/CUDA versions in CSV or filename.
- Consider memory-efficient attention/Flash-Attn for further large-resolution gains.
- Add throughput metrics (frames/s) alongside latency for easier comparisons.

