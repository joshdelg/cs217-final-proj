# GraphSAGE gather + scatter_add + mean (v1)

This experiment isolates GraphSAGE neighbor aggregation with on-device gather:

- `run_torch.py`: PyTorch baseline (`x[src_padded] -> mask -> sum -> mean`)
- `run_nki.py`: fused NKI kernel path for the same padded-segment representation

## Baseline run (Torch only)

Command used:

```bash
python -m profiler profile graphsage_gather_scatter_mean --mode torch --force
```

Artifacts:
- `artifacts/torch/model.neff`
- `artifacts/torch/profile.ntff`
- `artifacts/torch/summary.json`

Latest Torch kernel time from `summary.json`:
- `total_time ~= 0.001897 s` (about 1.90 ms)

## Profiler-driven observations -> NKI optimization choices

This section documents the actual loop we followed: run Torch, inspect profiler
metrics, form a bottleneck hypothesis, then change NKI.

### 1) Observation from Torch profile (v1)

From `artifacts/torch/summary.json`:
- `total_time ~= 1.897 ms`
- `dma_active_time_percent ~= 98.25%`
- `mbu_estimated_percent ~= 0.078%`
- `hbm_read_bytes ~= 35.7 MB`
- `hbm_write_bytes ~= 70.3 MB`
- `software_dynamic_dma_packet_count ~= 52,128`

Interpretation:
- The workload is strongly **DMA/memory paced** (not tensor-engine limited).
- Very low MBU + many DMA packets suggests command-stream/pacing overhead and
  irregular access behavior dominate more than raw arithmetic throughput.
- High HBM writes indicate heavy intermediate materialization in the staged
  Torch flow (`gather -> mask -> view -> sum -> mean`).

### 2) Optimization hypothesis

If we fuse gather+mask+segment-reduce+mean in one NKI kernel and keep the
partial reduction on-chip, we should:
- ****reduce intermediate HBM round-trips,
- reduce orchestration overhead between separate graph ops,
- improve end-to-end `total_time` even if still DMA-bound.

### 3) NKI design choices that came from those observations

- Keep gather on-device for fairness (same `src_padded/mask_padded` inputs).
- Accumulate in SBUF over degree chunks, store final output once.
- Expose `D_CHUNK` and tile sizes as tunables for DMA granularity/pipelining.
- Prefer `total_time` (device wall time) for optimization decisions, not only
  active-time counters.

### 4) What happened in practice

From `artifacts/compare_report.json`:
- Torch `total_time` mean: `0.001900 s`
- NKI `total_time` mean: `0.001737 s`
- Net: about **1.09x speedup** for NKI in this v1 experiment.

This matched the original hypothesis: gains came mostly from dataflow/overhead
cleanup rather than from higher compute-engine saturation.

## Why the v1 NKI kernel looked the way it did

The v1 NKI kernel is intentionally structured as:
- gather + mask inside the kernel,
- chunked degree reduction in on-chip buffers,
- final mean normalization and store.

That structure came directly from the Torch baseline bottlenecks above: minimize intermediate materialization, reduce overhead around irregular memory access, and expose tunables (`D_CHUNK`, tile sizes) for bandwidth/pacing tradeoffs.

