# GraphSAGE gather+mean (NKI) - Option 2

This directory is the `_v2` follow-up to `graphsage_gather_scatter_mean/`.

## What’s different from `graphsage_gather_scatter_mean`?

Both experiments profile a GraphSAGE neighbor aggregation expressed as:

1. CPU builds padded destination segments:
   - `src_padded[n*MAX_DEG + k]` contains the `k`-th source node index for destination node `n` (padded with 0)
   - `mask_padded[n*MAX_DEG + k]` is 1 for valid neighbors and 0 for padded slots
2. The profiled workload performs a *gather + mask + segment sum + mean*.

### `_v1` (Option 1)
`run_nki.py` used a two-step approach per `(node tile, degree chunk)`:
- Gather/mask `D_CHUNK` neighbors into an SBUF `chunk_buf`
- Reduce with a single `nl.sum(chunk_buf, axis=1)` over the `D_CHUNK` dimension

### `_v2` (Option 2)
`run_nki.py` switches to a single-step accumulation:
- Gather one `dk` neighbor at a time
- Immediately accumulate: `acc += gathered * mask`
- Eliminate the intermediate `chunk_buf` and the subsequent `nl.sum`

## Tuning knob

Both versions use `GATHER_MEAN_D_CHUNK` (default: `4`) as the degree chunk size used by the loop structure.

Autotune sweep over `D_CHUNK={4,8,16,32}` found the best `total_time` at `D_CHUNK=4` (differences are small; DMA traffic was essentially unchanged).

