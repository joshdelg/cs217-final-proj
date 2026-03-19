# GraphSAGE gather+mean (NKI) - v4 (src-block grouped ordering)

This experiment is a lighter-weight `_v4` step toward “2D blocking”.

## What changed vs `graphsage_gather_scatter_mean_v2`?

`v2` and `v1` both build a padded destination-segment representation:
- `src_padded[n * MAX_DEG + k]` holds the `k`-th neighbor source index for destination node `n`
- `mask_padded[n * MAX_DEG + k]` is 1 for valid slots else 0

In `v4`, we **reorder neighbor slots inside each destination segment** on the CPU by:
- `src_block = src // SRC_BLOCK_SIZE` (default `128`)
- secondary sort by `src_block` (and then stable by `src` / edge slot)

Then both `run_torch.py` and `run_nki.py` consume the **exact same** reordered `src_padded` and `mask_padded`.

## Why this approach first?

Full “load `x` blocks into SBUF and gather from SBUF” requires more aggressive refactoring and hits current NKI indexing constraints. Reordering is a low-risk way to introduce source-locality that may improve DMA coalescing without changing the NKI kernel’s structure.

## Kernel / tuning knob

- NKI uses the `v2` Option-2 style fused accumulation.
- `GATHER_MEAN_D_CHUNK` default is `4` (tuned for the earlier v2 kernel).

