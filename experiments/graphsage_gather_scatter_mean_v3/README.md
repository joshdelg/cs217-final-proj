# GraphSAGE gather+mean (NKI) - Option 3

This directory is the `_v3` follow-up to `graphsage_gather_scatter_mean_v2/`.

## What’s different from `graphsage_gather_scatter_mean_v2`?

`_v2` (Option 2) removed the intermediate `chunk_buf` + `nl.sum` reduction and
accumulated directly inside the `dk` loop:

- gather one neighbor feature tile
- `acc += gathered * mask`

In `_v3` we keep that direct accumulation, but optimize loading while staying
within current NKI constraints:

For each `(node tile, degree chunk)` we now:
1. Load the entire `mask_padded` chunk into SBUF in one shot as a 2D tile
2. Loop over `dk` to do:
   - load `src_idx` from `src_padded` (per `dk`) and gather `x[src_idx, :]`
   - multiply by the already-buffered mask value
   - accumulate into `acc`

Note: NKI currently does *not* allow reusing an SBUF-loaded, sliced index
tensor (“TensorView”) as the indirect gather indices for `nl.load(x[...])`.
So `src_idx` loads remain per-`dk` (same gather pattern as `_v2`).

This targets the remaining control/scalar overhead by reducing mask loads
for the degree chunk, while still minimizing HBM writes.

## Tuning knob

Both versions use `GATHER_MEAN_D_CHUNK` (default: `32`) as the degree chunk size
used by the loop structure.

