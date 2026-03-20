# Scatter-Add (NKI) — chunked degree reduction

This experiment profiles a single **scatter-add** / **segment-sum** style reduction on Trainium, comparing:
- `run_torch.py`: pure PyTorch baseline
- `run_nki.py`: NKI kernel with a tuned reduction strategy

The goal is to exercise and optimize the on-device reduction + memory access pattern for the padded segment tensor that both implementations share.

## Data representation (the shared input)

Both `run_torch.py` and `run_nki.py` receive the SAME padded segment matrix:

`seg_values: [NUM_NODES * MAX_DEG, FEAT_DIM]`

For each destination node `n`, its neighborhood values live in the contiguous block:
- `seg_values[n * MAX_DEG : n * MAX_DEG + MAX_DEG, :]`
- rows beyond the true degree are zero-padded, and therefore do not affect sums

The PyTorch baseline reshapes and sums:
- `seg_values.view(NUM_NODES, MAX_DEG, FEAT_DIM).sum(dim=1)`

## Kernel performance change (what we improved)

The NKI kernel (`scatter_add_nki` in `run_nki.py`) improves performance by **changing how it reduces over the degree dimension**:

### Before (conceptually)

Loading and reducing directly over the full padded degree dimension can require more complex indexing patterns for `nl.load` and can lead to less favorable compiler/verifier behavior.

### After (current approach)

We switched to a **chunked degree reduction** scheme:

1. Split `MAX_DEG` into blocks of size `D_CHUNK` (default: `32`, with `MAX_DEG=32` in this experiment).
2. For each chunk, load the slice into an on-chip buffer (SBUF) shaped:
   - `chunk_buf: [TILE_NODES, D_CHUNK, FEAT_DIM]`
3. Immediately reduce across the `D_CHUNK` dimension:
   - `chunk_sum = nl.sum(chunk_buf, axis=1)`
4. Accumulate `chunk_sum` into the output accumulator for the tile.

This is called out in `run_nki.py` as avoiding “complicated 3D indexing for `nl.load`” by loading chunk slices into a structured SBUF buffer and reducing right away.

## Tuning knobs

`run_nki.py` exposes two env vars:
- `SCATTER_ADD_TILE_NODES` (default `128`)
- `SCATTER_ADD_D_CHUNK` (default `32`)

Constraints:
- `MAX_DEG % D_CHUNK == 0` (the kernel asserts this)

Tuning note documented in `run_nki.py`:
- Sweeping `D_CHUNK` over `{4, 8, 16, 32}` with `MAX_DEG=32` showed the best device `total_time` at `D_CHUNK=32`.

## Run

From repo root:

```bash
# Baselines (correctness + quick sanity)
python experiments/scatter_add/run_torch.py
python experiments/scatter_add/run_nki.py

# Profiler compare (device `total_time`)
python -m profiler profile scatter_add --mode compare --trials 3 --no-ingest
```

Optional autotuning:

```bash
python experiments/scatter_add/autotune_scatter_add.py --d-chunks 4,8,16,32 --tile-nodes 128 --no-ingest
```

## Where to look

- NKI kernel: `experiments/scatter_add/run_nki.py` (`scatter_add_nki`)
- Autotuner: `experiments/scatter_add/autotune_scatter_add.py`
- Profiler outputs: `experiments/scatter_add/artifacts/`

