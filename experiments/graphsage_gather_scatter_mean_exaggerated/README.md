# GraphSAGE gather+scatter+mean (exaggerated)

This experiment is the same core operation as `graphsage_gather_scatter_mean`, but with defaults tuned to amplify aggregation costs so custom-kernel gains are easier to observe end-to-end.

## Why these defaults

Default shape knobs:

- `GMM_NUM_NODES=8192`
- `GMM_AVG_DEGREE=24`
- `GMM_FEAT_DIM=128`

These increase the volume of gather + masked reduction work while keeping the experiment fair:

- Torch and NKI both perform gather on device.
- Both use the same padded segment representation (`src_padded`, `mask_padded`, `inv_degree`).
- Both compute the same output (`mean` aggregation).

## Files

- `run_torch.py`: pure Torch baseline for fused gather+mask+sum+mean.
- `run_nki.py`: NKI kernel path (`gather_mean_nki`) with Torch reference check.

## Tunables

Workload:

- `GMM_NUM_NODES` (default `8192`)
- `GMM_AVG_DEGREE` (default `24`)
- `GMM_FEAT_DIM` (default `128`)
- `GMM_SEED` (default `42`)

NKI kernel:

- `GMM_TILE_NODES` (default `128`, must divide `NUM_NODES`)
- `GMM_D_CHUNK` (default `32`, must divide `MAX_DEG`)

## Run

From repo root:

```bash
python -m profiler profile graphsage_gather_scatter_mean_exaggerated --mode compare --trials 3
```

For quick direct runs:

```bash
python experiments/graphsage_gather_scatter_mean_exaggerated/run_torch.py
python experiments/graphsage_gather_scatter_mean_exaggerated/run_nki.py
```

## Notes

- This experiment intentionally stresses aggregation bandwidth/packet behavior to make custom-kernel impact easier to see.
- Keep comparisons fair by changing shape knobs for both implementations together.
