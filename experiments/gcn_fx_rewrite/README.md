# GCN with `torch.fx` auto-rewrite to NKI

This experiment demonstrates Approach B: detect a Torch aggregation pattern in an FX graph and replace it with the NKI aggregation kernel call.

## Goal

Start from a Torch-style aggregation expression:

- `messages = x[src_padded] * norm_padded[:, None]`
- `agg = messages.view(NUM_NODES, MAX_DEG, F).sum(dim=1)`
- `out = agg * inv_degree[:, None]`

Then rewrite this subgraph via `torch.fx` into:

- `gather_mean_nki(x, src_padded, norm_padded, inv_degree)`

## Files

- `run_torch.py`: pure Torch 2-layer GCN baseline.
- `run_nki.py`: same model, but aggregation is produced by FX rewrite and lowered to NKI kernel calls.

## Run

From repo root:

```bash
python -m profiler profile gcn_fx_rewrite --mode compare --trials 3
```

To inspect the rewrite-only path:

```bash
python experiments/gcn_fx_rewrite/run_nki.py
```

