# GraphSAGE forward (2-layer) — v5 end-to-end

This experiment profiles a full **2-layer GraphSAGE** forward end-to-end on Trainium2.

We compare:
- `run_torch.py`: fully PyTorch end-to-end (neighbor aggregation + linear transforms + ReLU)
- `run_nki.py`: replaces **only** the **neighbor aggregation** step with the NKI kernel from
  `graphsage_gather_scatter_mean/` (v1/Option-1 style), while keeping the linear layers and ReLU as PyTorch ops.

## Data representation

To ensure the PyTorch and NKI neighbor aggregation see the **same representation**, both scripts use the
padded destination-segment form:
- `src_padded[n*MAX_DEG + k]` = neighbor source index for destination node `n` (0-padded)
- `mask_padded[n*MAX_DEG + k]` = 1 for valid neighbor slots else 0
- `inv_degree[n] = 1 / degree[n]`

Both layers reuse the same `src_padded/mask_padded/inv_degree` (graph structure is fixed).

