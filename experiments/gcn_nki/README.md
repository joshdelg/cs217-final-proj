# GCN with reusable NKI aggregation kernel

This experiment tests whether the same fused gather+reduce NKI kernel used for
GraphSAGE can be reused for a real 2-layer GCN forward pass.

## Architecture

- Paper: Thomas N. Kipf, Max Welling, "Semi-Supervised Classification with
  Graph Convolutional Networks", ICLR 2017.
- Update rule:
  `X' = D_hat^{-1/2} A_hat D_hat^{-1/2} X W`, where `A_hat = A + I`.

## Kernel mapping

The NKI kernel computes:

`out[n] = (sum_k x[src_padded[n*MAX_DEG+k]] * mask_padded[n*MAX_DEG+k]) * inv_degree[n]`

For GCN we map this as:
- `mask_padded = 1 / sqrt(deg(src) * deg(dst))` for each padded edge slot.
- `inv_degree = 1.0` for every node (normalization already in mask).
- Self loops are added to match `A_hat = A + I`.

So the same kernel binary is reused; only preprocessing differs.

## Files

- `run_torch.py`: pure PyTorch 2-layer GCN baseline.
- `run_nki.py`: same model, replacing only aggregation with `gather_mean_nki`.

## Run

```bash
python -m profiler profile gcn_nki --mode compare --trials 3
```

