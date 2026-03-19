# GIN with reusable NKI aggregation kernel

This experiment tests reuse of the same fused gather+reduce NKI kernel for a
2-layer GIN forward pass.

## Architecture

- Paper: Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka,
  "How Powerful are Graph Neural Networks?", ICLR 2019.
- Update rule:
  `h' = MLP((1 + eps) * h_i + sum_{j in N(i)} h_j)`

## Kernel mapping

The NKI kernel computes:

`out[n] = (sum_k x[src_padded[n*MAX_DEG+k]] * mask_padded[n*MAX_DEG+k]) * inv_degree[n]`

For GIN we map this as:
- `mask_padded` is 0/1 (neighbor presence).
- `inv_degree = 1.0` for every node (sum aggregation).
- `(1 + eps) * h_i` and MLPs stay in PyTorch around the kernel call.

So again, same kernel code path, different preprocessing and surrounding ops.

## Files

- `run_torch.py`: pure PyTorch 2-layer GIN baseline.
- `run_nki.py`: same model, replacing only neighbor sum aggregation with NKI.

## Run

```bash
python -m profiler profile gin_nki --mode compare --trials 3
```

