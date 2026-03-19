# GNN kernel generalization summary

Goal: test a "mock compiler pass" idea by reusing one optimized NKI
gather+reduce kernel across multiple real GNN architectures.

## Architectures and citations

- GraphSAGE (mean aggregation): Hamilton et al., NeurIPS 2017  
  <https://arxiv.org/abs/1706.02216>
- GCN (degree-normalized aggregation): Kipf and Welling, ICLR 2017  
  <https://arxiv.org/abs/1609.02907>
- GIN (sum aggregation + MLP): Xu et al., ICLR 2019  
  <https://arxiv.org/abs/1810.00826>

## One kernel, different mappings

Reused kernel form:

`out[n] = (sum_k x[src_padded[n*MAX_DEG+k]] * mask_padded[n*MAX_DEG+k]) * inv_degree[n]`

Architecture-specific mapping:

| Architecture | `mask_padded` | `inv_degree` | Notes |
|---|---|---|---|
| GraphSAGE mean | `0/1` | `1/deg(n)` | Existing v5 end-to-end setup |
| GCN | `1/sqrt(deg(src)*deg(dst))` | `1.0` | Uses `A_hat = A + I` with self loops |
| GIN | `0/1` | `1.0` | Self term `(1+eps)*x` and MLP outside kernel |

This means the NKI kernel code path stays the same; only preprocessing and the
surrounding PyTorch ops change.

## Experimental setup

- Device/profile metric: `total_time` from `neuron-profile` summaries
- Mode: `python -m profiler profile <exp> --mode compare --trials 3 --force`
- Experiments:
  - `graphsage_gather_scatter_mean_v5`
  - `gcn_nki`
  - `gin_nki`

## Results (total_time)

| Architecture | Torch mean (s) | NKI mean (s) | Speedup (Torch/NKI) |
|---|---:|---:|---:|
| GraphSAGE v5 | 0.00592249 | 0.00353130 | 1.677x |
| GCN | 0.00518681 | 0.00350997 | 1.478x |
| GIN | 0.00522325 | 0.00352494 | 1.482x |

## Interpretation

- The same optimized aggregation kernel delivers consistent gains across three
  different, cited GNN architectures.
- This supports the "mock compiler pass" hypothesis: if a compiler can detect
  gather+weighted-reduce style subgraphs and substitute this kernel, it can
  improve end-to-end runtime without architecture-specific kernels.
- Practical caveat: robust compiler integration still needs graph-pattern
  matching, legality checks (dtype/layout/shape constraints), and fallback
  paths when assumptions do not hold.

## Relevant files

- GraphSAGE reference:
  - `experiments/graphsage_gather_scatter_mean_v5/run_torch.py`
  - `experiments/graphsage_gather_scatter_mean_v5/run_nki.py`
- GCN:
  - `experiments/gcn_nki/run_torch.py`
  - `experiments/gcn_nki/run_nki.py`
- GIN:
  - `experiments/gin_nki/run_torch.py`
  - `experiments/gin_nki/run_nki.py`

