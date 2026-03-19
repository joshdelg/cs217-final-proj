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

## HLO graph insights (before vs after optimizations)

The generated HLO images are useful for understanding graph-level optimization boundaries:
- `artifacts/torch/hlo_before_optimizations.png`
- `artifacts/torch/hlo_after_optimizations.png`
- `artifacts/nki/hlo_before_optimizations.png`
- `artifacts/nki/hlo_after_optimizations.png`

What we observed in this experiment:
- **Torch after-optimization graph is still larger** than NKI (roughly 59 vs 30 ENTRY instructions in our run), indicating more graph plumbing and less structural consolidation around neighbor aggregation.
- **Torch shows explicit `gather -> reduce -> multiply(inv_degree)` patterns** per layer, plus surrounding reshape/broadcast/index-canonicalization nodes.
- **NKI after-optimization graph shows `custom-call` nodes** where neighbor aggregation is encapsulated, with a cleaner surrounding graph for linear/ReLU stages.
- The extra nodes before Torch gather are mainly **index handling/canonicalization** from advanced indexing lowering (not additional model math).

Important caveat about interpreting HLO:
- HLO graphs show **what operations remain at graph level** after XLA optimization.
- HLO graphs do **not directly expose final low-level scheduling details** such as tile sizes, degree/node chunking, DMA packetization, or software pipelining strategy.
- Those lower-level choices must be inferred from profiler traces/counters (`neuron-profile` timeline + DMA/engine metrics), not from HLO topology alone.

Practical takeaway:
- Use HLO to identify missing high-level fusion and extra intermediates.
- Use profile metrics to validate whether backend lowering achieved efficient tiling/chunking in practice.

