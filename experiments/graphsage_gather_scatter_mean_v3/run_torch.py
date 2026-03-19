"""
GraphSAGE neighbor aggregation (gather + mean) — pure PyTorch.

This matches the padded destination-segment representation used for NKI:
  - CPU builds `src_padded` and `mask_padded` so that for each node n:
        src_padded[n*MAX_DEG + k] is the k-th neighbor source index (padded with 0)
        mask_padded[n*MAX_DEG + k] is 1 for valid neighbors else 0
  - Device profiled region does:
        messages = x[src_padded] * mask_padded
        out = sum_k messages / degree
"""

import os
import math
import torch
from torch_xla.core import xla_model as xm

os.environ.setdefault("NEURON_FRAMEWORK_DEBUG", "1")

NUM_NODES = 4096
AVG_DEGREE = 10
NUM_EDGES = NUM_NODES * AVG_DEGREE
FEAT_DIM = 64
SEED = 42


def generate_random_graph(num_nodes: int, avg_degree: int, seed: int = 42):
    gen = torch.Generator().manual_seed(seed)
    num_edges = num_nodes * avg_degree
    src = torch.randint(0, num_nodes, (num_edges,), generator=gen)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=gen)

    degree = torch.zeros(num_nodes, dtype=torch.float32)
    degree.scatter_add_(0, dst, torch.ones(num_edges, dtype=torch.float32))
    degree.clamp_(min=1.0)
    return src, dst, degree


def build_src_padded(src: torch.Tensor, dst: torch.Tensor, degree: torch.Tensor, num_nodes: int, max_deg: int):
    """Build src_padded + mask_padded for destination segments.

    This matches `graphsage_gather_scatter_mean_v2/run_torch.py` exactly:
      - sort edges by destination so each node's neighbors are contiguous
      - build CSR `rowptr` over the sorted destinations
      - fill a padded `[num_nodes * max_deg]` destination-segment matrix
    """
    perm = dst.argsort(stable=True)
    src_sorted = src[perm]
    dst_sorted = dst[perm]

    rowptr = torch.zeros(num_nodes + 1, dtype=torch.int64)
    ones = torch.ones_like(dst_sorted, dtype=torch.int64)
    rowptr[1:].scatter_add_(0, dst_sorted.to(torch.int64), ones)
    rowptr = rowptr.cumsum(0)

    src_padded = torch.zeros(num_nodes * max_deg, dtype=torch.int64)
    mask_padded = torch.zeros(num_nodes * max_deg, dtype=torch.float32)

    for n in range(num_nodes):
        start = int(rowptr[n].item())
        end = int(rowptr[n + 1].item())
        deg_n = end - start
        if deg_n > 0:
            row_start = n * max_deg
            src_padded[row_start : row_start + deg_n] = src_sorted[start:end].to(torch.int64)
            mask_padded[row_start : row_start + deg_n] = 1.0

    return src_padded, mask_padded


def main():
    device = xm.xla_device()
    torch.manual_seed(SEED)

    # ---- build inputs on CPU ----
    src, dst, degree = generate_random_graph(NUM_NODES, AVG_DEGREE, SEED)
    x = torch.randn(NUM_NODES, FEAT_DIM, dtype=torch.float32)

    raw_max_deg = int(degree.max().item())
    max_deg = 1 << int(math.ceil(math.log2(max(raw_max_deg, 1))))

    src_padded, mask_padded = build_src_padded(src, dst, degree, NUM_NODES, max_deg)
    inv_degree = 1.0 / degree

    # ---- move to device and materialize ----
    x = x.to(device)
    src_padded = src_padded.to(device)
    mask_padded = mask_padded.to(device)
    inv_degree = inv_degree.to(device)
    xm.mark_step()
    xm.wait_device_ops()

    # ---- profiled aggregation ----
    messages = x[src_padded]  # [NUM_NODES * MAX_DEG, FEAT_DIM]
    messages = messages * mask_padded[:, None]

    messages_3d = messages.view(NUM_NODES, max_deg, FEAT_DIM)
    agg = messages_3d.sum(dim=1)  # [NUM_NODES, FEAT_DIM]
    out = agg * inv_degree[:, None]

    xm.mark_step()
    xm.wait_device_ops()

    out_cpu_sum = out.cpu().sum().item()
    print("output shape:", out.shape)
    print("output sample:", out.cpu()[0, :4].tolist())
    print("output sum:", out_cpu_sum)


if __name__ == "__main__":
    main()

