"""
GraphSAGE forward (2-layer) — pure PyTorch baseline.

We express mean neighbor aggregation using padded destination segments:
  - messages = x[src_padded] * mask_padded[:, None]
  - agg = messages.view(NUM_NODES, MAX_DEG, F).sum(dim=1)
  - out = agg * inv_degree[:, None]

Then each GraphSAGE layer does:
  h'_neigh = W_neigh @ agg + b
  h'_self  = W_self  @ x
  h' = h'_neigh + h'_self
  (ReLU after layer 1)
"""

import os
import math
import torch
import torch.nn.functional as F
from torch_xla.core import xla_model as xm

os.environ.setdefault("NEURON_FRAMEWORK_DEBUG", "1")

NUM_NODES = 4096
AVG_DEGREE = 10
IN_FEATURES = 64
HIDDEN = 64
OUT_CLASSES = 16
SEED = 42

TWO = 2


def generate_random_graph(num_nodes: int, avg_degree: int, seed: int = 42):
    gen = torch.Generator().manual_seed(seed)
    num_edges = num_nodes * avg_degree
    src = torch.randint(0, num_nodes, (num_edges,), generator=gen)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=gen)

    degree = torch.zeros(num_nodes, dtype=torch.float32)
    degree.scatter_add_(0, dst, torch.ones(num_edges, dtype=torch.float32))
    degree.clamp_(min=1.0)
    inv_degree = 1.0 / degree
    return src, dst, degree, inv_degree


def build_src_padded(src: torch.Tensor, dst: torch.Tensor, num_nodes: int, max_deg: int):
    """Build padded destination segments: src_padded + mask_padded."""
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


def neighbor_agg_mean_pytorch(x: torch.Tensor, src_padded: torch.Tensor, mask_padded: torch.Tensor, inv_degree: torch.Tensor, max_deg: int):
    """Mean aggregation for all nodes, using the padded segment representation."""
    # messages: [NUM_NODES * MAX_DEG, F]
    messages = x[src_padded] * mask_padded[:, None]
    messages_3d = messages.view(NUM_NODES, max_deg, x.shape[1])
    agg = messages_3d.sum(dim=1)
    return agg * inv_degree[:, None]


def main():
    device = xm.xla_device()
    torch.manual_seed(SEED)

    # ---- build graph + inputs on CPU ----
    src, dst, degree, inv_degree = generate_random_graph(NUM_NODES, AVG_DEGREE, SEED)
    x0 = torch.randn(NUM_NODES, IN_FEATURES, dtype=torch.float32)

    raw_max_deg = int(degree.max().item())
    max_deg = 1 << int(math.ceil(math.log2(max(raw_max_deg, 1))))

    src_padded_cpu, mask_padded_cpu = build_src_padded(src, dst, NUM_NODES, max_deg)

    # ---- weights (match original graphsage parameters) ----
    W_neigh1 = torch.randn(HIDDEN, IN_FEATURES, dtype=torch.float32) * 0.01
    W_self1 = torch.randn(HIDDEN, IN_FEATURES, dtype=torch.float32) * 0.01
    b1 = torch.zeros(HIDDEN, dtype=torch.float32)

    W_neigh2 = torch.randn(OUT_CLASSES, HIDDEN, dtype=torch.float32) * 0.01
    W_self2 = torch.randn(OUT_CLASSES, HIDDEN, dtype=torch.float32) * 0.01
    b2 = torch.zeros(OUT_CLASSES, dtype=torch.float32)

    # ---- move to device ----
    x = x0.to(device)
    src_padded = src_padded_cpu.to(device)
    mask_padded = mask_padded_cpu.to(device)
    inv_degree_dev = inv_degree.to(device)

    W_neigh1 = W_neigh1.to(device)
    W_self1 = W_self1.to(device)
    b1 = b1.to(device)
    W_neigh2 = W_neigh2.to(device)
    W_self2 = W_self2.to(device)
    b2 = b2.to(device)

    xm.mark_step()
    xm.wait_device_ops()

    # ---- profiled forward ----
    agg1 = neighbor_agg_mean_pytorch(x, src_padded, mask_padded, inv_degree_dev, max_deg)
    h1 = F.linear(agg1, W_neigh1, b1) + F.linear(x, W_self1)
    h1 = F.relu(h1)

    agg2 = neighbor_agg_mean_pytorch(h1, src_padded, mask_padded, inv_degree_dev, max_deg)
    out = F.linear(agg2, W_neigh2, b2) + F.linear(h1, W_self2)

    xm.mark_step()
    xm.wait_device_ops()

    out_cpu_sum = out.cpu().sum().item()
    print("output shape:", out.shape)
    print("output sample:", out.cpu()[0, :4].tolist())
    print("output sum:", out_cpu_sum)


if __name__ == "__main__":
    main()

