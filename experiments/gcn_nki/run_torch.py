"""
2-layer GCN baseline (pure PyTorch) on Trainium.

Aggregation pattern:
  out[n] = sum_k x[src_padded[n*MAX_DEG+k], :] * norm_padded[n*MAX_DEG+k]

where norm_padded stores symmetric GCN normalization weights:
  norm_{j->i} = 1 / sqrt(deg(i) * deg(j))
for A_hat = A + I.
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


def generate_random_graph(num_nodes: int, avg_degree: int, seed: int = 42):
    gen = torch.Generator().manual_seed(seed)
    num_edges = num_nodes * avg_degree
    src = torch.randint(0, num_nodes, (num_edges,), generator=gen, dtype=torch.int64)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=gen, dtype=torch.int64)

    # GCN uses A_hat = A + I, so add self loops explicitly.
    nodes = torch.arange(num_nodes, dtype=torch.int64)
    src = torch.cat([src, nodes], dim=0)
    dst = torch.cat([dst, nodes], dim=0)
    return src, dst


def build_gcn_padded(src: torch.Tensor, dst: torch.Tensor, num_nodes: int, max_deg: int):
    """Build padded src indices and symmetric GCN normalization weights."""
    # Degree over destination index for A_hat.
    deg = torch.zeros(num_nodes, dtype=torch.float32)
    deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    deg.clamp_(min=1.0)

    norm = 1.0 / torch.sqrt(deg[src] * deg[dst])  # per edge j->i

    # Group edges by destination so each node is a contiguous segment.
    perm = dst.argsort(stable=True)
    src_sorted = src[perm]
    dst_sorted = dst[perm]
    norm_sorted = norm[perm]

    rowptr = torch.zeros(num_nodes + 1, dtype=torch.int64)
    ones = torch.ones_like(dst_sorted, dtype=torch.int64)
    rowptr[1:].scatter_add_(0, dst_sorted, ones)
    rowptr = rowptr.cumsum(0)

    src_padded = torch.zeros(num_nodes * max_deg, dtype=torch.int64)
    norm_padded = torch.zeros(num_nodes * max_deg, dtype=torch.float32)

    for n in range(num_nodes):
        start = int(rowptr[n].item())
        end = int(rowptr[n + 1].item())
        deg_n = end - start
        if deg_n > 0:
            row_start = n * max_deg
            src_padded[row_start:row_start + deg_n] = src_sorted[start:end]
            norm_padded[row_start:row_start + deg_n] = norm_sorted[start:end]

    return src_padded, norm_padded


def gcn_aggregate_torch(x: torch.Tensor, src_padded: torch.Tensor, norm_padded: torch.Tensor, max_deg: int):
    messages = x[src_padded] * norm_padded[:, None]
    messages_3d = messages.view(NUM_NODES, max_deg, x.shape[1])
    return messages_3d.sum(dim=1)


def main():
    device = xm.xla_device()
    torch.manual_seed(SEED)

    src, dst = generate_random_graph(NUM_NODES, AVG_DEGREE, SEED)
    x0 = torch.randn(NUM_NODES, IN_FEATURES, dtype=torch.float32)

    # MAX_DEG from destination in-degree under A_hat.
    deg = torch.zeros(NUM_NODES, dtype=torch.int64)
    deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.int64))
    raw_max_deg = int(deg.max().item())
    max_deg = 1 << int(math.ceil(math.log2(max(raw_max_deg, 1))))

    src_padded_cpu, norm_padded_cpu = build_gcn_padded(src, dst, NUM_NODES, max_deg)

    # GCN weights.
    W1 = torch.randn(HIDDEN, IN_FEATURES, dtype=torch.float32) * 0.01
    b1 = torch.zeros(HIDDEN, dtype=torch.float32)
    W2 = torch.randn(OUT_CLASSES, HIDDEN, dtype=torch.float32) * 0.01
    b2 = torch.zeros(OUT_CLASSES, dtype=torch.float32)

    x = x0.to(device)
    src_padded = src_padded_cpu.to(device)
    norm_padded = norm_padded_cpu.to(device)
    W1 = W1.to(device)
    b1 = b1.to(device)
    W2 = W2.to(device)
    b2 = b2.to(device)

    xm.mark_step()
    xm.wait_device_ops()

    # ---- profiled forward ----
    agg1 = gcn_aggregate_torch(x, src_padded, norm_padded, max_deg)
    h1 = F.linear(agg1, W1, b1)
    h1 = F.relu(h1)

    agg2 = gcn_aggregate_torch(h1, src_padded, norm_padded, max_deg)
    out = F.linear(agg2, W2, b2)

    xm.mark_step()
    xm.wait_device_ops()

    out_cpu_sum = out.cpu().sum().item()
    print("output shape:", out.shape)
    print("output sample:", out.cpu()[0, :4].tolist())
    print("output sum:", out_cpu_sum)


if __name__ == "__main__":
    main()

