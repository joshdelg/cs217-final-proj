"""
2-layer GIN baseline (pure PyTorch) on Trainium.

GIN aggregation uses an unweighted sum:
  agg[n] = sum_{j in N(n)} x[j]
  h' = MLP((1 + eps) * x + agg)
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
EPS = 0.1


def generate_random_graph(num_nodes: int, avg_degree: int, seed: int = 42):
    gen = torch.Generator().manual_seed(seed)
    num_edges = num_nodes * avg_degree
    src = torch.randint(0, num_nodes, (num_edges,), generator=gen, dtype=torch.int64)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=gen, dtype=torch.int64)
    return src, dst


def build_src_padded(src: torch.Tensor, dst: torch.Tensor, num_nodes: int, max_deg: int):
    perm = dst.argsort(stable=True)
    src_sorted = src[perm]
    dst_sorted = dst[perm]

    rowptr = torch.zeros(num_nodes + 1, dtype=torch.int64)
    ones = torch.ones_like(dst_sorted, dtype=torch.int64)
    rowptr[1:].scatter_add_(0, dst_sorted, ones)
    rowptr = rowptr.cumsum(0)

    src_padded = torch.zeros(num_nodes * max_deg, dtype=torch.int64)
    mask_padded = torch.zeros(num_nodes * max_deg, dtype=torch.float32)

    for n in range(num_nodes):
        start = int(rowptr[n].item())
        end = int(rowptr[n + 1].item())
        deg_n = end - start
        if deg_n > 0:
            row_start = n * max_deg
            src_padded[row_start:row_start + deg_n] = src_sorted[start:end]
            mask_padded[row_start:row_start + deg_n] = 1.0

    return src_padded, mask_padded


def gin_aggregate_torch(x: torch.Tensor, src_padded: torch.Tensor, mask_padded: torch.Tensor, max_deg: int):
    messages = x[src_padded] * mask_padded[:, None]
    messages_3d = messages.view(NUM_NODES, max_deg, x.shape[1])
    return messages_3d.sum(dim=1)


def mlp_block(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor):
    return F.linear(F.relu(F.linear(x, w1, b1)), w2, b2)


def main():
    device = xm.xla_device()
    torch.manual_seed(SEED)

    src, dst = generate_random_graph(NUM_NODES, AVG_DEGREE, SEED)
    x0 = torch.randn(NUM_NODES, IN_FEATURES, dtype=torch.float32)

    deg_i64 = torch.zeros(NUM_NODES, dtype=torch.int64)
    deg_i64.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.int64))
    raw_max_deg = int(deg_i64.max().item())
    max_deg = 1 << int(math.ceil(math.log2(max(raw_max_deg, 1))))

    src_padded_cpu, mask_padded_cpu = build_src_padded(src, dst, NUM_NODES, max_deg)

    # Layer 1 MLP: 64 -> 64 -> 64
    l1_w1 = torch.randn(HIDDEN, IN_FEATURES, dtype=torch.float32) * 0.01
    l1_b1 = torch.zeros(HIDDEN, dtype=torch.float32)
    l1_w2 = torch.randn(HIDDEN, HIDDEN, dtype=torch.float32) * 0.01
    l1_b2 = torch.zeros(HIDDEN, dtype=torch.float32)

    # Layer 2 MLP: 64 -> 64 -> 16
    l2_w1 = torch.randn(HIDDEN, HIDDEN, dtype=torch.float32) * 0.01
    l2_b1 = torch.zeros(HIDDEN, dtype=torch.float32)
    l2_w2 = torch.randn(OUT_CLASSES, HIDDEN, dtype=torch.float32) * 0.01
    l2_b2 = torch.zeros(OUT_CLASSES, dtype=torch.float32)

    x = x0.to(device)
    src_padded = src_padded_cpu.to(device)
    mask_padded = mask_padded_cpu.to(device)
    l1_w1 = l1_w1.to(device)
    l1_b1 = l1_b1.to(device)
    l1_w2 = l1_w2.to(device)
    l1_b2 = l1_b2.to(device)
    l2_w1 = l2_w1.to(device)
    l2_b1 = l2_b1.to(device)
    l2_w2 = l2_w2.to(device)
    l2_b2 = l2_b2.to(device)

    xm.mark_step()
    xm.wait_device_ops()

    # ---- profiled forward ----
    agg1 = gin_aggregate_torch(x, src_padded, mask_padded, max_deg)
    h1_in = (1.0 + EPS) * x + agg1
    h1 = mlp_block(h1_in, l1_w1, l1_b1, l1_w2, l1_b2)

    agg2 = gin_aggregate_torch(h1, src_padded, mask_padded, max_deg)
    h2_in = (1.0 + EPS) * h1 + agg2
    out = mlp_block(h2_in, l2_w1, l2_b1, l2_w2, l2_b2)

    xm.mark_step()
    xm.wait_device_ops()

    out_cpu_sum = out.cpu().sum().item()
    print("output shape:", out.shape)
    print("output sample:", out.cpu()[0, :4].tolist())
    print("output sum:", out_cpu_sum)


if __name__ == "__main__":
    main()

