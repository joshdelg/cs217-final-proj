"""
Torch baseline for fused gather + mask + segment sum + mean (exaggerated setting).

This keeps gather on-device for fairness:
  messages = x[src_padded]
  messages *= mask_padded[:, None]
  out = messages.view(NUM_NODES, MAX_DEG, FEAT_DIM).sum(dim=1) * inv_degree[:, None]
"""

import math
import os
import torch
from torch_xla.core import xla_model as xm

os.environ.setdefault("NEURON_FRAMEWORK_DEBUG", "1")

# Tunable workload shape (defaults chosen to emphasize aggregation bottleneck).
NUM_NODES = int(os.getenv("GMM_NUM_NODES", "8192"))
AVG_DEGREE = int(os.getenv("GMM_AVG_DEGREE", "24"))
FEAT_DIM = int(os.getenv("GMM_FEAT_DIM", "128"))
SEED = int(os.getenv("GMM_SEED", "42"))


def generate_random_graph(num_nodes: int, avg_degree: int, seed: int = 42):
    gen = torch.Generator().manual_seed(seed)
    num_edges = num_nodes * avg_degree
    src = torch.randint(0, num_nodes, (num_edges,), generator=gen, dtype=torch.int64)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=gen, dtype=torch.int64)
    degree = torch.zeros(num_nodes, dtype=torch.float32)
    degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    degree.clamp_(min=1.0)
    inv_degree = 1.0 / degree
    return src, dst, degree, inv_degree


def build_src_padded(src: torch.Tensor, dst: torch.Tensor, num_nodes: int, max_deg: int):
    perm = dst.argsort(stable=True)
    src_sorted = src[perm]
    dst_sorted = dst[perm]

    rowptr = torch.zeros(num_nodes + 1, dtype=torch.int64)
    rowptr[1:].scatter_add_(0, dst_sorted, torch.ones_like(dst_sorted, dtype=torch.int64))
    rowptr = rowptr.cumsum(0)

    src_padded = torch.zeros(num_nodes * max_deg, dtype=torch.int64)
    mask_padded = torch.zeros(num_nodes * max_deg, dtype=torch.float32)
    for n in range(num_nodes):
        start = int(rowptr[n].item())
        end = int(rowptr[n + 1].item())
        deg_n = end - start
        if deg_n > 0:
            off = n * max_deg
            src_padded[off:off + deg_n] = src_sorted[start:end]
            mask_padded[off:off + deg_n] = 1.0
    return src_padded, mask_padded


def main():
    device = xm.xla_device()
    torch.manual_seed(SEED)

    src, dst, degree, inv_degree = generate_random_graph(NUM_NODES, AVG_DEGREE, SEED)
    x_cpu = torch.randn(NUM_NODES, FEAT_DIM, dtype=torch.float32)

    raw_max_deg = int(degree.max().item())
    max_deg = 1 << int(math.ceil(math.log2(max(raw_max_deg, 1))))
    src_padded_cpu, mask_padded_cpu = build_src_padded(src, dst, NUM_NODES, max_deg)

    x = x_cpu.to(device)
    src_padded = src_padded_cpu.to(device)
    mask_padded = mask_padded_cpu.to(device)
    inv_degree = inv_degree.to(device)

    xm.mark_step()
    xm.wait_device_ops()

    # ---- profiled aggregation ----
    messages = x[src_padded]
    messages = messages * mask_padded[:, None]
    messages_3d = messages.view(NUM_NODES, max_deg, FEAT_DIM)
    agg = messages_3d.sum(dim=1)
    out = agg * inv_degree[:, None]

    xm.mark_step()
    xm.wait_device_ops()

    out_cpu_sum = out.cpu().sum().item()
    print("shape config:", {"NUM_NODES": NUM_NODES, "AVG_DEGREE": AVG_DEGREE, "FEAT_DIM": FEAT_DIM, "MAX_DEG": max_deg})
    print("output shape:", out.shape)
    print("output sample:", out.cpu()[0, :4].tolist())
    print("output sum:", out_cpu_sum)


if __name__ == "__main__":
    main()

