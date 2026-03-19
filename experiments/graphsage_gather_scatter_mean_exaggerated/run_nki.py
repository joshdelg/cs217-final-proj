"""
NKI implementation for fused gather + mask + segment sum + mean (exaggerated setting).

Kernel form:
  out[n] = (sum_k x[src_padded[n*MAX_DEG+k], :] * mask_padded[n*MAX_DEG+k]) * inv_degree[n]
"""

import math
import os
import torch
from torch_xla.core import xla_model as xm

os.environ.setdefault("NEURON_FRAMEWORK_DEBUG", "1")
os.environ.setdefault("XLA_IR_DEBUG", "1")
os.environ.setdefault("XLA_HLO_DEBUG", "1")

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
except ImportError:
    raise ImportError("run_nki.py requires neuronxcc (Neuron venv)")

# Tunable workload shape (defaults chosen to emphasize aggregation bottleneck).
NUM_NODES = int(os.getenv("GMM_NUM_NODES", "8192"))
AVG_DEGREE = int(os.getenv("GMM_AVG_DEGREE", "24"))
FEAT_DIM = int(os.getenv("GMM_FEAT_DIM", "128"))
SEED = int(os.getenv("GMM_SEED", "42"))

TILE_NODES = int(os.getenv("GMM_TILE_NODES", "128"))
GATHER_MEAN_D_CHUNK = int(os.getenv("GMM_D_CHUNK", "32"))
MAX_DEG = 32  # overwritten in main()


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

    src_padded = torch.zeros(num_nodes * max_deg, dtype=torch.int32)
    mask_padded = torch.zeros(num_nodes * max_deg, dtype=torch.float32)
    for n in range(num_nodes):
        start = int(rowptr[n].item())
        end = int(rowptr[n + 1].item())
        deg_n = end - start
        if deg_n > 0:
            off = n * max_deg
            src_padded[off:off + deg_n] = src_sorted[start:end].to(torch.int32)
            mask_padded[off:off + deg_n] = 1.0
    return src_padded, mask_padded


@nki.jit
def gather_mean_nki(x, src_padded, mask_padded, inv_degree):
    d_chunk = GATHER_MEAN_D_CHUNK
    assert MAX_DEG % d_chunk == 0, "MAX_DEG must be divisible by GMM_D_CHUNK"
    assert NUM_NODES % TILE_NODES == 0, "NUM_NODES must be divisible by GMM_TILE_NODES"

    out = nl.ndarray((NUM_NODES, FEAT_DIM), dtype=nl.float32, buffer=nl.shared_hbm)
    i_p = nl.arange(TILE_NODES)[:, None]
    i_f = nl.arange(FEAT_DIM)[None, :]
    num_node_tiles = NUM_NODES // TILE_NODES

    for nt in nl.affine_range(num_node_tiles):
        acc = nl.zeros((TILE_NODES, FEAT_DIM), dtype=nl.float32, buffer=nl.sbuf)
        node_base = nt * TILE_NODES
        for c in nl.affine_range(MAX_DEG // d_chunk):
            chunk_buf = nl.ndarray((TILE_NODES, d_chunk, FEAT_DIM), dtype=nl.float32, buffer=nl.sbuf)
            for dk in nl.affine_range(d_chunk):
                k = c * d_chunk + dk
                row_idx = (node_base + i_p) * MAX_DEG + k
                src_idx = nl.load(src_padded[row_idx])
                m = nl.load(mask_padded[row_idx])
                gathered = nl.load(x[src_idx, i_f])
                chunk_buf[i_p, dk, i_f] = gathered * m
            acc += nl.sum(chunk_buf, axis=1, keepdims=False)
        inv_tile = nl.load(inv_degree[node_base + i_p])
        nl.store(out[node_base + i_p, i_f], value=acc * inv_tile)
    return out


def main():
    global MAX_DEG

    device = xm.xla_device()
    torch.manual_seed(SEED)

    src, dst, degree, inv_degree = generate_random_graph(NUM_NODES, AVG_DEGREE, SEED)
    x_cpu = torch.randn(NUM_NODES, FEAT_DIM, dtype=torch.float32)

    raw_max_deg = int(degree.max().item())
    MAX_DEG = 1 << int(math.ceil(math.log2(max(raw_max_deg, 1))))
    src_padded_cpu, mask_padded_cpu = build_src_padded(src, dst, NUM_NODES, MAX_DEG)

    # CPU reference
    with torch.no_grad():
        messages = x_cpu[src_padded_cpu.long()] * mask_padded_cpu[:, None]
        ref = messages.view(NUM_NODES, MAX_DEG, FEAT_DIM).sum(dim=1) * inv_degree[:, None]

    x = x_cpu.to(device)
    src_padded = src_padded_cpu.to(device)
    mask_padded = mask_padded_cpu.to(device)
    inv_degree = inv_degree.to(device)

    xm.mark_step()
    xm.wait_device_ops()

    # ---- profiled aggregation ----
    out = gather_mean_nki(x, src_padded, mask_padded, inv_degree)

    xm.mark_step()
    xm.wait_device_ops()

    out_cpu = out.cpu()
    diff = (out_cpu - ref).abs().max().item()
    out_cpu_sum = out_cpu.sum().item()
    print("shape config:", {"NUM_NODES": NUM_NODES, "AVG_DEGREE": AVG_DEGREE, "FEAT_DIM": FEAT_DIM, "MAX_DEG": MAX_DEG})
    print("kernel config:", {"TILE_NODES": TILE_NODES, "D_CHUNK": GATHER_MEAN_D_CHUNK})
    print("output shape:", out.shape)
    print("output sample:", out_cpu[0, :4].tolist())
    print("max abs diff vs torch ref:", diff)
    print("output sum:", out_cpu_sum)


if __name__ == "__main__":
    main()

