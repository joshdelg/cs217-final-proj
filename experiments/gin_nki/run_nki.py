"""
2-layer GIN with NKI aggregation + PyTorch MLP blocks.

Kernel mapping for GIN:
  - mask_padded is 0/1 neighbor presence
  - inv_degree is ones (sum aggregation, no mean normalization)
  - self-term (1 + eps) * x is handled outside the kernel
"""

import os
import math
import torch
import torch.nn.functional as F

os.environ.setdefault("NEURON_FRAMEWORK_DEBUG", "1")
os.environ.setdefault("XLA_IR_DEBUG", "1")
os.environ.setdefault("XLA_HLO_DEBUG", "1")

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
except ImportError:
    raise ImportError("run_nki.py requires neuronxcc (e.g. from /opt/aws_neuron_... venv)")

from torch_xla.core import xla_model as xm

NUM_NODES = 4096
AVG_DEGREE = 10
IN_FEATURES = 64
HIDDEN = 64
OUT_CLASSES = 16
SEED = 42
EPS = 0.1

TILE_NODES = 128
GATHER_MEAN_D_CHUNK = int(os.getenv("GATHER_MEAN_D_CHUNK", "32"))
MAX_DEG = 32  # overwritten in main()


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

    src_padded = torch.zeros(num_nodes * max_deg, dtype=torch.int32)
    mask_padded = torch.zeros(num_nodes * max_deg, dtype=torch.float32)

    for n in range(num_nodes):
        start = int(rowptr[n].item())
        end = int(rowptr[n + 1].item())
        deg_n = end - start
        if deg_n > 0:
            row_start = n * max_deg
            src_padded[row_start:row_start + deg_n] = src_sorted[start:end].to(torch.int32)
            mask_padded[row_start:row_start + deg_n] = 1.0

    inv_degree = torch.ones(num_nodes, dtype=torch.float32)
    return src_padded, mask_padded, inv_degree


def gin_aggregate_torch_cpu(x: torch.Tensor, src_padded: torch.Tensor, mask_padded: torch.Tensor, inv_degree: torch.Tensor, max_deg: int):
    messages = x[src_padded.long()] * mask_padded[:, None]
    messages_3d = messages.view(NUM_NODES, max_deg, x.shape[1])
    agg = messages_3d.sum(dim=1)
    return agg * inv_degree[:, None]


def mlp_block(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor):
    return F.linear(F.relu(F.linear(x, w1, b1)), w2, b2)


@nki.jit
def gather_mean_nki(x, src_padded, mask_padded, inv_degree):
    d_chunk = GATHER_MEAN_D_CHUNK
    assert MAX_DEG % d_chunk == 0, "MAX_DEG must be divisible by GATHER_MEAN_D_CHUNK"

    out = nl.ndarray((NUM_NODES, x.shape[1]), dtype=nl.float32, buffer=nl.shared_hbm)

    i_p = nl.arange(TILE_NODES)[:, None]
    i_f = nl.arange(x.shape[1])[None, :]
    num_node_tiles = NUM_NODES // TILE_NODES

    for nt in nl.affine_range(num_node_tiles):
        acc = nl.zeros((TILE_NODES, x.shape[1]), dtype=nl.float32, buffer=nl.sbuf)
        node_base = nt * TILE_NODES

        for c in nl.affine_range(MAX_DEG // d_chunk):
            chunk_buf = nl.ndarray(
                (TILE_NODES, d_chunk, x.shape[1]),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            for dk in nl.affine_range(d_chunk):
                k = c * d_chunk + dk
                row_idx = (node_base + i_p) * MAX_DEG + k
                src_idx = nl.load(src_padded[row_idx])
                m = nl.load(mask_padded[row_idx])
                gathered = nl.load(x[src_idx, i_f])
                chunk_buf[i_p, dk, i_f] = gathered * m

            chunk_sum = nl.sum(chunk_buf, axis=1, keepdims=False)
            acc += chunk_sum

        inv_tile = nl.load(inv_degree[node_base + i_p])
        acc = acc * inv_tile
        nl.store(out[node_base + i_p, i_f], value=acc)

    return out


def main():
    global MAX_DEG

    device = xm.xla_device()
    torch.manual_seed(SEED)

    src, dst = generate_random_graph(NUM_NODES, AVG_DEGREE, SEED)
    x0 = torch.randn(NUM_NODES, IN_FEATURES, dtype=torch.float32)

    deg_i64 = torch.zeros(NUM_NODES, dtype=torch.int64)
    deg_i64.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.int64))
    raw_max_deg = int(deg_i64.max().item())
    MAX_DEG = 1 << int(math.ceil(math.log2(max(raw_max_deg, 1))))

    src_padded_cpu, mask_padded_cpu, inv_degree_cpu = build_src_padded(src, dst, NUM_NODES, MAX_DEG)

    l1_w1 = torch.randn(HIDDEN, IN_FEATURES, dtype=torch.float32) * 0.01
    l1_b1 = torch.zeros(HIDDEN, dtype=torch.float32)
    l1_w2 = torch.randn(HIDDEN, HIDDEN, dtype=torch.float32) * 0.01
    l1_b2 = torch.zeros(HIDDEN, dtype=torch.float32)

    l2_w1 = torch.randn(HIDDEN, HIDDEN, dtype=torch.float32) * 0.01
    l2_b1 = torch.zeros(HIDDEN, dtype=torch.float32)
    l2_w2 = torch.randn(OUT_CLASSES, HIDDEN, dtype=torch.float32) * 0.01
    l2_b2 = torch.zeros(OUT_CLASSES, dtype=torch.float32)

    # CPU reference.
    with torch.no_grad():
        agg1 = gin_aggregate_torch_cpu(x0, src_padded_cpu, mask_padded_cpu, inv_degree_cpu, MAX_DEG)
        h1_in = (1.0 + EPS) * x0 + agg1
        h1 = mlp_block(h1_in, l1_w1, l1_b1, l1_w2, l1_b2)
        agg2 = gin_aggregate_torch_cpu(h1, src_padded_cpu, mask_padded_cpu, inv_degree_cpu, MAX_DEG)
        h2_in = (1.0 + EPS) * h1 + agg2
        out_ref = mlp_block(h2_in, l2_w1, l2_b1, l2_w2, l2_b2)

    x = x0.to(device)
    src_padded = src_padded_cpu.to(device)
    mask_padded = mask_padded_cpu.to(device)
    inv_degree = inv_degree_cpu.to(device)
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
    agg1 = gather_mean_nki(x, src_padded, mask_padded, inv_degree)
    h1_in = (1.0 + EPS) * x + agg1
    h1 = mlp_block(h1_in, l1_w1, l1_b1, l1_w2, l1_b2)

    agg2 = gather_mean_nki(h1, src_padded, mask_padded, inv_degree)
    h2_in = (1.0 + EPS) * h1 + agg2
    out = mlp_block(h2_in, l2_w1, l2_b1, l2_w2, l2_b2)

    xm.mark_step()
    xm.wait_device_ops()

    out_cpu = out.cpu()
    diff = (out_cpu - out_ref).abs().max().item()
    out_cpu_sum = out_cpu.sum().item()
    print("output sample:", out_cpu[0, :4].tolist())
    print("max abs diff vs torch ref:", diff)
    print("output sum:", out_cpu_sum)


if __name__ == "__main__":
    main()

