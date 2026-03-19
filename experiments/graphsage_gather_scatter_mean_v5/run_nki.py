"""
GraphSAGE forward (2-layer) — NKI neighbor aggregation + PyTorch linear layers.

Only the mean neighbor aggregation step is computed by the custom NKI kernel.
All linear layers and activations remain as PyTorch ops (so we can attribute
end-to-end differences mainly to the aggregation kernel).
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

TILE_NODES = 128
GATHER_MEAN_D_CHUNK = int(os.getenv("GATHER_MEAN_D_CHUNK", "32"))  # v1/Option-1 style default

# Overwritten in main() based on padded graph.
MAX_DEG = 32


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

    src_padded = torch.zeros(num_nodes * max_deg, dtype=torch.int32)
    mask_padded = torch.zeros(num_nodes * max_deg, dtype=torch.float32)

    for n in range(num_nodes):
        start = int(rowptr[n].item())
        end = int(rowptr[n + 1].item())
        deg_n = end - start
        if deg_n > 0:
            row_start = n * max_deg
            src_padded[row_start : row_start + deg_n] = src_sorted[start:end].to(torch.int32)
            mask_padded[row_start : row_start + deg_n] = 1.0

    return src_padded, mask_padded


def neighbor_agg_mean_pytorch_cpu(x: torch.Tensor, src_padded: torch.Tensor, mask_padded: torch.Tensor, inv_degree: torch.Tensor, max_deg: int):
    messages = x[src_padded.long()] * mask_padded[:, None]
    messages_3d = messages.view(NUM_NODES, max_deg, x.shape[1])
    agg = messages_3d.sum(dim=1)
    return agg * inv_degree[:, None]


@nki.jit
def gather_mean_nki(x, src_padded, mask_padded, inv_degree):
    """Mean aggregation using padded destination segments (v1/Option-1 style)."""
    D_CHUNK = GATHER_MEAN_D_CHUNK
    assert MAX_DEG % D_CHUNK == 0, "MAX_DEG must be divisible by GATHER_MEAN_D_CHUNK"

    out = nl.ndarray((NUM_NODES, x.shape[1]), dtype=nl.float32, buffer=nl.shared_hbm)

    # Partition = TILE_NODES, free = feature dim.
    i_p = nl.arange(TILE_NODES)[:, None]  # [TILE_NODES, 1]
    i_f = nl.arange(x.shape[1])[None, :]  # [1, F]

    num_node_tiles = NUM_NODES // TILE_NODES

    for nt in nl.affine_range(num_node_tiles):
        acc = nl.zeros((TILE_NODES, x.shape[1]), dtype=nl.float32, buffer=nl.sbuf)
        node_base = nt * TILE_NODES

        for c in nl.affine_range(MAX_DEG // D_CHUNK):
            # chunk_buf: [TILE_NODES, D_CHUNK, F] in SBUF
            chunk_buf = nl.ndarray(
                (TILE_NODES, D_CHUNK, x.shape[1]),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )

            for dk in nl.affine_range(D_CHUNK):
                k = c * D_CHUNK + dk
                row_idx = (node_base + i_p) * MAX_DEG + k  # [TILE_NODES, 1]

                src_idx = nl.load(src_padded[row_idx])  # [TILE_NODES, 1]
                m = nl.load(mask_padded[row_idx])        # [TILE_NODES, 1]

                gathered = nl.load(x[src_idx, i_f])     # [TILE_NODES, F]
                chunk_buf[i_p, dk, i_f] = gathered * m

            chunk_sum = nl.sum(chunk_buf, axis=1, keepdims=False)  # [TILE_NODES, F]
            acc += chunk_sum

        inv_tile = nl.load(inv_degree[node_base + i_p])  # [TILE_NODES, 1]
        acc = acc * inv_tile
        nl.store(out[node_base + i_p, i_f], value=acc)

    return out


def main():
    global MAX_DEG

    device = xm.xla_device()
    torch.manual_seed(SEED)

    # ---- build graph + CPU inputs ----
    src, dst, degree, inv_degree = generate_random_graph(NUM_NODES, AVG_DEGREE, SEED)
    x0 = torch.randn(NUM_NODES, IN_FEATURES, dtype=torch.float32)

    raw_max_deg = int(degree.max().item())
    MAX_DEG = 1 << int(math.ceil(math.log2(max(raw_max_deg, 1))))
    src_padded_cpu, mask_padded_cpu = build_src_padded(src, dst, NUM_NODES, MAX_DEG)

    # ---- weights: match `graphsage/run_torch.py` ----
    W_neigh1 = torch.randn(HIDDEN, IN_FEATURES, dtype=torch.float32) * 0.01
    W_self1 = torch.randn(HIDDEN, IN_FEATURES, dtype=torch.float32) * 0.01
    b1 = torch.zeros(HIDDEN, dtype=torch.float32)

    W_neigh2 = torch.randn(OUT_CLASSES, HIDDEN, dtype=torch.float32) * 0.01
    W_self2 = torch.randn(OUT_CLASSES, HIDDEN, dtype=torch.float32) * 0.01
    b2 = torch.zeros(OUT_CLASSES, dtype=torch.float32)

    # ---- CPU reference (for correctness only; keep it off-device to avoid polluting NEFF) ----
    with torch.no_grad():
        agg1 = neighbor_agg_mean_pytorch_cpu(x0, src_padded_cpu, mask_padded_cpu, inv_degree, MAX_DEG)
        h1 = F.linear(agg1, W_neigh1, b1) + F.linear(x0, W_self1)
        h1 = F.relu(h1)
        agg2 = neighbor_agg_mean_pytorch_cpu(h1, src_padded_cpu, mask_padded_cpu, inv_degree, MAX_DEG)
        out_ref = F.linear(agg2, W_neigh2, b2) + F.linear(h1, W_self2)

    # ---- move inputs to device ----
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
    h1_neigh = gather_mean_nki(x, src_padded, mask_padded, inv_degree_dev)
    h1 = F.linear(h1_neigh, W_neigh1, b1) + F.linear(x, W_self1)
    h1 = F.relu(h1)

    h2_neigh = gather_mean_nki(h1, src_padded, mask_padded, inv_degree_dev)
    out = F.linear(h2_neigh, W_neigh2, b2) + F.linear(h1, W_self2)

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

