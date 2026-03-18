"""
GraphSAGE neighbor aggregation: fused gather + mean (NKI scaffold).

CPU setup builds `src_padded` and `mask_padded` for destination segments so that:
  out[n] = (sum_k x[src_padded[n*MAX_DEG+k], :] * mask_padded[n*MAX_DEG+k]) * inv_degree[n]

The current NKI kernel is only a scaffold (writes zeros). You will implement
the actual indirect gather + segment sum next.
"""

import os
import math
import torch

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
FEAT_DIM = 64
SEED = 42

TILE_NODES = 128
GATHER_MEAN_D_CHUNK = int(os.getenv("GATHER_MEAN_D_CHUNK", "32"))
MAX_DEG = 32  # overwritten in main()


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


@nki.jit
def gather_mean_nki(x, src_padded, mask_padded, inv_degree):
    # Option 1:
    #   For each node tile and degree chunk:
    #     - gather x[src_padded] into a temporary SBUF chunk_buf
    #     - mask in SBUF
    #     - reduce across D_CHUNK in SBUF
    #     - accumulate into acc
    #   Finally multiply by inv_degree and store.
    D_CHUNK = GATHER_MEAN_D_CHUNK
    assert MAX_DEG % D_CHUNK == 0, "MAX_DEG must be divisible by GATHER_MEAN_D_CHUNK"

    out = nl.ndarray((NUM_NODES, FEAT_DIM), dtype=nl.float32, buffer=nl.shared_hbm)

    i_p = nl.arange(TILE_NODES)[:, None]     # [TILE_NODES, 1] partition indices
    i_f = nl.arange(FEAT_DIM)[None, :]      # [1, FEAT_DIM] free indices

    num_node_tiles = NUM_NODES // TILE_NODES

    for nt in nl.affine_range(num_node_tiles):
        # acc holds partial sums for this node tile across all degree chunks.
        acc = nl.zeros((TILE_NODES, FEAT_DIM), dtype=nl.float32, buffer=nl.sbuf)

        node_base = nt * TILE_NODES
        for c in nl.affine_range(MAX_DEG // D_CHUNK):
            # chunk_buf: [TILE_NODES, D_CHUNK, FEAT_DIM]
            chunk_buf = nl.ndarray((TILE_NODES, D_CHUNK, FEAT_DIM),
                                     dtype=nl.float32,
                                     buffer=nl.sbuf)

            # Load and gather one dk slot at a time.
            for dk in nl.affine_range(D_CHUNK):
                k = c * D_CHUNK + dk
                row_idx = (node_base + i_p) * MAX_DEG + k  # [TILE_NODES, 1]

                # Indirect gather of x rows: use src_padded indices in SBUF.
                src_idx = nl.load(src_padded[row_idx])     # [TILE_NODES, 1] int32 in SBUF
                m = nl.load(mask_padded[row_idx])          # [TILE_NODES, 1] float32 in SBUF

                gathered = nl.load(x[src_idx, i_f])      # [TILE_NODES, FEAT_DIM]

                # Apply mask: broadcast over feature dim.
                gathered = gathered * m

                # Store into chunk buffer at this dk.
                chunk_buf[i_p, dk, i_f] = gathered

            # Sum across dk dimension (first free axis after partition).
            chunk_sum = nl.sum(chunk_buf, axis=1, keepdims=False)  # [TILE_NODES, FEAT_DIM]
            acc += chunk_sum

        # Multiply by inv_degree to get mean aggregation.
        inv_tile = nl.load(inv_degree[node_base + i_p])  # [TILE_NODES, 1]
        acc = acc * inv_tile

        nl.store(out[node_base + i_p, i_f], value=acc)

    return out


def main():
    global MAX_DEG
    device = xm.xla_device()
    torch.manual_seed(SEED)

    src, dst, degree, inv_degree = generate_random_graph(NUM_NODES, AVG_DEGREE, SEED)
    x = torch.randn(NUM_NODES, FEAT_DIM, dtype=torch.float32)

    raw_max_deg = int(degree.max().item())
    MAX_DEG = 1 << int(math.ceil(math.log2(max(raw_max_deg, 1))))
    print(f"raw_max_deg={raw_max_deg}, padded MAX_DEG={MAX_DEG}")

    src_padded, mask_padded = build_src_padded(src, dst, NUM_NODES, MAX_DEG)

    # Torch reference using the same padded semantics.
    with torch.no_grad():
        messages = x[src_padded.long()] * mask_padded[:, None]
        messages_3d = messages.view(NUM_NODES, MAX_DEG, FEAT_DIM)
        agg = messages_3d.sum(dim=1)
        ref = agg * inv_degree[:, None]

    # Move inputs to device and materialize.
    x = x.to(device)
    src_padded = src_padded.to(device)
    mask_padded = mask_padded.to(device)
    inv_degree = inv_degree.to(device)
    xm.mark_step()
    xm.wait_device_ops()

    out = gather_mean_nki(x, src_padded, mask_padded, inv_degree)
    xm.mark_step()
    xm.wait_device_ops()

    out_cpu = out.cpu()
    diff = (out_cpu - ref).abs().max().item()
    print("output sample:", out_cpu[0, :4].tolist())
    print("max abs diff vs torch ref:", diff)


if __name__ == "__main__":
    main()
    raise SystemExit(0)

"""
GraphSAGE neighbor aggregation: fused gather + mean (NKI scaffold).

We refactor this experiment so that the gather is part of the profiled dataflow
for a fair comparison with a future fully-fused NKI implementation.

On CPU (setup, outside the profiled NEFF):
  - Sort edges by destination (`dst`) so each node has a contiguous segment.
  - Build:
      src_padded[n*MAX_DEG + k]  = src_sorted[edge_slot]  (padded with 0)
      mask_padded[n*MAX_DEG + k] = 1 for valid edge slots else 0

Within the NKI kernel (to be implemented next):
  - gather x[src_padded] (indirect gather),
  - multiply by mask_padded,
  - reduce across k (segment sum),
  - multiply by inv_degree (mean normalization).

Current status:
  - This file provides a compile-able scaffold that returns zeros.
  - Next step is for you to implement the gather+segment reduction in
    `gather_mean_nki`.
"""

import os

os.environ.setdefault("NEURON_FRAMEWORK_DEBUG", "1")
os.environ.setdefault("XLA_IR_DEBUG", "1")
os.environ.setdefault("XLA_HLO_DEBUG", "1")

import math
import torch

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
except ImportError:
    raise ImportError("run_nki.py requires neuronxcc (e.g. from /opt/aws_neuron_... venv)")

from torch_xla.core import xla_model as xm

NUM_NODES = 4096
AVG_DEGREE = 10
FEAT_DIM = 64
SEED = 42

# Kernel tunables for later performance work.
TILE_NODES = 128


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
    """Build src_padded + mask_padded for destination segments."""
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


# Overwritten in main() based on the graph.
MAX_DEG = 32


@nki.jit
def gather_mean_nki(x, src_padded, mask_padded, inv_degree):
    """
    TODO implement fused gather+mean:
      1. For each node n and feature f:
           acc[n,f] = sum_k x[src_padded[n*MAX_DEG+k], f] * mask_padded[n*MAX_DEG+k]
      2. out[n,f] = acc[n,f] * inv_degree[n]

    Current placeholder returns zeros (compile-able scaffold).
    """
    # Option 1: gather+mask, buffer per chunk, then reduce across D_CHUNK.
    D_CHUNK = GATHER_MEAN_D_CHUNK
    assert MAX_DEG % D_CHUNK == 0, "MAX_DEG must be divisible by GATHER_MEAN_D_CHUNK"

    out = nl.ndarray((NUM_NODES, FEAT_DIM), dtype=nl.float32, buffer=nl.shared_hbm)

    i_p = nl.arange(TILE_NODES)[:, None]     # [TILE_NODES, 1] partition indices
    i_f = nl.arange(FEAT_DIM)[None, :]      # [1, FEAT_DIM] free indices

    num_node_tiles = NUM_NODES // TILE_NODES

    for nt in nl.affine_range(num_node_tiles):
        acc = nl.zeros((TILE_NODES, FEAT_DIM), dtype=nl.float32, buffer=nl.sbuf)
        node_base = nt * TILE_NODES

        for c in nl.affine_range(MAX_DEG // D_CHUNK):
            chunk_buf = nl.ndarray((TILE_NODES, D_CHUNK, FEAT_DIM),
                                     dtype=nl.float32,
                                     buffer=nl.sbuf)

            for dk in nl.affine_range(D_CHUNK):
                k = c * D_CHUNK + dk
                row_idx = (node_base + i_p) * MAX_DEG + k  # [TILE_NODES, 1]

                src_idx = nl.load(src_padded[row_idx])      # [TILE_NODES, 1] int32
                m = nl.load(mask_padded[row_idx])           # [TILE_NODES, 1] float32

                gathered = nl.load(x[src_idx, i_f])        # [TILE_NODES, FEAT_DIM]
                gathered = gathered * m                      # broadcast over FEAT_DIM

                chunk_buf[i_p, dk, i_f] = gathered

            chunk_sum = nl.sum(chunk_buf, axis=1, keepdims=False)  # [TILE_NODES, FEAT_DIM]
            acc += chunk_sum

        inv_tile = nl.load(inv_degree[node_base + i_p])  # [TILE_NODES, 1]
        acc = acc * inv_tile                              # broadcast
        nl.store(out[node_base + i_p, i_f], value=acc)

    return out


def main():
    global MAX_DEG
    device = xm.xla_device()
    torch.manual_seed(SEED)

    src, dst, degree, inv_degree = generate_random_graph(NUM_NODES, AVG_DEGREE, SEED)
    x = torch.randn(NUM_NODES, FEAT_DIM, dtype=torch.float32)

    raw_max_deg = int(degree.max().item())
    MAX_DEG = 1 << int(math.ceil(math.log2(max(raw_max_deg, 1))))
    print(f"raw_max_deg={raw_max_deg}, padded MAX_DEG={MAX_DEG}")

    src_padded, mask_padded = build_src_padded(src, dst, NUM_NODES, MAX_DEG)

    # Torch reference (same semantics as the padded segment reduction).
    with torch.no_grad():
        messages = x[src_padded.long()] * mask_padded[:, None]
        messages_3d = messages.view(NUM_NODES, MAX_DEG, FEAT_DIM)
        agg = messages_3d.sum(dim=1)
        ref = agg * inv_degree[:, None]

    # Move to device and materialize.
    x = x.to(device)
    src_padded = src_padded.to(device)
    mask_padded = mask_padded.to(device)
    inv_degree = inv_degree.to(device)

    xm.mark_step()
    xm.wait_device_ops()

    out = gather_mean_nki(x, src_padded, mask_padded, inv_degree)
    xm.mark_step()
    xm.wait_device_ops()

    out_cpu = out.cpu()
    diff = (out_cpu - ref).abs().max().item()
    print("output shape:", out_cpu.shape)
    print("output sample:", out_cpu[0, :4].tolist())
    print("max abs diff vs torch ref:", diff)


if __name__ == "__main__":
    main()

"""
GraphSAGE neighbor aggregation: fused gather + mean (NKI stub).

This experiment refactors the earlier (non-fused) version so the *inputs* and
dataflow are compatible with an NKI kernel that will fuse gather too.

Data representation (built on CPU; gather happens in the eventual NKI kernel):
  - Sort edges by destination (`dst`) on CPU.
  - Build a padded destination-segment index tensor:
        src_padded[n*MAX_DEG + k] = src_sorted[edge_slot]
    and a corresponding mask:
        mask_padded[n*MAX_DEG + k] = 1 for valid edges, else 0.

Then the neighbor aggregation is:
  messages = x[src_padded] * mask_padded[:, None]
  agg[n] = sum_k messages[n*MAX_DEG + k]
  out[n] = agg[n] / degree[n]   (implemented as multiply by inv_degree)

For now, the NKI kernel is a scaffold (returns zeros) so you can implement
the gather+segment reduction logic next.
"""

import os

os.environ.setdefault("NEURON_FRAMEWORK_DEBUG", "1")
os.environ.setdefault("XLA_IR_DEBUG", "1")
os.environ.setdefault("XLA_HLO_DEBUG", "1")

import math
import torch

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
except ImportError:
    raise ImportError("run_nki.py requires neuronxcc (e.g. from /opt/aws_neuron_... venv)")

from torch_xla.core import xla_model as xm

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
    inv_degree = 1.0 / degree
    return src, dst, degree, inv_degree


def build_src_padded(src: torch.Tensor, dst: torch.Tensor, num_nodes: int, max_deg: int):
    """Build src_padded + mask_padded for destination segments."""
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


MAX_DEG = 32  # overwritten in main() based on graph
TILE_NODES = 128


@nki.jit
def gather_mean_nki(x, src_padded, mask_padded, inv_degree):
    """
    TODO implement:
      - gather: x[src_padded] (indirect HBM->SBUF loads)
      - multiply by mask_padded
      - reduce over MAX_DEG for each node
      - multiply by inv_degree
    """
    # Option 1: gather+mask, buffer per chunk, then reduce across D_CHUNK.
    D_CHUNK = GATHER_MEAN_D_CHUNK
    assert MAX_DEG % D_CHUNK == 0, "MAX_DEG must be divisible by GATHER_MEAN_D_CHUNK"

    out = nl.ndarray((NUM_NODES, FEAT_DIM), dtype=nl.float32, buffer=nl.shared_hbm)

    i_p = nl.arange(TILE_NODES)[:, None]     # [TILE_NODES, 1] partition indices
    i_f = nl.arange(FEAT_DIM)[None, :]      # [1, FEAT_DIM] free indices

    num_node_tiles = NUM_NODES // TILE_NODES

    for nt in nl.affine_range(num_node_tiles):
        acc = nl.zeros((TILE_NODES, FEAT_DIM), dtype=nl.float32, buffer=nl.sbuf)
        node_base = nt * TILE_NODES

        for c in nl.affine_range(MAX_DEG // D_CHUNK):
            chunk_buf = nl.ndarray((TILE_NODES, D_CHUNK, FEAT_DIM),
                                     dtype=nl.float32,
                                     buffer=nl.sbuf)

            for dk in nl.affine_range(D_CHUNK):
                k = c * D_CHUNK + dk
                row_idx = (node_base + i_p) * MAX_DEG + k  # [TILE_NODES, 1]

                src_idx = nl.load(src_padded[row_idx])      # [TILE_NODES, 1]
                m = nl.load(mask_padded[row_idx])           # [TILE_NODES, 1]

                gathered = nl.load(x[src_idx, i_f])        # [TILE_NODES, FEAT_DIM]
                gathered = gathered * m

                chunk_buf[i_p, dk, i_f] = gathered

            chunk_sum = nl.sum(chunk_buf, axis=1, keepdims=False)  # [TILE_NODES, FEAT_DIM]
            acc += chunk_sum

        inv_tile = nl.load(inv_degree[node_base + i_p])  # [TILE_NODES, 1]
        acc = acc * inv_tile
        nl.store(out[node_base + i_p, i_f], value=acc)

    return out


def main():
    global MAX_DEG
    device = xm.xla_device()
    torch.manual_seed(SEED)

    src, dst, degree, inv_degree = generate_random_graph(NUM_NODES, AVG_DEGREE, SEED)
    x = torch.randn(NUM_NODES, FEAT_DIM, dtype=torch.float32)

    # Compute padded MAX_DEG (next power of 2 of max in-degree)
    raw_max_deg = int(degree.max().item())
    MAX_DEG = 1 << int(math.ceil(math.log2(max(raw_max_deg, 1))))
    print(f"raw_max_deg={raw_max_deg}, padded MAX_DEG={MAX_DEG}")

    src_padded, mask_padded = build_src_padded(src, dst, NUM_NODES, MAX_DEG)

    # Torch reference (same padded segment semantics)
    with torch.no_grad():
        messages = x[src_padded.long()] * mask_padded[:, None]
        messages_3d = messages.view(NUM_NODES, MAX_DEG, FEAT_DIM)
        agg = messages_3d.sum(dim=1)
        ref = agg * inv_degree[:, None]

    # Move to device
    x = x.to(device)
    src_padded = src_padded.to(device)
    mask_padded = mask_padded.to(device)
    inv_degree = inv_degree.to(device)

    xm.mark_step()
    xm.wait_device_ops()

    # Profiled kernel (currently placeholder)
    out = gather_mean_nki(x, src_padded, mask_padded, inv_degree)
    xm.mark_step()
    xm.wait_device_ops()

    out_cpu = out.cpu()
    diff = (out_cpu - ref).abs().max().item()
    print("output shape:", out_cpu.shape)
    print("output sample:", out_cpu[0, :4].tolist())
    print("max abs diff vs torch ref:", diff)


if __name__ == "__main__":
    main()

"""
GraphSAGE neighbor aggregation (gather + scatter_add + mean) — NKI kernel.

This first version fuses:
  - scatter_add into per-node buckets
  - mean normalization (division by degree)

Gather (`x[src]`) is done on CPU to construct a padded "messages" tensor that
lets the scatter-add run efficiently in NKI.

Kernel does:
  out[n, f] = sum_{k=0..MAX_DEG-1} seg_values[n*MAX_DEG + k, f] * inv_degree[n]

So caller provides:
  seg_values: [NUM_NODES * MAX_DEG, FEAT_DIM] float32 (padded sorted segments)
  inv_degree_expand: [NUM_NODES, FEAT_DIM] float32
"""

import os
os.environ.setdefault("NEURON_FRAMEWORK_DEBUG", "1")
os.environ.setdefault("XLA_IR_DEBUG", "1")
os.environ.setdefault("XLA_HLO_DEBUG", "1")

import math
import torch

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
except ImportError:
    raise ImportError("run_nki.py requires neuronxcc (e.g. from /opt/aws_neuron_... venv)")

from torch_xla.core import xla_model as xm

NUM_NODES = 4096
AVG_DEGREE = 10
NUM_EDGES = NUM_NODES * AVG_DEGREE  # 40960
FEAT_DIM = 64
SEED = 42

# Tunables (degree chunk size) to reuse the scatter_add tuning approach.
MAX_DEG = 32  # set from data in main()
TILE_NODES = int(os.getenv("SCATTER_ADD_TILE_NODES", "128"))
D_CHUNK = int(os.getenv("SCATTER_ADD_D_CHUNK", "32"))


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


def build_sorted_inputs(src, dst, x):
    """Sort edges by destination (stable), returning src_sorted, dst_sorted."""
    # stable sort by dst for deterministic degree segmentation
    perm = dst.argsort(stable=True)
    return src[perm], dst[perm]


def build_padded_messages(x, src_sorted, dst_sorted, num_nodes, max_deg):
    """Build seg_values padded by destination segments.

    seg_values shape: [num_nodes * max_deg, feat_dim]
    """
    feat_dim = x.shape[1]

    # rowptr from sorted dst
    rowptr = torch.zeros(num_nodes + 1, dtype=torch.int64)
    ones = torch.ones_like(dst_sorted, dtype=torch.int64)
    rowptr[1:].scatter_add_(0, dst_sorted.to(torch.int64), ones)
    rowptr = rowptr.cumsum(0)

    seg = torch.zeros(num_nodes * max_deg, feat_dim, dtype=torch.float32)
    for i in range(num_nodes):
        start = int(rowptr[i].item())
        end = int(rowptr[i + 1].item())
        deg = end - start
        if deg > 0:
            row_start = i * max_deg
            seg[row_start : row_start + deg, :] = x[src_sorted[start:end], :]
    return seg


@nki.jit
def scatter_add_mean_nki(seg_values, inv_degree_expand):
    """Scatter-add + mean (multiply by inv_degree).

    seg_values: [NUM_NODES * MAX_DEG, FEAT_DIM] float32
    inv_degree_expand: [NUM_NODES, FEAT_DIM] float32
    """
    feat_dim = seg_values.shape[1]
    num_node_tiles = NUM_NODES // TILE_NODES
    assert MAX_DEG % D_CHUNK == 0, "MAX_DEG must be divisible by D_CHUNK"

    out = nl.ndarray((NUM_NODES, feat_dim), dtype=seg_values.dtype, buffer=nl.shared_hbm)

    i_p = nl.arange(TILE_NODES)[:, None]   # [128, 1] partition
    i_f = nl.arange(feat_dim)[None, :]    # [1, 64] free

    for nt in nl.affine_range(num_node_tiles):
        acc = nl.zeros((TILE_NODES, feat_dim), dtype=nl.float32, buffer=nl.sbuf)

        for c in nl.affine_range(MAX_DEG // D_CHUNK):
            chunk_buf = nl.ndarray((TILE_NODES, D_CHUNK, feat_dim),
                                     dtype=seg_values.dtype,
                                     buffer=nl.sbuf)
            for dk in nl.affine_range(D_CHUNK):
                row_idx = (nt * TILE_NODES + i_p) * MAX_DEG + (c * D_CHUNK + dk)  # [128, 1]
                tile = nl.load(seg_values[row_idx, i_f])  # [128, 64]
                chunk_buf[i_p, dk, i_f] = tile

            chunk_sum = nl.sum(chunk_buf, axis=1, keepdims=False)  # [128, 64]
            acc += chunk_sum

        # Multiply by inv degree (mean normalization).
        inv_tile = nl.load(inv_degree_expand[nt * TILE_NODES + i_p, i_f])  # [128, 64]
        acc = acc * inv_tile

        nl.store(out[nt * TILE_NODES + i_p, i_f], value=acc)

    return out


def main():
    global MAX_DEG
    device = xm.xla_device()
    torch.manual_seed(SEED)

    src, dst, degree, inv_degree = generate_random_graph(NUM_NODES, AVG_DEGREE, SEED)

    x = torch.randn(NUM_NODES, FEAT_DIM, dtype=torch.float32)

    # Sort edges by dst and build padded messages on CPU.
    src_sorted, dst_sorted = build_sorted_inputs(src, dst, x)

    # Compute raw max degree from dst_sorted via CSR counts.
    # (Matches how padded segments are built.)
    tmp_rowptr = torch.zeros(NUM_NODES + 1, dtype=torch.int64)
    ones = torch.ones_like(dst_sorted, dtype=torch.int64)
    tmp_rowptr[1:].scatter_add_(0, dst_sorted.to(torch.int64), ones)
    tmp_rowptr = tmp_rowptr.cumsum(0)
    degrees_int = tmp_rowptr[1:] - tmp_rowptr[:-1]
    raw_max_deg = int(degrees_int.max().item())
    MAX_DEG = 1 << int(math.ceil(math.log2(max(raw_max_deg, 1))))
    assert MAX_DEG % D_CHUNK == 0, f"With MAX_DEG={MAX_DEG}, need MAX_DEG%D_CHUNK==0 (D_CHUNK={D_CHUNK})"
    print(f"raw_max_deg={raw_max_deg}, padded MAX_DEG={MAX_DEG}, D_CHUNK={D_CHUNK}")

    seg_values = build_padded_messages(x, src_sorted, dst_sorted, NUM_NODES, MAX_DEG)

    inv_degree_expand = inv_degree.unsqueeze(1).expand(NUM_NODES, FEAT_DIM).contiguous()

    # Torch reference (CPU) for correctness check.
    with torch.no_grad():
        messages = x[src]
        idx = dst.unsqueeze(1).expand_as(messages)
        agg = torch.zeros(NUM_NODES, FEAT_DIM, dtype=messages.dtype)
        agg.scatter_add_(0, idx, messages)
        ref = agg / degree.unsqueeze(1)

    # ---- move inputs to device and materialize ----
    seg_values = seg_values.to(device)
    inv_degree_expand = inv_degree_expand.to(device)
    xm.mark_step()
    xm.wait_device_ops()

    # ---- profiled kernel ----
    out = scatter_add_mean_nki(seg_values, inv_degree_expand)
    xm.mark_step()
    xm.wait_device_ops()

    out_cpu = out.cpu()
    diff = (out_cpu - ref).abs().max().item()
    print("output shape:", out_cpu.shape)
    print("output sample:", out_cpu[0, :4].tolist())
    print("max abs diff vs torch ref:", diff)


if __name__ == "__main__":
    main()

