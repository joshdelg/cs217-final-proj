"""
GraphSAGE neighbor aggregation: fused gather + mean (NKI implementation).

This is the `_v3` optimization compared to `graphsage_gather_scatter_mean_v2/`:

Option 2 (v2):
  - inside (node tile, degree) loop:
      src_idx = nl.load(src_padded[row_idx])
      mask    = nl.load(mask_padded[row_idx])
      gathered = nl.load(x[src_idx, :])
      acc += gathered * mask

Option 3 (v3):
  - For each (node tile, degree *chunk*):
      preload src_idx chunk and mask chunk into SBUF using a *single* 2D nl.load each
  - Then loop over dk only for the actual `x[src_idx, :]` gathers and accumulation.
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

# overwritten in main() based on runtime graph max degree.
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
    """Build src_padded (int32) and mask_padded (float32) for destination segments."""
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
    """Option 3:
    preload `src_padded` + `mask_padded` for an entire degree chunk into SBUF.
    """
    D_CHUNK = GATHER_MEAN_D_CHUNK
    assert MAX_DEG % D_CHUNK == 0, "MAX_DEG must be divisible by GATHER_MEAN_D_CHUNK"

    out = nl.ndarray((NUM_NODES, FEAT_DIM), dtype=nl.float32, buffer=nl.shared_hbm)

    # Partition dim = TILE_NODES, free dim = FEAT_DIM.
    i_p = nl.arange(TILE_NODES)[:, None]  # [TILE_NODES, 1]
    i_f = nl.arange(FEAT_DIM)[None, :]   # [1, FEAT_DIM]
    i_dk = nl.arange(D_CHUNK)[None, :]   # [1, D_CHUNK]

    num_node_tiles = NUM_NODES // TILE_NODES

    for nt in nl.affine_range(num_node_tiles):
        node_base = nt * TILE_NODES
        acc = nl.zeros((TILE_NODES, FEAT_DIM), dtype=nl.float32, buffer=nl.sbuf)

        # Iterate over degree chunks for this node tile.
        for c in nl.affine_range(MAX_DEG // D_CHUNK):
            k_base = c * D_CHUNK

            # row_idx_chunk is 2D: [TILE_NODES, D_CHUNK]
            row_idx_chunk = (node_base + i_p) * MAX_DEG + (k_base + i_dk)

            # Preload only masks for the whole chunk.
            # NKI doesn't allow using an SBUF "TensorView" as indirect indices for
            # indexing `x`, so we keep `src_idx` loads per-dk (as in v2).
            mask_chunk = nl.load(mask_padded[row_idx_chunk])    # [TILE_NODES, D_CHUNK]

            for dk in nl.affine_range(D_CHUNK):
                k = k_base + dk
                row_idx = (node_base + i_p) * MAX_DEG + k  # [TILE_NODES, 1]

                src_idx = nl.load(src_padded[row_idx])      # [TILE_NODES, 1]
                m_dk = mask_chunk[:, dk][:, None]           # [TILE_NODES, 1]

                gathered = nl.load(x[src_idx, i_f])          # [TILE_NODES, FEAT_DIM]
                acc += gathered * m_dk                      # broadcast m_dk over FEAT_DIM

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

    # Torch reference (same padded semantics).
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
    print("output sample:", out_cpu[0, :4].tolist())
    print("max abs diff vs torch ref:", diff)


if __name__ == "__main__":
    main()

