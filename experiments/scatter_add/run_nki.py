"""
Scatter-add — NKI kernel on Trainium (chunked degree reduction).

Both run_torch.py and run_nki.py receive the SAME padded segment tensor:
    seg_values: [NUM_NODES * MAX_DEG, FEAT_DIM]

Strategy:
    - Node tiling: TILE_NODES=128 consecutive destination nodes.
    - Degree reduction in chunks: process MAX_DEG edges in blocks of D_CHUNK.
    - For each chunk, load a tile [TILE_NODES, D_CHUNK, FEAT_DIM] using the
      same strided row indexing as the compiler, then reduce along D_CHUNK
      and accumulate chunk sums.

Tuning note (for this scatter_add workload):
  - Sweeping `D_CHUNK` over {4, 8, 16, 32} on `MAX_DEG=32` showed best
    `total_time` at `D_CHUNK=32`.
  - So by default, we set `SCATTER_ADD_D_CHUNK=32` (override via env var).

Sizes:
    num_nodes  = 4096
    num_edges  = 40960
    feat_dim   = 64
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
    import neuronxcc.nki.isa as nisa
except ImportError:
    raise ImportError("run_nki.py requires neuronxcc (e.g. from /opt/aws_neuron_... venv)")

from torch_xla.core import xla_model as xm

NUM_NODES = 4096
NUM_EDGES = 40960
FEAT_DIM = 64
SEED = 42

MAX_DEG = 32   # set from data in main()
TILE_NODES = int(os.getenv("SCATTER_ADD_TILE_NODES", "128"))
D_CHUNK = int(os.getenv("SCATTER_ADD_D_CHUNK", "32"))


def build_sorted_inputs(num_nodes, num_edges, feat_dim, seed=42):
    """Sort edges by dst, build CSR rowptr. Identical to run_torch.py."""
    gen = torch.Generator().manual_seed(seed)
    values_unsorted = torch.randn(num_edges, feat_dim, dtype=torch.float32, generator=gen)
    dst_unsorted = torch.randint(0, num_nodes, (num_edges,), generator=gen)

    sort_perm = dst_unsorted.argsort(stable=True)
    dst = dst_unsorted[sort_perm]
    values = values_unsorted[sort_perm]

    rowptr = torch.zeros(num_nodes + 1, dtype=torch.int64)
    ones = torch.ones(num_edges, dtype=torch.int64)
    rowptr[1:].scatter_add_(0, dst, ones)
    rowptr = rowptr.cumsum(0)

    return values, dst, rowptr


def build_padded_segments(values, rowptr, num_nodes, feat_dim, max_deg):
    """Build padded segment matrix [NUM_NODES * MAX_DEG, FEAT_DIM] on CPU.

    Identical to run_torch.py — both kernels receive the same tensor.
    """
    seg = torch.zeros(num_nodes * max_deg, feat_dim, dtype=torch.float32)
    for i in range(num_nodes):
        start = int(rowptr[i].item())
        end = int(rowptr[i + 1].item())
        deg = end - start
        if deg > 0:
            row_start = i * max_deg
            seg[row_start:row_start + deg, :] = values[start:end, :]
    return seg


@nki.jit
def scatter_add_nki(seg_values):
    """NKI scatter-add via chunked degree reduction (two-step load).

    We avoid complicated 3D indexing for `nl.load` by:
      - loading each edge-within-chunk slice (size D_CHUNK) as a 2D [TILE_NODES, FEAT_DIM]
      - placing it into an SBUF buffer [TILE_NODES, D_CHUNK, FEAT_DIM]
      - calling nl.sum over the D_CHUNK dimension
    """
    feat_dim = seg_values.shape[1]
    num_node_tiles = NUM_NODES // TILE_NODES
    assert MAX_DEG % D_CHUNK == 0, "MAX_DEG must be divisible by D_CHUNK"

    out = nl.ndarray((NUM_NODES, feat_dim), dtype=seg_values.dtype, buffer=nl.shared_hbm)

    # 2D indices for loads/stores (partition dim = TILE_NODES, free dim = FEAT_DIM)
    i_p = nl.arange(TILE_NODES)[:, None]     # [128, 1]
    i_f = nl.arange(feat_dim)[None, :]      # [1, 64]

    for nt in nl.affine_range(num_node_tiles):
        acc = nl.zeros((TILE_NODES, feat_dim), dtype=nl.float32, buffer=nl.sbuf)

        for c in nl.affine_range(MAX_DEG // D_CHUNK):
            # chunk_buf: [TILE_NODES, D_CHUNK, FEAT_DIM] in SBUF
            chunk_buf = nl.ndarray((TILE_NODES, D_CHUNK, feat_dim),
                                     dtype=seg_values.dtype,
                                     buffer=nl.sbuf)

            # Load each dk slice into chunk_buf
            for dk in nl.affine_range(D_CHUNK):
                row_idx = (nt * TILE_NODES + i_p) * MAX_DEG + (c * D_CHUNK + dk)  # [128, 1]
                tile = nl.load(seg_values[row_idx, i_f])                         # [128, 64]
                chunk_buf[i_p, dk, i_f] = tile

            # Reduce across dk dimension (axis=1 is first free axis after partition)
            chunk_sum = nl.sum(chunk_buf, axis=1, keepdims=False)               # [128, 64]
            acc += chunk_sum

        nl.store(out[nt * TILE_NODES + i_p, i_f], value=acc)

    return out


def main():
    global MAX_DEG
    device = xm.xla_device()

    # --- CPU preprocessing (identical to run_torch.py) ---
    values, dst, rowptr = build_sorted_inputs(NUM_NODES, NUM_EDGES, FEAT_DIM, SEED)

    degrees = rowptr[1:] - rowptr[:-1]
    raw_max_deg = int(degrees.max().item())
    MAX_DEG = 1 << int(math.ceil(math.log2(max(raw_max_deg, 1))))
    assert MAX_DEG % D_CHUNK == 0, f"MAX_DEG={MAX_DEG} must be divisible by D_CHUNK={D_CHUNK}"
    print(f"max_degree={raw_max_deg}, padded to MAX_DEG={MAX_DEG}")

    seg_values = build_padded_segments(values, rowptr, NUM_NODES, FEAT_DIM, MAX_DEG)
    print(f"seg_values shape: {seg_values.shape} "
          f"({seg_values.nelement() * 4 / 1e6:.1f} MB)")

    # Correctness check against the padded segment semantics:
    #   out[n, :] = sum_k seg_values[n*MAX_DEG + k, :]
    out_ref = seg_values.view(NUM_NODES, MAX_DEG, FEAT_DIM).sum(dim=1)

    # --- move to device and materialize ---
    seg_values = seg_values.to(device)
    xm.mark_step()
    xm.wait_device_ops()

    # --- NKI kernel (this is the NEFF we profile) ---
    out = scatter_add_nki(seg_values)
    xm.mark_step()
    xm.wait_device_ops()

    out_cpu = out.cpu()
    diff = (out_cpu - out_ref).abs().max().item()
    print("output shape:", out_cpu.shape)
    print("output sample:", out_cpu[0, :4].tolist())
    print("max abs diff vs ref:", diff)


if __name__ == "__main__":
    main()
