"""
Scatter-add baseline — pure PyTorch on Trainium.

Edges are pre-sorted by destination on the CPU. The sorted edges are packed
into a padded segment matrix of shape [NUM_NODES * MAX_DEG, FEAT_DIM] where
each node gets MAX_DEG contiguous rows (zero-padded beyond its actual degree).
The kernel reshapes to [NUM_NODES, MAX_DEG, FEAT_DIM] and sums along dim=1.

Both run_torch.py and run_nki.py receive the SAME padded tensor so the
comparison is apples-to-apples (only the on-device kernel differs).

Sizes (matching graphsage experiment):
    num_nodes  = 4096
    num_edges  = 40960  (avg_degree=10)
    feat_dim   = 64
"""
import os
os.environ.setdefault("NEURON_FRAMEWORK_DEBUG", "1")

import math
import torch
from torch_xla.core import xla_model as xm

NUM_NODES = 4096
NUM_EDGES = 40960
FEAT_DIM = 64
SEED = 42


def build_sorted_inputs(num_nodes, num_edges, feat_dim, seed=42):
    """Build scatter-add inputs, pre-sorted by destination (CPU).

    Returns:
        values:  [num_edges, feat_dim]  float32 — sorted by dst
        dst:     [num_edges]            int64   — sorted destination indices
        rowptr:  [num_nodes + 1]        int64   — CSR row pointers
    """
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

    For node i, rows [i*max_deg : i*max_deg + degree[i]] hold that node's
    edge features.  Remaining rows are zero-padded (don't affect the sum).
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


def scatter_add_torch(seg_values, num_nodes, max_deg):
    """Sum each node's padded segment.

    seg_values: [num_nodes * max_deg, feat_dim]
    Returns:    [num_nodes, feat_dim]
    """
    return seg_values.view(num_nodes, max_deg, -1).sum(dim=1)


def main():
    device = xm.xla_device()

    # --- CPU preprocessing: sort edges, build CSR, pad segments ---
    values, dst, rowptr = build_sorted_inputs(NUM_NODES, NUM_EDGES, FEAT_DIM, SEED)

    degrees = rowptr[1:] - rowptr[:-1]
    raw_max_deg = int(degrees.max().item())
    max_deg = 1 << int(math.ceil(math.log2(max(raw_max_deg, 1))))
    print(f"max_degree={raw_max_deg}, padded to MAX_DEG={max_deg}")

    seg_values = build_padded_segments(values, rowptr, NUM_NODES, FEAT_DIM, max_deg)
    print(f"seg_values shape: {seg_values.shape} "
          f"({seg_values.nelement() * 4 / 1e6:.1f} MB)")

    # --- move to device and materialize ---
    seg_values = seg_values.to(device)
    xm.mark_step()
    xm.wait_device_ops()

    # --- kernel (this is the NEFF we profile) ---
    out = scatter_add_torch(seg_values, NUM_NODES, max_deg)
    xm.mark_step()
    xm.wait_device_ops()

    print("output shape:", out.shape)
    print("output sample:", out.cpu()[0, :4].tolist())


if __name__ == "__main__":
    main()
