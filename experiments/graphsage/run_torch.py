"""
GraphSAGE forward pass — manual PyTorch on Trainium (no PyG dependency).

Two-layer GraphSAGE with mean aggregation:
    h_i' = W_l @ mean(h_j for j in N(i)) + W_r @ h_i + b

Graph is stored in COO format (src, dst tensors) which maps cleanly to
scatter_add on XLA.

Sizes (tractable for profiling):
    num_nodes   = 4096
    avg_degree  = 10  (~40k edges)
    in_features = 64
    hidden      = 64
    out_classes  = 16
"""
import os
os.environ.setdefault("NEURON_FRAMEWORK_DEBUG", "1")

import torch
import torch.nn.functional as F
from torch_xla.core import xla_model as xm

# --------------- graph parameters ---------------
NUM_NODES = 4096
AVG_DEGREE = 10
IN_FEATURES = 64
HIDDEN = 64
OUT_CLASSES = 16
SEED = 42


def generate_random_graph(num_nodes: int, avg_degree: int, seed: int = 42):
    """Return COO edge tensors (src, dst) and per-node degree for dst nodes."""
    gen = torch.Generator().manual_seed(seed)
    num_edges = num_nodes * avg_degree
    src = torch.randint(0, num_nodes, (num_edges,), generator=gen)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=gen)
    degree = torch.zeros(num_nodes, dtype=torch.float32)
    degree.scatter_add_(0, dst, torch.ones(num_edges, dtype=torch.float32))
    degree.clamp_(min=1.0)
    return src, dst, degree


def sage_conv(
    x: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    degree: torch.Tensor,
    W_neigh: torch.Tensor,
    W_self: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Single SAGEConv layer (mean aggregation).

    x:        [num_nodes, F_in]
    src, dst: [num_edges]  COO edge list
    degree:   [num_nodes]  in-degree of each node (for mean)
    W_neigh:  [F_out, F_in]
    W_self:   [F_out, F_in]
    bias:     [F_out]
    """
    num_nodes = x.shape[0]

    # 1. Gather source features for each edge
    neigh_feats = x[src]  # [num_edges, F_in]

    # 2. Scatter-add into destination buckets
    idx = dst.unsqueeze(1).expand_as(neigh_feats)  # [num_edges, F_in]
    agg = torch.zeros(num_nodes, x.shape[1], dtype=x.dtype, device=x.device)
    agg.scatter_add_(0, idx, neigh_feats)

    # 3. Mean aggregation (divide by degree)
    agg = agg / degree.unsqueeze(1)

    # 4. Linear transforms + combine
    out = F.linear(agg, W_neigh, bias) + F.linear(x, W_self)
    return out


def main():
    device = xm.xla_device()
    torch.manual_seed(SEED)

    # ---- build graph (CPU, then move) ----
    src, dst, degree = generate_random_graph(NUM_NODES, AVG_DEGREE, SEED)

    # ---- node features ----
    x = torch.randn(NUM_NODES, IN_FEATURES, dtype=torch.float32)

    # ---- layer 1 weights: (IN_FEATURES -> HIDDEN) ----
    W_neigh1 = torch.randn(HIDDEN, IN_FEATURES, dtype=torch.float32) * 0.01
    W_self1 = torch.randn(HIDDEN, IN_FEATURES, dtype=torch.float32) * 0.01
    b1 = torch.zeros(HIDDEN, dtype=torch.float32)

    # ---- layer 2 weights: (HIDDEN -> OUT_CLASSES) ----
    W_neigh2 = torch.randn(OUT_CLASSES, HIDDEN, dtype=torch.float32) * 0.01
    W_self2 = torch.randn(OUT_CLASSES, HIDDEN, dtype=torch.float32) * 0.01
    b2 = torch.zeros(OUT_CLASSES, dtype=torch.float32)

    # ---- move everything to device and materialize ----
    x = x.to(device)
    src = src.to(device)
    dst = dst.to(device)
    degree = degree.to(device)
    W_neigh1 = W_neigh1.to(device)
    W_self1 = W_self1.to(device)
    b1 = b1.to(device)
    W_neigh2 = W_neigh2.to(device)
    W_self2 = W_self2.to(device)
    b2 = b2.to(device)
    xm.mark_step()
    xm.wait_device_ops()

    # ---- forward pass (this is the NEFF we profile) ----
    h = sage_conv(x, src, dst, degree, W_neigh1, W_self1, b1)
    h = F.relu(h)
    out = sage_conv(h, src, dst, degree, W_neigh2, W_self2, b2)
    xm.mark_step()
    xm.wait_device_ops()

    print("output shape:", out.shape)
    print("output sample:", out.cpu()[0, :4].tolist())


if __name__ == "__main__":
    main()
