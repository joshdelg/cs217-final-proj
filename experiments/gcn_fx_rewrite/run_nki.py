"""
2-layer GCN where aggregation is auto-rewritten via torch.fx to NKI kernel call.
"""

import importlib.util
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_xla.core import xla_model as xm

os.environ.setdefault("NEURON_FRAMEWORK_DEBUG", "1")
os.environ.setdefault("XLA_IR_DEBUG", "1")
os.environ.setdefault("XLA_HLO_DEBUG", "1")

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
except ImportError:
    raise ImportError("run_nki.py requires neuronxcc (from Neuron venv)")

NUM_NODES = 4096
AVG_DEGREE = 10
IN_FEATURES = 64
HIDDEN = 64
OUT_CLASSES = 16
SEED = 42

TILE_NODES = 128
GATHER_MEAN_D_CHUNK = int(os.getenv("GATHER_MEAN_D_CHUNK", "32"))
MAX_DEG = 32  # set at runtime


def _load_fx_rewrite():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "fx_gnn_rewrite.py"
    spec = importlib.util.spec_from_file_location("fx_gnn_rewrite", str(mod_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load FX rewrite utility from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.trace_and_rewrite_gnn_aggregation


def generate_random_graph(num_nodes: int, avg_degree: int, seed: int = 42):
    gen = torch.Generator().manual_seed(seed)
    num_edges = num_nodes * avg_degree
    src = torch.randint(0, num_nodes, (num_edges,), generator=gen, dtype=torch.int64)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=gen, dtype=torch.int64)
    nodes = torch.arange(num_nodes, dtype=torch.int64)  # A_hat = A + I
    src = torch.cat([src, nodes], dim=0)
    dst = torch.cat([dst, nodes], dim=0)
    return src, dst


def build_gcn_padded(src: torch.Tensor, dst: torch.Tensor, num_nodes: int, max_deg: int):
    deg = torch.zeros(num_nodes, dtype=torch.float32)
    deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    deg.clamp_(min=1.0)
    norm = 1.0 / torch.sqrt(deg[src] * deg[dst])

    perm = dst.argsort(stable=True)
    src_sorted = src[perm]
    dst_sorted = dst[perm]
    norm_sorted = norm[perm]

    rowptr = torch.zeros(num_nodes + 1, dtype=torch.int64)
    rowptr[1:].scatter_add_(0, dst_sorted, torch.ones_like(dst_sorted, dtype=torch.int64))
    rowptr = rowptr.cumsum(0)

    src_padded = torch.zeros(num_nodes * max_deg, dtype=torch.int32)
    norm_padded = torch.zeros(num_nodes * max_deg, dtype=torch.float32)
    for n in range(num_nodes):
        start = int(rowptr[n].item())
        end = int(rowptr[n + 1].item())
        deg_n = end - start
        if deg_n > 0:
            off = n * max_deg
            src_padded[off:off + deg_n] = src_sorted[start:end].to(torch.int32)
            norm_padded[off:off + deg_n] = norm_sorted[start:end]
    inv_degree = torch.ones(num_nodes, dtype=torch.float32)
    return src_padded, norm_padded, inv_degree


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
            chunk_buf = nl.ndarray((TILE_NODES, d_chunk, x.shape[1]), dtype=nl.float32, buffer=nl.sbuf)
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


def gcn_aggregate_pattern(
    x: torch.Tensor,
    src_padded: torch.Tensor,
    norm_padded: torch.Tensor,
    inv_degree: torch.Tensor,
    max_deg: int,
):
    messages = x[src_padded] * norm_padded[:, None]
    messages_3d = messages.view(NUM_NODES, max_deg, x.shape[1])
    agg = messages_3d.sum(dim=1)
    return agg * inv_degree[:, None]


def gcn_aggregate_nki_replacement(
    x: torch.Tensor,
    src_padded: torch.Tensor,
    mask_like: torch.Tensor,
    max_deg: int,
    inv_degree_or_none: torch.Tensor | None,
):
    if int(max_deg) != MAX_DEG:
        raise RuntimeError(f"max_deg mismatch: expected {MAX_DEG}, got {max_deg}")
    if inv_degree_or_none is None:
        inv_degree_or_none = torch.ones(NUM_NODES, dtype=x.dtype, device=x.device)
    return gather_mean_nki(x, src_padded, mask_like, inv_degree_or_none)


def main():
    global MAX_DEG

    trace_and_rewrite_gnn_aggregation = _load_fx_rewrite()

    device = xm.xla_device()
    torch.manual_seed(SEED)

    src, dst = generate_random_graph(NUM_NODES, AVG_DEGREE, SEED)
    x0 = torch.randn(NUM_NODES, IN_FEATURES, dtype=torch.float32)

    deg_i64 = torch.zeros(NUM_NODES, dtype=torch.int64)
    deg_i64.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.int64))
    raw_max_deg = int(deg_i64.max().item())
    MAX_DEG = 1 << int(math.ceil(math.log2(max(raw_max_deg, 1))))

    src_padded_cpu, norm_padded_cpu, inv_degree_cpu = build_gcn_padded(src, dst, NUM_NODES, MAX_DEG)

    W1 = torch.randn(HIDDEN, IN_FEATURES, dtype=torch.float32) * 0.01
    b1 = torch.zeros(HIDDEN, dtype=torch.float32)
    W2 = torch.randn(OUT_CLASSES, HIDDEN, dtype=torch.float32) * 0.01
    b2 = torch.zeros(OUT_CLASSES, dtype=torch.float32)

    gm, info = trace_and_rewrite_gnn_aggregation(gcn_aggregate_pattern, gcn_aggregate_nki_replacement)
    if not info.matched:
        raise RuntimeError("FX rewrite did not match the GCN aggregation pattern")
    print(f"FX rewrite matched: inv_tail={info.has_inv_degree_tail} replacement={info.replacement_target}")

    with torch.no_grad():
        agg1 = gcn_aggregate_pattern(x0, src_padded_cpu.long(), norm_padded_cpu, inv_degree_cpu, MAX_DEG)
        h1 = F.relu(F.linear(agg1, W1, b1))
        agg2 = gcn_aggregate_pattern(h1, src_padded_cpu.long(), norm_padded_cpu, inv_degree_cpu, MAX_DEG)
        out_ref = F.linear(agg2, W2, b2)

    x = x0.to(device)
    src_padded = src_padded_cpu.to(device)
    norm_padded = norm_padded_cpu.to(device)
    inv_degree = inv_degree_cpu.to(device)
    W1 = W1.to(device)
    b1 = b1.to(device)
    W2 = W2.to(device)
    b2 = b2.to(device)

    xm.mark_step()
    xm.wait_device_ops()

    agg1 = gm(x, src_padded, norm_padded, inv_degree, MAX_DEG)
    h1 = F.relu(F.linear(agg1, W1, b1))
    agg2 = gm(h1, src_padded, norm_padded, inv_degree, MAX_DEG)
    out = F.linear(agg2, W2, b2)

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

