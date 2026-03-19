"""
Single-script GCN experiment with toggleable graph interpolation (FX rewrite).

Usage:
  python experiments/gcn_fx_toggle/run.py --graph-interpolation on
  python experiments/gcn_fx_toggle/run.py --graph-interpolation off
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import statistics
import time
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
    raise ImportError("run.py requires neuronxcc (Neuron venv)")

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


def _build_mode_aggregators(
    x: torch.Tensor,
    src_padded: torch.Tensor,
    norm_padded: torch.Tensor,
    inv_degree: torch.Tensor,
    max_deg: int,
):
    trace_and_rewrite_gnn_aggregation = _load_fx_rewrite()
    gm, info = trace_and_rewrite_gnn_aggregation(gcn_aggregate_pattern, gcn_aggregate_nki_replacement)
    if not info.matched:
        raise RuntimeError("FX rewrite did not match aggregation pattern")

    def agg_torch(inp: torch.Tensor):
        return gcn_aggregate_pattern(inp, src_padded.long(), norm_padded, inv_degree, max_deg)

    def agg_manual_nki(inp: torch.Tensor):
        return gather_mean_nki(inp, src_padded, norm_padded, inv_degree)

    def agg_fx_nki(inp: torch.Tensor):
        return gm(inp, src_padded, norm_padded, inv_degree, max_deg)

    return {
        "torch": agg_torch,
        "manual_nki": agg_manual_nki,
        "fx_nki": agg_fx_nki,
    }


def _forward_two_layer(
    aggregate_fn,
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
):
    agg1 = aggregate_fn(x)
    h1 = F.relu(F.linear(agg1, w1, b1))
    agg2 = aggregate_fn(h1)
    return F.linear(agg2, w2, b2)


def _run_mode_trials(mode_name: str, aggregate_fn, x, w1, b1, w2, b2, warmup: int, trials: int):
    # Compile/warmup path (not timed)
    for _ in range(max(1, warmup)):
        _ = _forward_two_layer(aggregate_fn, x, w1, b1, w2, b2)
        xm.mark_step()
        xm.wait_device_ops()

    times = []
    out_last = None
    for i in range(trials):
        t0 = time.perf_counter()
        out_last = _forward_two_layer(aggregate_fn, x, w1, b1, w2, b2)
        xm.mark_step()
        xm.wait_device_ops()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"[{mode_name}] trial_{i + 1}: {elapsed:.6f} s")

    assert out_last is not None
    return times, out_last.cpu()


def main():
    global MAX_DEG

    parser = argparse.ArgumentParser(description="Fair A/B harness for Torch vs manual NKI vs FX-rewrite NKI")
    parser.add_argument(
        "--mode",
        choices=["torch", "manual_nki", "fx_nki", "compare"],
        default="compare",
        help="Single mode or compare all three (default: compare)",
    )
    parser.add_argument("--trials", type=int, default=5, help="Timed trials per mode (default: 5)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations per mode (default: 1)")
    args = parser.parse_args()
    trials = max(1, args.trials)
    warmup = max(1, args.warmup)
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

    x = x0.to(device)
    src_padded = src_padded_cpu.to(device)
    norm_padded = norm_padded_cpu.to(device)
    inv_degree = inv_degree_cpu.to(device)
    W1 = W1.to(device)
    b1 = b1.to(device)
    W2 = W2.to(device)
    b2 = b2.to(device)

    aggregators = _build_mode_aggregators(x, src_padded, norm_padded, inv_degree, MAX_DEG)

    modes = [args.mode] if args.mode != "compare" else ["torch", "manual_nki", "fx_nki"]
    results = {}
    outputs = {}

    print(f"harness_config: warmup={warmup}, trials={trials}, max_deg={MAX_DEG}")
    for mode in modes:
        times, out_cpu = _run_mode_trials(mode, aggregators[mode], x, W1, b1, W2, b2, warmup, trials)
        results[mode] = times
        outputs[mode] = out_cpu

    print("\n=== Timing Summary (wall clock) ===")
    for mode in modes:
        t = results[mode]
        mean_t = statistics.mean(t)
        std_t = statistics.pstdev(t) if len(t) > 1 else 0.0
        print(
            f"{mode}: mean={mean_t:.6f}s std={std_t:.6f}s min={min(t):.6f}s max={max(t):.6f}s"
        )

    if "torch" in outputs:
        torch_out = outputs["torch"]
        for mode in modes:
            if mode == "torch":
                continue
            diff = (outputs[mode] - torch_out).abs()
            print(
                f"{mode}_vs_torch: max_abs_diff={diff.max().item():.6g}, mean_abs_diff={diff.mean().item():.6g}"
            )

    if "torch" in results and "manual_nki" in results:
        print(f"speedup_torch_over_manual_nki: {statistics.mean(results['torch']) / statistics.mean(results['manual_nki']):.3f}x")
    if "torch" in results and "fx_nki" in results:
        print(f"speedup_torch_over_fx_nki: {statistics.mean(results['torch']) / statistics.mean(results['fx_nki']):.3f}x")
    if "manual_nki" in results and "fx_nki" in results:
        print(f"speedup_manual_nki_over_fx_nki: {statistics.mean(results['manual_nki']) / statistics.mean(results['fx_nki']):.3f}x")


if __name__ == "__main__":
    main()

