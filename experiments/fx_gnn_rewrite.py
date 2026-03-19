"""
torch.fx rewrite utility for GNN aggregation pattern replacement.

Detected pattern (shared by GraphSAGE/GCN/GIN helpers):
  gathered = x[src_padded]
  masked = gathered * mask_like[:, None]
  reduced = masked.view(NUM_NODES, max_deg, F).sum(dim=1)
  out = reduced * inv_degree[:, None]   # optional

This module replaces that subgraph with a single call:
  replacement(x, src_padded, mask_like, max_deg, inv_degree_or_none)

Usage:
  from experiments.fx_gnn_rewrite import trace_and_rewrite_gnn_aggregation
  gm, info = trace_and_rewrite_gnn_aggregation(fn, my_nki_wrapper)
"""

from __future__ import annotations

import builtins
import operator
from dataclasses import dataclass
from typing import Callable

import torch
import torch.fx as fx


@dataclass
class RewriteInfo:
    matched: bool
    has_inv_degree_tail: bool
    replacement_target: str


def _is_unsqueeze_slice_node(node: fx.Node) -> bool:
    # matches: something[:, None]
    if node.op != "call_function" or node.target != operator.getitem:
        return False
    if len(node.args) != 2:
        return False
    index = node.args[1]
    if not isinstance(index, tuple) or len(index) != 2:
        return False
    return index[0] == slice(None, None, None) and index[1] is None


def _find_placeholder_by_name(gm: fx.GraphModule, name: str) -> fx.Node | None:
    for n in gm.graph.nodes:
        if n.op == "placeholder" and n.target == name:
            return n
    return None


def trace_and_rewrite_gnn_aggregation(
    fn: Callable,
    replacement: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor | None], torch.Tensor],
) -> tuple[fx.GraphModule, RewriteInfo]:
    """
    Trace `fn` with torch.fx and replace gather+mask+sum(+optional inv_degree scale)
    with one `replacement(...)` call.

    `replacement` must accept:
      (x, src_padded, mask_like, max_deg, inv_degree_or_none)
    """
    gm = fx.symbolic_trace(fn)

    x_ph = _find_placeholder_by_name(gm, "x")
    src_ph = _find_placeholder_by_name(gm, "src_padded")
    max_deg_ph = _find_placeholder_by_name(gm, "max_deg")
    inv_deg_ph = _find_placeholder_by_name(gm, "inv_degree")
    if x_ph is None or src_ph is None or max_deg_ph is None:
        return gm, RewriteInfo(False, False, replacement.__name__)

    output_node = next(n for n in gm.graph.nodes if n.op == "output")
    ret = output_node.args[0]
    if not isinstance(ret, fx.Node):
        return gm, RewriteInfo(False, False, replacement.__name__)

    has_inv_tail = False
    sum_node = ret
    inv_unsqueeze = None

    # Optional tail: reduced * inv_degree[:, None]
    if ret.op == "call_function" and ret.target == operator.mul:
        lhs, rhs = ret.args
        if isinstance(lhs, fx.Node) and isinstance(rhs, fx.Node):
            if _is_unsqueeze_slice_node(lhs) and lhs.args[0] is inv_deg_ph:
                has_inv_tail = True
                inv_unsqueeze = lhs
                sum_node = rhs
            elif _is_unsqueeze_slice_node(rhs) and rhs.args[0] is inv_deg_ph:
                has_inv_tail = True
                inv_unsqueeze = rhs
                sum_node = lhs

    if not (isinstance(sum_node, fx.Node) and sum_node.op == "call_method" and sum_node.target == "sum"):
        return gm, RewriteInfo(False, False, replacement.__name__)
    if sum_node.kwargs.get("dim", None) != 1:
        return gm, RewriteInfo(False, False, replacement.__name__)

    view_node = sum_node.args[0]
    if not (isinstance(view_node, fx.Node) and view_node.op == "call_method" and view_node.target == "view"):
        return gm, RewriteInfo(False, False, replacement.__name__)

    mul_node = view_node.args[0]
    if not (isinstance(mul_node, fx.Node) and mul_node.op == "call_function" and mul_node.target == operator.mul):
        return gm, RewriteInfo(False, False, replacement.__name__)

    a, b = mul_node.args
    gather_node = None
    mask_unsqueeze = None
    if isinstance(a, fx.Node) and isinstance(b, fx.Node):
        if a.op == "call_function" and a.target == operator.getitem and a.args == (x_ph, src_ph) and _is_unsqueeze_slice_node(b):
            gather_node, mask_unsqueeze = a, b
        elif b.op == "call_function" and b.target == operator.getitem and b.args == (x_ph, src_ph) and _is_unsqueeze_slice_node(a):
            gather_node, mask_unsqueeze = b, a
    if gather_node is None or mask_unsqueeze is None:
        return gm, RewriteInfo(False, False, replacement.__name__)

    mask_like = mask_unsqueeze.args[0]
    if not isinstance(mask_like, fx.Node):
        return gm, RewriteInfo(False, False, replacement.__name__)

    with gm.graph.inserting_before(output_node):
        new_out = gm.graph.call_function(
            replacement,
            args=(x_ph, src_ph, mask_like, max_deg_ph, inv_deg_ph if has_inv_tail else None),
        )

    output_node.args = (new_out,)

    # Remove matched nodes (in reverse dependency order where needed).
    for n in [ret if has_inv_tail else None, inv_unsqueeze, sum_node, view_node, mul_node, gather_node, mask_unsqueeze]:
        if isinstance(n, fx.Node):
            gm.graph.erase_node(n)

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm, RewriteInfo(True, has_inv_tail, replacement.__name__)


def nki_style_fallback(
    x: torch.Tensor,
    src_padded: torch.Tensor,
    mask_like: torch.Tensor,
    max_deg: int,
    inv_degree: torch.Tensor | None,
) -> torch.Tensor:
    """Reference wrapper with the same callable signature as replacement."""
    messages = x[src_padded] * mask_like[:, None]
    out = messages.view(x.shape[0], max_deg, x.shape[1]).sum(dim=1)
    if inv_degree is not None:
        out = out * inv_degree[:, None]
    return out


def _demo() -> None:
    # Local dynamic imports keep this utility standalone.
    import importlib.util
    from pathlib import Path

    def _load_fn(path: str, fn_name: str):
        spec = importlib.util.spec_from_file_location(fn_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, fn_name)

    root = Path(__file__).resolve().parent
    gcn_aggregate_torch = _load_fn(str(root / "gcn_nki" / "run_torch.py"), "gcn_aggregate_torch")
    gin_aggregate_torch = _load_fn(str(root / "gin_nki" / "run_torch.py"), "gin_aggregate_torch")
    neighbor_agg_mean_pytorch = _load_fn(
        str(root / "graphsage_gather_scatter_mean_v5" / "run_torch.py"), "neighbor_agg_mean_pytorch"
    )

    cases = [
        ("gcn", gcn_aggregate_torch, False),
        ("gin", gin_aggregate_torch, False),
        ("sage", neighbor_agg_mean_pytorch, True),
    ]

    torch.manual_seed(0)
    n = 4096
    f = 16
    max_deg = 8
    x = torch.randn(n, f)
    src_padded = torch.randint(0, n, (n * max_deg,), dtype=torch.int64)
    mask_like = torch.rand(n * max_deg)
    inv_degree = torch.rand(n) + 0.1

    for name, fn, uses_inv in cases:
        gm, info = trace_and_rewrite_gnn_aggregation(fn, nki_style_fallback)
        if not info.matched:
            print(f"[{name}] no pattern match")
            continue

        if uses_inv:
            y_ref = fn(x, src_padded, mask_like, inv_degree, max_deg)
            y_new = gm(x, src_padded, mask_like, inv_degree, max_deg)
        else:
            y_ref = fn(x, src_padded, mask_like, max_deg)
            y_new = gm(x, src_padded, mask_like, max_deg)

        max_abs = (y_ref - y_new).abs().max().item()
        print(f"[{name}] matched={info.matched} inv_tail={info.has_inv_degree_tail} max_abs_err={max_abs:.6g}")
        print(gm.graph)


if __name__ == "__main__":
    _demo()
