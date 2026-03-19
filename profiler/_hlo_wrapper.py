#!/usr/bin/env python3
"""Wrapper that captures pre-optimisation HLO from experiment scripts.

Usage (called by hlo_extract.py, not directly):
    python profiler/_hlo_wrapper.py <experiment_script.py> <pre_opt_output_dir>

Monkey-patches ``xm.mark_step`` so that, immediately before each sync, the
pending XLA computation graph is dumped via ``_get_xla_tensors_hlo``.
"""

from __future__ import annotations

import gc
import os
import sys
import runpy
from pathlib import Path


def _collect_xla_tensors():
    """Return all live XLA tensors that may carry pending lazy ops."""
    import torch
    tensors = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and "xla" in str(obj.device):
                tensors.append(obj)
        except Exception:
            pass
    return tensors


def main() -> None:
    script_path = sys.argv[1]
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch          # noqa: F401
    import torch_xla
    import torch_xla.core.xla_model as xm

    capture_idx = [0]
    _inside = [False]  # reentrancy guard
    _real_mark_step = xm.mark_step

    def _patched_mark_step(*args, **kwargs):
        if _inside[0]:
            return _real_mark_step(*args, **kwargs)
        _inside[0] = True
        try:
            xla_tensors = _collect_xla_tensors()
            if xla_tensors:
                try:
                    hlo_text = torch_xla._XLAC._get_xla_tensors_hlo(xla_tensors)
                    if hlo_text and hlo_text.strip() and "ENTRY" in hlo_text:
                        idx = capture_idx[0]
                        dest = output_dir / f"pre_opt_{idx:04d}.hlo.txt"
                        dest.write_text(hlo_text)
                        capture_idx[0] += 1
                except Exception as e:
                    print(f"[hlo_wrapper] pre-opt capture failed: {e}", file=sys.stderr)
            return _real_mark_step(*args, **kwargs)
        finally:
            _inside[0] = False

    xm.mark_step = _patched_mark_step

    sys.argv = [script_path]
    runpy.run_path(script_path, run_name="__main__")


if __name__ == "__main__":
    main()
