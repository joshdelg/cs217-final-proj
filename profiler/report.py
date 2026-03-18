"""Compare mode: aggregate timings and write report."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


def aggregate_timings(timings: list[float]) -> dict:
    """Compute mean and optional std from a list of wall-clock times."""
    if not timings:
        return {"mean": None, "std": None, "n": 0}
    n = len(timings)
    mean = statistics.mean(timings)
    std = statistics.stdev(timings) if n > 1 else None
    return {"mean": mean, "std": std, "n": n}


def print_compare_report(torch_timings: list[float], nki_timings: list[float]) -> None:
    """Print a short comparison table to stdout."""
    t = aggregate_timings(torch_timings)
    n = aggregate_timings(nki_timings)
    print("\n--- Compare report ---")
    print(f"  torch  mean = {t['mean']:.4f} s  (n={t['n']})" + (f"  std = {t['std']:.4f} s" if t['std'] is not None else ""))
    print(f"  nki    mean = {n['mean']:.4f} s  (n={n['n']})" + (f"  std = {n['std']:.4f} s" if n['std'] is not None else ""))
    if t["mean"] and n["mean"] and n["mean"] > 0:
        ratio = t["mean"] / n["mean"]
        print(f"  torch/nki  = {ratio:.3f}x")
    print("---\n")


def save_compare_report(
    artifacts_dir: Path,
    torch_timings: list[float],
    nki_timings: list[float],
) -> Path:
    """Write compare_report.json to artifacts dir and return path."""
    out = {
        "torch": aggregate_timings(torch_timings),
        "nki": aggregate_timings(nki_timings),
    }
    report_path = artifacts_dir / "compare_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(out, f, indent=2)
    return report_path
