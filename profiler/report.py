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
    if t["mean"] is None or t["n"] == 0:
        print("  torch  no kernel timings captured")
    else:
        print(
            f"  torch  mean = {t['mean']:.6f} s  (n={t['n']})"
            + (f"  std = {t['std']:.6f} s" if t["std"] is not None else "")
        )
    if n["mean"] is None or n["n"] == 0:
        print("  nki    no kernel timings captured")
    else:
        print(
            f"  nki    mean = {n['mean']:.6f} s  (n={n['n']})"
            + (f"  std = {n['std']:.6f} s" if n["std"] is not None else "")
        )
    if (
        t["mean"] is not None
        and n["mean"] is not None
        and t["mean"] > 0
        and n["mean"] > 0
    ):
        ratio = t["mean"] / n["mean"]
        print(f"  torch/nki  = {ratio:.3f}x")
    print("---\n")


def save_compare_report(
    artifacts_dir: Path,
    torch_timings: list[float],
    nki_timings: list[float],
    *,
    torch_source: str | None = None,
    nki_source: str | None = None,
) -> Path:
    """Write compare_report.json to artifacts dir and return path.

    The report includes both summary stats and the raw timings for each run.
    """
    torch_summary = aggregate_timings(torch_timings)
    torch_summary["runs"] = torch_timings
    if torch_source:
        torch_summary["time_source"] = torch_source

    nki_summary = aggregate_timings(nki_timings)
    nki_summary["runs"] = nki_timings
    if nki_source:
        nki_summary["time_source"] = nki_source

    out = {
        "torch": torch_summary,
        "nki": nki_summary,
    }
    # Clarify what we measure: total_time = device wall time (includes memory latency).
    if torch_source or nki_source:
        out["_note"] = (
            "total_time = device wall time (includes memory latency, DMA, stalls). "
            "total_active_time = sum of engine-active intervals only (would hide memory cost)."
        )
    report_path = artifacts_dir / "compare_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(out, f, indent=2)
    return report_path
