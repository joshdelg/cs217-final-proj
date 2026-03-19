"""
Autotune GATHER_MEAN_D_CHUNK for graphsage_gather_scatter_mean_v2 (Option 2).

For each candidate D_CHUNK:
  1) Set env var `GATHER_MEAN_D_CHUNK`
  2) Run profiler in NKI mode to generate NEFF/NTFF artifacts
  3) Run `neuron-profile view --output-format summary-json` and parse `total_time`

We minimize `total_time` (device wall time for the profiled kernel).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_NAME = "graphsage_gather_scatter_mean_v2"
EXP_DIR = REPO_ROOT / "experiments" / EXP_NAME
ART_DIR = EXP_DIR / "artifacts"
NKI_DIR = ART_DIR / "nki"


@dataclass
class Result:
    d_chunk: int
    total_time_s: float
    vector_engine_instruction_count: int | None


def _run(cmd: list[str], *, env: dict[str, str], cwd: Path, log_path: Path | None, verbose: bool) -> subprocess.CompletedProcess:
    if verbose:
        print("Running:", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "COMMAND:\n"
            + " ".join(cmd)
            + "\n\nSTDOUT:\n"
            + (proc.stdout or "")
            + "\n\nSTDERR:\n"
            + (proc.stderr or ""),
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")
    return proc


def _parse_summary_json(output: str) -> tuple[float, int | None]:
    # neuron-profile view prints logs before JSON; extract first '{'
    first_brace = output.find("{")
    if first_brace < 0:
        raise ValueError("Could not find JSON payload in neuron-profile output")
    payload = output[first_brace:]
    data = json.loads(payload)
    if not isinstance(data, dict) or not data:
        raise ValueError("summary-json payload was empty or invalid")
    first_entry = next(iter(data.values()))
    if not isinstance(first_entry, dict):
        raise ValueError("summary-json first entry was not an object")
    if "total_time" not in first_entry:
        raise ValueError("summary-json did not contain total_time")
    total_time_s = float(first_entry["total_time"])
    vec_count = first_entry.get("vector_engine_instruction_count", None)
    vec_count_i = int(vec_count) if vec_count is not None else None
    return total_time_s, vec_count_i


def autotune(
    *,
    d_chunks: list[int],
    trials_note: str,
    verbose: bool,
    output_dir: Path,
) -> list[Result]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[Result] = []

    neff_path = NKI_DIR / "model.neff"
    ntff_path = NKI_DIR / "profile.ntff"

    for d in d_chunks:
        run_dir = output_dir / f"d_chunk_{d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["GATHER_MEAN_D_CHUNK"] = str(d)

        # Run profiler to capture a fresh profile for this D_CHUNK.
        profiler_log = run_dir / "profiler_cli.log"
        cli_cmd = [
            sys.executable,
            "-m",
            "profiler",
            "profile",
            EXP_NAME,
            "--mode",
            "nki",
            "--force",
        ]

        _run(
            cli_cmd,
            env=env,
            cwd=REPO_ROOT,
            log_path=profiler_log,
            verbose=verbose,
        )

        if not neff_path.exists():
            raise FileNotFoundError(f"Expected NEFF not found: {neff_path}")
        if not ntff_path.exists():
            raise FileNotFoundError(f"Expected NTFF not found: {ntff_path}")

        # Extract summary metrics from the captured artifacts.
        summary_log = run_dir / "neuron_profile_summary.json.log"
        summary_cmd = [
            "neuron-profile",
            "view",
            "--output-format",
            "summary-json",
            "-n",
            str(neff_path),
            "-s",
            str(ntff_path),
            "--json-pretty-print",
        ]

        proc2 = _run(
            summary_cmd,
            env=os.environ.copy(),
            cwd=REPO_ROOT,
            log_path=summary_log,
            verbose=verbose,
        )

        total_time_s, vec_count = _parse_summary_json(proc2.stdout or "")
        results.append(Result(d_chunk=d, total_time_s=total_time_s, vector_engine_instruction_count=vec_count))
        print(f"D_CHUNK={d}: total_time={total_time_s:.6f}s, vector_inst={vec_count}")

    results.sort(key=lambda r: r.total_time_s)
    best = results[0]
    print(f"\nBest: D_CHUNK={best.d_chunk} total_time={best.total_time_s:.6f}s")
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-chunks", type=str, default="4,8,16,32", help="Comma-separated candidate D_CHUNK values")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--trials-note", type=str, default="nki-only capture total_time")
    args = ap.parse_args()

    d_chunks = [int(x) for x in args.d_chunks.split(",") if x.strip()]

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = EXP_DIR / "artifacts" / f"autotune_d_chunk_{ts}"

    autotune(d_chunks=d_chunks, trials_note=args.trials_note, verbose=args.verbose, output_dir=out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

