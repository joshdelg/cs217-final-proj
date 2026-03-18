#!/usr/bin/env python3
"""
Autotune scatter_add NKI kernel parameters (currently D_CHUNK sweep).

This script runs the repo's profiler CLI for each configuration, then reads
neuron-profile's `summary-json` to extract:
  - total_time (device wall time)
  - vector_engine_instruction_count

All stdout/stderr from the profiler run and the captured summary-json are saved
to a timestamped output directory so you can archive execution logs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Result:
    d_chunk: int
    tile_nodes: int
    total_time_s: float
    vector_engine_instruction_count: int


def _strip_json_payload(s: str) -> str:
    """neuron-profile prints logs before JSON; strip to first JSON object."""
    first = s.find("{")
    if first < 0:
        raise ValueError("No JSON object found in output")
    return s[first:]


def _parse_summary_json(output: str) -> dict:
    payload = _strip_json_payload(output)
    return json.loads(payload)


def _extract_primary_entry(summary_json: dict) -> dict:
    if not isinstance(summary_json, dict) or not summary_json:
        raise ValueError("summary-json payload was empty or not an object")
    return next(iter(summary_json.values()))


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path, verbose: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[run] cwd={cwd} cmd={' '.join(cmd)}")

    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )

    log_path.write_text(
        "STDOUT:\n"
        + (proc.stdout or "")
        + "\n\nSTDERR:\n"
        + (proc.stderr or ""),
        encoding="utf-8",
    )

    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (rc={proc.returncode}). See: {log_path}")


def tune_scatter_add(
    *,
    d_chunks: list[int],
    tile_nodes: int,
    force: bool,
    no_ingest: bool,
    trials_note: str,
    output_dir: Path,
    verbose: bool,
) -> list[Result]:
    repo_root = Path(__file__).resolve().parents[2]  # .../cs217-final-proj
    exp_name = "scatter_add"
    exp_dir = repo_root / "experiments" / exp_name
    artifacts_dir = exp_dir / "artifacts"

    # Where profiler CLI writes for this experiment.
    nki_art = artifacts_dir / "nki"
    neff_path = nki_art / "model.neff"
    ntff_path = nki_art / "profile.ntff"

    py = sys.executable
    cli_cmd = [
        py,
        "-m",
        "profiler.cli",
        "profile",
        exp_name,
        "--mode",
        "nki",
    ]
    if force:
        cli_cmd.append("--force")
    if no_ingest:
        cli_cmd.append("--no-ingest")

    neuron_venv_bin = Path("/opt/aws_neuronx_venv_pytorch_2_9/bin")
    venv_path = str(neuron_venv_bin)

    results: list[Result] = []

    for d in d_chunks:
        run_dir = output_dir / f"d_chunk_{d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Run profiler for this configuration.
        env = os.environ.copy()
        env["SCATTER_ADD_TILE_NODES"] = str(tile_nodes)
        env["SCATTER_ADD_D_CHUNK"] = str(d)
        if neuron_venv_bin.exists():
            # Ensure `neuron-profile` and related tools resolve from the venv.
            env["PATH"] = venv_path + os.pathsep + env.get("PATH", "")

        profiler_log = run_dir / "profiler_cli.log"
        _run(
            cli_cmd,
            cwd=repo_root,
            env=env,
            log_path=profiler_log,
            verbose=verbose,
        )

        if not neff_path.exists() or not ntff_path.exists():
            raise FileNotFoundError(
                f"Expected profiler artifacts not found under {nki_art}. "
                f"neff={neff_path.exists()} ntff={ntff_path.exists()}"
            )

        # Capture summary-json (to extract metrics).
        summary_log = run_dir / "neuron_profile_summary_json.log"
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
        if verbose:
            print(f"[run] summary_cmd={' '.join(summary_cmd)}")

        proc2 = subprocess.run(
            summary_cmd,
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
        )
        summary_log.write_text(
            "STDOUT:\n" + (proc2.stdout or "") + "\n\nSTDERR:\n" + (proc2.stderr or ""),
            encoding="utf-8",
        )
        if proc2.returncode != 0:
            raise RuntimeError(
                f"neuron-profile view failed for D_CHUNK={d} (rc={proc2.returncode}). "
                f"See {summary_log}"
            )

        summary_json = _parse_summary_json(proc2.stdout or "")
        entry = _extract_primary_entry(summary_json)

        total_time_s = float(entry["total_time"])
        vec_count = int(entry["vector_engine_instruction_count"])

        results.append(
            Result(
                d_chunk=d,
                tile_nodes=tile_nodes,
                total_time_s=total_time_s,
                vector_engine_instruction_count=vec_count,
            )
        )

        # Write per-run result json for easy parsing.
        (run_dir / "result.json").write_text(
            json.dumps(results[-1].__dict__, indent=2),
            encoding="utf-8",
        )

        print(
            f"[result] D_CHUNK={d} total_time={total_time_s * 1e6:.3f} us "
            f"vector_engine_instruction_count={vec_count} ({trials_note})"
        )

    # Select best.
    results_sorted = sorted(results, key=lambda r: r.total_time_s)
    best = results_sorted[0]
    (output_dir / "best.json").write_text(
        json.dumps(best.__dict__, indent=2),
        encoding="utf-8",
    )

    print("\n=== Best config ===")
    print(
        f"D_CHUNK={best.d_chunk}, TILE_NODES={best.tile_nodes}, "
        f"total_time={best.total_time_s * 1e6:.3f} us, "
        f"vector_engine_instruction_count={best.vector_engine_instruction_count}"
    )
    return results_sorted


def main() -> int:
    parser = argparse.ArgumentParser(description="Autotune scatter_add NKI kernel parameters")
    parser.add_argument("--d-chunks", default="4,8,16,32", type=str, help="Comma-separated D_CHUNK list")
    parser.add_argument("--tile-nodes", default=128, type=int, help="SCATTER_ADD_TILE_NODES")
    parser.add_argument("--force", action="store_true", help="Pass --force to profiler.cli (clears stale NEFF)")
    parser.add_argument("--ingest", action="store_true", help="Also ingest into InfluxDB (slower; requires configured DB endpoint/org)")
    parser.add_argument("--output-dir", default=None, type=str, help="Output root for logs (timestamped default)")
    parser.add_argument("--verbose", action="store_true", help="Print more command details to console")
    args = parser.parse_args()

    d_chunks = [int(x.strip()) for x in args.d_chunks.split(",") if x.strip()]
    if not d_chunks:
        raise ValueError("No D_CHUNK values provided")

    ts = time.strftime("%Y%m%d-%H%M%S")
    if args.output_dir:
        out_root = Path(args.output_dir).expanduser().resolve()
    else:
        out_root = Path(__file__).resolve().parent / "artifacts" / "autotune" / ts

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config.json").write_text(
        json.dumps(
            {
                "d_chunks": d_chunks,
                "tile_nodes": args.tile_nodes,
                "force": args.force,
                "ingest": args.ingest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # For now, profiler CLI is always doing exactly one capture per configuration.
    trials_note = "trials=1"

    tune_scatter_add(
        d_chunks=d_chunks,
        tile_nodes=args.tile_nodes,
        force=args.force,
        no_ingest=not bool(args.ingest),
        trials_note=trials_note,
        output_dir=out_root,
        verbose=bool(args.verbose),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

