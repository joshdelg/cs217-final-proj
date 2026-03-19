from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Cell:
    torch_mean_s: float
    nki_mean_s: float
    speedup: float


def render_speedup_matrix_md(
    *,
    tile_nodes_list: list[int],
    d_chunk_list: list[int],
    results: dict[tuple[int, int], Cell],
) -> str:
    # Deterministic column order for stable diffs.
    cols = d_chunk_list

    header = "| TILE_NODES \\ GMM_D_CHUNK | " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    for t in tile_nodes_list:
        row = [f"| {t} |"]
        for d in cols:
            cell = results.get((t, d))
            if cell is None:
                row.append(" - |")
            else:
                row.append(f" {cell.speedup:.2f}x |")
        lines.append("".join(row))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3, help="Profiler compare trials per cell")
    ap.add_argument("--force", action="store_true", help="Force recompile for both impls (slower)")
    ap.add_argument(
        "--tile-nodes",
        type=str,
        default="64,128",
        help="Comma-separated GMM_TILE_NODES values (must divide NUM_NODES)",
    )
    ap.add_argument(
        "--d-chunks",
        type=str,
        default="16,32",
        help="Comma-separated GMM_D_CHUNK values (must divide MAX_DEG)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Only validate constraints; do not run profiler")
    ap.add_argument(
        "--update-readme",
        action="store_true",
        help="If set, script writes the matrix into the experiment README",
    )
    args = ap.parse_args()

    exp_name = "graphsage_gather_scatter_mean_exaggerated"
    exp_dir = Path(__file__).resolve().parent
    repo_root = exp_dir.parents[1]
    artifacts_dir = exp_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Keep shape defaults constant (fair workload).
    # These should match graphsage_gather_scatter_mean_exaggerated/run_torch.py + run_nki.py defaults.
    NUM_NODES = 8192
    AVG_DEGREE = 24
    FEAT_DIM = 128
    SEED = 42

    tile_nodes_list = [int(x) for x in args.tile_nodes.split(",") if x.strip()]
    d_chunk_list = [int(x) for x in args.d_chunks.split(",") if x.strip()]

    if NUM_NODES % min(tile_nodes_list) != 0:
        raise ValueError(f"NUM_NODES={NUM_NODES} must be divisible by every TILE_NODES")
    for t in tile_nodes_list:
        if NUM_NODES % t != 0:
            raise ValueError(f"NUM_NODES={NUM_NODES} must be divisible by TILE_NODES={t}")

    results: dict[tuple[int, int], Cell] = {}

    for tile_nodes in tile_nodes_list:
        for d_chunk in d_chunk_list:
            print(f"\nRunning cell TILE_NODES={tile_nodes}, GMM_D_CHUNK={d_chunk} ...")

            if args.dry_run:
                # Constraint checks require torch + the same random graph,
                # so dry-run just validates simple divisibility checks.
                results[(tile_nodes, d_chunk)] = Cell(torch_mean_s=float("nan"), nki_mean_s=float("nan"), speedup=float("nan"))  # type: ignore[arg-type]
                continue

            env = os.environ.copy()
            env.update(
                {
                    "GMM_NUM_NODES": str(NUM_NODES),
                    "GMM_AVG_DEGREE": str(AVG_DEGREE),
                    "GMM_FEAT_DIM": str(FEAT_DIM),
                    "GMM_SEED": str(SEED),
                    "GMM_TILE_NODES": str(tile_nodes),
                    "GMM_D_CHUNK": str(d_chunk),
                    # Keep profiler from doing DB ingest (faster, less noise).
                    # We'll pass this flag to profiler as well.
                }
            )

            cmd = [
                sys.executable,
                "-m",
                "profiler",
                "profile",
                exp_name,
                "--mode",
                "compare",
                "--trials",
                str(args.trials),
                "--no-ingest",
            ]
            if args.force:
                cmd.append("--force")

            proc = subprocess.run(
                cmd,
                cwd=str(repo_root),
                env=env,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                # Likely a kernel compile-time assertion (e.g., MAX_DEG % D_CHUNK).
                # Keep the matrix runnable by marking this cell as unavailable.
                print(proc.stdout)
                print(proc.stderr, file=sys.stderr)
                continue

            report_path = artifacts_dir / "compare_report.json"
            if not report_path.exists():
                # Unexpected; skip cell.
                continue
            report = json.loads(report_path.read_text())

            torch_mean = float(report["torch"]["mean"])
            nki_mean = float(report["nki"]["mean"])
            if nki_mean == 0:
                continue
            speedup = torch_mean / nki_mean
            results[(tile_nodes, d_chunk)] = Cell(
                torch_mean_s=torch_mean,
                nki_mean_s=nki_mean,
                speedup=speedup,
            )

    # Persist results for auditability.
    out_json = artifacts_dir / "speedup_matrix.json"
    out_json_payload = {
        "shape_defaults": {
            "NUM_NODES": NUM_NODES,
            "AVG_DEGREE": AVG_DEGREE,
            "FEAT_DIM": FEAT_DIM,
            "SEED": SEED,
            "MAX_DEG": None,
        },
        "trials": args.trials,
        "tile_nodes": tile_nodes_list,
        "d_chunks": d_chunk_list,
        "results": {
            f"{t},{d}": {
                "torch_mean_s": (results[(t, d)].torch_mean_s if (t, d) in results else None),
                "nki_mean_s": (results[(t, d)].nki_mean_s if (t, d) in results else None),
                "speedup": (results[(t, d)].speedup if (t, d) in results else None),
            }
            for t in tile_nodes_list
            for d in d_chunk_list
        },
    }
    out_json.write_text(json.dumps(out_json_payload, indent=2))

    md = render_speedup_matrix_md(
        tile_nodes_list=tile_nodes_list,
        d_chunk_list=d_chunk_list,
        results={k: v for k, v in results.items() if (not math.isnan(v.speedup))},
    )
    out_md = artifacts_dir / "speedup_matrix.md"
    out_md.write_text(md)

    print("\nSpeedup matrix (torch/nki, mean of compare total_time):")
    print(md)

    if args.update_readme:
        readme_path = exp_dir / "README.md"
        readme = readme_path.read_text()

        # Replace an existing section if present; otherwise append.
        heading = "## Speedup matrix"
        if heading in readme:
            before = readme.split(heading, 1)[0]
            # Drop everything from the heading to the next "## " heading (or EOF).
            remainder = readme.split(heading, 1)[1]
            next_heading_idx = remainder.find("\n## ")
            rest = remainder[next_heading_idx + 1 :] if next_heading_idx >= 0 else ""
            updated = before + heading + "\n\n"  # keep heading, replace body
            updated += md + "\n\n"
            updated += (
                "Speedup definition: `torch_total_time_mean / nki_total_time_mean` "
                "(profiler compare mode, Neuron `total_time`), with fixed workload defaults and "
                "sweeping `GMM_TILE_NODES` + `GMM_D_CHUNK`.\n\n"
            )
            updated += rest
        else:
            updated = readme.rstrip() + "\n\n" + "## Speedup matrix" + "\n\n" + md + "\n\n"
            updated += (
                "Speedup definition: `torch_total_time_mean / nki_total_time_mean` "
                "(profiler compare mode, Neuron `total_time`), with fixed workload defaults and "
                "sweeping `GMM_TILE_NODES` + `GMM_D_CHUNK`.\n"
            )

        readme_path.write_text(updated)
        print(f"\nUpdated {readme_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

