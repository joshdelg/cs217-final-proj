"""CLI: profile <experiment_name> --mode torch|nki|compare [--trials N] [--ingest] ..."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import capture, hlo_extract, report, runner

CONFIG_FILENAME = "profiler_influx.json"


def _load_influx_config() -> dict:
    """Load InfluxDB defaults from profiler_influx.json (repo root or cwd)."""
    repo_root = Path(__file__).resolve().parent.parent
    for base in (repo_root, Path.cwd()):
        path = base / CONFIG_FILENAME
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
    return {}


def _apply_influx_defaults(args: argparse.Namespace) -> None:
    """Fill in db_endpoint, db_org from config when not set on CLI.

    We intentionally do NOT pull db_bucket from config so that each implementation
    (torch/nki) can use a descriptive bucket name like <experiment>_torch / <experiment>_nki.
    """
    config = _load_influx_config()
    if not config:
        return
    if getattr(args, "db_endpoint", None) is None and config.get("db_endpoint"):
        args.db_endpoint = config["db_endpoint"]
    if getattr(args, "db_org", None) is None and config.get("db_org"):
        args.db_org = config["db_org"]


def _artifact_dir(experiment_dir: Path, impl: str) -> Path:
    return experiment_dir / "artifacts" / impl


def _run_one_impl(
    experiment_dir: Path,
    impl: str,
    *,
    do_capture: bool = True,
    do_ingest: bool = False,
    enable_dge_notifs: bool = False,
    force_recompile: bool = False,
    db_endpoint: str | None = None,
    db_org: str | None = None,
    db_bucket: str | None = None,
) -> tuple[float, Path | None]:
    """Run script, optionally capture and ingest. Returns (wall_clock, ntff_path or None)."""
    neff, proc, elapsed = runner.run_and_discover_neff(experiment_dir, impl, force_recompile=force_recompile)
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"{impl}.py exited with code {proc.returncode}")

    if not neff or not do_capture:
        return elapsed, None

    art = _artifact_dir(experiment_dir, impl)
    art.mkdir(parents=True, exist_ok=True)
    # Copy NEFF to artifacts with a stable name for neuron-profile
    dest_neff = art / "model.neff"
    shutil.copy2(neff, dest_neff)
    ntff_path = art / "profile.ntff"

    cap = capture.capture(dest_neff, ntff_path, enable_dge_notifs=enable_dge_notifs)
    if cap.returncode != 0:
        if cap.stderr:
            print(cap.stderr, file=sys.stderr)
        raise RuntimeError(f"neuron-profile capture failed: {cap.returncode}")

    if do_ingest:
        ing = capture.ingest(
            dest_neff, ntff_path,
            db_endpoint=db_endpoint, db_org=db_org, db_bucket=db_bucket,
        )
        if ing.returncode != 0 and ing.stderr:
            print(ing.stderr, file=sys.stderr)

    return elapsed, ntff_path


def _db_bucket_for_impl(
    args: argparse.Namespace, experiment_dir: Path, impl: str, *, do_ingest: bool = False
) -> str | None:
    """Human-readable InfluxDB bucket for this profile when ingesting."""
    if args.db_bucket:
        return args.db_bucket
    if not do_ingest:
        return None
    name = getattr(args, "profile_name", None) or experiment_dir.name
    return f"{name}_{impl}"


def cmd_profile(args: argparse.Namespace) -> int:
    _apply_influx_defaults(args)
    enable_dge = True
    experiments_root = Path(args.experiments_root).resolve() if args.experiments_root else None
    experiment_dir = runner.get_experiment_dir(args.experiment_name, experiments_root)

    force = getattr(args, "force", False)

    if args.mode == "torch":
        do_ingest = args.ingest
        _run_one_impl(
            experiment_dir, "torch",
            do_capture=True, do_ingest=do_ingest, enable_dge_notifs=enable_dge,
            force_recompile=force,
            db_endpoint=args.db_endpoint, db_org=args.db_org,
            db_bucket=args.db_bucket or _db_bucket_for_impl(args, experiment_dir, "torch", do_ingest=do_ingest),
        )
        print("Torch: run + capture done. Artifacts in", _artifact_dir(experiment_dir, "torch"))
        return 0

    if args.mode == "nki":
        do_ingest = args.ingest
        _run_one_impl(
            experiment_dir, "nki",
            do_capture=True, do_ingest=do_ingest, enable_dge_notifs=enable_dge,
            force_recompile=force,
            db_endpoint=args.db_endpoint, db_org=args.db_org,
            db_bucket=args.db_bucket or _db_bucket_for_impl(args, experiment_dir, "nki", do_ingest=do_ingest),
        )
        print("NKI: run + capture done. Artifacts in", _artifact_dir(experiment_dir, "nki"))
        return 0

    # compare: ingest is on by default when DB is configured (use --no-ingest to disable)
    have_db = bool(args.db_endpoint and args.db_org)
    do_ingest = (args.ingest or (not getattr(args, "no_ingest", False) and have_db))
    trials = max(1, int(args.trials))
    torch_kernel_times: list[float] = []
    nki_kernel_times: list[float] = []
    torch_time_source: str | None = None
    nki_time_source: str | None = None

    print("Compare mode: running hardware-profiled trials (first run may take 5–15+ min for Neuron compilation)...")

    # Helper: ensure NEFF exists and copied into artifacts once per impl
    def _prepare_neff(impl: str) -> tuple[Path, Path]:
        neff, proc, _ = runner.run_and_discover_neff(experiment_dir, impl, force_recompile=force)
        if proc.returncode != 0:
            if proc.stderr:
                print(proc.stderr, file=sys.stderr)
            raise RuntimeError(f"{impl}.py exited with code {proc.returncode}")
        if not neff:
            raise RuntimeError(f"No NEFF found for {impl}")
        art = _artifact_dir(experiment_dir, impl)
        art.mkdir(parents=True, exist_ok=True)
        dest_neff = art / "model.neff"
        shutil.copy2(neff, dest_neff)
        return dest_neff, art

    # Torch trials
    dest_neff_torch, art_torch = _prepare_neff("torch")
    last_ntff_torch: Path | None = None
    for i in range(trials):
        print(f"  [torch] trial {i + 1}/{trials} (kernel time)...", flush=True)
        ntff_path = art_torch / f"profile_torch_trial_{i + 1}.ntff"
        cap = capture.capture(dest_neff_torch, ntff_path, enable_dge_notifs=enable_dge)
        if cap.returncode != 0:
            if cap.stderr:
                print(cap.stderr, file=sys.stderr)
            return 1
        kt, src = capture.summarize_kernel_time(dest_neff_torch, ntff_path)
        if kt is None:
            print("Warning: unable to read kernel time for torch trial", i + 1, file=sys.stderr)
        else:
            torch_kernel_times.append(kt)
            torch_time_source = src or torch_time_source
        last_ntff_torch = ntff_path

    # NKI trials
    dest_neff_nki, art_nki = _prepare_neff("nki")
    last_ntff_nki: Path | None = None
    for i in range(trials):
        print(f"  [nki]   trial {i + 1}/{trials} (kernel time)...", flush=True)
        ntff_path = art_nki / f"profile_nki_trial_{i + 1}.ntff"
        cap = capture.capture(dest_neff_nki, ntff_path, enable_dge_notifs=enable_dge)
        if cap.returncode != 0:
            if cap.stderr:
                print(cap.stderr, file=sys.stderr)
            return 1
        kt, src = capture.summarize_kernel_time(dest_neff_nki, ntff_path)
        if kt is None:
            print("Warning: unable to read kernel time for nki trial", i + 1, file=sys.stderr)
        else:
            nki_kernel_times.append(kt)
            nki_time_source = src or nki_time_source
        last_ntff_nki = ntff_path

    report.print_compare_report(torch_kernel_times, nki_kernel_times)
    artifacts_dir = experiment_dir / "artifacts"
    report_path = report.save_compare_report(
        artifacts_dir,
        torch_kernel_times,
        nki_kernel_times,
        torch_source=torch_time_source,
        nki_source=nki_time_source,
    )
    print("Report saved to", report_path)

    # For UI / ingest, keep the last NTFF for each impl as profile.ntff
    if last_ntff_torch:
        (art_torch / "profile.ntff").write_bytes(last_ntff_torch.read_bytes())
    if last_ntff_nki:
        (art_nki / "profile.ntff").write_bytes(last_ntff_nki.read_bytes())

    # Optionally ingest profiles into InfluxDB
    if do_ingest:
        if last_ntff_torch:
            capture.ingest(
                dest_neff_torch,
                art_torch / "profile.ntff",
                db_endpoint=args.db_endpoint,
                db_org=args.db_org,
                db_bucket=args.db_bucket
                or _db_bucket_for_impl(args, experiment_dir, "torch", do_ingest=True),
            )
        if last_ntff_nki:
            capture.ingest(
                dest_neff_nki,
                art_nki / "profile.ntff",
                db_endpoint=args.db_endpoint,
                db_org=args.db_org,
                db_bucket=args.db_bucket
                or _db_bucket_for_impl(args, experiment_dir, "nki", do_ingest=True),
            )

    print("Compare artifacts in", experiment_dir / "artifacts")
    if do_ingest and args.db_endpoint and args.db_org:
        print("Profiles ingested to InfluxDB. Start the viewer with the same org to see them:")
        print(f"  neuron-profile view --db-endpoint {args.db_endpoint} --db-org {args.db_org}")
        print("  Then open http://localhost:3001")
    elif have_db:
        print("Tip: Ingest skipped (use without --no-ingest to ingest). Run neuron-profile view to open UI.")
    else:
        print("Tip: To ingest into InfluxDB, set --db-endpoint and --db-org (or pass --no-ingest to suppress this)")
    return 0


def cmd_hlo(args: argparse.Namespace) -> int:
    """Extract and visualise HLO graphs for an experiment."""
    experiments_root = Path(args.experiments_root).resolve() if args.experiments_root else None
    experiment_dir = runner.get_experiment_dir(args.experiment_name, experiments_root)
    force = getattr(args, "force", True)

    images = hlo_extract.extract_and_visualize(
        experiment_dir, args.impl, force_recompile=force,
    )
    if not images:
        return 1

    for stage, path in images.items():
        print(f"  {stage}: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile PyTorch vs NKI experiments on Trainium.")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── profile subcommand ──
    p = sub.add_parser("profile", help="Run and profile an experiment")
    p.add_argument("experiment_name", help="Name of experiment (subdir under experiments/)")
    p.add_argument("--mode", choices=["torch", "nki", "compare"], default="compare",
                    help="Profile torch only, nki only, or both and compare (default: compare)")
    p.add_argument("--trials", type=int, default=1,
                    help="Number of timing trials in compare mode (default: 1)")
    p.add_argument("--experiments-root", type=str, default=None,
                    help="Root directory for experiments (default: repo experiments/)")
    p.add_argument("--ingest", action="store_true", help="Ingest profile into InfluxDB after capture")
    p.add_argument("--no-ingest", action="store_true",
                    help="In compare mode, disable ingest (ingest is on by default for compare)")
    p.add_argument("--profile-name", type=str, default=None,
                    help="Human-readable name for the profile (used as InfluxDB bucket when ingesting; default: experiment_impl e.g. example_torch)")
    p.add_argument("--db-endpoint", type=str, default=None, help="InfluxDB endpoint (e.g. http://localhost:8086)")
    p.add_argument("--db-org", type=str, default=None, help="InfluxDB org")
    p.add_argument("--db-bucket", type=str, default=None,
                    help="InfluxDB bucket (overrides --profile-name when set)")
    p.add_argument("--force", action="store_true",
                    help="Force recompile all NEFFs (clears stale binaries for both torch and NKI; "
                         "by default only NKI NEFFs are cleaned)")
    p.set_defaults(handler=cmd_profile)

    # ── hlo subcommand ──
    h = sub.add_parser("hlo", help="Extract and visualise XLA HLO graphs (before/after optimisation)")
    h.add_argument("experiment_name", help="Name of experiment (subdir under experiments/)")
    h.add_argument("--impl", choices=["torch", "nki"], required=True,
                   help="Which implementation to extract HLO for")
    h.add_argument("--experiments-root", type=str, default=None,
                   help="Root directory for experiments (default: repo experiments/)")
    h.add_argument("--force", action="store_true", default=True,
                   help="Force recompile to ensure fresh HLO dumps (default: true)")
    h.add_argument("--no-force", action="store_false", dest="force",
                   help="Skip recompile if cached NEFF exists")
    h.set_defaults(handler=cmd_hlo)

    args = parser.parse_args()
    handler = getattr(args, "handler", None)
    if handler:
        return handler(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
