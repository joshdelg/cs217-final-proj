"""Run neuron-profile capture and optional view --ingest-only."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def find_neuron_profile() -> str:
    """Return path to neuron-profile (prefer system install)."""
    return shutil.which("neuron-profile") or "neuron-profile"


def capture(
    neff_path: Path,
    ntff_path: Path,
    *,
    ignore_exec_errors: bool = True,
    enable_dge_notifs: bool = False,
) -> subprocess.CompletedProcess:
    """Run neuron-profile capture -n <neff> -s <ntff>."""
    cmd = [
        find_neuron_profile(),
        "capture",
        "-n", str(neff_path),
        "-s", str(ntff_path),
    ]
    if ignore_exec_errors:
        cmd.append("--ignore-exec-errors")
    if enable_dge_notifs:
        cmd.append("--enable-dge-notifs")
    return subprocess.run(cmd, capture_output=True, text=True)


def ingest(
    neff_path: Path,
    ntff_path: Path,
    *,
    db_endpoint: str | None = None,
    db_org: str | None = None,
    db_bucket: str | None = None,
    force: bool = False,
) -> subprocess.CompletedProcess:
    """Run neuron-profile view --ingest-only to push profile into InfluxDB."""
    cmd = [
        find_neuron_profile(),
        "view",
        "-n", str(neff_path),
        "-s", str(ntff_path),
        "--ingest-only",
        "--output-format", "db",
    ]
    if db_endpoint:
        cmd.extend(["--db-endpoint", db_endpoint])
    if db_org:
        cmd.extend(["--db-org", db_org])
    if db_bucket:
        cmd.extend(["--db-bucket", db_bucket])
    if force:
        cmd.append("--force")
    return subprocess.run(cmd, capture_output=True, text=True)
