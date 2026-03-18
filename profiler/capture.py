"""Run neuron-profile capture and optional view --ingest-only."""

from __future__ import annotations

import json
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


def summarize_kernel_time(neff_path: Path, ntff_path: Path) -> float | None:
    """Return kernel execution time (seconds) from summary-json, or None if unavailable."""
    cmd = [
        find_neuron_profile(),
        "view",
        "-n",
        str(neff_path),
        "-s",
        str(ntff_path),
        "--output-format",
        "summary-json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    # neuron-profile prints log lines before JSON; strip everything before first '{'
    text = proc.stdout
    first_brace = text.find("{")
    if first_brace == -1:
        return None
    payload = text[first_brace:]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None

    # Current summary-json format is a dict keyed by profile/bucket id, e.g.:
    # { "n_xxx": { "total_time": ..., "total_active_time": ..., ... } }
    if not isinstance(data, dict) or not data:
        return None
    first_entry = next(iter(data.values()))
    if isinstance(first_entry, dict):
        if "total_time" in first_entry:
            return float(first_entry["total_time"])
        if "total_active_time" in first_entry:
            return float(first_entry["total_active_time"])
    return None

