"""Run experiment scripts (run_torch.py / run_nki.py) and discover NEFF artifacts."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


def get_experiment_dir(experiment_name: str, experiments_root: Path | None = None) -> Path:
    """Resolve experiment name to directory under experiments_root."""
    if experiments_root is None:
        experiments_root = Path(__file__).resolve().parent.parent / "experiments"
    path = experiments_root / experiment_name
    if not path.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {path}")
    return path


def get_profiler_config(experiment_dir: Path, impl: str):
    """Read optional profiler_config.json for neff_path / neff_dir override."""
    config_path = experiment_dir / "profiler_config.json"
    if not config_path.exists():
        return None
    import json
    with open(config_path) as f:
        config = json.load(f)
    return config.get(impl) or config.get("neff_path") or config.get("neff_dir")


def discover_neff(experiment_dir: Path, impl: str) -> Path | None:
    """Find the NEFF to use: config override, or newest *.neff under experiment dir."""
    config = get_profiler_config(experiment_dir, impl)
    if config:
        if isinstance(config, str):
            p = experiment_dir / config if not os.path.isabs(config) else Path(config)
        else:
            p = experiment_dir / config.get("neff_path", config.get("neff_dir", ""))
        if p.exists():
            return p.resolve()
    # Glob *.neff under experiment dir, newest by mtime
    neffs = list(experiment_dir.rglob("*.neff"))
    if not neffs:
        return None
    return max(neffs, key=lambda p: p.stat().st_mtime)


def run_experiment(
    experiment_dir: Path,
    impl: str,
    *,
    env: dict | None = None,
    timeout: int | None = None,
) -> tuple[subprocess.CompletedProcess, float]:
    """Run run_torch.py or run_nki.py from experiment dir; return (result, wall_clock_seconds)."""
    script = experiment_dir / f"run_{impl}.py"
    if not script.exists():
        raise FileNotFoundError(f"Experiment script not found: {script}")

    run_env = os.environ.copy()
    run_env["NEURON_FRAMEWORK_DEBUG"] = "1"
    if impl == "nki":
        run_env["XLA_IR_DEBUG"] = "1"
        run_env["XLA_HLO_DEBUG"] = "1"
    if env:
        run_env.update(env)

    start = time.perf_counter()
    proc = subprocess.run(
        [os.environ.get("PYTHON", "python3"), str(script)],
        cwd=str(experiment_dir),
        env=run_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    return proc, elapsed


def run_and_discover_neff(
    experiment_dir: Path,
    impl: str,
    *,
    env: dict | None = None,
    timeout: int | None = None,
) -> tuple[Path | None, subprocess.CompletedProcess, float]:
    """Run experiment script and return (neff_path or None, process result, wall_clock)."""
    proc, elapsed = run_experiment(experiment_dir, impl, env=env, timeout=timeout)
    neff = discover_neff(experiment_dir, impl)
    return neff, proc, elapsed
