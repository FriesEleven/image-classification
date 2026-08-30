"""Validate and launch the three validation-only position experiments."""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_classification.config import load_config
from image_classification.paths import RunPaths

SWEEP_PATH = ROOT / "configs/sweeps/position_screening.yaml"
RUNNER_PATH = ROOT / "scripts/run_experiments.py"
LOG_DIRECTORY = ROOT / "artifacts/launcher_logs"
PID_PATH = ROOT / "artifacts/position_screening_launcher.pid"


def build_command(*extra_arguments: str) -> list[str]:
    return [
        sys.executable,
        str(RUNNER_PATH),
        "--sweep",
        str(SWEEP_PATH),
        *extra_arguments,
    ]


def position_run_directories() -> list[Path]:
    with SWEEP_PATH.open(encoding="utf-8") as handle:
        experiments = (yaml.safe_load(handle) or {}).get("experiments", [])
    if len(experiments) != 3:
        raise ValueError(f"Expected three position experiments in {SWEEP_PATH}")
    directories = []
    for config_value in experiments:
        config_path = (ROOT / config_value).resolve()
        config = load_config(["--config", str(config_path)])
        if config.evaluate_test:
            raise ValueError(f"Position config must be validation-only: {config_path}")
        directories.append(RunPaths(config.experiment_id).root)
    return directories


def _read_active_pid() -> int | None:
    if not PID_PATH.exists():
        return None
    try:
        pid = int(PID_PATH.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return None
    return pid


def _ensure_outputs_are_new() -> None:
    existing = [path for path in position_run_directories() if path.exists()]
    if existing:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise RuntimeError(
            "Position experiment output already exists. Inspect and move it before restarting:\n"
            f"{formatted}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the three training commands without launching them",
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="run in the current terminal instead of a detached background session",
    )
    args = parser.parse_args()

    # Resolve every config here so dry-run also enforces validation-only semantics.
    position_run_directories()
    dry_run = subprocess.run(build_command("--dry-run"), cwd=ROOT, check=False)
    if dry_run.returncode or args.dry_run:
        return dry_run.returncode

    active_pid = _read_active_pid()
    if active_pid is not None:
        raise RuntimeError(f"Position screening is already running with PID {active_pid}")
    _ensure_outputs_are_new()

    command = build_command()
    if args.foreground:
        return subprocess.run(command, cwd=ROOT, check=False).returncode

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIRECTORY / f"position_screening_{timestamp}.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Command: {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(f"{process.pid}\n", encoding="utf-8")
    print(f"Position screening started with PID {process.pid}")
    print(f"Log: {log_path}")
    print(f"Monitor: tail -f {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
