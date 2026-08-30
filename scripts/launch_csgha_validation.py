"""Validate and launch the validation-only bounded CSGHA v3 experiment."""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_classification.config import ExperimentConfig, load_config
from image_classification.paths import RunPaths

CONFIG_PATH = ROOT / "configs/experiments/csgha_se_shallow_cbam_middle.yaml"
TRAIN_PATH = ROOT / "scripts/train.py"
LOG_DIRECTORY = ROOT / "artifacts/launcher_logs"
PID_PATH = ROOT / "artifacts/csgha_v3_validation_launcher.pid"


def build_command() -> list[str]:
    return [sys.executable, str(TRAIN_PATH), "--config", str(CONFIG_PATH)]


def load_target_config() -> ExperimentConfig:
    config = load_config(["--config", str(CONFIG_PATH)])
    expected = {
        "experiment_name": "csgha_v3_se1-2_cbam7-8",
        "model_type": "csgha",
        "dataset": "cifar10",
        "evaluate_test": False,
        "se_positions": (1, 2),
        "cbam_positions": (7, 8),
        "guidance_position": 2,
    }
    mismatches = [
        f"{key}={getattr(config, key)!r} (expected {value!r})"
        for key, value in expected.items()
        if getattr(config, key) != value
    ]
    if mismatches:
        raise ValueError("Invalid CSGHA candidate config: " + "; ".join(mismatches))
    return config


def target_run_directory() -> Path:
    return RunPaths(load_target_config().experiment_id).root


def _read_active_pid() -> int | None:
    if not PID_PATH.exists():
        return None
    try:
        pid = int(PID_PATH.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return None
    return pid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the training command without launching it",
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="run in the current terminal instead of a detached background session",
    )
    args = parser.parse_args()

    run_directory = target_run_directory()
    command = build_command()
    print(" ".join(command), flush=True)
    print(f"Run directory: {run_directory}", flush=True)
    if args.dry_run:
        return 0

    active_pid = _read_active_pid()
    if active_pid is not None:
        raise RuntimeError(f"CSGHA v3 validation is already running with PID {active_pid}")
    if run_directory.exists():
        raise RuntimeError(
            "CSGHA output already exists. Inspect and move it before restarting: "
            f"{run_directory}"
        )

    if args.foreground:
        return subprocess.run(command, cwd=ROOT, check=False).returncode

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIRECTORY / f"csgha_v3_validation_{timestamp}.log"
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
    print(f"CSGHA v3 validation started with PID {process.pid}")
    print(f"Log: {log_path}")
    print(f"Monitor: tail -f {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
