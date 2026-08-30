"""Validate and launch the six-run CIFAR-10 attention stability sweep."""

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from image_classification.paths import RunPaths
from scripts.run_baselines import build_plan, load_sweep

SWEEP_PATH = ROOT / "configs/sweeps/cifar10_stability.yaml"
RUNNER_PATH = ROOT / "scripts/run_baselines.py"
LOG_DIRECTORY = ROOT / "artifacts/launcher_logs"
PID_PATH = ROOT / "artifacts/cifar10_stability_launcher.pid"

EXPECTED_CONFIGS = {
    "configs/experiments/position_se_shallow_cbam_shallow.yaml",
    "configs/experiments/position_se_shallow_cbam_middle.yaml",
    "configs/experiments/csgha_se_shallow_cbam_middle.yaml",
}
EXPECTED_VARIANTS = Counter(
    {
        ("hybrid", (1, 2), (1, 2)): 2,
        ("hybrid", (1, 2), (7, 8)): 2,
        ("csgha", (1, 2), (7, 8)): 2,
    }
)


def build_command(*extra_arguments: str) -> list[str]:
    return [
        sys.executable,
        str(RUNNER_PATH),
        "--sweep",
        str(SWEEP_PATH),
        *extra_arguments,
    ]


def validated_plan() -> list[dict]:
    sweep = load_sweep(SWEEP_PATH)
    plan = build_plan(sweep)
    config_paths = {run["config_path"] for run in plan}
    variants = Counter(
        (
            run["resolved_config"]["model_type"],
            tuple(run["resolved_config"]["se_positions"]),
            tuple(run["resolved_config"]["cbam_positions"]),
        )
        for run in plan
    )
    if len(plan) != 6:
        raise ValueError(f"Expected six stability runs, got {len(plan)}")
    if {run["seed"] for run in plan} != {43, 44}:
        raise ValueError("Stability sweep must contain only seeds 43 and 44")
    if config_paths != EXPECTED_CONFIGS:
        raise ValueError(f"Unexpected stability configs: {sorted(config_paths)}")
    if variants != EXPECTED_VARIANTS:
        raise ValueError(f"Unexpected stability variants: {variants}")
    for run in plan:
        config = run["resolved_config"]
        if config["dataset"] != "cifar10" or config["evaluate_test"]:
            raise ValueError(f"Run must be CIFAR-10 validation-only: {run['experiment_id']}")
        protocol = {
            "validation_size": 5000,
            "epochs": 200,
            "batch_size": 128,
            "lr": 0.01,
            "amp": True,
            "accumulation_steps": 1,
            "num_workers": 8,
        }
        if any(config[key] != value for key, value in protocol.items()):
            raise ValueError(f"Unexpected training protocol: {run['experiment_id']}")
        if config["seed"] != run["seed"]:
            raise ValueError(f"Seed mismatch: {run['experiment_id']}")
        if config["model_type"] == "csgha" and (
            config["guidance_position"] != 2 or config["guidance_reduction"] != 4
        ):
            raise ValueError(f"Unexpected CSGHA guidance config: {run['experiment_id']}")
    return plan


def _read_active_pid() -> int | None:
    if not PID_PATH.exists():
        return None
    try:
        pid = int(PID_PATH.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return None
    return pid


def _ensure_restartable(plan: list[dict]) -> None:
    for run in plan:
        paths = RunPaths(run["experiment_id"])
        if not paths.root.exists():
            continue
        summary_path = paths.root / "summary.json"
        if not summary_path.exists():
            raise RuntimeError(
                f"Incomplete output exists for {run['experiment_id']}: {paths.root}"
            )
        saved_config_path = paths.root / "config.yaml"
        with saved_config_path.open(encoding="utf-8") as handle:
            saved_config = yaml.safe_load(handle) or {}
        saved_config.pop("runtime", None)
        if saved_config != run["resolved_config"]:
            raise RuntimeError(f"Completed output has a different config: {paths.root}")
        json.loads(summary_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate, record, and print the six commands without launching training",
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="run sequentially in the current terminal instead of a detached session",
    )
    args = parser.parse_args()

    plan = validated_plan()
    if args.dry_run:
        return subprocess.run(build_command("--dry-run"), cwd=ROOT, check=False).returncode

    active_pid = _read_active_pid()
    if active_pid is not None:
        raise RuntimeError(f"CIFAR-10 stability sweep is already running with PID {active_pid}")
    _ensure_restartable(plan)

    command = build_command()
    if args.foreground:
        return subprocess.run(command, cwd=ROOT, check=False).returncode

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIRECTORY / f"cifar10_stability_{timestamp}.log"
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
    print(f"CIFAR-10 stability sweep started with PID {process.pid}")
    print(f"Log: {log_path}")
    print(f"Monitor: tail -f {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
