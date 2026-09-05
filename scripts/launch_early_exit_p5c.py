"""Launch the frozen serial P5-C training-component ablation batch."""

import argparse
import fcntl
import hashlib
import json
import shlex
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_classification.paths import ARTIFACTS_DIR, PROJECT_ROOT
from scripts.run_baselines import build_plan, load_sweep

PROTOCOL = PROJECT_ROOT / "reports/experiments/2026-09-05-early-exit-p5c-design/protocol_manifest.json"
SWEEPS = (
    PROJECT_ROOT / "configs/sweeps/early_exit_p5c_cifar10.yaml",
    PROJECT_ROOT / "configs/sweeps/early_exit_p5c_cifar100.yaml",
)
EXPERIMENT_TAG = "p5c_r1"
JOBS = 1
EXPECTED_GPU = "NVIDIA GeForce RTX 3080 Ti"

VARIANT_CONTRACTS = {
    "a1": {"exit_positions": [8], "exit_loss_weights": [0.1], "exit_distillation_alpha": 0.0},
    "a2": {"exit_positions": [8], "exit_loss_weights": [0.2], "exit_distillation_alpha": 0.5},
    "a3": {"exit_positions": [8, 16], "exit_loss_weights": [0.1, 0.15], "exit_distillation_alpha": 0.0},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_protocol() -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_p5c_training":
        raise ValueError("P5-C protocol is not frozen")
    if protocol.get("expected_new_runs") != 18:
        raise ValueError("P5-C protocol must declare exactly 18 new runs")
    gate = protocol["gate_a_evidence"]
    gate_path = PROJECT_ROOT / gate["path"]
    if not gate_path.is_file() or _sha256(gate_path) != gate["sha256"]:
        raise ValueError("P5-A/B Gate A evidence is missing or changed")
    gate_result = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate_result.get("status") != "go_p5c" or gate_result.get("gate_a") != gate["required_gates"]:
        raise ValueError("P5-A/B did not authorize P5-C")
    for relative, expected in protocol["existing_evidence"].items():
        path = PROJECT_ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Existing P5-C evidence mismatch: {relative}")
    return protocol


def _variant(config_path: str) -> str:
    stem = Path(config_path).stem
    for variant in VARIANT_CONTRACTS:
        if stem.endswith(f"_{variant}"):
            return variant
    raise ValueError(f"Unknown P5-C variant config: {config_path}")


def validated_plans(experiment_tag: str = EXPERIMENT_TAG) -> list[dict]:
    _load_protocol()
    runs = []
    for sweep in SWEEPS:
        runs.extend(build_plan(load_sweep(sweep), experiment_tag))
    common = {
        "validation_size": 5000,
        "calibration_size": 5000,
        "evaluate_test": False,
        "epochs": 200,
        "batch_size": 128,
        "lr": 0.01,
        "amp": True,
        "cuda_graph": True,
        "torch_num_threads": 1,
        "measure_inference": False,
        "accumulation_steps": 1,
        "num_workers": 8,
        "prefetch_factor": 8,
        "model_type": "multi_exit",
        "exit_temperature": 3.0,
    }
    counts = Counter()
    for run in runs:
        config = run["resolved_config"]
        if any(config[key] != expected for key, expected in common.items()):
            raise ValueError(f"Unexpected common P5-C config: {run['experiment_id']}")
        expected_split = 20_260_902 if config["dataset"] == "cifar10" else 20_260_904
        expected_seeds = (54, 55, 56) if config["dataset"] == "cifar10" else (66, 67, 68)
        if config["split_seed"] != expected_split or run["seed"] not in expected_seeds:
            raise ValueError(f"Unexpected P5-C seed protocol: {run['experiment_id']}")
        variant = _variant(run["config_path"])
        contract = VARIANT_CONTRACTS[variant]
        if any(config[key] != expected for key, expected in contract.items()):
            raise ValueError(f"Unexpected P5-C {variant.upper()} contract: {run['experiment_id']}")
        counts[(config["dataset"], variant, run["seed"])] += 1
        if not config["experiment_name"].endswith(f"_{experiment_tag}_seed{run['seed']}"):
            raise ValueError("P5-C requires a fresh tagged experiment ID")
    expected_counts = Counter(
        (dataset, variant, seed)
        for dataset, seeds in (("cifar10", (54, 55, 56)), ("cifar100", (66, 67, 68)))
        for variant in VARIANT_CONTRACTS
        for seed in seeds
    )
    if counts != expected_counts or len(runs) != 18:
        raise ValueError("Expected exactly two datasets x three variants x three seeds")
    return runs


def _git_status() -> str:
    return subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True).strip()


def _require_tracked() -> None:
    required = [PROTOCOL, Path(__file__).resolve(), *SWEEPS]
    required.extend(PROJECT_ROOT / run["config_path"] for run in validated_plans())
    for path in required:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(path.relative_to(PROJECT_ROOT))],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
        )


def _run_sweeps(experiment_tag: str) -> int:
    for sweep in SWEEPS:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run_baselines.py"),
            "--sweep",
            str(sweep),
            "--jobs",
            str(JOBS),
            "--experiment-tag",
            experiment_tag,
        ]
        return_code = subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
        if return_code:
            return return_code
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--experiment-tag", default=EXPERIMENT_TAG)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    plan = validated_plans(args.experiment_tag)
    if args.worker:
        return _run_sweeps(args.experiment_tag)
    for run in plan:
        print(shlex.join(run["command"]), flush=True)
    if args.dry_run:
        print(
            f"Validated: {len(plan)} new P5-C runs, concurrency={JOBS}, "
            "datasets=cifar10/cifar100, variants=A1/A2/A3; no test or training started."
        )
        return 0

    changes = _git_status()
    if changes:
        raise RuntimeError(f"Commit all P5-C protocol, code, and configs before execution:\n{changes}")
    _require_tracked()
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != EXPECTED_GPU:
        actual = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable"
        raise RuntimeError(f"P5-C requires {EXPECTED_GPU}; found {actual}")
    processes = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    if any("scripts/train.py" in line or "scripts/run_baselines.py" in line for line in processes.splitlines()):
        raise RuntimeError("An existing training/sweep process was found; do not overlap experiments")
    for run in plan:
        target = ARTIFACTS_DIR / "runs" / run["experiment_id"]
        if target.exists():
            raise RuntimeError(f"Existing output will not be overwritten: {target}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = ARTIFACTS_DIR / "early_exit_p5c_launcher.lock"
    with lock_path.open("a") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Early-exit P5-C is already running") from error
        worker_command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--experiment-tag", args.experiment_tag]
        if args.foreground:
            return subprocess.run(worker_command, cwd=PROJECT_ROOT, check=False).returncode
        log_directory = ARTIFACTS_DIR / "launcher_logs"
        log_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        log_path = log_directory / f"early_exit_p5c_serial_{args.experiment_tag}_{timestamp}.log"
        with log_path.open("x") as log:
            log.write(f"Command: {shlex.join(worker_command)}\n")
            log.write(f"Frozen protocol SHA-256: {_sha256(PROTOCOL)}\n")
            log.write("18 serial development-only runs on RTX 3080 Ti; official test disabled.\n")
            log.flush()
            process = subprocess.Popen(
                worker_command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(lock_handle.fileno(),),
            )
        print(f"Early-exit P5-C started with PID {process.pid}")
        print(f"Matrix: {len(plan)} runs; concurrency={JOBS}; GPU={EXPECTED_GPU}")
        print("Order: CIFAR-10 A1/A2/A3, then CIFAR-100 A1/A2/A3; three seeds each")
        print(f"Log: {log_path}")
        print(f"Monitor: tail -f {shlex.quote(str(log_path))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
