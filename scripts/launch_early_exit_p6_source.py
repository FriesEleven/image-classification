"""Launch the frozen serial P6 ResNet-18 source stage on RTX 3080 Ti."""

from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from image_classification.paths import ARTIFACTS_DIR
from scripts.run_baselines import build_plan, load_sweep

PROTOCOL = ROOT / "reports/experiments/2026-09-05-early-exit-p6-design/protocol_manifest.json"
SWEEPS = (
    ROOT / "configs/sweeps/early_exit_p6_source_cifar10.yaml",
    ROOT / "configs/sweeps/early_exit_p6_source_cifar100.yaml",
)
EXPERIMENT_TAG = "p6_source_r1"
EXPECTED_GPU = "NVIDIA GeForce RTX 3080 Ti"
EXPECTED_RUNS = 12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol() -> dict:
    protocol = json.loads(PROTOCOL.read_text())
    if protocol.get("status") != "frozen_before_p6_source_training":
        raise ValueError("P6 source protocol is not frozen")
    if protocol["run_accounting"]["source_runs"] != EXPECTED_RUNS:
        raise ValueError("P6 source run count changed")
    for evidence in protocol["upstream_gates"]:
        path = ROOT / evidence["path"]
        if sha256(path) != evidence["sha256"]:
            raise ValueError(f"Upstream P6 evidence changed: {evidence['path']}")
        if json.loads(path.read_text()).get("status") != evidence["required_status"]:
            raise ValueError(f"Upstream P6 gate failed: {evidence['path']}")
    architecture = protocol["architecture_profile"]
    profile_path = ROOT / architecture["path"]
    if sha256(profile_path) != architecture["sha256"]:
        raise ValueError("P6 architecture profile changed")
    profile = json.loads(profile_path.read_text())
    selections = {
        (item["selected"]["deployable"]["position"],
         item["selected"]["training_only_auxiliary"]["position"])
        for item in profile["profiles"]
    }
    if selections != {(2, 6)} or profile.get("data_or_labels_accessed"):
        raise ValueError("P6 architecture-selection contract failed")
    return protocol


def validated_plans(experiment_tag: str = EXPERIMENT_TAG) -> list[dict]:
    protocol = load_protocol()
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
    }
    counts = Counter()
    for run in runs:
        config = run["resolved_config"]
        if any(config[key] != expected for key, expected in common.items()):
            raise ValueError(f"Unexpected P6 training config: {run['experiment_id']}")
        dataset = config["dataset"]
        expected_split = protocol["datasets"][dataset]["split_seed"]
        expected_seeds = protocol["datasets"][dataset]["source_training_seeds"]
        if config["split_seed"] != expected_split or run["seed"] not in expected_seeds:
            raise ValueError(f"Unexpected P6 source seed protocol: {run['experiment_id']}")
        variant = "A0" if config["model_type"] == "resnet18" else "A3"
        expected = protocol["models_per_seed"][variant]
        if config["model_type"] != expected["model_type"]:
            raise ValueError(f"Unexpected P6 model: {run['experiment_id']}")
        if variant == "A3":
            for field in ("exit_positions", "exit_loss_weights", "exit_distillation_alpha"):
                if config[field] != expected[field]:
                    raise ValueError(f"Unexpected P6 A3 contract: {run['experiment_id']} {field}")
        counts[(dataset, variant, run["seed"])] += 1
        if not config["experiment_name"].endswith(f"_{experiment_tag}_seed{run['seed']}"):
            raise ValueError("P6 requires fresh tagged experiment IDs")
    expected_counts = Counter(
        (dataset, variant, seed)
        for dataset, values in protocol["datasets"].items()
        for variant in ("A0", "A3")
        for seed in values["source_training_seeds"]
    )
    if counts != expected_counts or len(runs) != EXPECTED_RUNS:
        raise ValueError("P6 source matrix must be 2 datasets x 2 variants x 3 seeds")
    return runs


def git_status() -> str:
    return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip()


def require_tracked(plan: list[dict]) -> None:
    required = [PROTOCOL, Path(__file__).resolve(),
                ROOT / "scripts/analysis/profile_early_exit_p6_architecture.py", *SWEEPS]
    required.extend(ROOT / run["config_path"] for run in plan)
    for path in required:
        subprocess.run(["git", "ls-files", "--error-unmatch", "--", str(path.relative_to(ROOT))],
                       cwd=ROOT, capture_output=True, check=True)


def run_sweeps(experiment_tag: str) -> int:
    for sweep in SWEEPS:
        command = [sys.executable, str(ROOT / "scripts/run_baselines.py"),
                   "--sweep", str(sweep), "--jobs", "1", "--experiment-tag", experiment_tag]
        return_code = subprocess.run(command, cwd=ROOT, check=False).returncode
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
        return run_sweeps(args.experiment_tag)
    for run in plan:
        print(shlex.join(run["command"]), flush=True)
    if args.dry_run:
        print(f"Validated: {len(plan)} P6 source runs, serial RTX 3080 Ti, no test/training started.")
        return 0
    changes = git_status()
    if changes:
        raise RuntimeError(f"Commit P6 source code and protocol before execution:\n{changes}")
    require_tracked(plan)
    actual_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable"
    if actual_gpu != EXPECTED_GPU:
        raise RuntimeError(f"P6 source requires {EXPECTED_GPU}; found {actual_gpu}")
    processes = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    conflicts = ("scripts/train.py", "scripts/run_baselines.py", "benchmark_early_exit_p8")
    if any(any(name in line for name in conflicts) for line in processes.splitlines()):
        raise RuntimeError("A conflicting training or benchmark process is running")
    for run in plan:
        if (ARTIFACTS_DIR / "runs" / run["experiment_id"]).exists():
            raise RuntimeError(f"Existing output will not be overwritten: {run['experiment_id']}")

    lock_path = ARTIFACTS_DIR / "early_exit_p6_source_launcher.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker = [sys.executable, str(Path(__file__).resolve()),
                  "--worker", "--experiment-tag", args.experiment_tag]
        if args.foreground:
            return subprocess.run(worker, cwd=ROOT, check=False).returncode
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        log_dir = ARTIFACTS_DIR / "launcher_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"early_exit_p6_source_{args.experiment_tag}_{timestamp}.log"
        with log_path.open("x") as handle:
            handle.write(f"Command: {shlex.join(worker)}\nFrozen protocol SHA-256: {sha256(PROTOCOL)}\n")
            handle.write("12 serial development-only source runs; official test disabled.\n")
            handle.flush()
            process = subprocess.Popen(worker, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True, pass_fds=(lock.fileno(),))
        print(f"P6 source PID: {process.pid}")
        print(f"Matrix: {len(plan)} runs; concurrency=1; GPU={EXPECTED_GPU}")
        print(f"Log: {log_path}")
        print(f"Monitor: tail -f {shlex.quote(str(log_path))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
