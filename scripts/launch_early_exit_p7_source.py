"""Launch the frozen serial P7 ImageNet-100 source stage on RTX 3080 Ti."""

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

from image_classification.paths import ARTIFACTS_DIR, DATA_DIR
from scripts.run_baselines import build_plan, load_sweep

PROTOCOL = ROOT / "reports/experiments/2026-09-06-early-exit-p7-imagenet100-design/protocol_manifest.json"
ARCHITECTURE = ROOT / "reports/experiments/2026-09-06-early-exit-p7-imagenet100-design/architecture_profile.json"
SWEEP = ROOT / "configs/sweeps/early_exit_p7_imagenet100_source.yaml"
EXPERIMENT_TAG = "p7_source_r1"
EXPECTED_GPU = "NVIDIA GeForce RTX 3080 Ti"
EXPECTED_RUNS = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol() -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_p7_source_training":
        raise ValueError("P7 source protocol is not frozen")
    if protocol["run_accounting"]["source_runs"] != EXPECTED_RUNS:
        raise ValueError("P7 source run count changed")
    for evidence in protocol["upstream_gates"]:
        path = ROOT / evidence["path"]
        if sha256(path) != evidence["sha256"]:
            raise ValueError(f"Upstream P7 evidence changed: {evidence['path']}")
        if json.loads(path.read_text())["status"] != evidence["required_status"]:
            raise ValueError(f"Upstream P7 gate failed: {evidence['path']}")
    boundary = protocol["prior_boundary"]
    boundary_path = ROOT / boundary["path"]
    if sha256(boundary_path) != boundary["sha256"]:
        raise ValueError("P6 boundary receipt changed")
    if json.loads(boundary_path.read_text())["status"] != boundary["required_status"]:
        raise ValueError("P6 boundary status changed")
    if sha256(ARCHITECTURE) != protocol["architecture_profile"]["sha256"]:
        raise ValueError("P7 architecture profile changed")
    profile = json.loads(ARCHITECTURE.read_text())
    positions = (
        profile["selected"]["deployable"]["position"],
        profile["selected"]["training_only_auxiliary"]["position"],
    )
    if positions != (8, 15) or profile.get("data_or_labels_accessed"):
        raise ValueError("P7 architecture-selection contract failed")
    return protocol


def validate_prepared_dataset(protocol: dict) -> None:
    manifest_path = DATA_DIR / "imagenet100/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = protocol["dataset"]
    expected = {
        "status": "prepared_train_pool",
        "source_archive_sha256": dataset["source_archive_sha256"],
        "central_directory_sha256": dataset["central_directory_sha256"],
        "image_count": dataset["image_count"],
        "class_names": dataset["class_names"],
        "official_test_prepared": False,
        "official_test_accessed": False,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Prepared ImageNet-100 data differs from the frozen protocol")
    if manifest.get("protocol_sha256") != sha256(PROTOCOL):
        raise ValueError("Prepared data was bound to a different P7 protocol")


def validated_plan(experiment_tag: str = EXPERIMENT_TAG) -> list[dict]:
    protocol = load_protocol()
    runs = build_plan(load_sweep(SWEEP), experiment_tag)
    common = {
        "dataset": "imagenet100",
        "validation_size": 10_000,
        "calibration_size": 10_000,
        "split_seed": 20_261_003,
        "evaluate_test": False,
        "epochs": 100,
        "batch_size": 128,
        "lr": 0.01,
        "amp": True,
        "cuda_graph": True,
        "torch_num_threads": 1,
        "measure_inference": False,
        "accumulation_steps": 1,
        "num_workers": 12,
        "prefetch_factor": 4,
    }
    counts = Counter()
    source_seeds = protocol["training_seeds"]["source"]
    for run in runs:
        config = run["resolved_config"]
        if any(config[key] != value for key, value in common.items()):
            raise ValueError(f"Unexpected P7 config: {run['experiment_id']}")
        variant = "A0" if config["model_type"] == "mobilenetv2" else "A3"
        expected = protocol["models_per_seed"][variant]
        if config["model_type"] != expected["model_type"]:
            raise ValueError("Unexpected P7 model type")
        if variant == "A3":
            for field in ("exit_positions", "exit_loss_weights", "exit_distillation_alpha"):
                if config[field] != expected[field]:
                    raise ValueError(f"Unexpected P7 A3 field: {field}")
        if run["seed"] not in source_seeds:
            raise ValueError("Unexpected P7 source seed")
        counts[(variant, run["seed"])] += 1
    expected_counts = Counter((variant, seed) for variant in ("A0", "A3") for seed in source_seeds)
    if counts != expected_counts or len(runs) != EXPECTED_RUNS:
        raise ValueError("P7 source matrix must be 2 variants x 3 seeds")
    return runs


def git_status() -> str:
    return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip()


def require_tracked(plan: list[dict]) -> None:
    required = [PROTOCOL, ARCHITECTURE, SWEEP, Path(__file__).resolve()]
    required.extend(ROOT / run["config_path"] for run in plan)
    for path in required:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(path.relative_to(ROOT))],
            cwd=ROOT,
            capture_output=True,
            check=True,
        )


def run_sweep(experiment_tag: str) -> int:
    command = [
        sys.executable,
        str(ROOT / "scripts/run_baselines.py"),
        "--sweep",
        str(SWEEP),
        "--jobs",
        "1",
        "--experiment-tag",
        experiment_tag,
    ]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--experiment-tag", default=EXPERIMENT_TAG)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    protocol = load_protocol()
    plan = validated_plan(args.experiment_tag)
    if args.worker:
        validate_prepared_dataset(protocol)
        return run_sweep(args.experiment_tag)
    for run in plan:
        print(shlex.join(run["command"]), flush=True)
    if args.dry_run:
        validate_prepared_dataset(protocol)
        print(
            "Validated: 6 P7 source runs, prepared ImageNet-100 pool, "
            "serial RTX 3080 Ti, official validation locked."
        )
        return 0
    changes = git_status()
    if changes:
        raise RuntimeError(f"Commit P7 source code and protocol before execution:\n{changes}")
    require_tracked(plan)
    validate_prepared_dataset(protocol)
    actual_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable"
    if actual_gpu != EXPECTED_GPU:
        raise RuntimeError(f"P7 source requires {EXPECTED_GPU}; found {actual_gpu}")
    processes = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    conflicts = ("scripts/train.py", "scripts/run_baselines.py", "benchmark_early_exit")
    if any(any(name in line for name in conflicts) for line in processes.splitlines()):
        raise RuntimeError("A conflicting training or benchmark process is running")
    for run in plan:
        if (ARTIFACTS_DIR / "runs" / run["experiment_id"]).exists():
            raise RuntimeError(f"Existing output will not be overwritten: {run['experiment_id']}")
    lock_path = ARTIFACTS_DIR / "early_exit_p7_source_launcher.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--experiment-tag",
            args.experiment_tag,
        ]
        if args.foreground:
            return subprocess.run(worker, cwd=ROOT, check=False).returncode
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        log_dir = ARTIFACTS_DIR / "launcher_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"early_exit_p7_source_{args.experiment_tag}_{timestamp}.log"
        with log_path.open("x") as handle:
            handle.write(f"Command: {shlex.join(worker)}\nFrozen protocol SHA-256: {sha256(PROTOCOL)}\n")
            handle.write("6 serial development-only source runs; official ImageNet validation locked.\n")
            handle.flush()
            process = subprocess.Popen(
                worker,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(lock.fileno(),),
            )
        print(f"P7 source PID: {process.pid}")
        print(f"Matrix: {len(plan)} runs; concurrency=1; GPU={EXPECTED_GPU}")
        print(f"Log: {log_path}")
        print(f"Monitor: tail -f {shlex.quote(str(log_path))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
