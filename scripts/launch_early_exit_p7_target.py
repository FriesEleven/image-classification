"""Launch the frozen serial P7 ImageNet-100 target stage on RTX 3080 Ti."""

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

PROTOCOL = ROOT / "reports/experiments/2026-09-07-early-exit-p7-target-design/protocol_manifest.json"
SWEEP = ROOT / "configs/sweeps/early_exit_p7_imagenet100_target.yaml"
EXPERIMENT_TAG = "p7_target_r1"
EXPECTED_GPU = "NVIDIA GeForce RTX 3080 Ti"
EXPECTED_RUNS = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_receipt(item: dict, label: str) -> Path:
    path = ROOT / item["path"]
    if not path.is_file() or sha256(path) != item["sha256"]:
        raise ValueError(f"P7 target {label} changed: {item['path']}")
    return path


def load_protocol() -> dict:
    protocol = read_json(PROTOCOL)
    if protocol.get("status") != "frozen_before_p7_target_training":
        raise ValueError("P7 target protocol is not frozen")

    training = read_json(validate_receipt(protocol["training_protocol"], "training protocol"))
    receipt = read_json(validate_receipt(protocol["source_analysis_receipt"], "source receipt"))
    source_lock = read_json(validate_receipt(protocol["source_lock"], "source lock"))
    dataset = read_json(validate_receipt(protocol["prepared_dataset_manifest"], "dataset manifest"))
    validate_receipt(protocol["target_sweep"], "sweep")
    for config in protocol["experiment_configs"]:
        validate_receipt(config, "experiment config")

    source_receipt = protocol["source_analysis_receipt"]
    source_lock_receipt = protocol["source_lock"]
    if receipt.get("status") != source_receipt["required_status"]:
        raise ValueError("P7 source analysis did not permit target training")
    if source_lock.get("status") != source_lock_receipt["required_status"]:
        raise ValueError("P7 source lock did not permit target training")
    if receipt["artifact_sha256"].get(source_lock_receipt["path"]) != source_lock_receipt["sha256"]:
        raise ValueError("P7 source receipt does not bind the source lock")
    threshold = source_lock["decision"]["selected_policy"]["threshold"]
    if threshold != protocol["source_gate"]["selected_shared_threshold"]:
        raise ValueError("P7 source threshold differs from the target freeze")
    if source_lock.get("official_test_accessed") is not False or source_lock.get("target_data_accessed") is not False:
        raise ValueError("P7 source analysis crossed a forbidden data boundary")
    if training.get("status") != "frozen_before_p7_source_training":
        raise ValueError("P7 training protocol changed status")
    if training["training_seeds"]["target_after_source_gate"] != protocol["target_training"]["seeds"]:
        raise ValueError("P7 target seeds differ from the original training freeze")
    if (
        dataset.get("status") != "prepared_train_pool"
        or dataset.get("official_test_prepared") is not False
        or dataset.get("official_test_accessed") is not False
        or protocol["official_test"] != {
            "prepared": False,
            "accessed": False,
            "authorization_required_after_target_pass": True,
        }
    ):
        raise ValueError("P7 official-test boundary changed")
    return protocol


def validate_prepared_dataset(protocol: dict) -> None:
    path = DATA_DIR / "imagenet100/manifest.json"
    if sha256(path) != protocol["prepared_dataset_manifest"]["sha256"]:
        raise ValueError("Prepared ImageNet-100 dataset manifest changed")
    manifest = read_json(path)
    if (
        manifest.get("status") != "prepared_train_pool"
        or manifest.get("official_test_prepared") is not False
        or manifest.get("official_test_accessed") is not False
        or (DATA_DIR / "imagenet100/locked_test").exists()
    ):
        raise ValueError("Official ImageNet validation data must remain unprepared")


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
    for run in runs:
        config = run["resolved_config"]
        if any(config[key] != value for key, value in common.items()):
            raise ValueError(f"Unexpected P7 target config: {run['experiment_id']}")
        variant = "A0" if config["model_type"] == "mobilenetv2" else "A3"
        if variant == "A3" and (
            config.get("exit_positions") != [8, 15]
            or config.get("exit_loss_weights") != [0.1, 0.15]
            or config.get("exit_distillation_alpha") != 0.0
        ):
            raise ValueError("Unexpected P7 target A3 architecture")
        if run["seed"] not in protocol["target_training"]["seeds"]:
            raise ValueError("Unexpected P7 target seed")
        counts[(variant, run["seed"])] += 1
    expected = Counter(
        (variant, seed)
        for variant in ("A0", "A3")
        for seed in protocol["target_training"]["seeds"]
    )
    if counts != expected or len(runs) != EXPECTED_RUNS:
        raise ValueError("P7 target matrix must be A0/A3 x seeds 80-82")
    return runs


def git_status() -> str:
    return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip()


def require_tracked(plan: list[dict]) -> None:
    required = [PROTOCOL, SWEEP, Path(__file__).resolve()]
    required.extend(ROOT / run["config_path"] for run in plan)
    required.extend(
        ROOT / item["path"]
        for item in (
            load_protocol()["training_protocol"],
            load_protocol()["source_analysis_receipt"],
        )
    )
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
            "Validated: 6 serial P7 target runs on seeds 80-82, source threshold 0.850 locked, "
            "official ImageNet validation unprepared."
        )
        return 0

    changes = git_status()
    if changes:
        raise RuntimeError(f"Commit P7 target code and protocol before execution:\n{changes}")
    require_tracked(plan)
    validate_prepared_dataset(protocol)
    actual_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable"
    if actual_gpu != EXPECTED_GPU:
        raise RuntimeError(f"P7 target requires {EXPECTED_GPU}; found {actual_gpu}")
    processes = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    conflicts = ("scripts/train.py", "scripts/run_baselines.py", "benchmark_early_exit")
    if any(any(name in line for name in conflicts) for line in processes.splitlines()):
        raise RuntimeError("A conflicting training or benchmark process is running")
    for run in plan:
        if (ARTIFACTS_DIR / "runs" / run["experiment_id"]).exists():
            raise RuntimeError(f"Existing output will not be overwritten: {run['experiment_id']}")

    lock_path = ARTIFACTS_DIR / "early_exit_p7_target_launcher.lock"
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
        log_path = log_dir / f"early_exit_p7_target_{args.experiment_tag}_{timestamp}.log"
        with log_path.open("x") as handle:
            handle.write(
                f"Command: {shlex.join(worker)}\n"
                f"Frozen target protocol SHA-256: {sha256(PROTOCOL)}\n"
                "6 serial target runs; threshold=0.850; official ImageNet validation locked.\n"
            )
            handle.flush()
            process = subprocess.Popen(
                worker,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(lock.fileno(),),
            )
        print(f"P7 target PID: {process.pid}")
        print(f"Matrix: {len(plan)} runs; concurrency=1; GPU={EXPECTED_GPU}")
        print(f"Log: {log_path}")
        print(f"Monitor: tail -f {shlex.quote(str(log_path))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
