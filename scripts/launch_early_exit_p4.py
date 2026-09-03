"""Launch the serial CIFAR-100 P4 independent confirmation batch."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_classification.paths import ARTIFACTS_DIR, PROJECT_ROOT
from scripts.run_baselines import build_plan, load_sweep

SWEEP = PROJECT_ROOT / "configs/sweeps/early_exit_p4.yaml"
POLICY_LOCK = PROJECT_ROOT / ("reports/experiments/2026-09-03-early-exit-p4-design/locked_policy.json")
EXPERIMENT_TAG = "p4a"
JOBS = 1
SEEDS = (66, 67, 68)
SPLIT_SEED = 20_260_904
LOCKED_THRESHOLD = 0.903
EXPECTED_SOURCE_HASHES = {
    "reports/audits/2026-09-03-early-exit-p3a/audit_results.json": (
        "1f9415e8c3e81cd6e16bcdc6c10fd6f008a0e2d7681a5129f9d978a4968645fa"
    ),
    "reports/experiments/2026-09-03-early-exit-p3-cifar100/selection.json": (
        "c71ad43122d5fec4a08679f80a616b7e3015ef730d4c0465829a51329edb6bfe"
    ),
    "reports/diagnostics/2026-09-03-early-exit-p3-boundary-v2/diagnostic.json": (
        "25bbcf5365be9b860b0919b3b1160cfefe5b1adfd609e54f158bf4803fb73910"
    ),
    "reports/diagnostics/2026-09-03-early-exit-p3-class-guard-v2/diagnostic.json": (
        "ece0c80b6790245715440991a57e109b62ee61f1fc66b208546397f7fbd603dc"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_policy_lock() -> dict:
    lock = json.loads(POLICY_LOCK.read_text(encoding="utf-8"))
    if lock.get("status") != "ready_for_independent_p4_confirmation":
        raise ValueError("P4 policy lock is not ready")
    policy = lock.get("frozen_policy", {})
    expected_policy = {
        "policy_version": "shared_exit8_softmax_threshold_p4_v1",
        "exit_position": 8,
        "confidence": "maximum softmax probability",
        "confidence_threshold": LOCKED_THRESHOLD,
        "protected_predicted_classes": [],
        "fallback": "final head",
        "p4_threshold_candidates": 0,
        "per_model_recalibration": False,
    }
    for key, expected in expected_policy.items():
        if policy.get(key) != expected:
            raise ValueError(f"Unexpected P4 policy field: {key}")
    protocol = lock.get("confirmation_protocol", {})
    expected_protocol = {
        "dataset": "cifar100",
        "training_seeds": list(SEEDS),
        "split_seed": SPLIT_SEED,
        "train_samples": 40_000,
        "model_selection_validation_samples": 5000,
        "policy_confirmation_samples": 5000,
        "samples_per_class_in_policy_confirmation": 50,
        "final_head_mean_gain_minimum": -0.003,
        "final_head_each_gain_minimum": -0.0075,
        "maximum_accuracy_drop": 0.0,
        "maximum_balanced_accuracy_drop": 0.0,
        "maximum_worst_class_accuracy_drop": 0.04,
        "minimum_early_fraction": 0.15,
        "maximum_early_fraction": 0.95,
        "minimum_mac_saving_fraction": 0.15,
    }
    if protocol != expected_protocol:
        raise ValueError("Unexpected P4 confirmation protocol")
    if lock.get("p3_frozen_result", {}).get("status") != "stop_without_test":
        raise ValueError("P3 failure is not frozen in the P4 lock")
    if lock.get("p3_frozen_result", {}).get("official_test_accessed") is not False:
        raise ValueError("P4 requires P3 official test to remain unopened")
    for relative, expected_hash in EXPECTED_SOURCE_HASHES.items():
        path = PROJECT_ROOT / relative
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"P4 source evidence mismatch: {relative}")
    return lock


def validated_plan(experiment_tag: str = EXPERIMENT_TAG):
    validate_policy_lock()
    plan = build_plan(load_sweep(SWEEP), experiment_tag)
    protocol = {
        "dataset": "cifar100",
        "validation_size": 5000,
        "calibration_size": 5000,
        "split_seed": SPLIT_SEED,
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
    for run in plan:
        config = run["resolved_config"]
        if any(config[key] != value for key, value in protocol.items()):
            raise ValueError(f"Unexpected early-exit P4 protocol: {run['experiment_id']}")
        if run["seed"] not in SEEDS or config["seed"] != run["seed"]:
            raise ValueError(f"Unexpected P4 training seed: {run['experiment_id']}")
        model_type = config["model_type"]
        if model_type not in {"mobilenetv2", "multi_exit"}:
            raise ValueError(f"Unexpected P4 model: {run['experiment_id']}")
        if model_type == "multi_exit":
            expected = {
                "exit_positions": [8, 16],
                "exit_loss_weights": [0.2, 0.3],
                "exit_distillation_alpha": 0.5,
                "exit_temperature": 3.0,
            }
            if any(config[key] != value for key, value in expected.items()):
                raise ValueError(f"Unexpected P4 multi-exit contract: {run['experiment_id']}")
        counts[(model_type, run["seed"])] += 1
        if not config["experiment_name"].endswith(f"_{experiment_tag}_seed{run['seed']}"):
            raise ValueError("P4 requires a fresh tagged experiment ID")
    expected_counts = Counter({(model_type, seed): 1 for model_type in ("mobilenetv2", "multi_exit") for seed in SEEDS})
    if counts != expected_counts:
        raise ValueError("Expected exactly two matched models x P4 seeds 66 through 68")
    return plan


def _git_status() -> str:
    return subprocess.check_output(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


def _require_policy_lock_tracked() -> None:
    relative = str(POLICY_LOCK.relative_to(PROJECT_ROOT))
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--experiment-tag", default=EXPERIMENT_TAG)
    args = parser.parse_args()
    plan = validated_plan(args.experiment_tag)
    for run in plan:
        print(shlex.join(run["command"]), flush=True)
    if args.dry_run:
        print(
            f"Validated: {len(plan)} new {args.experiment_tag} runs, one serial job, "
            f"seeds={SEEDS}, split_seed={SPLIT_SEED}, threshold={LOCKED_THRESHOLD}; "
            "no test or training started."
        )
        return 0

    changes = _git_status()
    if changes:
        raise RuntimeError(f"Commit all evidence and experiment files before P4:\n{changes}")
    _require_policy_lock_tracked()
    forbidden = (
        PROJECT_ROOT / "reports/experiments/2026-09-03-early-exit-p3-cifar100-test",
        PROJECT_ROOT / "artifacts/official_tests/early_exit_p3_cifar100_20260903",
    )
    p3_access_markers = list(
        (ARTIFACTS_DIR / "official_test_access_registry").glob("early_exit_p3_cifar100_*.started.json")
    )
    if any(path.exists() for path in forbidden) or p3_access_markers:
        raise RuntimeError("P3 official-test output exists despite its frozen stop decision")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with (ARTIFACTS_DIR / "early_exit_p4_launcher.lock").open("a") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Early-exit P4 is already running") from error
        processes = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
        if any("scripts/train.py" in line or "scripts/run_baselines.py" in line for line in processes.splitlines()):
            raise RuntimeError("An existing training/sweep process was found; do not overlap experiments")
        for run in plan:
            target = ARTIFACTS_DIR / "runs" / run["experiment_id"]
            if target.exists():
                raise RuntimeError(f"Existing output will not be overwritten: {target}")

        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run_baselines.py"),
            "--sweep",
            str(SWEEP),
            "--jobs",
            str(JOBS),
            "--experiment-tag",
            args.experiment_tag,
        ]
        if args.foreground:
            return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
        log_directory = ARTIFACTS_DIR / "launcher_logs"
        log_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        log_path = log_directory / f"early_exit_p4_serial_{args.experiment_tag}_{timestamp}.log"
        with log_path.open("x") as log:
            log.write(f"Command: {shlex.join(command)}\n")
            log.write(f"Frozen P4 policy SHA-256: {_sha256(POLICY_LOCK)}\n")
            log.write(
                "Threshold 0.903; no class guard; zero P4 threshold candidates; "
                "new seeds 66/67/68 and split seed 20260904.\n"
            )
            log.write("Official CIFAR-100 test evaluation is disabled for every training run.\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(lock_handle.fileno(),),
            )
        print(f"Early-exit P4 started with PID {process.pid}")
        print(f"Matrix: {len(plan)} runs; concurrency={JOBS}")
        print(f"Seeds: {SEEDS}; split seed: {SPLIT_SEED}; frozen threshold: {LOCKED_THRESHOLD}")
        print(f"Log: {log_path}")
        print(f"Monitor: tail -f {shlex.quote(str(log_path))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
