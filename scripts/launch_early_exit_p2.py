"""Launch the serial P2 transfer test for the immutable P1b early-exit policy."""

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

SWEEP = PROJECT_ROOT / "configs/sweeps/early_exit_p2.yaml"
SOURCE_SELECTION = PROJECT_ROOT / "reports/experiments/2026-09-02-early-exit-p1b/locked_selection.json"
EXPECTED_SOURCE_SELECTION_SHA256 = "8dce5d938e06ae9d7432bd6baafd0c0ed9f978678fb53767ab02eeafecc723ab"
LOCKED_THRESHOLD = 0.984
SOURCE_SEEDS = (54, 55, 56)
EXPERIMENT_TAG = "p2a"
JOBS = 1
SEEDS = (57, 58, 59)
SPLIT_SEED = 20_260_902


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_policy() -> dict:
    if _sha256(SOURCE_SELECTION) != EXPECTED_SOURCE_SELECTION_SHA256:
        raise ValueError("The versioned P1b source-policy file is not the frozen selection")
    selection = json.loads(SOURCE_SELECTION.read_text(encoding="utf-8"))
    policy = selection.get("locked_policy") or {}
    expected = {
        "policy_version": "shared_exit8_softmax_threshold_v1",
        "exit_position": 8,
        "confidence": "maximum softmax probability",
        "confidence_threshold": LOCKED_THRESHOLD,
        "protected_predicted_classes": [],
        "fallback": "final head",
        "shared_across_training_seeds": list(SOURCE_SEEDS),
    }
    if selection.get("status") != "ready_for_locked_test":
        raise ValueError("The source P1b selection did not pass its frozen gates")
    for name, value in expected.items():
        if policy.get(name) != value:
            raise ValueError(f"Unexpected P1b source policy field: {name}")
    if not selection.get("gates") or not all(selection["gates"].values()):
        raise ValueError("Not all frozen P1b source-policy gates passed")
    return selection


def validated_plan(experiment_tag: str = EXPERIMENT_TAG):
    validate_source_policy()
    plan = build_plan(load_sweep(SWEEP), experiment_tag)
    protocol = {
        "dataset": "cifar10",
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
            raise ValueError(f"Unexpected early-exit P2 protocol: {run['experiment_id']}")
        if run["seed"] not in SEEDS or config["seed"] != run["seed"]:
            raise ValueError(f"Unexpected P2 training seed: {run['experiment_id']}")
        model_type = config["model_type"]
        if model_type not in {"mobilenetv2", "multi_exit"}:
            raise ValueError(f"Unexpected P2 model: {run['experiment_id']}")
        if model_type == "multi_exit":
            expected = {
                "exit_positions": [8, 16],
                "exit_loss_weights": [0.2, 0.3],
                "exit_distillation_alpha": 0.5,
                "exit_temperature": 3.0,
            }
            if any(config[key] != value for key, value in expected.items()):
                raise ValueError(f"Unexpected P2 multi-exit contract: {run['experiment_id']}")
        counts[(model_type, run["seed"])] += 1
        if not config["experiment_name"].endswith(f"_{experiment_tag}_seed{run['seed']}"):
            raise ValueError("P2 requires a fresh tagged experiment ID")
    expected_counts = Counter({(model_type, seed): 1 for model_type in ("mobilenetv2", "multi_exit") for seed in SEEDS})
    if counts != expected_counts:
        raise ValueError("Expected exactly 2 matched models x training seeds 57/58/59")
    return plan


def _tracked_experiment_changes() -> str:
    return subprocess.check_output(
        ["git", "status", "--short", "--", "src", "scripts", "configs", "pyproject.toml"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


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
            f"Validated: {len(plan)} new {args.experiment_tag} runs, {JOBS} serial job, "
            f"unseen seeds={SEEDS}; source threshold={LOCKED_THRESHOLD}; no training started."
        )
        return 0

    changes = _tracked_experiment_changes()
    if changes:
        raise RuntimeError(f"Commit experiment source/config changes before P2:\n{changes}")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with (ARTIFACTS_DIR / "early_exit_p2_launcher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Early-exit P2 is already running") from error
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
        log_path = log_directory / f"early_exit_p2_serial_{args.experiment_tag}_{timestamp}.log"
        with log_path.open("x") as log:
            log.write(f"Command: {shlex.join(command)}\n")
            log.write(
                f"Frozen source policy: sha256={EXPECTED_SOURCE_SELECTION_SHA256}; "
                f"exit8 threshold={LOCKED_THRESHOLD}; source_seeds={SOURCE_SEEDS}\n"
            )
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(lock.fileno(),),
            )
        print(f"Early-exit P2 started with PID {process.pid}")
        print(f"Matrix: {len(plan)} runs; concurrency={JOBS}; target_training_seeds={SEEDS}")
        print(
            f"Frozen source: {EXPECTED_SOURCE_SELECTION_SHA256}; "
            f"exit8 threshold={LOCKED_THRESHOLD}; no policy fitting in training"
        )
        print(f"Fixed split seed: {SPLIT_SEED}; train/validation/transfer=40k/5k/5k")
        print(f"Log: {log_path}")
        print(f"Monitor: tail -f {shlex.quote(str(log_path))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
