"""Launch the serial validation-only early-exit P0 feasibility matrix."""

import argparse
import fcntl
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

SWEEP = PROJECT_ROOT / "configs/sweeps/early_exit_p0.yaml"
EXPERIMENT_TAG = "p0a"
JOBS = 1
SEEDS = (51, 52, 53)


def validated_plan(experiment_tag: str = EXPERIMENT_TAG):
    plan = build_plan(load_sweep(SWEEP), experiment_tag)
    protocol = {
        "dataset": "cifar10",
        "validation_size": 5000,
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
        "prefetch_factor": 4,
    }
    counts = Counter()
    for run in plan:
        config = run["resolved_config"]
        if any(config[key] != value for key, value in protocol.items()):
            raise ValueError(f"Unexpected early-exit P0 protocol: {run['experiment_id']}")
        if run["seed"] not in SEEDS or config["seed"] != run["seed"]:
            raise ValueError(f"Unexpected P0 seed: {run['experiment_id']}")
        model_type = config["model_type"]
        if model_type not in {"mobilenetv2", "multi_exit"}:
            raise ValueError(f"Unexpected P0 model: {run['experiment_id']}")
        if model_type == "multi_exit":
            expected = {
                "exit_positions": [8, 16],
                "exit_loss_weights": [0.2, 0.3],
                "exit_distillation_alpha": 0.5,
                "exit_temperature": 3.0,
            }
            if any(config[key] != value for key, value in expected.items()):
                raise ValueError(f"Unexpected multi-exit contract: {run['experiment_id']}")
        counts[(model_type, run["seed"])] += 1
        if not config["experiment_name"].endswith(f"_{experiment_tag}_seed{run['seed']}"):
            raise ValueError("P0 requires a fresh tagged experiment ID")
    expected_counts = Counter(
        {(model_type, seed): 1 for model_type in ("mobilenetv2", "multi_exit") for seed in SEEDS}
    )
    if counts != expected_counts:
        raise ValueError("Expected exactly 2 matched models x seeds 51/52/53")
    return plan


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
            "from scratch, validation-only; no training started."
        )
        return 0

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with (ARTIFACTS_DIR / "early_exit_p0_launcher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Early-exit P0 is already running") from error
        processes = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
        if any(
            "scripts/train.py" in line or "scripts/run_baselines.py" in line
            for line in processes.splitlines()
        ):
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
        log_path = log_directory / f"early_exit_p0_serial_{args.experiment_tag}_{timestamp}.log"
        with log_path.open("x") as log:
            log.write(f"Command: {shlex.join(command)}\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(lock.fileno(),),
            )
        print(f"Early-exit P0 started with PID {process.pid}")
        print(f"Matrix: {len(plan)} runs; concurrency={JOBS}; seeds={SEEDS}")
        print(f"Log: {log_path}")
        print(f"Monitor: tail -f {shlex.quote(str(log_path))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
