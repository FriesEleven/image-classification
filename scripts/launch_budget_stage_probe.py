"""Launch the serial validation-only budget-aware singleton probe matrix."""

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
from image_classification.selection import candidate_from_positions, enumerate_stage_candidates
from scripts.run_baselines import build_plan, load_sweep

SWEEP = PROJECT_ROOT / "configs/sweeps/budget_stage_probe.yaml"
EXPERIMENT_TAG = "probe1"
JOBS = 1
SEEDS = (45, 46, 47)


def _probe_candidate_ids() -> set[str]:
    return {
        candidate["candidate_id"]
        for candidate in enumerate_stage_candidates()
        if candidate["active_stages"] <= 1
    }


def validated_plan(experiment_tag: str = EXPERIMENT_TAG):
    plan = build_plan(load_sweep(SWEEP), experiment_tag)
    expected_candidates = _probe_candidate_ids()
    counts = Counter()
    protocol = {
        "model_type": "stage_sparse",
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
    for run in plan:
        config = run["resolved_config"]
        if any(config[key] != value for key, value in protocol.items()):
            raise ValueError(f"Unexpected budget-probe protocol: {run['experiment_id']}")
        if run["seed"] not in SEEDS or config["seed"] != run["seed"]:
            raise ValueError(f"Unexpected calibration seed: {run['experiment_id']}")
        candidate = candidate_from_positions(
            config["eca_positions"], config["se_positions"], config["cbam_positions"]
        )
        if candidate["candidate_id"] not in expected_candidates:
            raise ValueError(f"Expected baseline or singleton probe: {run['experiment_id']}")
        counts[(candidate["candidate_id"], run["seed"])] += 1
        if not config["experiment_name"].endswith(f"_{experiment_tag}_seed{run['seed']}"):
            raise ValueError("Budget probe requires a fresh tagged experiment ID")
    expected = Counter(
        {(candidate_id, seed): 1 for candidate_id in expected_candidates for seed in SEEDS}
    )
    if counts != expected:
        raise ValueError("Expected exactly 10 candidates x seeds 45/46/47")
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
    with (ARTIFACTS_DIR / "budget_stage_probe_launcher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Budget stage probe is already running") from error
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
        log_path = log_directory / f"budget_stage_probe_serial_{args.experiment_tag}_{timestamp}.log"
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
        print(f"Budget stage probe started with PID {process.pid}")
        print(f"Matrix: {len(plan)} runs; concurrency={JOBS}; seeds={SEEDS}")
        print(f"Log: {log_path}")
        print(f"Monitor: tail -f {shlex.quote(str(log_path))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
