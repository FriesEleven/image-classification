"""Launch six validation-only CSGHA-v4 / matched-control experiments."""

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

SWEEP = PROJECT_ROOT / "configs/sweeps/csgha_v4_matched.yaml"
JOBS = 2


def validated_plan(experiment_tag: str | None = None):
    plan = build_plan(load_sweep(SWEEP), experiment_tag)
    counts = Counter((run["resolved_config"]["model_type"], run["seed"]) for run in plan)
    expected = Counter({(variant, seed): 1 for variant in ("hybrid_leaky", "csgha_v4") for seed in (42, 43, 44)})
    if counts != expected:
        raise ValueError("Expected exactly two matched variants x seeds 42/43/44")
    protocol = {"dataset": "cifar10", "evaluate_test": False, "epochs": 200, "batch_size": 128,
                "validation_size": 5000, "lr": 0.01, "amp": True, "cuda_graph": True,
                "torch_num_threads": 1, "measure_inference": False,
                "accumulation_steps": 1,
                "se_positions": [1, 2], "cbam_positions": [7, 8], "guidance_position": 2,
                "guidance_reduction": 4, "num_workers": 8, "prefetch_factor": 4}
    for run in plan:
        config = run["resolved_config"]
        if any(config[key] != value for key, value in protocol.items()) or config["seed"] != run["seed"]:
            raise ValueError(f"Unexpected protocol: {run['experiment_id']}")
        tag_part = f"_{experiment_tag}" if experiment_tag else ""
        if not config["experiment_name"].endswith(f"_perf2{tag_part}_seed{run['seed']}"):
            raise ValueError("Concurrent GPU training requires new perf2 run IDs; do not mix old outputs")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--experiment-tag")
    args = parser.parse_args()
    plan = validated_plan(args.experiment_tag)
    for run in plan:
        print(shlex.join(run["command"]), flush=True)
    if args.dry_run:
        print(f"Validated: 6 new perf2 runs, {JOBS} concurrent jobs, from scratch, validation-only; no training started.")
        return 0
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with (ARTIFACTS_DIR / "csgha_v4_launcher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("CSGHA v4 sweep is already running") from error
        processes = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
        if any("scripts/train.py" in line or "scripts/run_baselines.py" in line for line in processes.splitlines()):
            raise RuntimeError("An existing training/sweep process was found; do not overlap experiments")
        for run in plan:
            target = ARTIFACTS_DIR / "runs" / run["experiment_id"]
            if target.exists():
                raise RuntimeError(f"Existing output will not be overwritten: {target}")
        command = [sys.executable, str(PROJECT_ROOT / "scripts/run_baselines.py"), "--sweep", str(SWEEP),
                   "--jobs", str(JOBS)]
        if args.experiment_tag:
            command.extend(["--experiment-tag", args.experiment_tag])
        if args.foreground:
            return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
        log_directory = ARTIFACTS_DIR / "launcher_logs"
        log_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        log_path = log_directory / f"csgha_v4_matched_perf2_{timestamp}.log"
        with log_path.open("x") as log:
            log.write(f"Command: {shlex.join(command)}\n")
            log.flush()
            # The inherited open file description holds flock until the runner exits.
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, pass_fds=(lock.fileno(),))
        print(f"CSGHA v4 matched sweep started with PID {process.pid}")
        print(f"Concurrency: {JOBS} independent jobs; each retains batch size 128")
        print(f"Log: {log_path}")
        print(f"Monitor: tail -f {shlex.quote(str(log_path))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
