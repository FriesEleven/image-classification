"""Launch the clean P8 v3 rerun after validating failure and input evidence."""

import argparse
import fcntl
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.analysis.analyze_early_exit_p5c import discover_new_runs, read_json, sha256
from scripts.analysis.benchmark_early_exit_p8_v3 import PROTOCOL, validate_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    p = validate_protocol()
    runs, _ = discover_new_runs("p5c_r1")
    for dataset, matrix in p["datasets"].items():
        for seed in matrix["seeds"]:
            run = runs[(dataset, "A3", seed)]
            directory = ROOT / "artifacts/runs" / run["experiment_id"]
            summary = read_json(directory / "summary.json")
            if sha256(directory / "checkpoints/model_best.pth") != summary["best_checkpoint_sha256"]:
                raise ValueError("P8 checkpoint hash mismatch")
            split = read_json(directory / "split_indices.json")
            if len(set(split["calibration_indices"])) != 5000:
                raise ValueError("Calibration sample IDs invalid")
    if args.dry_run:
        print(json.dumps({"status": "ready", "checkpoints_verified": 6,
                          "raw_files_expected": p["expected_raw_files"], "new_training_runs": 0}, indent=2))
        return 0
    status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip()
    if status:
        raise RuntimeError(f"Commit before execution: {status}")
    processes = subprocess.check_output(["ps", "-eo", "pid=,comm=,args="], text=True)
    for line in processes.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3 and parts[1].startswith("python") and any(
            name in parts[2] for name in (
                "scripts/train.py", "run_baselines.py", "benchmark_early_exit_p8.py",
                "benchmark_early_exit_p8_v2.py", "benchmark_early_exit_p8_v3.py",
            )
        ):
            raise RuntimeError(f"Conflicting process: {line.strip()}")
    artifacts = ROOT / "artifacts"
    with (artifacts / "early_exit_p8_launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        output = artifacts / "analyses" / f"early_exit_p8_v3_{stamp}"
        log = artifacts / "launcher_logs" / f"early_exit_p8_v3_{stamp}.log"
        command = [sys.executable, str(ROOT / "scripts/analysis/benchmark_early_exit_p8_v3.py"), "--output", str(output)]
        with log.open("x") as handle:
            handle.write(f"Command: {shlex.join(command)}\nProtocol SHA-256: {sha256(PROTOCOL)}\n")
            handle.flush()
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True, pass_fds=(lock.fileno(),))
        print(f"P8 v3 PID: {process.pid}\nOutput: {output}\nLog: {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
