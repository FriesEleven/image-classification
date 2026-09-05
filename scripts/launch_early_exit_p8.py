"""Launch the frozen P8 hardware benchmark in a detached process."""

import argparse
import fcntl
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analysis.analyze_early_exit_p5c import read_json, sha256

PROTOCOL = ROOT / "reports/experiments/2026-09-05-early-exit-p8-design/protocol_manifest.json"


def validate() -> dict:
    protocol = read_json(PROTOCOL)
    if protocol.get("status") != "frozen_before_p8_execution":
        raise ValueError("P8 protocol is not frozen")
    receipt = protocol["method_decision_receipt"]
    if sha256(ROOT / receipt["path"]) != receipt["sha256"]:
        raise ValueError("P5-C decision receipt mismatch")
    if protocol["timing"] != {
        "input_residency": "device-resident main result",
        "warmup_batches_per_mode_round": 100,
        "timed_batches_per_mode_round": 1000,
        "paired_rounds": 5,
        "mode_order": "deterministically randomized within every paired round",
        "cuda_synchronize_each_batch": True,
        "precision": "FP32",
        "incomplete_last_batch": "excluded from timing",
    }:
        raise ValueError("Unexpected P8 timing contract")
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = validate()
    command = [sys.executable, str(ROOT / "scripts/analysis/benchmark_early_exit_p8.py")]
    if args.dry_run:
        print(json.dumps({"status": "ready", "models": protocol["expected_model_checkpoints"], "devices": protocol["matrix"]["devices"], "batch_sizes": protocol["matrix"]["batch_sizes"], "warmups": 100, "iterations": 1000, "rounds": 5}, indent=2))
        return 0
    changes = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip()
    if changes:
        raise RuntimeError(f"Commit P8 protocol and code before execution:\n{changes}")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3080 Ti":
        raise RuntimeError("P8 requires NVIDIA GeForce RTX 3080 Ti")
    processes = subprocess.check_output(["ps", "-eo", "args"], text=True)
    if any(value in processes for value in ("scripts/train.py", "run_baselines.py", "benchmark_early_exit_p8.py --output")):
        raise RuntimeError("Another training or P8 process is active")
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    with (artifacts / "early_exit_p8_launcher.lock").open("a") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("P8 is already running") from error
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        output = artifacts / "analyses" / f"early_exit_p8_{stamp}"
        log = artifacts / "launcher_logs" / f"early_exit_p8_{stamp}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        full_command = [*command, "--output", str(output)]
        with log.open("x") as handle:
            handle.write(f"Command: {shlex.join(full_command)}\nProtocol SHA-256: {sha256(PROTOCOL)}\n")
            handle.flush()
            process = subprocess.Popen(full_command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock_handle.fileno(),))
        print(f"P8 started with PID {process.pid}")
        print(f"Output: {output}")
        print(f"Log: {log}")
        print(f"Monitor: tail -f {shlex.quote(str(log))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
