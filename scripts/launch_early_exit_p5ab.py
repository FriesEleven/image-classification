"""Launch the frozen P5-A/B development-only batch in the background."""

import argparse
import fcntl
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "reports/experiments/2026-09-05-early-exit-p5-design/protocol_manifest.json"
ANALYZER = ROOT / "scripts/analysis/run_early_exit_p5ab.py"
ARTIFACTS = ROOT / "artifacts"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--reuse-logits", type=Path)
    args = parser.parse_args()
    subprocess.run([sys.executable, str(ANALYZER), "--output", "/tmp/p5-unused", "--verify-only"], cwd=ROOT, check=True)
    if args.dry_run:
        print(json.dumps({"status": "ready", "batch": "P5-A/P5-B", "training_runs": 0, "serial_development_inference_runs": 18, "official_or_external_evaluator_runs": 0}, indent=2))
        return 0
    changes = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip()
    if changes:
        raise RuntimeError(f"Commit P5 protocol and code before execution:\n{changes}")
    for path in (PROTOCOL, ANALYZER, Path(__file__).resolve()):
        subprocess.run(["git", "ls-files", "--error-unmatch", "--", str(path.relative_to(ROOT))], cwd=ROOT, check=True, capture_output=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with (ARTIFACTS / "early_exit_p5ab_launcher.lock").open("a") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("P5-A/B is already running") from error
        processes = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
        if any("scripts/train.py" in line or "scripts/run_baselines.py" in line for line in processes.splitlines()):
            raise RuntimeError("A training process is active; P5-A/B will not overlap it")
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        output = ARTIFACTS / "analyses" / f"early_exit_p5ab_{timestamp}"
        log_dir = ARTIFACTS / "launcher_logs"; log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"early_exit_p5ab_{timestamp}.log"
        command = [sys.executable, str(ANALYZER), "--output", str(output)]
        if args.reuse_logits is not None:
            command.extend(["--reuse-logits", str(args.reuse_logits)])
        if args.foreground:
            return subprocess.run(command, cwd=ROOT, check=False).returncode
        with log_path.open("x", encoding="utf-8") as log:
            log.write(f"Command: {shlex.join(command)}\n")
            log.write("Development-only P5-A/B; no training and no official/external evaluator.\n")
            log.flush()
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock_handle.fileno(),))
        print(f"P5-A/B started with PID {process.pid}")
        print("Matrix: 18 matched checkpoint pairs, processed serially; training runs=0")
        print(f"Output: {output}")
        print(f"Log: {log_path}")
        print(f"Monitor: tail -f {shlex.quote(str(log_path))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
