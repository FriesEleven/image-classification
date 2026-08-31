"""Validate and launch the P2 CSGHA-v4/control checkpoint diagnostics in background."""

import argparse
import fcntl
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC = ROOT / "scripts/diagnostics/audit_csgha_v4_retry1.py"
OUTPUT = ROOT / "artifacts/diagnostics/csgha_v4_retry1_information_20260831_v1"
LOCK = ROOT / "artifacts/csgha_v4_retry1_diagnostics.lock"


def command(dry_run=False):
    result = [sys.executable, str(DIAGNOSTIC)]
    if dry_run:
        result.append("--dry-run")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        return subprocess.run(command(True), cwd=ROOT, check=False).returncode
    if OUTPUT.exists():
        raise FileExistsError(f"Diagnostic output already exists: {OUTPUT}")
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("P2 diagnostics are already running") from error
        processes = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
        forbidden = ("scripts/train.py", "scripts/run_baselines.py", "audit_csgha_v4_retry1.py")
        if any(any(token in line for token in forbidden) for line in processes.splitlines()):
            raise RuntimeError("A training or P2 diagnostic process is already running")
        if args.foreground:
            return subprocess.run(command(), cwd=ROOT, check=False).returncode
        logs = ROOT / "artifacts/launcher_logs"
        logs.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        log_path = logs / f"csgha_v4_retry1_diagnostics_{stamp}.log"
        with log_path.open("x") as log:
            log.write(f"Command: {shlex.join(command())}\n")
            log.flush()
            process = subprocess.Popen(
                command(), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(lock.fileno(),),
            )
        print(f"P2 diagnostics started with PID {process.pid}")
        print(f"Output: {OUTPUT}")
        print(f"Log: {log_path}")
        print(f"Monitor: tail -f {shlex.quote(str(log_path))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
