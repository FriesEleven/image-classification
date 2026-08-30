"""Bounded, fail-fast process scheduling for independent training runs."""

import os
import signal
import subprocess
import time
from collections import deque
from datetime import datetime


def _timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _stop_owned(active, status, on_change):
    """Stop only process groups created by this queue, including data workers."""
    for sig, timeout in ((signal.SIGINT, 10), (signal.SIGTERM, 3), (signal.SIGKILL, 3)):
        for item in active.values():
            process = item["process"]
            if process.poll() is None:
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + timeout
        while any(item["process"].poll() is None for item in active.values()) and time.monotonic() < deadline:
            time.sleep(0.1)
    for item in active.values():
        process, run = item["process"], item["run"]
        run.update(status=status, finished_at=_timestamp(), return_code=process.poll())
        item["reader"].close()
        item["log"].close()
    active.clear()
    on_change()


def run_parallel_queue(runs, jobs, cwd, log_directory, on_change, check_source, completed_summary,
                       continue_on_error=False, poll_seconds=0.5):
    if not 2 <= jobs <= 3:
        raise ValueError("Independent shared-GPU queue supports only 2 or 3 jobs")
    if any(run["resolved_config"].get("measure_inference", True) for run in runs):
        raise ValueError("Disable inference timing for shared-GPU parallel runs")
    log_directory.mkdir(parents=True, exist_ok=True)
    pending = deque(runs)
    active = {}
    failures = 0

    def terminate(_signum, _frame):
        raise KeyboardInterrupt("Sweep termination requested")

    previous_term = signal.signal(signal.SIGTERM, terminate)
    try:
        while pending or active:
            while pending and len(active) < jobs:
                check_source()
                run = pending.popleft()
                summary = completed_summary(run)
                if summary is not None:
                    run.update(status="skipped_complete", finished_at=_timestamp(), return_code=0, summary=summary)
                    on_change()
                    continue
                log_path = log_directory / f"{run['experiment_id']}.log"
                log = log_path.open("x")
                reader = None
                try:
                    reader = log_path.open()
                    process = subprocess.Popen(run["command"], cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                except BaseException:
                    if reader is not None:
                        reader.close()
                    log.close()
                    raise
                run.update(status="running", started_at=_timestamp(), pid=process.pid,
                           process_group=process.pid, log_path=str(log_path))
                active[process.pid] = {"process": process, "run": run, "log": log,
                                       "reader": reader}
                on_change()
                print(f"Started PID {process.pid}: {run['experiment_id']} | Log: {log_path}", flush=True)

            for pid, item in list(active.items()):
                # Relay only epoch summaries; each full child log stays separate.
                for line in item["reader"]:
                    if line.startswith("Epoch "):
                        print(f"[{item['run']['experiment_id']}] {line.rstrip()}", flush=True)
                process, run = item["process"], item["run"]
                return_code = process.poll()
                if return_code is None:
                    continue
                check_source()
                if return_code == 0:
                    summary = completed_summary(run)
                    if summary is None:
                        raise RuntimeError(f"Process exited successfully without summary: {run['experiment_id']}")
                    run["summary"] = summary
                run.update(status="completed" if return_code == 0 else "failed",
                           finished_at=_timestamp(), return_code=return_code)
                item["reader"].close()
                item["log"].close()
                del active[pid]
                on_change()
                print(f"{run['status']}: {run['experiment_id']}", flush=True)
                if return_code:
                    failures += 1
                    if not continue_on_error:
                        _stop_owned(active, "cancelled_after_failure", on_change)
                        return failures
            if active:
                time.sleep(poll_seconds)
    except BaseException:
        _stop_owned(active, "interrupted", on_change)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    return failures
