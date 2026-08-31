import json
import signal
import sys
from pathlib import Path

import pytest

from image_classification.training.sweep import run_parallel_queue


def plan(tmp_path, delays=(0.1, 0.1, 0.1), fail_index=None):
    runs = []
    for index, delay in enumerate(delays):
        target = tmp_path / f"done{index}.json"
        code = (
            "import time,json; from pathlib import Path; "
            f"time.sleep({delay}); "
            + ("raise SystemExit(7)" if index == fail_index else
               f"Path({str(target)!r}).write_text(json.dumps({{'done': {index}}})); print('Epoch 1/1')")
        )
        runs.append({"experiment_id": f"run{index}", "status": "pending",
                     "resolved_config": {"measure_inference": False},
                     "command": [sys.executable, "-c", code], "summary_path": str(target)})
    return runs


def summary(run):
    path = Path(run["summary_path"])
    return json.loads(path.read_text()) if path.exists() else None


def test_parallel_queue_caps_concurrency_and_saves_distinct_logs(tmp_path, capsys):
    runs = plan(tmp_path)
    observed = []
    result = run_parallel_queue(
        runs, 2, tmp_path, tmp_path / "logs",
        lambda: observed.append(sum(row["status"] == "running" for row in runs)),
        lambda: None, summary, poll_seconds=0.01,
    )
    assert result == 0
    assert max(observed) == 2
    assert all(row["status"] == "completed" for row in runs)
    assert len({row["pid"] for row in runs}) == 3
    assert all(Path(row["log_path"]).read_text().strip() == "Epoch 1/1" for row in runs)
    assert "Epoch 1/1" in capsys.readouterr().out


def test_parallel_failure_stops_owned_siblings_and_leaves_unstarted_pending(tmp_path):
    runs = plan(tmp_path, delays=(0.1, 10, 0.1), fail_index=0)
    assert run_parallel_queue(runs, 2, tmp_path, tmp_path / "logs", lambda: None,
                              lambda: None, summary, poll_seconds=0.01) == 1
    assert [row["status"] for row in runs] == ["failed", "cancelled_after_failure", "pending"]
    assert runs[1]["return_code"] is not None
    assert not Path(runs[1]["summary_path"]).exists()
    assert "pid" not in runs[2]


def test_native_abort_records_signal_and_faulthandler_stack(tmp_path):
    runs = plan(tmp_path, delays=(0.1, 10, 0.1))
    runs[0]["command"] = [
        sys.executable,
        "-c",
        "import os,signal; os.kill(os.getpid(), signal.SIGABRT)",
    ]
    assert run_parallel_queue(
        runs, 2, tmp_path, tmp_path / "logs", lambda: None,
        lambda: None, summary, poll_seconds=0.01,
    ) == 1
    assert runs[0]["status"] == "failed"
    assert runs[0]["return_code"] == -signal.SIGABRT
    assert runs[0]["termination_signal"] == "SIGABRT"
    assert "Fatal Python error: Aborted" in Path(runs[0]["log_path"]).read_text()


def test_source_change_interrupts_active_children(tmp_path):
    runs = plan(tmp_path, delays=(10, 10))
    checks = 0

    def source_guard():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RuntimeError("source changed")

    with pytest.raises(RuntimeError, match="source changed"):
        run_parallel_queue(runs, 2, tmp_path, tmp_path / "logs", lambda: None,
                           source_guard, summary, poll_seconds=0.01)
    assert runs[0]["status"] == "interrupted"
    assert runs[0]["return_code"] is not None
    assert runs[1]["status"] == "pending"


def test_parallel_gpu_rejects_contended_inference_measurement(tmp_path):
    runs = plan(tmp_path)
    runs[0]["resolved_config"]["measure_inference"] = True
    with pytest.raises(ValueError, match="inference timing"):
        run_parallel_queue(runs, 2, tmp_path, tmp_path / "logs", lambda: None, lambda: None, summary)
    assert all("pid" not in row for row in runs)


def test_completed_run_is_skipped_without_creating_process(tmp_path):
    runs = plan(tmp_path)
    Path(runs[0]["summary_path"]).write_text('{"done": 0}')
    assert run_parallel_queue(runs, 2, tmp_path, tmp_path / "logs", lambda: None,
                              lambda: None, summary, poll_seconds=0.01) == 0
    assert runs[0]["status"] == "skipped_complete"
    assert "pid" not in runs[0]
    assert all(row["status"] == "completed" for row in runs[1:])


def test_explicit_continue_on_error_runs_remaining_jobs(tmp_path):
    runs = plan(tmp_path, fail_index=0)
    assert run_parallel_queue(runs, 2, tmp_path, tmp_path / "logs", lambda: None,
                              lambda: None, summary, continue_on_error=True, poll_seconds=0.01) == 1
    assert [row["status"] for row in runs] == ["failed", "completed", "completed"]
