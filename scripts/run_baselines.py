"""Run and record a multi-seed CIFAR experiment batch."""

import argparse
import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_classification.config import load_config
from image_classification.paths import RunPaths
from image_classification.training.provenance import runtime_provenance, snapshot_sources, source_fingerprint
from image_classification.training.sweep import (
    NATIVE_DIAGNOSTIC_ENVIRONMENT,
    exit_record,
    native_diagnostic_environment,
    run_owned_process,
    run_parallel_queue,
)

DEFAULT_SWEEP = ROOT / "configs/sweeps/baselines.yaml"


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _runtime_record() -> dict:
    cuda_available = torch.cuda.is_available()
    return {
        **runtime_provenance(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_status": _git_output("status", "--short"),
    }


def load_sweep(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        sweep = yaml.safe_load(handle) or {}
    if not sweep.get("name"):
        raise ValueError(f"Sweep name is missing in {path}")
    if not sweep.get("experiments"):
        raise ValueError(f"No experiments listed in {path}")
    if not sweep.get("seeds"):
        raise ValueError(f"No seeds listed in {path}")
    return sweep


def build_plan(sweep: dict, experiment_tag: str | None = None) -> list[dict]:
    if experiment_tag is not None and not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", experiment_tag):
        raise ValueError(
            "Experiment tag must be lowercase alphanumeric words separated by underscores"
        )
    plan = []
    for config_value in sweep["experiments"]:
        config_path = (ROOT / config_value).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        with config_path.open(encoding="utf-8") as handle:
            experiment_name = (yaml.safe_load(handle) or {}).get("experiment_name", config_path.stem)
        if experiment_tag:
            experiment_name = f"{experiment_name}_{experiment_tag}"
        for seed_value in sweep["seeds"]:
            seed = int(seed_value)
            seeded_name = f"{experiment_name}_seed{seed}"
            arguments = [
                "--config", str(config_path),
                "--seed", str(seed),
                "--experiment_name", seeded_name,
            ]
            config = load_config(arguments)
            plan.append(
                {
                    "config_path": str(config_path.relative_to(ROOT)),
                    "seed": seed,
                    "experiment_id": config.experiment_id,
                    "resolved_config": config.to_dict(),
                    "command": [sys.executable, str(ROOT / "scripts/train.py"), *arguments],
                    "status": "pending",
                    "started_at": None,
                    "finished_at": None,
                    "return_code": None,
                    "termination_signal": None,
                    "summary": None,
                }
            )
    experiment_ids = [run["experiment_id"] for run in plan]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("The seeded sweep contains duplicate experiment IDs")
    return plan


def _write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    os.replace(temporary_path, path)


def _completed_summary(run: dict) -> dict | None:
    paths = RunPaths(run["experiment_id"])
    summary_path = paths.root / "summary.json"
    if not summary_path.exists():
        if paths.root.exists():
            raise RuntimeError(
                f"Incomplete output already exists for {run['experiment_id']}: {paths.root}. "
                "Move or remove that directory after inspecting it before restarting."
            )
        return None
    saved_config_path = paths.root / "config.yaml"
    with saved_config_path.open(encoding="utf-8") as handle:
        saved_config = yaml.safe_load(handle) or {}
    saved_config.pop("runtime", None)
    if saved_config != run["resolved_config"]:
        raise RuntimeError(f"Completed output has a different config: {paths.root}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--jobs", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--experiment-tag")
    args = parser.parse_args()

    sweep_path = args.sweep.resolve()
    sweep = load_sweep(sweep_path)
    tagged_sweep_name = sweep["name"]
    if args.experiment_tag:
        tagged_sweep_name = f"{tagged_sweep_name}_{args.experiment_tag}"
    launch_stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    manifest_path = ROOT / "artifacts/sweeps" / f"{tagged_sweep_name}_{launch_stamp}" / "manifest.json"
    manifest = {
        "sweep_name": tagged_sweep_name,
        "description": sweep.get("description", ""),
        "sweep_config": str(sweep_path.relative_to(ROOT)),
        "status": "dry_run" if args.dry_run else "running",
        "created_at": _timestamp(),
        "finished_at": None,
        "runtime": _runtime_record(),
        "runs": build_plan(sweep, args.experiment_tag),
        "concurrent_jobs": args.jobs,
    }
    manifest["runtime"]["native_crash_diagnostics"] = NATIVE_DIAGNOSTIC_ENVIRONMENT
    _write_manifest(manifest_path, manifest)
    print(f"Manifest: {manifest_path}", flush=True)

    failures = 0
    try:
        if not args.dry_run:
            snapshot_sources(manifest_path.parent / "source_snapshot", manifest["runtime"]["source_sha256"])

        def check_source():
            if source_fingerprint() != manifest["runtime"]["source_sha256"]:
                raise RuntimeError("Source/config files changed since this sweep started; refusing mixed-version runs")

        if args.jobs > 1 and not args.dry_run:
            failures = run_parallel_queue(
                manifest["runs"], args.jobs, ROOT, manifest_path.parent / "run_logs",
                lambda: _write_manifest(manifest_path, manifest), check_source,
                _completed_summary, args.continue_on_error,
            )
        for run in manifest["runs"] if args.jobs == 1 or args.dry_run else []:
            print(" ".join(run["command"]), flush=True)
            if args.dry_run:
                continue
            if source_fingerprint() != manifest["runtime"]["source_sha256"]:
                raise RuntimeError("Source/config files changed since this sweep started; refusing mixed-version runs")
            completed_summary = _completed_summary(run)
            if completed_summary is not None:
                run.update(
                    status="skipped_complete",
                    finished_at=_timestamp(),
                    return_code=0,
                    summary=completed_summary,
                )
                _write_manifest(manifest_path, manifest)
                print(f"Already complete: {run['experiment_id']}", flush=True)
                continue
            run.update(status="running", started_at=_timestamp())

            def record_process(process, current_run=run):
                current_run.update(pid=process.pid, process_group=process.pid)
                _write_manifest(manifest_path, manifest)

            return_code = run_owned_process(
                run["command"], cwd=ROOT, env=native_diagnostic_environment(),
                on_start=record_process,
            )
            if source_fingerprint() != manifest["runtime"]["source_sha256"]:
                raise RuntimeError("Source/config files changed during training; inspect the run before reuse")
            run.update(
                status="completed" if return_code == 0 else "failed",
                finished_at=_timestamp(),
                **exit_record(return_code),
            )
            if return_code == 0:
                run["summary"] = _completed_summary(run)
            else:
                failures += 1
            _write_manifest(manifest_path, manifest)
            if return_code and not args.continue_on_error:
                break
    except (KeyboardInterrupt, Exception):
        manifest["status"] = "interrupted"
        manifest["finished_at"] = _timestamp()
        for run in manifest["runs"]:
            if run["status"] == "running":
                run.update(status="interrupted", finished_at=manifest["finished_at"])
        _write_manifest(manifest_path, manifest)
        raise

    manifest["status"] = "failed" if failures else ("dry_run" if args.dry_run else "completed")
    manifest["finished_at"] = _timestamp()
    _write_manifest(manifest_path, manifest)
    print(f"Sweep status: {manifest['status']}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
