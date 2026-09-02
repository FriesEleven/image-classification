"""Audit exactly the completed early-exit P1b batch without evaluating test data."""

import argparse
import csv
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(
    "artifacts/sweeps/cifar10_early_exit_p1_serial_p1b_20260902_144353/manifest.json"
)
DEFAULT_LAUNCHER_LOG = Path(
    "artifacts/launcher_logs/early_exit_p1_serial_p1b_20260902_144349_381510.log"
)
DEFAULT_SNAPSHOT = Path("artifacts/audits/2026-09-02-early-exit-p1b/snapshot.json")
DEFAULT_OUTPUT = Path("reports/audits/2026-09-02-early-exit-p1b")
EXPECTED_COMMIT = "278164e9d75aafba511ea02107f4ff0e7c2c67a8"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SMALL_FILES = (
    "config.yaml",
    "summary.json",
    "metrics.json",
    "benchmark.json",
    "provenance.json",
    "split_indices.json",
    "logs/training.csv",
)
EXPECTED_PROTOCOL = {
    "dataset": "cifar10",
    "validation_size": 5000,
    "calibration_size": 5000,
    "split_seed": 20_260_902,
    "batch_size": 128,
    "epochs": 200,
    "lr": 0.01,
    "amp": True,
    "cuda_graph": True,
    "torch_num_threads": 1,
    "measure_inference": False,
    "evaluate_test": False,
    "accumulation_steps": 1,
    "num_workers": 8,
    "prefetch_factor": 8,
}
EXPECTED_MULTI_EXIT = {
    "exit_positions": [8, 16],
    "exit_loss_weights": [0.2, 0.3],
    "exit_distillation_alpha": 0.5,
    "exit_temperature": 3.0,
}
EXPECTED_ARCHITECTURES = {
    "mobilenetv2": "mobilenetv2_v1",
    "multi_exit": "mobilenetv2_multi_exit_v1_detached_final_kd",
}
EXPECTED_SEEDS = (54, 55, 56)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_record(path: Path, *, include_text: bool = True) -> dict:
    if not path.is_file():
        return {"path": str(path.relative_to(PROJECT_ROOT)), "exists": False}
    data = path.read_bytes()
    record = {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "exists": True,
        "size_bytes": len(data),
        "sha256": sha256_bytes(data),
    }
    if include_text:
        record["text"] = data.decode("utf-8")
    return record


def _decode(record: dict) -> str:
    text = record.get("text", "")
    if record.get("exists") and sha256_bytes(text.encode()) != record["sha256"]:
        raise ValueError(f"Snapshot checksum mismatch: {record['path']}")
    return text


def sample_stats(values: list[float]) -> dict:
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("Sample statistics require at least two finite values")
    mean = math.fsum(values) / len(values)
    sample_sd = math.sqrt(
        math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    )
    return {"n": len(values), "mean_percent": mean, "sample_sd_percent": sample_sd}


def split_fingerprint(split: dict) -> str:
    canonical = json.dumps(
        {
            "train": split.get("train_indices", []),
            "validation": split.get("validation_indices", []),
            "calibration": split.get("calibration_indices", []),
        },
        separators=(",", ":"),
    ).encode()
    return sha256_bytes(canonical)


def collect(manifest_path: Path, launcher_log_path: Path) -> dict:
    manifest_record = file_record(manifest_path)
    manifest = json.loads(manifest_record["text"])
    if manifest.get("sweep_name") != "cifar10_early_exit_p1_serial_p1b":
        raise ValueError("Select the exact early-exit P1b manifest")
    if manifest.get("status") != "completed" or manifest.get("concurrent_jobs") != 1:
        raise ValueError("Early-exit P1b manifest is not a completed serial sweep")
    if len(manifest.get("runs", [])) != 6:
        raise ValueError("Early-exit P1b manifest must contain exactly six runs")

    runs = []
    for manifest_run in manifest["runs"]:
        run_id = manifest_run["experiment_id"]
        if "_p1b_seed" not in run_id:
            raise ValueError(f"Unexpected P1b ID: {run_id}")
        directory = PROJECT_ROOT / "artifacts/runs" / run_id
        runs.append(
            {
                "run_id": run_id,
                "manifest_run": manifest_run,
                "files": {name: file_record(directory / name) for name in SMALL_FILES},
                "checkpoints": {
                    name: file_record(
                        directory / "checkpoints" / name,
                        include_text=False,
                    )
                    for name in ("model_best.pth", "model_latest.pth", "final.pth")
                },
                "periodic_checkpoints": sorted(
                    str(path.relative_to(PROJECT_ROOT))
                    for path in (directory / "checkpoints").glob("epoch_*.pth")
                    if path.is_file()
                ),
                "prediction_files": sorted(
                    str(path.relative_to(PROJECT_ROOT))
                    for path in (directory / "predictions").glob("**/*")
                    if path.is_file()
                ),
            }
        )

    source_snapshot = manifest_path.parent / "source_snapshot"
    source_files = {
        str(path.relative_to(source_snapshot)): file_record(path, include_text=False)
        for path in sorted(source_snapshot.glob("**/*"))
        if path.is_file()
    }
    return {
        "schema_version": 1,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest_record,
        "launcher_log": file_record(launcher_log_path),
        "runs": runs,
        "source_snapshot_root": str(source_snapshot.relative_to(PROJECT_ROOT)),
        "source_snapshot": source_files,
        "scope": (
            "Exact completed P1b manifest only; 40k train, 5k checkpoint-selection "
            "validation and disjoint 5k policy calibration; no official-test evaluation."
        ),
    }


def _elapsed_seconds(start: str, finish: str) -> float:
    return (datetime.fromisoformat(finish) - datetime.fromisoformat(start)).total_seconds()


def analyze(snapshot: dict) -> dict:
    manifest = json.loads(_decode(snapshot["manifest"]))
    global_issues = []
    if not snapshot["launcher_log"].get("exists"):
        global_issues.append("missing_launcher_log")
    elif "Sweep status: completed" not in _decode(snapshot["launcher_log"]):
        global_issues.append("launcher_log_not_completed")

    runtime = manifest.get("runtime", {})
    if runtime.get("git_commit") != EXPECTED_COMMIT:
        global_issues.append("manifest_commit_mismatch")
    if runtime.get("git_status") != "":
        global_issues.append("manifest_git_not_clean")
    if runtime.get("tracked_source_diff_sha256") != EMPTY_SHA256:
        global_issues.append("manifest_source_diff_not_empty")
    runtime_sources = runtime.get("source_sha256", {})
    if set(runtime_sources) != set(snapshot["source_snapshot"]):
        global_issues.append("manifest_source_snapshot_file_set_mismatch")
    for relative, expected_hash in runtime_sources.items():
        record = snapshot["source_snapshot"].get(relative)
        if not record or record.get("sha256") != expected_hash:
            global_issues.append(f"manifest_source_snapshot_mismatch:{relative}")
            break

    rows = []
    matrix_keys = set()
    raw_split_hashes: dict[int, set[str]] = {}
    canonical_split_hashes = set()
    source_hash_sets = set()
    previous_finish = None
    for entry in snapshot["runs"]:
        run_id = entry["run_id"]
        manifest_run = entry["manifest_run"]
        issues = []
        if manifest_run.get("status") != "completed" or manifest_run.get("return_code") != 0:
            issues.append("manifest_run_not_completed")
        if manifest_run.get("termination_signal") is not None:
            issues.append("unexpected_termination_signal")
        start = manifest_run.get("started_at")
        finish = manifest_run.get("finished_at")
        if not start or not finish or _elapsed_seconds(start, finish) <= 0:
            issues.append("invalid_run_timestamps")
        if previous_finish is not None and start != previous_finish:
            issues.append("serial_timeline_gap_or_overlap")
        previous_finish = finish
        for name, record in entry["files"].items():
            if not record["exists"]:
                issues.append(f"missing:{name}")
        if issues and any(issue.startswith("missing:") for issue in issues):
            rows.append({"run_id": run_id, "issues": issues})
            continue

        config = yaml.safe_load(_decode(entry["files"]["config.yaml"]))
        config.pop("runtime", None)
        summary = json.loads(_decode(entry["files"]["summary.json"]))
        metrics = json.loads(_decode(entry["files"]["metrics.json"]))
        benchmark = json.loads(_decode(entry["files"]["benchmark.json"]))
        provenance = json.loads(_decode(entry["files"]["provenance.json"]))
        split = json.loads(_decode(entry["files"]["split_indices.json"]))
        training = list(csv.DictReader(io.StringIO(_decode(entry["files"]["logs/training.csv"]))))
        if config != manifest_run.get("resolved_config"):
            issues.append("manifest_config_mismatch")
        if summary != manifest_run.get("summary"):
            issues.append("manifest_summary_mismatch")
        for key, expected in EXPECTED_PROTOCOL.items():
            if config.get(key) != expected:
                issues.append(f"protocol_mismatch:{key}")

        model_type = config.get("model_type")
        seed = config.get("seed")
        matrix_keys.add((model_type, seed))
        if model_type not in EXPECTED_ARCHITECTURES:
            issues.append("unexpected_model_type")
        if seed not in EXPECTED_SEEDS:
            issues.append("unexpected_training_seed")
        if manifest_run.get("seed") != seed:
            issues.append("manifest_seed_mismatch")
        if model_type == "multi_exit":
            for key, expected in EXPECTED_MULTI_EXIT.items():
                if config.get(key) != expected:
                    issues.append(f"multi_exit_mismatch:{key}")

        epochs = [int(row["epoch"]) for row in training]
        metric_columns = (
            "learning_rate",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "train_precision",
            "train_recall",
            "train_f1",
            "val_precision",
            "val_recall",
            "val_f1",
        )
        metric_values = [[float(row[column]) for row in training] for column in metric_columns]
        val_accuracies = metric_values[4]
        if epochs != list(range(1, 201)):
            issues.append("epoch_sequence_mismatch")
        if not all(math.isfinite(value) for values in metric_values for value in values):
            issues.append("nonfinite_training_metric")
        if val_accuracies:
            best = max(val_accuracies)
            first_best_epoch = epochs[val_accuracies.index(best)]
            if not math.isclose(best, summary.get("best_validation_accuracy", math.nan)):
                issues.append("summary_best_accuracy_mismatch")
            if first_best_epoch != summary.get("best_epoch"):
                issues.append("summary_best_epoch_mismatch")

        expected_architecture = EXPECTED_ARCHITECTURES.get(model_type)
        if summary.get("architecture_version") != expected_architecture:
            issues.append("summary_architecture_mismatch")
        if metrics.get("architecture_version") != expected_architecture:
            issues.append("metrics_architecture_mismatch")
        if summary.get("experiment_id") != run_id:
            issues.append("summary_experiment_id_mismatch")
        expected_counts = {
            "train_samples": 40_000,
            "validation_samples": 5000,
            "calibration_samples": 5000,
            "test_samples": 10_000,
            "test_evaluated": False,
        }
        for key, expected in expected_counts.items():
            if summary.get(key) != expected:
                issues.append(f"summary_protocol_mismatch:{key}")

        if not all(record["exists"] for record in entry["checkpoints"].values()):
            issues.append("checkpoint_missing")
        best_checkpoint = entry["checkpoints"]["model_best.pth"]
        if best_checkpoint.get("sha256") != summary.get("best_checkpoint_sha256"):
            issues.append("best_checkpoint_hash_mismatch")
        if len(entry["periodic_checkpoints"]) != 20:
            issues.append("periodic_checkpoint_count_mismatch")

        split_record = entry["files"]["split_indices.json"]
        if split_record["sha256"] != summary.get("split_indices_sha256"):
            issues.append("split_hash_mismatch")
        train_indices = split.get("train_indices", [])
        validation_indices = split.get("validation_indices", [])
        calibration_indices = split.get("calibration_indices", [])
        partitions = [set(values) for values in (train_indices, validation_indices, calibration_indices)]
        if [len(values) for values in partitions] != [40_000, 5000, 5000]:
            issues.append("split_size_mismatch")
        if any(
            partitions[left] & partitions[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            issues.append("split_overlap")
        if set.union(*partitions) != set(range(50_000)):
            issues.append("split_coverage_mismatch")
        if split.get("split_seed") != EXPECTED_PROTOCOL["split_seed"]:
            issues.append("split_seed_mismatch")
        if split.get("training_seed") != seed:
            issues.append("split_training_seed_mismatch")
        raw_split_hashes.setdefault(seed, set()).add(split_record["sha256"])
        canonical_fingerprint = split_fingerprint(split)
        canonical_split_hashes.add(canonical_fingerprint)

        if summary.get("test_evaluated") is not False or entry["prediction_files"]:
            issues.append("unexpected_test_or_prediction_output")
        if benchmark.get("measurement_status") != "skipped":
            issues.append("benchmark_not_skipped")
        if any(
            benchmark.get(key) is not None
            for key in ("inference_latency_mean", "inference_latency_std", "throughput_fps")
        ):
            issues.append("benchmark_null_mismatch")

        if provenance.get("git_commit") != EXPECTED_COMMIT:
            issues.append("provenance_commit_mismatch")
        if provenance.get("git_status") != "":
            issues.append("provenance_git_not_clean")
        if provenance.get("tracked_source_diff_sha256") != EMPTY_SHA256:
            issues.append("provenance_source_diff_not_empty")
        if provenance.get("execution_backend") != "cuda_graph_training_v1":
            issues.append("execution_backend_mismatch")
        if provenance.get("amp_cache_enabled") is not False:
            issues.append("amp_cache_mismatch")
        if provenance.get("torch_num_threads") != 1:
            issues.append("torch_threads_mismatch")
        if provenance.get("split_seed") != EXPECTED_PROTOCOL["split_seed"]:
            issues.append("provenance_split_seed_mismatch")
        if provenance.get("training_seed") != seed:
            issues.append("provenance_training_seed_mismatch")
        if provenance.get("architecture_version") != expected_architecture:
            issues.append("provenance_architecture_mismatch")
        source_hashes = provenance.get("source_sha256", {})
        source_hash_sets.add(json.dumps(source_hashes, sort_keys=True))
        if source_hashes != runtime_sources:
            issues.append("provenance_manifest_source_mismatch")

        rows.append(
            {
                "run_id": run_id,
                "model_type": model_type,
                "seed": seed,
                "best_validation_percent": 100 * summary["best_validation_accuracy"],
                "final_validation_percent": 100 * val_accuracies[-1],
                "best_epoch": summary["best_epoch"],
                "elapsed_seconds": _elapsed_seconds(start, finish),
                "parameters_total": metrics["parameters_total"],
                "parameters_exit_heads": metrics.get("parameters_exit_heads", 0),
                "best_checkpoint_sha256": best_checkpoint.get("sha256"),
                "latest_checkpoint_sha256": entry["checkpoints"]["model_latest.pth"].get("sha256"),
                "final_checkpoint_sha256": entry["checkpoints"]["final.pth"].get("sha256"),
                "split_indices_sha256": split_record["sha256"],
                "canonical_split_sha256": canonical_fingerprint,
                "periodic_checkpoint_count": len(entry["periodic_checkpoints"]),
                "issues": issues,
            }
        )

    expected_matrix = {
        (model_type, seed) for model_type in EXPECTED_ARCHITECTURES for seed in EXPECTED_SEEDS
    }
    if matrix_keys != expected_matrix:
        global_issues.append("run_matrix_mismatch")
    for seed, hashes in raw_split_hashes.items():
        if len(hashes) != 1:
            global_issues.append(f"matched_split_file_mismatch:seed{seed}")
    if len(canonical_split_hashes) != 1:
        global_issues.append("fixed_split_content_mismatch")
    if len(source_hash_sets) != 1:
        global_issues.append("mixed_source_versions")
    if previous_finish != manifest.get("finished_at"):
        global_issues.append("manifest_finish_mismatch")
    first_start = snapshot["runs"][0]["manifest_run"].get("started_at")
    if (
        not first_start
        or not manifest.get("created_at")
        or not 0 <= _elapsed_seconds(manifest["created_at"], first_start) <= 10
    ):
        global_issues.append("manifest_start_mismatch")

    valid_rows = [row for row in rows if "model_type" in row]
    groups = {}
    for model_type in EXPECTED_ARCHITECTURES:
        selected = sorted(
            (row for row in valid_rows if row["model_type"] == model_type),
            key=lambda row: row["seed"],
        )
        if [row["seed"] for row in selected] != list(EXPECTED_SEEDS):
            raise ValueError(f"Incomplete seed set for {model_type}")
        groups[model_type] = {
            "seeds": list(EXPECTED_SEEDS),
            "best_validation": sample_stats(
                [row["best_validation_percent"] for row in selected]
            ),
            "parameters_total": selected[0]["parameters_total"],
            "parameters_exit_heads": selected[0]["parameters_exit_heads"],
        }

    controls = {row["seed"]: row for row in valid_rows if row["model_type"] == "mobilenetv2"}
    exits = {row["seed"]: row for row in valid_rows if row["model_type"] == "multi_exit"}
    deltas = [
        exits[seed]["best_validation_percent"] - controls[seed]["best_validation_percent"]
        for seed in EXPECTED_SEEDS
    ]
    issues = {row["run_id"]: row["issues"] for row in rows if row.get("issues")}
    if global_issues:
        issues["__batch__"] = global_issues
    return {
        "schema_version": 1,
        "scope": snapshot["scope"],
        "manifest_path": snapshot["manifest"]["path"],
        "manifest_sha256": snapshot["manifest"]["sha256"],
        "launcher_log": {
            key: snapshot["launcher_log"].get(key)
            for key in ("path", "size_bytes", "sha256")
        },
        "runtime_commit": runtime.get("git_commit"),
        "canonical_split_sha256": next(iter(canonical_split_hashes), None),
        "runs": rows,
        "groups": groups,
        "paired": {
            "comparison": "multi_exit_final - mobilenetv2",
            "unit": "percentage_points",
            "seeds": list(EXPECTED_SEEDS),
            "deltas": deltas,
            "statistics": sample_stats(deltas),
            "wins": sum(delta > 0 for delta in deltas),
        },
        "elapsed_seconds": _elapsed_seconds(manifest["created_at"], manifest["finished_at"]),
        "issues": issues,
    }


def write_outputs(snapshot: dict, snapshot_path: Path, output: Path) -> dict:
    result = analyze(snapshot)
    if result["issues"]:
        raise ValueError(f"Audit issues: {result['issues']}")
    if snapshot_path.exists() or output.exists():
        raise FileExistsError("Audit evidence is immutable; choose fresh output paths")
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    output.mkdir(parents=True)
    (output / "audit_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    source_index = {
        "snapshot_sha256": sha256_bytes(snapshot_path.read_bytes()),
        "manifest": {
            key: snapshot["manifest"][key] for key in ("path", "sha256", "size_bytes")
        },
        "launcher_log": result["launcher_log"],
        "checkpoint_hashes": {
            row["run_id"]: {
                "best": row["best_checkpoint_sha256"],
                "latest": row["latest_checkpoint_sha256"],
                "final": row["final_checkpoint_sha256"],
            }
            for row in result["runs"]
        },
        "split_hashes": {
            row["run_id"]: row["split_indices_sha256"] for row in result["runs"]
        },
        "canonical_split_sha256": result["canonical_split_sha256"],
        "runtime_commit": result["runtime_commit"],
        "scope": snapshot["scope"],
    }
    (output / "source_index.json").write_text(
        json.dumps(source_index, ensure_ascii=False, indent=2) + "\n"
    )

    controls = {row["seed"]: row for row in result["runs"] if row["model_type"] == "mobilenetv2"}
    exits = {row["seed"]: row for row in result["runs"] if row["model_type"] == "multi_exit"}
    lines = [
        "# Early-exit P1b 正式审计",
        "",
        "本报告只使用唯一 completed P1b manifest 的六组 validation/calibration-only run。",
        "",
        "| Seed | MobileNetV2 validation | Multi-exit final validation | 配对差值 |",
        "|---:|---:|---:|---:|",
    ]
    for seed, delta in zip(result["paired"]["seeds"], result["paired"]["deltas"]):
        lines.append(
            f"| {seed} | {controls[seed]['best_validation_percent']:.2f}% | "
            f"{exits[seed]['best_validation_percent']:.2f}% | {delta:+.2f} pp |"
        )
    control_stats = result["groups"]["mobilenetv2"]["best_validation"]
    exit_stats = result["groups"]["multi_exit"]["best_validation"]
    delta_stats = result["paired"]["statistics"]
    lines.extend(
        [
            (
                f"| 均值 ± 样本标准差 | {control_stats['mean_percent']:.3f} ± "
                f"{control_stats['sample_sd_percent']:.3f}% | "
                f"{exit_stats['mean_percent']:.3f} ± "
                f"{exit_stats['sample_sd_percent']:.3f}% | "
                f"{delta_stats['mean_percent']:+.3f} ± "
                f"{delta_stats['sample_sd_percent']:.3f} pp |"
            ),
            "",
            "六组均连续记录 200 epochs；清单、summary、首次 best、best/latest/final、",
            "40k/5k/5k 互斥划分、执行溯源与源码快照全部通过文件级核查。",
            "所有 run 使用同一 split seed，且同 seed 两模型的原始划分文件完全一致。",
            "官方 CIFAR-10 test 未评估、无预测文件、受争用训练期间未记录推理延迟。",
            "",
            f"总串行时长：`{result['elapsed_seconds'] / 3600:.3f}` 小时。",
            f"Manifest SHA-256：`{result['manifest_sha256']}`。",
            "",
        ]
    )
    (output / "experiment_summary.md").write_text("\n".join(lines))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--launcher-log", type=Path, default=DEFAULT_LAUNCHER_LOG)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    snapshot = collect(
        (PROJECT_ROOT / args.manifest).resolve(),
        (PROJECT_ROOT / args.launcher_log).resolve(),
    )
    result = analyze(snapshot)
    if result["issues"]:
        raise ValueError(result["issues"])
    if args.dry_run:
        print(
            json.dumps(
                {"runs": len(result["runs"]), "paired": result["paired"], "issues": {}},
                indent=2,
            )
        )
        return 0
    written = write_outputs(
        snapshot,
        PROJECT_ROOT / args.snapshot,
        PROJECT_ROOT / args.output,
    )
    print(
        json.dumps(
            {
                "runs": len(written["runs"]),
                "paired": written["paired"],
                "output": str(PROJECT_ROOT / args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
