"""Create an immutable manifest-driven audit of the CIFAR-100 P4 batch."""

import argparse
import csv
import io
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.audit_early_exit_p2 import (
    EMPTY_SHA256,
    SMALL_FILES,
    _decode,
    _elapsed_seconds,
    file_record,
    sample_stats,
    sha256_bytes,
    split_fingerprint,
)
from scripts.launch_early_exit_p4 import (
    EXPECTED_SOURCE_HASHES,
    LOCKED_THRESHOLD,
    POLICY_LOCK,
    SEEDS,
    SPLIT_SEED,
    _sha256,
)

EXPECTED_SWEEP_NAME = "cifar100_early_exit_p4_confirmation_serial_p4a"
EXPECTED_PROTOCOL = {
    "dataset": "cifar100",
    "validation_size": 5000,
    "calibration_size": 5000,
    "split_seed": SPLIT_SEED,
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


def collect(manifest_path: Path, launcher_log_path: Path) -> dict:
    manifest_record = file_record(manifest_path)
    manifest = json.loads(manifest_record["text"])
    if manifest.get("sweep_name") != EXPECTED_SWEEP_NAME:
        raise ValueError("Select the exact CIFAR-100 P4a manifest")
    if manifest.get("status") != "completed" or manifest.get("concurrent_jobs") != 1:
        raise ValueError("CIFAR-100 P4a manifest is not a completed serial sweep")
    if len(manifest.get("runs", [])) != 6:
        raise ValueError("CIFAR-100 P4a manifest must contain exactly six runs")

    runs = []
    for manifest_run in manifest["runs"]:
        run_id = manifest_run["experiment_id"]
        if "_p4a_seed" not in run_id:
            raise ValueError(f"Unexpected P4a ID: {run_id}")
        directory = PROJECT_ROOT / "artifacts/runs" / run_id
        runs.append(
            {
                "run_id": run_id,
                "manifest_run": manifest_run,
                "files": {name: file_record(directory / name) for name in SMALL_FILES},
                "checkpoints": {
                    name: file_record(directory / "checkpoints" / name, include_text=False)
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
    source_root = manifest_path.parent / "source_snapshot"
    source_snapshot = {
        relative: file_record(source_root / relative, include_text=False)
        for relative in manifest.get("runtime", {}).get("source_sha256", {})
    }
    return {
        "schema_version": 1,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest_record,
        "launcher_log": file_record(launcher_log_path),
        "policy_lock": file_record(POLICY_LOCK),
        "runs": runs,
        "source_snapshot_root": str(source_root.relative_to(PROJECT_ROOT)),
        "source_snapshot": source_snapshot,
        "scope": (
            "Exact completed CIFAR-100 P4a manifest only; seeds 66/67/68; frozen "
            "threshold 0.903; validation/confirmation only; no official-test evaluation."
        ),
    }


def analyze(snapshot: dict) -> dict:
    manifest = json.loads(_decode(snapshot["manifest"]))
    runtime = manifest.get("runtime", {})
    global_issues = []
    launcher_log = snapshot["launcher_log"]
    if not launcher_log.get("exists"):
        global_issues.append("missing_launcher_log")
    else:
        launcher_text = _decode(launcher_log)
        required_markers = (
            "Sweep status: completed",
            "Threshold 0.903",
            "new seeds 66/67/68",
            "split seed 20260904",
            "Official CIFAR-100 test evaluation is disabled",
        )
        for marker in required_markers:
            if marker not in launcher_text:
                global_issues.append(f"launcher_log_missing:{marker}")
        policy_hash = snapshot["policy_lock"].get("sha256")
        if policy_hash and f"Frozen P4 policy SHA-256: {policy_hash}" not in launcher_text:
            global_issues.append("launcher_policy_hash_mismatch")
    if not snapshot["policy_lock"].get("exists"):
        global_issues.append("missing_policy_lock")
    else:
        policy = json.loads(_decode(snapshot["policy_lock"]))
        if policy.get("status") != "ready_for_independent_p4_confirmation":
            global_issues.append("policy_lock_status_mismatch")
        if policy.get("frozen_policy", {}).get("confidence_threshold") != LOCKED_THRESHOLD:
            global_issues.append("policy_lock_threshold_mismatch")
        if policy.get("frozen_policy", {}).get("p4_threshold_candidates") != 0:
            global_issues.append("policy_lock_target_candidates_nonzero")
        for relative, expected_hash in EXPECTED_SOURCE_HASHES.items():
            if not (PROJECT_ROOT / relative).is_file() or _sha256(PROJECT_ROOT / relative) != expected_hash:
                global_issues.append(f"policy_source_mismatch:{relative}")
    if not runtime.get("git_commit") or runtime.get("git_commit") == "unavailable":
        global_issues.append("missing_runtime_commit")
    if runtime.get("git_status") != "":
        global_issues.append("manifest_git_not_clean")
    if runtime.get("tracked_source_diff_sha256") != EMPTY_SHA256:
        global_issues.append("manifest_source_diff_not_empty")
    runtime_sources = runtime.get("source_sha256", {})
    if set(runtime_sources) != set(snapshot["source_snapshot"]):
        global_issues.append("source_snapshot_file_set_mismatch")
    for relative, expected_hash in runtime_sources.items():
        record = snapshot["source_snapshot"].get(relative, {})
        if not record.get("exists") or record.get("sha256") != expected_hash:
            global_issues.append(f"source_snapshot_mismatch:{relative}")
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
        if any(issue.startswith("missing:") for issue in issues):
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
        if seed not in SEEDS or manifest_run.get("seed") != seed:
            issues.append("training_seed_mismatch")
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
        validation_accuracies = metric_values[4]
        if epochs != list(range(1, 201)):
            issues.append("epoch_sequence_mismatch")
        if not all(math.isfinite(value) for values in metric_values for value in values):
            issues.append("nonfinite_training_metric")
        if validation_accuracies:
            best = max(validation_accuracies)
            first_best_epoch = epochs[validation_accuracies.index(best)]
            if not math.isclose(best, summary.get("best_validation_accuracy", math.nan)):
                issues.append("summary_best_accuracy_mismatch")
            if first_best_epoch != summary.get("best_epoch"):
                issues.append("summary_best_epoch_mismatch")

        architecture = EXPECTED_ARCHITECTURES.get(model_type)
        if summary.get("architecture_version") != architecture:
            issues.append("summary_architecture_mismatch")
        if metrics.get("architecture_version") != architecture:
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
        partitions = [
            set(split.get("train_indices", [])),
            set(split.get("validation_indices", [])),
            set(split.get("calibration_indices", [])),
        ]
        if [len(values) for values in partitions] != [40_000, 5000, 5000]:
            issues.append("split_size_mismatch")
        if any(partitions[left] & partitions[right] for left in range(3) for right in range(left + 1, 3)):
            issues.append("split_overlap")
        if set.union(*partitions) != set(range(50_000)):
            issues.append("split_coverage_mismatch")
        if split.get("split_seed") != SPLIT_SEED:
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
        if provenance.get("git_commit") != runtime.get("git_commit"):
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
        if provenance.get("split_seed") != SPLIT_SEED:
            issues.append("provenance_split_seed_mismatch")
        if provenance.get("training_seed") != seed:
            issues.append("provenance_training_seed_mismatch")
        if provenance.get("architecture_version") != architecture:
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
                "final_validation_percent": 100 * validation_accuracies[-1],
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

    expected_matrix = {(model_type, seed) for model_type in EXPECTED_ARCHITECTURES for seed in SEEDS}
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

    valid_rows = [row for row in rows if "model_type" in row]
    controls = {row["seed"]: row for row in valid_rows if row["model_type"] == "mobilenetv2"}
    exits = {row["seed"]: row for row in valid_rows if row["model_type"] == "multi_exit"}
    if set(controls) != set(SEEDS) or set(exits) != set(SEEDS):
        global_issues.append("incomplete_paired_matrix")
        deltas = []
    else:
        deltas = [exits[seed]["best_validation_percent"] - controls[seed]["best_validation_percent"] for seed in SEEDS]
    issues = {row["run_id"]: row["issues"] for row in rows if row.get("issues")}
    if global_issues:
        issues["__batch__"] = global_issues
    return {
        "schema_version": 1,
        "scope": snapshot["scope"],
        "manifest_path": snapshot["manifest"]["path"],
        "manifest_sha256": snapshot["manifest"]["sha256"],
        "launcher_log": {key: launcher_log.get(key) for key in ("path", "size_bytes", "sha256")},
        "policy_lock": {key: snapshot["policy_lock"].get(key) for key in ("path", "size_bytes", "sha256")},
        "runtime_commit": runtime.get("git_commit"),
        "canonical_split_sha256": next(iter(canonical_split_hashes), None),
        "runs": rows,
        "paired": {
            "comparison": "multi_exit_final - mobilenetv2",
            "unit": "percentage_points",
            "seeds": list(SEEDS),
            "deltas": deltas,
            "statistics": sample_stats(deltas) if deltas else None,
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
        raise FileExistsError("P4 audit evidence is immutable; choose fresh output paths")
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    output.mkdir(parents=True)
    (output / "audit_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    source_index = {
        "snapshot_sha256": sha256_bytes(snapshot_path.read_bytes()),
        "manifest": {key: snapshot["manifest"][key] for key in ("path", "sha256", "size_bytes")},
        "launcher_log": result["launcher_log"],
        "policy_lock": result["policy_lock"],
        "checkpoint_hashes": {
            row["run_id"]: {
                "best": row["best_checkpoint_sha256"],
                "latest": row["latest_checkpoint_sha256"],
                "final": row["final_checkpoint_sha256"],
            }
            for row in result["runs"]
        },
        "split_hashes": {row["run_id"]: row["split_indices_sha256"] for row in result["runs"]},
        "canonical_split_sha256": result["canonical_split_sha256"],
        "runtime_commit": result["runtime_commit"],
        "scope": snapshot["scope"],
    }
    (output / "source_index.json").write_text(json.dumps(source_index, ensure_ascii=False, indent=2) + "\n")
    controls = {row["seed"]: row for row in result["runs"] if row["model_type"] == "mobilenetv2"}
    exits = {row["seed"]: row for row in result["runs"] if row["model_type"] == "multi_exit"}
    lines = [
        "# CIFAR-100 early-exit P4a formal audit",
        "",
        "This report audits only the completed serial P4a manifest; official test data are excluded.",
        "",
        "| seed | baseline validation | multi-exit final validation | paired delta |",
        "|---:|---:|---:|---:|",
    ]
    for seed, delta in zip(result["paired"]["seeds"], result["paired"]["deltas"]):
        lines.append(
            f"| {seed} | {controls[seed]['best_validation_percent']:.2f}% | "
            f"{exits[seed]['best_validation_percent']:.2f}% | {delta:+.2f} pp |"
        )
    stats = result["paired"]["statistics"]
    lines.extend(
        [
            "",
            (
                f"Paired gain: `{stats['mean_percent']:+.3f} ± "
                f"{stats['sample_sd_percent']:.3f} pp` (sample SD); "
                f"wins `{result['paired']['wins']}/3`."
            ),
            f"Total serial duration: `{result['elapsed_seconds'] / 3600:.3f}` hours.",
            f"Manifest SHA-256: `{result['manifest_sha256']}`.",
            "",
        ]
    )
    (output / "experiment_summary.md").write_text("\n".join(lines))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--launcher-log", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    snapshot = collect(args.manifest.resolve(), args.launcher_log.resolve())
    result = analyze(snapshot)
    if result["issues"]:
        raise ValueError(result["issues"])
    if args.dry_run:
        print(json.dumps({"runs": len(result["runs"]), "paired": result["paired"]}, indent=2))
        return 0
    written = write_outputs(snapshot, args.snapshot.resolve(), args.output.resolve())
    print(
        json.dumps(
            {"runs": len(written["runs"]), "paired": written["paired"], "output": str(args.output)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
