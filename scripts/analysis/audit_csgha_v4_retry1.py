"""Audit exactly the completed CSGHA-v4 retry1 manifest without training or test use."""

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
    "artifacts/sweeps/cifar10_csgha_v4_matched_perf2_retry1_20260831_112717/manifest.json"
)
DEFAULT_SNAPSHOT = Path("artifacts/audits/2026-08-31-csgha-v4-retry1/snapshot.json")
DEFAULT_OUTPUT = Path("reports/audits/2026-08-31-csgha-v4-retry1")
SMALL_FILES = (
    "config.yaml", "summary.json", "metrics.json", "benchmark.json", "provenance.json",
    "split_indices.json", "logs/training.csv",
)
EXPECTED_PROTOCOL = {
    "dataset": "cifar10", "validation_size": 5000, "batch_size": 128, "epochs": 200,
    "evaluate_test": False, "cuda_graph": True, "torch_num_threads": 1,
    "measure_inference": False, "amp": True, "accumulation_steps": 1,
    "num_workers": 8, "prefetch_factor": 4,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_record(path: Path, include_text: bool = True) -> dict:
    if not path.is_file():
        return {"path": str(path.relative_to(PROJECT_ROOT)), "exists": False}
    data = path.read_bytes()
    record = {
        "path": str(path.relative_to(PROJECT_ROOT)), "exists": True,
        "size_bytes": len(data), "sha256": sha256_bytes(data),
    }
    if include_text:
        record["text"] = data.decode("utf-8")
    return record


def sample_stats(values: list[float]) -> dict:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Statistics require finite values")
    mean = math.fsum(values) / len(values)
    sample_sd = None
    if len(values) > 1:
        sample_sd = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return {
        "n": len(values), "mean_percent": mean, "sample_sd_percent": sample_sd,
    }


def collect(manifest_path: Path) -> dict:
    manifest_record = file_record(manifest_path)
    manifest = json.loads(manifest_record["text"])
    if manifest.get("sweep_name") != "cifar10_csgha_v4_matched_perf2_retry1":
        raise ValueError("Select the exact retry1 manifest")
    if manifest.get("status") != "completed" or manifest.get("concurrent_jobs") != 2:
        raise ValueError("retry1 manifest is not a completed two-job sweep")
    if len(manifest.get("runs", [])) != 6:
        raise ValueError("retry1 manifest must contain exactly six runs")
    runs = []
    for manifest_run in manifest["runs"]:
        run_id = manifest_run["experiment_id"]
        if "_perf2_retry1_seed" not in run_id:
            raise ValueError(f"Unexpected retry1 ID: {run_id}")
        directory = PROJECT_ROOT / "artifacts/runs" / run_id
        runs.append({
            "run_id": run_id,
            "manifest_run": manifest_run,
            "files": {name: file_record(directory / name) for name in SMALL_FILES},
            "checkpoints": {
                name: file_record(directory / "checkpoints" / name, include_text=False)
                for name in ("model_best.pth", "model_latest.pth", "final.pth")
            },
            "prediction_files": sorted(
                str(path.relative_to(PROJECT_ROOT))
                for path in (directory / "predictions").glob("**/*") if path.is_file()
            ) if (directory / "predictions").exists() else [],
        })
    source_snapshot = manifest_path.parent / "source_snapshot"
    source_files = {
        str(path.relative_to(source_snapshot)): file_record(path, include_text=False)
        for path in sorted(source_snapshot.glob("**/*")) if path.is_file()
    }
    historical_path = PROJECT_ROOT / "reports/audits/2026-08-30/audit_results.json"
    return {
        "schema_version": 1,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest_record,
        "runs": runs,
        "source_snapshot_root": str(source_snapshot.relative_to(PROJECT_ROOT)),
        "source_snapshot": source_files,
        "historical_audit": file_record(historical_path),
        "scope": "Exact retry1 manifest only; no smoke, failed first batch, training, or test evaluation.",
    }


def _decode(record: dict) -> str:
    text = record.get("text", "")
    if record.get("exists") and sha256_bytes(text.encode()) != record["sha256"]:
        raise ValueError(f"Snapshot checksum mismatch: {record['path']}")
    return text


def analyze(snapshot: dict) -> dict:
    _decode(snapshot["manifest"])
    source_snapshot = snapshot["source_snapshot"]
    rows = []
    split_hashes: dict[int, set[str]] = {}
    source_hash_sets = set()
    for entry in snapshot["runs"]:
        issues = []
        run_id = entry["run_id"]
        manifest_run = entry["manifest_run"]
        if manifest_run.get("status") != "completed" or manifest_run.get("return_code") != 0:
            issues.append("manifest_run_not_completed")
        if manifest_run.get("termination_signal") is not None:
            issues.append("unexpected_termination_signal")
        for name, record in entry["files"].items():
            if not record["exists"]:
                issues.append(f"missing:{name}")
        if issues:
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
        if config != manifest_run["resolved_config"]:
            issues.append("manifest_config_mismatch")
        if summary != manifest_run["summary"]:
            issues.append("manifest_summary_mismatch")
        for key, expected in EXPECTED_PROTOCOL.items():
            if config.get(key) != expected:
                issues.append(f"protocol_mismatch:{key}")
        epochs = [int(row["epoch"]) for row in training]
        accuracies = [float(row["val_acc"]) for row in training]
        if epochs != list(range(1, 201)):
            issues.append("epoch_sequence_mismatch")
        if accuracies:
            best = max(accuracies)
            first_best_epoch = epochs[accuracies.index(best)]
            if not math.isclose(best, summary["best_validation_accuracy"], abs_tol=1e-12):
                issues.append("summary_best_accuracy_mismatch")
            if first_best_epoch != summary["best_epoch"]:
                issues.append("summary_best_epoch_mismatch")
        best_checkpoint = entry["checkpoints"]["model_best.pth"]
        if not all(record["exists"] for record in entry["checkpoints"].values()):
            issues.append("checkpoint_missing")
        if best_checkpoint.get("sha256") != summary.get("best_checkpoint_sha256"):
            issues.append("best_checkpoint_hash_mismatch")
        split_record = entry["files"]["split_indices.json"]
        if split_record["sha256"] != summary.get("split_indices_sha256"):
            issues.append("split_hash_mismatch")
        train_indices = split["train_indices"]
        validation_indices = split["validation_indices"]
        if len(train_indices) != 45000 or len(validation_indices) != 5000:
            issues.append("split_size_mismatch")
        if set(train_indices) & set(validation_indices):
            issues.append("split_overlap")
        if set(train_indices) | set(validation_indices) != set(range(50000)):
            issues.append("split_coverage_mismatch")
        split_hashes.setdefault(config["seed"], set()).add(split_record["sha256"])
        if summary.get("test_evaluated") is not False or entry["prediction_files"]:
            issues.append("unexpected_test_or_prediction_output")
        if benchmark.get("measurement_status") != "skipped":
            issues.append("benchmark_not_skipped")
        if any(benchmark.get(key) is not None for key in (
            "inference_latency_mean", "inference_latency_std", "throughput_fps"
        )):
            issues.append("benchmark_null_mismatch")
        if provenance.get("execution_backend") != "cuda_graph_training_v1":
            issues.append("execution_backend_mismatch")
        if provenance.get("amp_cache_enabled") is not False or provenance.get("torch_num_threads") != 1:
            issues.append("execution_provenance_mismatch")
        if provenance.get("architecture_version") != summary.get("architecture_version"):
            issues.append("architecture_version_mismatch")
        source_hashes = provenance.get("source_sha256", {})
        source_hash_sets.add(json.dumps(source_hashes, sort_keys=True))
        for relative, expected_hash in source_hashes.items():
            record = source_snapshot.get(relative)
            if not record or record.get("sha256") != expected_hash:
                issues.append(f"source_snapshot_mismatch:{relative}")
                break
        rows.append({
            "run_id": run_id, "model_type": config["model_type"], "seed": config["seed"],
            "best_validation_percent": 100 * summary["best_validation_accuracy"],
            "final_validation_percent": 100 * accuracies[-1], "best_epoch": summary["best_epoch"],
            "parameters_total": metrics["parameters_total"], "flops_estimate": metrics["flops_total"],
            "architecture_version": summary["architecture_version"],
            "best_checkpoint_sha256": best_checkpoint["sha256"],
            "split_indices_sha256": split_record["sha256"],
            "config": config, "provenance": provenance, "issues": issues,
        })
    for seed, hashes in split_hashes.items():
        if len(hashes) != 1:
            for row in rows:
                if row.get("seed") == seed:
                    row.setdefault("issues", []).append("matched_split_mismatch")
    if len(source_hash_sets) != 1:
        for row in rows:
            row.setdefault("issues", []).append("mixed_source_versions")
    valid_rows = [row for row in rows if "model_type" in row]
    groups = {}
    for model_type in ("hybrid_leaky", "csgha_v4"):
        selected = sorted((row for row in valid_rows if row["model_type"] == model_type), key=lambda row: row["seed"])
        if [row["seed"] for row in selected] != [42, 43, 44]:
            raise ValueError(f"Incomplete seed set for {model_type}")
        groups[model_type] = {
            "seeds": [42, 43, 44],
            "best_validation": sample_stats([row["best_validation_percent"] for row in selected]),
            "parameters_total": selected[0]["parameters_total"],
            "flops_estimate": selected[0]["flops_estimate"],
        }
    controls = {row["seed"]: row for row in valid_rows if row["model_type"] == "hybrid_leaky"}
    guided = {row["seed"]: row for row in valid_rows if row["model_type"] == "csgha_v4"}
    deltas = [guided[seed]["best_validation_percent"] - controls[seed]["best_validation_percent"] for seed in (42, 43, 44)]
    paired = {
        "comparison": "csgha_v4 - hybrid_leaky", "unit": "percentage_points",
        "seeds": [42, 43, 44], "deltas": deltas, "statistics": sample_stats(deltas),
        "wins": sum(delta > 0 for delta in deltas),
    }
    historical = json.loads(_decode(snapshot["historical_audit"]))
    historical_groups = {
        group["variant"]: group for group in historical["groups"]
        if group["dataset"] == "cifar10" and group["variant"] in {
            "CSGHA v3", "Independent middle", "Independent shallow"
        }
    }
    return {
        "schema_version": 1, "manifest_path": snapshot["manifest"]["path"],
        "manifest_sha256": snapshot["manifest"]["sha256"], "runs": rows,
        "groups": groups, "paired": paired, "historical_reference": historical_groups,
        "historical_reference_sha256": snapshot["historical_audit"]["sha256"],
        "issues": {row["run_id"]: row["issues"] for row in rows if row.get("issues")},
    }


def write_outputs(snapshot: dict, snapshot_path: Path, output: Path) -> dict:
    result = analyze(snapshot)
    if result["issues"]:
        raise ValueError(f"Audit issues: {result['issues']}")
    if snapshot_path.exists() or output.exists():
        raise FileExistsError("Use new P1 snapshot/report paths; existing evidence is immutable")
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    source_root = snapshot_path.parent / "source_files"
    for entry in snapshot["runs"]:
        for record in entry["files"].values():
            target = source_root / record["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(record["text"])
    for record in (snapshot["manifest"], snapshot["historical_audit"]):
        target = source_root / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(record["text"])
    output.mkdir(parents=True)
    (output / "audit_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    source_index = {
        "snapshot_sha256": sha256_bytes(snapshot_path.read_bytes()),
        "manifest": {key: snapshot["manifest"][key] for key in ("path", "sha256", "size_bytes")},
        "checkpoint_hashes": {row["run_id"]: row["best_checkpoint_sha256"] for row in result["runs"]},
        "split_hashes": {row["run_id"]: row["split_indices_sha256"] for row in result["runs"]},
        "scope": snapshot["scope"],
    }
    (output / "source_index.json").write_text(json.dumps(source_index, ensure_ascii=False, indent=2) + "\n")
    control = result["groups"]["hybrid_leaky"]
    guided = result["groups"]["csgha_v4"]
    lines = [
        "# CSGHA v4 retry1 正式审计", "",
        "本报告只由 retry1 completed manifest 的六个 run 生成；未收集 smoke、首批失败/取消目录或 test 结果。", "",
        "## 核心结果", "", "| Seed | Matched control | CSGHA v4 | v4 − control |", "|---:|---:|---:|---:|",
    ]
    controls = {row["seed"]: row for row in result["runs"] if row["model_type"] == "hybrid_leaky"}
    guided_rows = {row["seed"]: row for row in result["runs"] if row["model_type"] == "csgha_v4"}
    for seed, delta in zip(result["paired"]["seeds"], result["paired"]["deltas"]):
        lines.append(f"| {seed} | {controls[seed]['best_validation_percent']:.2f}% | {guided_rows[seed]['best_validation_percent']:.2f}% | {delta:+.2f} pp |")
    lines += [
        (
            f"| 均值 ± 样本标准差 | {control['best_validation']['mean_percent']:.2f} ± {control['best_validation']['sample_sd_percent']:.2f}% | "
            f"{guided['best_validation']['mean_percent']:.2f} ± {guided['best_validation']['sample_sd_percent']:.2f}% | "
            f"{result['paired']['statistics']['mean_percent']:+.3f} ± {result['paired']['statistics']['sample_sd_percent']:.3f} pp |"
        ),
        "", f"v4 胜出 {result['paired']['wins']}/3；平均优势很小且 seed44 为负，不能声称稳定提升。", "",
        "## 完整性与协议", "",
        "六组均连续记录200 epochs；summary与首次best一致；best/latest/final checkpoint、划分哈希和source snapshot通过。",
        "同seed两模型使用相同45k/5k划分；CUDA Graph、AMP no-cache、单线程和validation-only协议一致，未生成test预测。", "",
        f"Control参数量 {control['parameters_total']:,}；v4参数量 {guided['parameters_total']:,}（+{guided['parameters_total'] - control['parameters_total']:,}）。FLOPs为项目解析估计，不是profiler实测。", "",
        "## 历史参照边界", "",
        "旧v3、Independent middle、Independent shallow来自2026-08-30审计，执行后端与retry1不同，只作历史参照；不能把跨批改善归因于LeakyReLU或guidance。", "",
        "## 下一步", "", "进入版本匹配checkpoint诊断：先严格复现完整5k validation，再做guidance置零/置换/训练均值及deep置零干预，同时统计v4与control的deep分支。", "",
        f"Manifest SHA-256：`{result['manifest_sha256']}`；旧审计SHA-256：`{result['historical_reference_sha256']}`。",
    ]
    (output / "experiment_summary.md").write_text("\n".join(lines) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = (PROJECT_ROOT / args.manifest).resolve()
    snapshot = collect(manifest)
    result = analyze(snapshot)
    if result["issues"]:
        raise ValueError(result["issues"])
    if args.dry_run:
        print(json.dumps({"runs": len(result["runs"]), "paired": result["paired"], "issues": result["issues"]}, indent=2))
        return 0
    written = write_outputs(snapshot, PROJECT_ROOT / args.snapshot, PROJECT_ROOT / args.output)
    print(json.dumps({"runs": len(written["runs"]), "paired": written["paired"], "output": str(PROJECT_ROOT / args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
