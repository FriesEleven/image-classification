"""Collect read-only experiment evidence over SSH and build a reproducible audit.

Only small metadata are copied; best checkpoint bytes are hashed on the server,
not downloaded or loaded into a model. No training or test evaluation is run.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import runpy
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

FORMAL_PREFIXES = ("baseline_mobilenetv2_seed", "position_se1-2_", "csgha_se1-2_", "csgha_v2_", "csgha_v3_")
SMALL_FILES = ("config.yaml", "summary.json", "metrics.json", "benchmark.json", "logs/training.csv")
CODE_FILES = (
    "src/image_classification/config.py",
    "src/image_classification/models/attention.py",
    "src/image_classification/models/mobilenetv2.py",
    "src/image_classification/training/engine.py",
    "src/image_classification/training/benchmark.py",
    "src/image_classification/data/cifar.py",
    "src/image_classification/utils/reproducibility.py",
    "scripts/diagnostics/check_csgha_guidance.py",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_record(path: Path, root: Path, include_text: bool = True) -> dict:
    record = {"path": path.relative_to(root).as_posix(), "exists": path.is_file()}
    if not record["exists"]:
        return record
    before = path.stat()
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    record.update(size_bytes=before.st_size, sha256=checksum.hexdigest())
    if include_text:
        data = path.read_bytes()
        if digest(data) != record["sha256"]:
            raise RuntimeError(f"File changed during collection: {path}")
        record["text"] = data.decode("utf-8")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed during collection: {path}")
    return record


def git_output(root: Path, *arguments: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def collect(root: Path) -> dict:
    import yaml

    runs = []
    excluded = []
    for directory in sorted((root / "artifacts/runs").iterdir()):
        if not directory.is_dir():
            continue
        if "smoke" in directory.name or not directory.name.startswith(FORMAL_PREFIXES):
            excluded.append(directory.name)
            continue
        files = {name: file_record(directory / name, root) for name in SMALL_FILES}
        config = yaml.safe_load(files["config.yaml"].get("text", "")) or {}
        runs.append({
            "run_id": directory.name, "config": config, "files": files,
            "best_checkpoint": file_record(directory / "checkpoints/model_best.pth", root, False),
            "other_checkpoints": {
                name: (directory / "checkpoints" / name).is_file()
                for name in ("model_latest.pth", "final.pth")
            },
        })
    manifests = [
        file_record(path, root)
        for path in sorted((root / "artifacts/sweeps").glob("*/manifest.json"))
        if path.parent.name.startswith(("mobilenetv2_baselines_", "cifar10_attention_stability_"))
    ]
    diagnostics = [
        file_record(path, root) for path in sorted((root / "artifacts/diagnostics").glob("csgha*.json"))
    ]
    logs = []
    for path in sorted((root / "artifacts/launcher_logs").glob("*.log")):
        if not path.name.startswith(("baselines_", "position_screening_", "csgha_", "cifar10_stability_")):
            continue
        record = file_record(path, root, False)
        with path.open(encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
        record["tail"] = "".join(lines[-40:])
        record["markers"] = [line.strip() for line in lines if (
            "Epoch 200/200" in line or "Sweep status:" in line
            or line.startswith("Manifest:") or line.startswith("/root/miniconda3/bin/python")
        )]
        logs.append(record)
    head = git_output(root, "rev-parse", "HEAD")
    revisions = {head}
    for record in manifests:
        manifest = json.loads(record["text"])
        revision = manifest.get("runtime", {}).get("git_commit")
        if revision and manifest.get("status") == "completed":
            revisions.add(revision)
    # Historical version labels are references to known commits, not inferred
    # run-time provenance. Missing per-run commits remain unknown in the audit.
    revisions.update(("5bad80c", "6dc4c57", "82625b4", "f11d0af"))
    source_code = {}
    for revision in sorted(revisions):
        canonical = git_output(root, "rev-parse", revision)
        source_code[canonical] = {}
        for path in CODE_FILES:
            result = subprocess.run(
                ["git", "-C", str(root), "show", f"{canonical}:{path}"],
                capture_output=True, check=False,
            )
            if result.returncode == 0:
                source_code[canonical][path] = {
                    "text": result.stdout.decode("utf-8"), "sha256": digest(result.stdout),
                }
    return {
        "schema_version": 1, "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "remote_root": str(root), "collection_git_head": head,
        "collection_git_status": git_output(root, "status", "--short"),
        "runs": runs, "excluded_run_directories": excluded,
        "manifests": manifests, "diagnostics": diagnostics, "logs": logs,
        "source_code": source_code,
        "backup_scope": "Metadata text + best-checkpoint SHA-256 only; checkpoint bytes and dataset not backed up.",
    }


def classify(config: dict, run_id: str) -> str:
    if config.get("model_type") == "mobilenetv2":
        return "Baseline"
    if config.get("model_type") == "hybrid":
        return { (1, 2): "Independent shallow", (7, 8): "Independent middle",
                 (15, 16): "Independent deep" }.get(tuple(config.get("cbam_positions", [])), "Other hybrid")
    if run_id.startswith("csgha_v3_"):
        return "CSGHA v3"
    if run_id.startswith("csgha_v2_"):
        return "CSGHA v2"
    return "CSGHA v1"


def stats(values: list[float]) -> dict:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Statistics require nonempty finite values")
    average = math.fsum(values) / len(values)
    deviation = math.sqrt(math.fsum((value - average) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else None
    return {
        "n": len(values), "mean_percent": average,
        "sample_sd_percent": deviation,
    }


def analyze(snapshot: dict) -> dict:
    indexed_manifests = {}
    for evidence in snapshot["manifests"]:
        if digest(evidence["text"].encode()) != evidence["sha256"]:
            raise ValueError(f"Manifest checksum mismatch: {evidence['path']}")
        manifest = json.loads(evidence["text"])
        for run in manifest["runs"]:
            if run["status"] == "completed" and run.get("return_code") == 0:
                indexed_manifests.setdefault(run["experiment_id"], []).append((evidence, manifest, run))
    rows = []
    for index, run in enumerate(snapshot["runs"], 1):
        issues = []
        warnings = []
        for name, evidence in run["files"].items():
            if not evidence["exists"]:
                issues.append(f"missing:{name}")
            elif digest(evidence["text"].encode()) != evidence["sha256"]:
                raise ValueError(f"Source checksum mismatch: {evidence['path']}")
        config = run["config"]
        summary = json.loads(run["files"]["summary.json"].get("text", "{}"))
        metrics = json.loads(run["files"]["metrics.json"].get("text", "{}"))
        benchmark = json.loads(run["files"]["benchmark.json"].get("text", "{}"))
        training = list(csv.DictReader(io.StringIO(run["files"]["logs/training.csv"].get("text", ""))))
        epochs = [int(row["epoch"]) for row in training]
        if epochs != list(range(1, config.get("epochs", 0) + 1)):
            issues.append("epoch_sequence_mismatch")
        accuracies = [float(row["val_acc"]) for row in training]
        if not accuracies or not all(math.isfinite(value) for value in accuracies):
            issues.append("missing_or_nonfinite_validation_accuracy")
        else:
            best = max(accuracies)
            first_epoch = epochs[accuracies.index(best)]
            if not math.isclose(best, summary.get("best_validation_accuracy", -1), abs_tol=1e-10):
                issues.append("summary_vs_log_accuracy_mismatch")
            if first_epoch != summary.get("best_epoch"):
                issues.append("summary_vs_log_best_epoch_mismatch")
        if summary.get("experiment_id") != run["run_id"]:
            issues.append("summary_run_id_mismatch")
        if summary.get("dataset") != config.get("dataset"):
            issues.append("dataset_mismatch")
        recorded_test_flag = summary.get("test_evaluated")
        test_evaluated = recorded_test_flag
        test_evaluated_source = "summary_flag"
        if recorded_test_flag is None and "test_accuracy" in summary:
            test_evaluated = True
            test_evaluated_source = "legacy_saved_test_metrics"
            warnings.append("legacy_test_flag_absent_inferred_from_saved_test_metrics")
        if test_evaluated != config.get("evaluate_test", True):
            issues.append("test_flag_mismatch")
        if summary.get("test_evaluated") is False and "test_accuracy" in summary:
            issues.append("test_metric_in_validation_only_run")
        if not run["best_checkpoint"]["exists"]:
            issues.append("best_checkpoint_missing")
        if not all(run.get("other_checkpoints", {}).values()):
            issues.append("final_or_latest_checkpoint_missing")
        if [summary.get(key) for key in ("train_samples", "validation_samples", "test_samples")] != [45000, 5000, 10000]:
            issues.append("split_sizes_mismatch")
        provenance = []
        saved_config = {key: value for key, value in config.items() if key != "runtime"}
        for evidence, manifest, manifest_run in indexed_manifests.get(run["run_id"], []):
            # Earlier baseline manifests predate the new guidance/evaluate_test
            # default fields; do not silently normalize them to exact equality.
            if manifest_run["summary"] != summary:
                issues.append("manifest_summary_mismatch")
            if manifest_run["resolved_config"] != saved_config:
                issues.append("manifest_config_mismatch")
            provenance.append({
                "path": evidence["path"], "sha256": evidence["sha256"],
                "git_commit": manifest["runtime"].get("git_commit"),
                "config_equal": manifest_run["resolved_config"] == saved_config,
                "runtime": manifest["runtime"],
            })
        if not provenance:
            warnings.append("run_time_git_commit_not_recorded")
        rows.append({
            "id": f"E{index:02}", "run_id": run["run_id"], "dataset": config.get("dataset"),
            "variant": classify(config, run["run_id"]), "seed": config.get("seed"),
            "epochs_logged": len(training), "best_epoch": summary.get("best_epoch"),
            "best_validation_percent": summary.get("best_validation_accuracy", 0) * 100,
            "final_validation_percent": accuracies[-1] * 100 if accuracies else None,
            "final_train_percent": float(training[-1]["train_acc"]) * 100 if training else None,
            "test_evaluated": test_evaluated, "test_evaluated_source": test_evaluated_source,
            "test_accuracy_percent": summary["test_accuracy"] * 100 if "test_accuracy" in summary else None,
            "parameters_total": metrics.get("parameters_total"), "benchmark": benchmark,
            "config": config, "summary_source": run["files"]["summary.json"]["path"],
            "best_checkpoint": run["best_checkpoint"], "provenance": provenance,
            "issues": issues, "warnings": warnings,
        })
    groups = []
    for dataset, variant in sorted({(row["dataset"], row["variant"]) for row in rows}):
        group = sorted([row for row in rows if (row["dataset"], row["variant"]) == (dataset, variant)], key=lambda r: r["seed"])
        if len({row["seed"] for row in group}) != len(group):
            raise ValueError(f"Duplicate seed in {dataset}/{variant}")
        test_values = [row["test_accuracy_percent"] for row in group if row["test_accuracy_percent"] is not None]
        groups.append({
            "dataset": dataset, "variant": variant, "run_ids": [row["id"] for row in group],
            "seeds": [row["seed"] for row in group],
            "validation": stats([row["best_validation_percent"] for row in group]),
            "test": stats(test_values) if len(test_values) == len(group) else None,
        })
    paired = []
    comparisons = [("CSGHA v3", control) for control in ("Baseline", "Independent shallow", "Independent middle")]
    comparisons.append(("Independent shallow", "Baseline"))
    for treatment, control in comparisons:
        candidate = {row["seed"]: row for row in rows if row["variant"] == treatment and row["dataset"] == "cifar10"}
        reference = {row["seed"]: row for row in rows if row["variant"] == control and row["dataset"] == "cifar10"}
        if set(candidate) != set(reference) or not candidate:
            raise ValueError(f"Incomplete paired seed set for {control}")
        seeds = sorted(candidate)
        deltas = [candidate[seed]["best_validation_percent"] - reference[seed]["best_validation_percent"] for seed in seeds]
        paired.append({
            "comparison": f"{treatment} - {control}", "unit": "percentage_points",
            "seeds": seeds, "deltas": deltas, "mean_delta": stats(deltas)["mean_percent"],
            "sample_sd_delta": stats(deltas)["sample_sd_percent"], "wins": sum(delta > 0 for delta in deltas),
        })
    return {"schema_version": 1, "runs": rows, "groups": groups, "paired": paired}


def write_results(snapshot_path: Path, output: Path) -> dict:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    result = analyze(snapshot)
    for revision, files in snapshot["source_code"].items():
        for name, evidence in files.items():
            if digest(evidence["text"].encode()) != evidence["sha256"]:
                raise ValueError(f"Code checksum mismatch: {revision}:{name}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    records = []
    for run in snapshot["runs"]:
        records.extend(run["files"].values())
        records.append(run["best_checkpoint"])
    records.extend(snapshot["manifests"] + snapshot["diagnostics"] + snapshot["logs"])
    sources = {record["path"]: {k: v for k, v in record.items() if k not in ("text", "tail", "markers")} for record in records}
    (output / "source_index.json").write_text(json.dumps({
        "snapshot_sha256": digest(snapshot_path.read_bytes()), "collected_at_utc": snapshot["collected_at_utc"],
        "collection_git_head_not_run_provenance": snapshot["collection_git_head"],
        "remote_root": snapshot["remote_root"],
        "collection_git_status": snapshot["collection_git_status"],
        "excluded_run_directories": snapshot["excluded_run_directories"],
        "historical_source_references_not_inferred_run_provenance": {
            revision: {name: evidence["sha256"] for name, evidence in files.items()}
            for revision, files in snapshot["source_code"].items()
        },
        "backup_scope": snapshot["backup_scope"], "sources": sources,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for record in records:
        if "text" not in record:
            continue
        if digest(record["text"].encode()) != record["sha256"]:
            raise ValueError(f"Source checksum mismatch: {record['path']}")
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "artifacts":
            raise ValueError(f"Unsafe artifact path: {relative}")
        target = snapshot_path.parent / "source_files" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(record["text"].encode())
    lines = [
        "# 正式实验审计汇总", "", "本表从服务器原始 summary、逐 epoch 日志及 manifest 复算；百分比仅展示四舍五入值。", "",
        f"- 本次采集快照：`{snapshot_path}`（本地、Git 忽略）。",
        "- 原文副本：同目录 `source_files/`；权重未下载，只有服务器文件 SHA-256。",
        "- `source_index.json` 为源文件校验索引；`audit_results.json` 包含未舍入统计和逐 run 核查。", "",
        "## 分组结果", "", "均值 ± 样本标准差（ddof=1）；n=1 不计算标准差。所有比较只用 validation。旧 baseline 的 test 标志字段缺失，但已保存 test 指标；单独标记其推断来源，不将字段缺失当作未评估。", "",
        "| 数据集 | 变体 | Seeds | n | Best validation (%) | 已有 test (%) |",
        "|---|---|---|---:|---:|---:|",
    ]
    def formatted(values):
        if values is None:
            return "未评估"
        mean = values["mean_percent"]
        sd = values["sample_sd_percent"]
        return f"{mean:.2f} ± {sd:.2f}" if sd is not None else f"{mean:.2f}（单次）"
    for group in result["groups"]:
        lines.append(f"| {group['dataset']} | {group['variant']} | {','.join(map(str, group['seeds']))} | {group['validation']['n']} | {formatted(group['validation'])} | {formatted(group['test'])} |")
    pair_seeds = result["paired"][0]["seeds"]
    if any(pair["seeds"] != pair_seeds for pair in result["paired"]):
        raise ValueError("Comparison seed sets differ")
    lines += ["", "## 同 seed 配对差值", "", "单位：百分点；胜出次数不能代替显著性检验。", "",
              "| 比较 | " + " | ".join(f"Seed {seed}" for seed in pair_seeds) + " | 平均差值 | 胜出 |",
              "|---|" + "---:|" * (len(pair_seeds) + 2)]
    for pair in result["paired"]:
        lines.append(f"| {pair['comparison']} | " + " | ".join(f"{value:+.2f}" for value in pair["deltas"]) + f" | {pair['mean_delta']:+.3f} | {pair['wins']}/{len(pair['seeds'])} |")
    lines += ["", "## 逐次实验台账", "", "| ID | 数据集 / 变体 | Seed | Epochs | Best val (%) | Best epoch | Test (%) | Params | 核查 |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in result["runs"]:
        test = "—" if row["test_accuracy_percent"] is None else f"{row['test_accuracy_percent']:.2f}"
        issues = ", ".join(row["issues"]) or ("通过；有溯源提示" if row["warnings"] else "通过")
        lines.append(f"| [{row['id']}](#{row['id'].lower()}) | {row['dataset']} / {row['variant']} | {row['seed']} | {row['epochs_logged']} | {row['best_validation_percent']:.2f} | {row['best_epoch']} | {test} | {row['parameters_total']} | {issues} |")
    lines += ["", "## 可追溯来源", "", "当前采集时 HEAD 不等于每次运行时 HEAD。无正式批次 commit 记录的运行标为未记录，不能靠时间戳补写。"]
    for row in result["runs"]:
        lines += ["", f"### {row['id']}", "", f"- Run ID：`{row['run_id']}`", f"- Summary：`{row['summary_source']}`", f"- Best checkpoint SHA-256：`{row['best_checkpoint'].get('sha256', 'missing')}`"]
        if row["provenance"]:
            for entry in row["provenance"]:
                lines.append(f"- 运行时 commit（manifest 记录）：`{entry['git_commit']}`；来源 `{entry['path']}`；配置完全一致：{entry['config_equal']}。")
        else:
            lines.append("- 运行时 commit：未记录；checkpoint/配置/日志文件可追溯，但精确代码版本不可由产物单独确证。")
        if row["warnings"]:
            lines.append(f"- 提示：`{', '.join(row['warnings'])}`。")
    (output / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-local", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--ssh-host")
    parser.add_argument("--remote-root", default="/root/autodl-tmp/image-classification")
    parser.add_argument("--remote-python", default="/root/miniconda3/bin/python")
    parser.add_argument("--snapshot", type=Path, default=Path("artifacts/audits/2026-08-30/snapshot.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/audits/2026-08-30"))
    args = parser.parse_args()
    if args.collect_local:
        print(json.dumps(collect(args.collect_local.resolve()), ensure_ascii=False))
        return 0
    root = Path(__file__).resolve().parents[2]
    # Reuse the central path definitions without importing the package's YAML
    # configuration dependency; offline report generation needs stdlib only.
    paths = runpy.run_path(str(root / "src/image_classification/paths.py"))
    snapshot_path = (paths["PROJECT_ROOT"] / args.snapshot).resolve()
    output = (paths["PROJECT_ROOT"] / args.output).resolve()
    if args.ssh_host:
        if snapshot_path.exists():
            raise FileExistsError(f"Use a new snapshot path instead of replacing {snapshot_path}")
        command = [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.ssh_host,
            shlex.join([args.remote_python, "-", "--collect-local", args.remote_root]),
        ]
        completed = subprocess.run(command, input=Path(__file__).read_text(encoding="utf-8"), capture_output=True, text=True, check=True)
        snapshot = json.loads(completed.stdout)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result = write_results(snapshot_path, output)
    print(json.dumps({"runs": len(result["runs"]), "groups": result["groups"], "paired": result["paired"], "issues": {row['id']: row['issues'] for row in result['runs'] if row['issues']}, "output": str(output)}, indent=2, ensure_ascii=False))
    return int(any(row["issues"] for row in result["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
