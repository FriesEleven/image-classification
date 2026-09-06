"""Apply the frozen P6 source gate and lock ResNet-18 thresholds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from image_classification.selection.early_exit import softmax_confidence
from scripts.analysis.analyze_early_exit_p0 import _collect_logits

PROTOCOL = ROOT / "reports/experiments/2026-09-06-early-exit-p6-source-analysis-design/protocol_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def configure_numerics() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def route_metrics(record: dict, early: np.ndarray) -> dict:
    labels = np.asarray(record["labels"], dtype=np.int64)
    final_predictions = record["final_logits"].argmax(1)
    exit_predictions = record["exit_logits"].argmax(1)
    predictions = np.where(early, exit_predictions, final_predictions)
    classes = record["num_classes"]
    support = np.bincount(labels, minlength=classes)
    final_correct = final_predictions == labels
    policy_correct = predictions == labels
    final_per_class = np.bincount(labels, weights=final_correct, minlength=classes) / support
    policy_per_class = np.bincount(labels, weights=policy_correct, minlength=classes) / support
    early_fraction = float(np.mean(early))
    expected_cost = (
        early_fraction * record["exit_cost_fraction"]
        + (1 - early_fraction) * record["fallback_cost_fraction"]
    )
    return {
        "accuracy": float(policy_correct.mean()),
        "reference_accuracy": float(final_correct.mean()),
        "accuracy_drop": float(final_correct.mean() - policy_correct.mean()),
        "balanced_accuracy": float(policy_per_class.mean()),
        "reference_balanced_accuracy": float(final_per_class.mean()),
        "balanced_accuracy_drop": float((final_per_class - policy_per_class).mean()),
        "worst_class_accuracy_drop": float((final_per_class - policy_per_class).max()),
        "early_fraction": early_fraction,
        "expected_cost_fraction": expected_cost,
        "mac_saving_fraction": 1 - expected_cost,
        "premature_count": int(np.sum(early & ~policy_correct & final_correct)),
        "rescue_count": int(np.sum(early & policy_correct & ~final_correct)),
        "samples": len(labels),
    }


def feasible(metrics: dict, budget: dict, selection: dict) -> bool:
    return (
        metrics["accuracy_drop"] <= budget["overall_drop"] + 1e-12
        and metrics["balanced_accuracy_drop"] <= budget["balanced_drop"] + 1e-12
        and metrics["worst_class_accuracy_drop"] <= budget["worst_class_drop"] + 1e-12
        and metrics["early_fraction"] >= selection["minimum_early_fraction_each_seed"] - 1e-12
        and metrics["early_fraction"] <= selection["maximum_early_fraction_each_seed"] + 1e-12
        and metrics["mac_saving_fraction"] >= selection["minimum_mac_saving_each_seed"] - 1e-12
    )


def select_shared_threshold(records: list[dict], budget: dict, selection: dict) -> tuple[dict | None, list[dict]]:
    thresholds = np.linspace(0.0, 1.0, 1001)
    scores = [softmax_confidence(record["exit_logits"]) for record in records]
    best = None
    frontier = []
    for threshold in thresholds:
        per_seed = [route_metrics(record, score >= threshold) for record, score in zip(records, scores)]
        is_feasible = all(feasible(metrics, budget, selection) for metrics in per_seed)
        savings = [metrics["mac_saving_fraction"] for metrics in per_seed]
        frontier.append({
            "threshold": float(threshold),
            "feasible": is_feasible,
            "minimum_saving": min(savings),
            "mean_saving": float(np.mean(savings)),
            "maximum_accuracy_drop": max(metrics["accuracy_drop"] for metrics in per_seed),
            "maximum_balanced_accuracy_drop": max(metrics["balanced_accuracy_drop"] for metrics in per_seed),
            "maximum_worst_class_accuracy_drop": max(metrics["worst_class_accuracy_drop"] for metrics in per_seed),
            "minimum_early_fraction": min(metrics["early_fraction"] for metrics in per_seed),
            "maximum_early_fraction": max(metrics["early_fraction"] for metrics in per_seed),
        })
        if is_feasible:
            objective = (min(savings), float(np.mean(savings)), float(threshold))
            if best is None or objective > best[0]:
                best = (objective, float(threshold), per_seed)
    selected = None if best is None else {
        "threshold": best[1],
        "minimum_saving": best[0][0],
        "mean_saving": best[0][1],
        "source_metrics": best[2],
        "threshold_candidates_evaluated": len(thresholds),
    }
    return selected, frontier


def validate_inputs() -> tuple[dict, dict[tuple[str, str, int], dict], list[dict]]:
    protocol = read_json(PROTOCOL)
    if protocol.get("status") != "frozen_before_p6_source_analysis":
        raise ValueError("P6 source analysis protocol is not frozen")
    training = protocol["training_protocol"]
    if sha256(ROOT / training["path"]) != training["sha256"]:
        raise ValueError("P6 training protocol changed")
    runs = {}
    receipts = []
    for item in protocol["source_manifests"]:
        path = ROOT / item["path"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"P6 source manifest changed: {item['path']}")
        manifest = read_json(path)
        if manifest.get("status") != "completed" or manifest.get("concurrent_jobs") != 1:
            raise ValueError("P6 source manifest is incomplete or nonserial")
        receipts.append(item)
        for run in manifest["runs"]:
            config = run["resolved_config"]
            variant = "A0" if config["model_type"] == "resnet18" else "A3"
            key = (config["dataset"], variant, int(run["seed"]))
            if key in runs or run.get("status") != "completed" or run.get("return_code") != 0:
                raise ValueError(f"Invalid P6 source run: {run['experiment_id']}")
            if config.get("evaluate_test") is not False:
                raise ValueError("P6 source attempted test evaluation")
            root = ROOT / "artifacts/runs" / run["experiment_id"]
            summary = read_json(root / "summary.json")
            if summary.get("test_evaluated") is not False or sha256(root / "checkpoints/model_best.pth") != summary["best_checkpoint_sha256"]:
                raise ValueError(f"P6 source checkpoint boundary failed: {root}")
            runs[key] = run
    expected = {(dataset, variant, seed) for dataset, item in protocol["datasets"].items()
                for variant in ("A0", "A3") for seed in item["seeds"]}
    if set(runs) != expected:
        raise ValueError("P6 source run matrix mismatch")
    return protocol, runs, receipts


def best_head_metrics(root: Path, best_epoch: int) -> dict[str, float]:
    rows = list(csv.DictReader((root / "logs/per_head_training.csv").open()))
    selected = {row["head"]: row for row in rows if row["split"] == "validation" and int(row["epoch"]) == best_epoch}
    return {f"{head}_best_epoch_validation_accuracy": float(row["accuracy"]) for head, row in selected.items()}


def execute(output: Path) -> dict:
    protocol, runs, manifest_receipts = validate_inputs()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    logits_dir = output / "calibration_logits"
    logits_dir.mkdir()
    configure_numerics()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or torch.cuda.get_device_name(0) != protocol["numerics"]["device"]:
        raise RuntimeError("P6 source analysis requires the frozen RTX 3080 Ti")
    seed_rows = []
    decisions = {}
    checkpoint_receipts = {}
    logits_receipts = {}
    frontier_rows = []
    for dataset, design in protocol["datasets"].items():
        records = []
        final_gains = []
        dataset_rows = []
        final_macs = design["final_path_macs"]
        for seed in design["seeds"]:
            a0 = runs[(dataset, "A0", seed)]
            a3 = runs[(dataset, "A3", seed)]
            a0_root = ROOT / "artifacts/runs" / a0["experiment_id"]
            a3_root = ROOT / "artifacts/runs" / a3["experiment_id"]
            a0_summary = read_json(a0_root / "summary.json")
            a3_summary = read_json(a3_root / "summary.json")
            gain = a3_summary["best_validation_accuracy"] - a0_summary["best_validation_accuracy"]
            final_gains.append(gain)
            labels, values = _collect_logits(a3, device, split="calibration")
            if len(values) != 3:
                raise ValueError("P6 A3 must return final, exit2, and exit6 logits")
            final_logits, exit_logits, auxiliary_logits = values
            path = logits_dir / f"{dataset}_a3_seed{seed}.npz"
            np.savez_compressed(path, labels=labels, final_logits=final_logits,
                                exit2_logits=exit_logits, exit6_logits=auxiliary_logits)
            record = {
                "seed": seed, "labels": labels, "final_logits": final_logits,
                "exit_logits": exit_logits, "num_classes": 10 if dataset == "cifar10" else 100,
                "exit_cost_fraction": design["exit_path_macs"] / final_macs,
                "fallback_cost_fraction": (final_macs + design["exit_head_macs"]) / final_macs,
            }
            records.append(record)
            row = {
                "dataset": dataset, "seed": seed,
                "a0_experiment_id": a0["experiment_id"], "a3_experiment_id": a3["experiment_id"],
                "a0_final_validation_accuracy": a0_summary["best_validation_accuracy"],
                "a3_final_validation_accuracy": a3_summary["best_validation_accuracy"],
                "paired_final_gain": gain,
                "calibration_final_accuracy": float(np.mean(final_logits.argmax(1) == labels)),
                "calibration_exit2_accuracy": float(np.mean(exit_logits.argmax(1) == labels)),
                "calibration_exit6_accuracy": float(np.mean(auxiliary_logits.argmax(1) == labels)),
                **best_head_metrics(a3_root, a3_summary["best_epoch"]),
            }
            dataset_rows.append(row)
            checkpoint_receipts[f"{dataset}/A0/seed{seed}"] = a0_summary["best_checkpoint_sha256"]
            checkpoint_receipts[f"{dataset}/A3/seed{seed}"] = a3_summary["best_checkpoint_sha256"]
            logits_receipts[f"{dataset}/A3/seed{seed}"] = {
                "path": str(path.relative_to(ROOT)), "sha256": sha256(path), "samples": len(labels),
                "checkpoint_sha256": a3_summary["best_checkpoint_sha256"],
            }
        selected, frontier = select_shared_threshold(records, design["risk_budget"], protocol["selection"])
        for row in frontier:
            frontier_rows.append({"dataset": dataset, **row})
        final_gate = (
            float(np.mean(final_gains)) >= protocol["final_head_gate"]["paired_a3_minus_a0_mean_minimum"] - 1e-12
            and min(final_gains) >= protocol["final_head_gate"]["paired_a3_minus_a0_each_seed_minimum"] - 1e-12
        )
        policy_gate = selected is not None
        status = "ready_for_target_training" if final_gate and policy_gate else "stop_without_target"
        decisions[dataset] = {
            "status": status,
            "final_head_gate_passed": final_gate,
            "policy_gate_passed": policy_gate,
            "paired_final_gain_mean": float(np.mean(final_gains)),
            "paired_final_gain_minimum": min(final_gains),
            "selected_policy": selected,
        }
        if selected:
            for row, metrics in zip(dataset_rows, selected["source_metrics"]):
                row.update({"selected_threshold": selected["threshold"],
                            **{f"policy_{key}": value for key, value in metrics.items()}})
        seed_rows.extend(dataset_rows)
    write_csv(output / "source_seed_metrics.csv", seed_rows)
    write_csv(output / "threshold_frontier.csv", frontier_rows)
    overall = "ready_for_p6_target_training" if all(
        value["status"] == "ready_for_target_training" for value in decisions.values()
    ) else "stop_or_partial_target"
    result = {
        "schema_version": 1,
        "status": overall,
        "official_test_accessed": False,
        "target_data_accessed": False,
        "protocol_sha256": sha256(PROTOCOL),
        "source_manifests": manifest_receipts,
        "checkpoint_sha256": checkpoint_receipts,
        "calibration_logits": logits_receipts,
        "decisions": decisions,
    }
    (output / "source_lock.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# P6 ResNet-18 source gate", "", f"Overall status: **{overall}**.", ""]
    for dataset, decision in decisions.items():
        lines.extend([
            f"## {dataset.upper().replace('CIFAR', 'CIFAR-')}", "",
            f"Status: **{decision['status']}**. Mean paired A3-A0 final-validation gain: "
            f"{100 * decision['paired_final_gain_mean']:.2f} pp; minimum seed gain: "
            f"{100 * decision['paired_final_gain_minimum']:.2f} pp.", "",
            ("Selected shared threshold: " + f"{decision['selected_policy']['threshold']:.3f}; "
             + f"minimum source-seed MAC saving: {100 * decision['selected_policy']['minimum_saving']:.2f}%."
             if decision["selected_policy"] else "No feasible shared threshold was found."), "",
        ])
    lines.append("Selection used only source calibration samples from seeds 71-73. No target or official-test data was accessed.\n")
    (output / "README.md").write_text("\n".join(lines))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.output.resolve()), indent=2))
