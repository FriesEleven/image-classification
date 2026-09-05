"""Run the frozen development-only P5-A/B reconstruction and policy comparison batch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from image_classification.selection.early_exit import softmax_confidence
from scripts.analysis.analyze_early_exit_p0 import _collect_logits

PROTOCOL = ROOT / "reports/experiments/2026-09-05-early-exit-p5-design/protocol_manifest.json"
AMENDMENT = ROOT / "reports/experiments/2026-09-05-early-exit-p5-design/protocol_amendment_1.json"
COHORTS = {
    "cifar10_source": ("cifar10", (54, 55, 56), "p1"),
    "cifar10_target": ("cifar10", (57, 58, 59), "p2"),
    "cifar100_source": ("cifar100", (60, 61, 62), "p3"),
    "cifar100_target": ("cifar100", (63, 64, 65), "p3"),
    "cifar100_confirmation": ("cifar100", (66, 67, 68), "p4"),
}
REPORTS = {
    "p1": ROOT / "reports/experiments/2026-09-02-early-exit-p1b/locked_selection.json",
    "p2": ROOT / "reports/experiments/2026-09-03-early-exit-p2a-transfer/transfer_results.json",
    "p3": ROOT / "reports/experiments/2026-09-03-early-exit-p3-cifar100/selection.json",
    "p4": ROOT / "reports/experiments/2026-09-03-early-exit-p4-cifar100/confirmation.json",
}
_WORKER_COHORTS: dict[str, list[dict]] = {}


def _initialize_worker(cohorts: dict[str, list[dict]]) -> None:
    global _WORKER_COHORTS
    _WORKER_COHORTS = cohorts
    torch.set_num_threads(1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        try:
            return ROOT / path.relative_to("/root/autodl-tmp/image-classification")
        except ValueError as error:
            raise ValueError(f"Recorded path is outside the project: {path}") from error
    return ROOT / path


def validate_protocol() -> dict:
    protocol = json_load(PROTOCOL)
    if protocol.get("status") != "frozen_before_p5ab_execution":
        raise ValueError("P5 protocol is not frozen")
    for relative, expected in protocol["frozen_evidence"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Frozen evidence mismatch: {relative}")
    if json_load(REPORTS["p3"]).get("status") != "stop_without_test":
        raise ValueError("P3 stop_without_test boundary was not preserved")
    amendment = json_load(AMENDMENT)
    if amendment.get("status") != "documented_after_first_reproduction_attempt":
        raise ValueError("P5 replay amendment is missing")
    protocol["replay_amendment"] = amendment
    return protocol


def load_runs(report_path: Path, expected_seeds: tuple[int, ...]) -> dict[tuple[str, int], dict]:
    report = json_load(report_path)
    manifest_path = resolve_recorded_path(report["manifest"])
    if sha256(manifest_path) != report["manifest_sha256"]:
        raise ValueError(f"Manifest hash mismatch: {manifest_path}")
    manifest = json_load(manifest_path)
    if manifest.get("status") != "completed" or manifest.get("concurrent_jobs") != 1:
        raise ValueError(f"Expected a completed serial manifest: {manifest_path}")
    runs = {}
    for run in manifest.get("runs", []):
        model_type = run.get("resolved_config", {}).get("model_type")
        seed = int(run.get("seed", -1))
        if model_type in {"mobilenetv2", "multi_exit"} and seed in expected_seeds:
            runs[(model_type, seed)] = run
    expected = {(model_type, seed) for model_type in ("mobilenetv2", "multi_exit") for seed in expected_seeds}
    if set(runs) != expected:
        raise ValueError(f"Manifest does not contain the expected matched runs: {manifest_path}")
    return runs


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / temperature
    values -= values.max(axis=1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(axis=1, keepdims=True)


def macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    classes = int(max(labels.max(), predictions.max())) + 1
    support = np.bincount(labels, minlength=classes)
    predicted = np.bincount(predictions, minlength=classes)
    true_positive = np.bincount(labels[labels == predictions], minlength=classes)
    denominator = support + predicted
    populated = support > 0
    values = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros(classes, dtype=float),
        where=denominator > 0,
    )
    return float(values[populated].mean())


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1
        start = end
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def calibration_metrics(logits: np.ndarray, labels: np.ndarray, bins: int = 15) -> dict:
    probabilities = softmax(logits)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    reliability = []
    for index in range(bins):
        selected = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if index == 0:
            selected |= confidence == 0
        count = int(selected.sum())
        if not count:
            continue
        accuracy = float(correct[selected].mean())
        mean_confidence = float(confidence[selected].mean())
        ece += count / len(labels) * abs(accuracy - mean_confidence)
        reliability.append({"bin": index, "count": count, "accuracy": accuracy, "confidence": mean_confidence})
    chosen = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    one_hot = np.eye(probabilities.shape[1])[labels]
    return {
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(np.mean([correct[labels == value].mean() for value in np.unique(labels)])),
        "macro_f1": macro_f1(labels, predictions),
        "ece_15": float(ece),
        "nll": float(-np.log(chosen).mean()),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "reliability": reliability,
    }


def score_values(name: str, logits: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    probabilities = softmax(logits, temperature)
    if name == "msp":
        return probabilities.max(axis=1)
    if name == "entropy":
        # Higher values must always mean "safer to exit"; negative entropy has that order.
        return np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)
    if name == "margin":
        top = np.partition(probabilities, -2, axis=1)[:, -2:]
        return top[:, 1] - top[:, 0]
    raise ValueError(f"Unknown score: {name}")


def route_metrics(record: dict, early: np.ndarray) -> dict:
    labels = np.asarray(record["labels"], dtype=np.int64)
    exit_predictions = record.get("exit8_predictions")
    if exit_predictions is None:
        exit_predictions = record["exit8_logits"].argmax(axis=1)
    final_predictions = record.get("final_predictions")
    if final_predictions is None:
        final_predictions = record["final_logits"].argmax(axis=1)
    predictions = np.where(early, exit_predictions, final_predictions)
    classes = int(labels.max()) + 1
    support = np.bincount(labels, minlength=classes)
    policy_correct = predictions == labels
    final_correct = final_predictions == labels
    policy_class_correct = np.bincount(labels, weights=policy_correct, minlength=classes)
    final_class_correct = np.bincount(labels, weights=final_correct, minlength=classes)
    populated = support > 0
    policy_class_accuracy = np.divide(policy_class_correct, support, out=np.zeros(classes), where=populated)
    final_class_accuracy = np.divide(final_class_correct, support, out=np.zeros(classes), where=populated)
    early_fraction = float(np.mean(early))
    expected_cost = early_fraction * record["exit_cost"] + (1.0 - early_fraction)
    values = {
        "accuracy": float(policy_correct.mean()),
        "reference_accuracy": float(final_correct.mean()),
        "accuracy_drop": float(final_correct.mean() - policy_correct.mean()),
        "balanced_accuracy": float(policy_class_accuracy[populated].mean()),
        "reference_balanced_accuracy": float(final_class_accuracy[populated].mean()),
        "balanced_accuracy_drop": float((final_class_accuracy[populated] - policy_class_accuracy[populated]).mean()),
        "worst_class_accuracy_drop": float((final_class_accuracy[populated] - policy_class_accuracy[populated]).max()),
        "route_fractions": [early_fraction, 1.0 - early_fraction],
        "expected_cost_fraction": expected_cost,
        "cost_saving_fraction": 1.0 - expected_cost,
    }
    values.update(
        {
            "macro_f1": macro_f1(labels, predictions),
            "premature_count": int(np.sum(early & (exit_predictions != labels) & (final_predictions == labels))),
            "rescue_count": int(np.sum(early & (exit_predictions == labels) & (final_predictions != labels))),
            "delayed_exit_count": int(np.sum(~early & (exit_predictions == labels))),
            "exit_disagreement_count": int(np.sum(exit_predictions != final_predictions)),
            "samples": len(labels),
        }
    )
    return values


def thresholds_for(records: list[dict], score_name: str, temperature: float = 1.0) -> np.ndarray:
    if score_name in {"msp", "temperature_msp"}:
        return np.linspace(0.0, 1.0, 1001)
    pooled = np.concatenate([score_values("entropy" if score_name == "entropy" else "margin", row["exit8_logits"]) for row in records])
    return np.unique(np.quantile(pooled, np.linspace(0, 1, 1001)))


def select_shared(
    records: list[dict],
    score_name: str,
    budgets: dict,
    *,
    temperature: float = 1.0,
    thresholds: np.ndarray | None = None,
) -> dict | None:
    candidates = thresholds_for(records, score_name, temperature) if thresholds is None else thresholds
    base_name = "msp" if score_name == "temperature_msp" else score_name
    scores = [score_values(base_name, record["exit8_logits"], temperature=temperature) for record in records]
    best = None
    for threshold in candidates:
        per_seed = []
        for record, score in zip(records, scores):
            metrics = route_metrics(record, score >= threshold)
            if (
                metrics["accuracy_drop"] > budgets["overall_drop"] + 1e-12
                or metrics["balanced_accuracy_drop"] > budgets["balanced_drop"] + 1e-12
                or metrics["worst_class_accuracy_drop"] > budgets["worst_class_drop"] + 1e-12
            ):
                break
            per_seed.append(metrics)
        if len(per_seed) != len(records):
            continue
        savings = [row["cost_saving_fraction"] for row in per_seed]
        objective = (min(savings), float(np.mean(savings)), -max(row["accuracy_drop"] for row in per_seed), threshold)
        if best is None or objective > best[0]:
            best = (objective, float(threshold), per_seed)
    if best is None:
        return None
    return {"threshold": best[1], "source_metrics": best[2], "min_saving": best[0][0], "mean_saving": best[0][1]}


def evaluate_selected(selected: dict | None, records: list[dict], score_name: str, temperature: float = 1.0) -> list[dict]:
    if selected is None:
        return []
    base_name = "msp" if score_name == "temperature_msp" else score_name
    scores = [score_values(base_name, row["exit8_logits"], temperature=temperature) for row in records]
    return [route_metrics(row, score >= selected["threshold"]) for row, score in zip(records, scores)]


def fit_temperature(records: list[dict]) -> float:
    labels = np.concatenate([row["labels"] for row in records])
    logits = np.concatenate([row["exit8_logits"] for row in records])
    best = None
    for temperature in np.linspace(0.5, 3.0, 101):
        probabilities = softmax(logits, float(temperature))
        nll = float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())
        if best is None or nll < best[0]:
            best = (nll, float(temperature))
    return best[1]


def fit_pcee(records: list[dict], bins: int = 20) -> dict:
    confidence = np.concatenate([softmax_confidence(row["exit8_logits"]) for row in records])
    correct = np.concatenate([row["exit8_logits"].argmax(axis=1) == row["labels"] for row in records])
    edges = np.unique(np.quantile(confidence, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        edges = np.array([0.0, 1.0])
    indices = np.clip(np.searchsorted(edges, confidence, side="right") - 1, 0, len(edges) - 2)
    accuracy = np.array([correct[indices == index].mean() if np.any(indices == index) else 0.0 for index in range(len(edges) - 1)])
    return {"edges": edges, "accuracy": accuracy}


def pcee_score(logits: np.ndarray, mapping: dict) -> np.ndarray:
    confidence = softmax_confidence(logits)
    indices = np.clip(np.searchsorted(mapping["edges"], confidence, side="right") - 1, 0, len(mapping["accuracy"]) - 1)
    return mapping["accuracy"][indices]


def select_from_precomputed(records: list[dict], scores: list[np.ndarray], budgets: dict) -> dict | None:
    candidates = np.unique(np.quantile(np.concatenate(scores), np.linspace(0, 1, 1001)))
    best = None
    for threshold in candidates:
        metrics = [route_metrics(record, score >= threshold) for record, score in zip(records, scores)]
        if any(
            row["accuracy_drop"] > budgets["overall_drop"] + 1e-12
            or row["balanced_accuracy_drop"] > budgets["balanced_drop"] + 1e-12
            or row["worst_class_accuracy_drop"] > budgets["worst_class_drop"] + 1e-12
            for row in metrics
        ):
            continue
        savings = [row["cost_saving_fraction"] for row in metrics]
        objective = (min(savings), float(np.mean(savings)), float(threshold))
        if best is None or objective > best[0]:
            best = (objective, float(threshold), metrics)
    return None if best is None else {"threshold": best[1], "source_metrics": best[2], "min_saving": best[0][0], "mean_saving": best[0][1]}


def summarize_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"feasible": False}
    keys = ("accuracy", "balanced_accuracy", "macro_f1", "accuracy_drop", "balanced_accuracy_drop", "worst_class_accuracy_drop", "cost_saving_fraction")
    result = {"feasible": True, "seeds": len(rows)}
    for key in keys:
        values = [float(row[key]) for row in rows]
        result[f"{key}_mean"] = float(np.mean(values))
        result[f"{key}_min"] = float(np.min(values))
        result[f"{key}_max"] = float(np.max(values))
    return result


def risk_feasible(summary: dict, budget: dict) -> bool:
    """Return whether every summarized seed satisfies the preregistered risk budget."""
    return bool(summary.get("feasible")) and (
        summary["accuracy_drop_max"] <= budget["overall_drop"] + 1e-12
        and summary["balanced_accuracy_drop_max"] <= budget["balanced_drop"] + 1e-12
        and summary["worst_class_accuracy_drop_max"] <= budget["worst_class_drop"] + 1e-12
    )


def collect_all(protocol: dict, output: Path, device: torch.device) -> tuple[dict[str, list[dict]], dict]:
    logits_dir = output / "development_logits"
    logits_dir.mkdir(parents=True)
    stage_runs = {
        "p1": load_runs(REPORTS["p1"], (54, 55, 56)),
        "p2": load_runs(REPORTS["p2"], (57, 58, 59)),
        "p3": load_runs(REPORTS["p3"], (60, 61, 62, 63, 64, 65)),
        "p4": load_runs(REPORTS["p4"], (66, 67, 68)),
    }
    cohorts = defaultdict(list)
    files = {}
    for cohort, (dataset, seeds, stage) in COHORTS.items():
        exit_cost = protocol["path_costs"][f"{dataset}_exit8"]
        for seed in seeds:
            baseline_run = stage_runs[stage][("mobilenetv2", seed)]
            multi_run = stage_runs[stage][("multi_exit", seed)]
            baseline_labels, baseline_values = _collect_logits(baseline_run, device, split="calibration")
            labels, multi_values = _collect_logits(multi_run, device, split="calibration")
            if not np.array_equal(labels, baseline_labels) or len(baseline_values) != 1 or len(multi_values) != 3:
                raise ValueError(f"Mismatched development inference for seed {seed}")
            split_path = ROOT / "artifacts/runs" / multi_run["experiment_id"] / "split_indices.json"
            split = json_load(split_path)
            sample_ids = np.asarray(split["calibration_indices"], dtype=np.int64)
            if len(sample_ids) != len(labels):
                raise ValueError(f"Calibration index count mismatch for seed {seed}")
            final_logits, exit8_logits, exit16_logits = multi_values
            path = logits_dir / f"{cohort}_seed{seed}.npz"
            np.savez_compressed(
                path,
                sample_ids=sample_ids,
                labels=labels,
                baseline_logits=baseline_values[0],
                final_logits=final_logits,
                exit8_logits=exit8_logits,
                exit16_logits=exit16_logits,
            )
            record = {
                "cohort": cohort,
                "dataset": dataset,
                "seed": seed,
                "labels": labels,
                "baseline_logits": baseline_values[0],
                "final_logits": final_logits,
                "exit8_logits": exit8_logits,
                "exit16_logits": exit16_logits,
                "final_predictions": final_logits.argmax(axis=1),
                "exit8_predictions": exit8_logits.argmax(axis=1),
                "sample_ids": sample_ids,
                "exit_cost": exit_cost,
                "baseline_experiment_id": baseline_run["experiment_id"],
                "multi_exit_experiment_id": multi_run["experiment_id"],
            }
            cohorts[cohort].append(record)
            files[f"{cohort}/seed{seed}"] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "samples": len(labels),
                "split_indices_sha256": sha256(split_path),
                "baseline_checkpoint_sha256": sha256(ROOT / "artifacts/runs" / baseline_run["experiment_id"] / "checkpoints/model_best.pth"),
                "multi_exit_checkpoint_sha256": sha256(ROOT / "artifacts/runs" / multi_run["experiment_id"] / "checkpoints/model_best.pth"),
            }
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL),
        "device": str(device),
        "model_or_test_inference_performed": True,
        "inference_scope": "calibration split only; no official/external test loader iterated",
        "files": files,
    }
    (output / "development_logits_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return dict(cohorts), manifest


def reuse_logits(protocol: dict, source: Path, output: Path, device: torch.device) -> tuple[dict[str, list[dict]], dict]:
    source_manifest_path = source / "development_logits_manifest.json"
    source_manifest = json_load(source_manifest_path)
    if source_manifest.get("protocol_sha256") != sha256(PROTOCOL):
        raise ValueError("Reused logits were produced under a different original P5 protocol")
    stage_runs = {
        "p1": load_runs(REPORTS["p1"], (54, 55, 56)),
        "p2": load_runs(REPORTS["p2"], (57, 58, 59)),
        "p3": load_runs(REPORTS["p3"], (60, 61, 62, 63, 64, 65)),
        "p4": load_runs(REPORTS["p4"], (66, 67, 68)),
    }
    cohorts = defaultdict(list)
    verified_files = {}
    for cohort, (dataset, seeds, stage) in COHORTS.items():
        for seed in seeds:
            key = f"{cohort}/seed{seed}"
            evidence = source_manifest["files"][key]
            path = resolve_recorded_path(evidence["path"])
            if sha256(path) != evidence["sha256"]:
                raise ValueError(f"Reused logits hash mismatch: {path}")
            values = np.load(path)
            arrays = {name: values[name] for name in values.files}
            if len(arrays["labels"]) != evidence["samples"]:
                raise ValueError(f"Reused logits sample count mismatch: {path}")
            baseline_run = stage_runs[stage][("mobilenetv2", seed)]
            multi_run = stage_runs[stage][("multi_exit", seed)]
            cohorts[cohort].append(
                {
                    "cohort": cohort,
                    "dataset": dataset,
                    "seed": seed,
                    **arrays,
                    "final_predictions": arrays["final_logits"].argmax(axis=1),
                    "exit8_predictions": arrays["exit8_logits"].argmax(axis=1),
                    "exit_cost": protocol["path_costs"][f"{dataset}_exit8"],
                    "baseline_experiment_id": baseline_run["experiment_id"],
                    "multi_exit_experiment_id": multi_run["experiment_id"],
                }
            )
            verified_files[key] = evidence
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL),
        "protocol_amendment": str(AMENDMENT.relative_to(ROOT)),
        "protocol_amendment_sha256": sha256(AMENDMENT),
        "device": str(device),
        "model_inference_performed": False,
        "reused_from": str(source.relative_to(ROOT)),
        "reused_manifest_sha256": sha256(source_manifest_path),
        "files": verified_files,
    }
    (output / "development_logits_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return dict(cohorts), manifest


def reproduction_row_passes(cohort: str, differences: dict, amendment: dict) -> bool:
    if cohort != "cifar10_source":
        return max(differences.values()) <= amendment["unchanged_requirements"]["all_non_p1_cohorts_absolute_difference_max"]
    unchanged = amendment["unchanged_requirements"]
    replay = amendment["p1_cross_hardware_replay_disclosure"]
    return (
        differences["accuracy_drop"] <= unchanged["accuracy_drop_absolute_difference_max"]
        and differences["balanced_accuracy_drop"] <= unchanged["balanced_accuracy_drop_absolute_difference_max"]
        and differences["worst_class_accuracy_drop"] <= unchanged["worst_class_accuracy_drop_absolute_difference_max"]
        and differences["accuracy"] <= replay["absolute_accuracy_difference_max"]
        and differences["cost_saving_fraction"] <= replay["cost_saving_fraction_difference_max"]
    )


def frozen_point_checks(cohorts: dict[str, list[dict]], amendment: dict) -> dict:
    checks = {}
    references = {
        "cifar10_source": (0.984, REPORTS["p1"], "locked_policy.calibration_metrics"),
        "cifar10_target": (0.984, REPORTS["p2"], "seed_results.frozen_policy_transfer_metrics"),
        "cifar100_confirmation": (0.903, REPORTS["p4"], "seed_results.policy_confirmation_metrics"),
    }
    for cohort, (threshold, report_path, metric_path) in references.items():
        report = json_load(report_path)
        if metric_path.startswith("locked_policy"):
            expected_by_seed = report["locked_policy"]["calibration_metrics"]
        else:
            field = metric_path.split(".")[-1]
            expected_by_seed = {str(row["seed"]): row[field] for row in report["seed_results"]}
        rows = []
        for record in cohorts[cohort]:
            actual = route_metrics(record, softmax_confidence(record["exit8_logits"]) >= threshold)
            expected = expected_by_seed[str(record["seed"])]
            differences = {key: abs(float(actual[key]) - float(expected[key])) for key in ("accuracy", "accuracy_drop", "balanced_accuracy_drop", "worst_class_accuracy_drop", "cost_saving_fraction")}
            rows.append({"seed": record["seed"], "maximum_absolute_difference": max(differences.values()), "differences": differences})
        exact = all(row["maximum_absolute_difference"] <= 1e-12 for row in rows)
        checks[cohort] = {
            "threshold": threshold,
            "rows": rows,
            "exact": exact,
            "passed": all(reproduction_row_passes(cohort, row["differences"], amendment) for row in rows),
            "status": "exact" if exact else "accepted_documented_cross_hardware_replay_drift",
        }
    p3 = json_load(ROOT / "reports/diagnostics/2026-09-03-early-exit-p3-boundary-v2/diagnostic.json")
    p3_candidate = p3["lowest_risk_shared_candidate_with_15_percent_mac_saving"]
    for cohort, side in (("cifar100_source", "source"), ("cifar100_target", "target_at_same_threshold")):
        expected_by_seed = p3_candidate[side]["metrics"]
        rows = []
        for record in cohorts[cohort]:
            actual = route_metrics(record, softmax_confidence(record["exit8_logits"]) >= 0.903)
            expected = expected_by_seed[str(record["seed"])]
            differences = {key: abs(float(actual[key]) - float(expected[key])) for key in ("accuracy", "accuracy_drop", "balanced_accuracy_drop", "worst_class_accuracy_drop", "cost_saving_fraction")}
            rows.append({"seed": record["seed"], "maximum_absolute_difference": max(differences.values()), "differences": differences})
        checks[cohort] = {"threshold": 0.903, "rows": rows, "exact": True, "passed": all(row["maximum_absolute_difference"] <= 1e-12 for row in rows), "status": "exact"}
    checks["p3_stop_without_test_preserved"] = json_load(REPORTS["p3"])["status"] == "stop_without_test"
    checks["all_passed"] = all(value["passed"] for value in checks.values() if isinstance(value, dict) and "passed" in value) and checks["p3_stop_without_test_preserved"]
    checks["all_exact"] = all(value.get("exact", False) for value in checks.values() if isinstance(value, dict) and "passed" in value)
    return checks


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def latex_from_csv(csv_path: Path, tex_path: Path) -> None:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    columns = "l" + "r" * (len(rows[0]) - 1)
    body = [" & ".join(value.replace("_", "\\_") for value in row) + r" \\" for row in rows]
    tex_path.write_text("\\begin{tabular}{" + columns + "}\n\\toprule\n" + body[0] + "\n\\midrule\n" + "\n".join(body[1:]) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def diagnostics(cohorts: dict[str, list[dict]], tables: Path) -> dict:
    per_class = []
    complementarity = []
    calibration = []
    for cohort, records in cohorts.items():
        threshold = 0.984 if records[0]["dataset"] == "cifar10" else 0.903
        for record in records:
            labels = record["labels"]
            final_predictions = record["final_logits"].argmax(axis=1)
            exit_predictions = record["exit8_logits"].argmax(axis=1)
            early = softmax_confidence(record["exit8_logits"]) >= threshold
            policy_predictions = np.where(early, exit_predictions, final_predictions)
            values = route_metrics(record, early)
            complementarity.append({"cohort": cohort, "seed": record["seed"], **{key: values[key] for key in ("samples", "premature_count", "rescue_count", "delayed_exit_count", "exit_disagreement_count")}})
            calibration.append({"cohort": cohort, "seed": record["seed"], "head": "exit8", **{key: value for key, value in calibration_metrics(record["exit8_logits"], labels).items() if key != "reliability"}, "failure_ranking_auroc": binary_auc(exit_predictions != labels, -score_values("msp", record["exit8_logits"]))})
            calibration.append({"cohort": cohort, "seed": record["seed"], "head": "final", **{key: value for key, value in calibration_metrics(record["final_logits"], labels).items() if key != "reliability"}, "failure_ranking_auroc": binary_auc(final_predictions != labels, -score_values("msp", record["final_logits"]))})
            for class_id in np.unique(labels):
                selected = labels == class_id
                support = int(selected.sum())
                per_class.append({
                    "cohort": cohort,
                    "seed": record["seed"],
                    "class_id": int(class_id),
                    "support": support,
                    "final_accuracy": float(np.mean(final_predictions[selected] == labels[selected])),
                    "exit8_accuracy": float(np.mean(exit_predictions[selected] == labels[selected])),
                    "policy_accuracy": float(np.mean(policy_predictions[selected] == labels[selected])),
                    "policy_drop_vs_final": float(np.mean(final_predictions[selected] == labels[selected]) - np.mean(policy_predictions[selected] == labels[selected])),
                    "early_fraction": float(np.mean(early[selected])),
                    "empirical_premature_risk": float(np.mean(early[selected] & (exit_predictions[selected] != labels[selected]) & (final_predictions[selected] == labels[selected]))),
                })
    for name, rows in (("per_class_route_risk", per_class), ("decision_complementarity", complementarity), ("calibration_metrics", calibration)):
        csv_path = tables / f"{name}.csv"
        write_csv(csv_path, rows)
        latex_from_csv(csv_path, tables / f"{name}.tex")
    return {"per_class_rows": len(per_class), "decision_rows": len(complementarity), "calibration_rows": len(calibration)}


def _shared_strategy_task(payload: tuple) -> tuple[str, str, dict]:
    design, source_key, target_key, method, score_name, budgets, scale = payload
    source = _WORKER_COHORTS[source_key]
    target = _WORKER_COHORTS[target_key]
    selected = select_shared(source, score_name, budgets, temperature=scale)
    target_metrics = evaluate_selected(selected, target, score_name, temperature=scale)
    return design, method, {
        "selection": selected,
        "source": summarize_metrics([] if selected is None else selected["source_metrics"]),
        "target": summarize_metrics(target_metrics),
        "temperature": scale if score_name == "temperature_msp" else None,
    }


def strategy_comparison(cohorts: dict[str, list[dict]], protocol: dict, tables: Path, pool) -> tuple[dict, list[dict]]:
    comparisons = {design: {} for design in ("cifar10", "cifar100_strict", "cifar100_relaxed")}
    flat_rows = []
    designs = (
        ("cifar10", "cifar10_source", "cifar10_target", protocol["risk_budgets"]["cifar10"]),
        ("cifar100_strict", "cifar100_source", "cifar100_target", protocol["risk_budgets"]["cifar100_strict"]),
        ("cifar100_relaxed", "cifar100_source", "cifar100_confirmation", protocol["risk_budgets"]["cifar100_relaxed_boundary"]),
    )
    contexts = {}
    shared_tasks = []
    for design, source_key, target_key, full_budget in designs:
        source = cohorts[source_key]
        variants = {
            "shared_msp_overall": ("msp", {**full_budget, "balanced_drop": 1.0, "worst_class_drop": 1.0}, 1.0),
            "shared_msp_overall_balanced": ("msp", {**full_budget, "worst_class_drop": 1.0}, 1.0),
            "shared_msp_full": ("msp", full_budget, 1.0),
            "shared_entropy_full": ("entropy", full_budget, 1.0),
            "shared_logit_margin_full": ("margin", full_budget, 1.0),
        }
        temperature = fit_temperature(source)
        variants["temperature_scaled_msp_full"] = ("temperature_msp", full_budget, temperature)
        contexts[design] = (source_key, target_key, full_budget)
        shared_tasks.extend((design, source_key, target_key, method, score_name, budgets, scale) for method, (score_name, budgets, scale) in variants.items())
    for design, method, values in pool.map(_shared_strategy_task, shared_tasks):
        comparisons[design][method] = values

    for design, (source_key, target_key, full_budget) in contexts.items():
        source = cohorts[source_key]
        target = cohorts[target_key]
        result = comparisons[design]
        mapping = fit_pcee(source)
        source_scores = [pcee_score(row["exit8_logits"], mapping) for row in source]
        selected = select_from_precomputed(source, source_scores, full_budget)
        target_metrics = [] if selected is None else [route_metrics(row, pcee_score(row["exit8_logits"], mapping) >= selected["threshold"]) for row in target]
        result["pcee_binned_accuracy_adapter"] = {"selection": selected, "source": summarize_metrics([] if selected is None else selected["source_metrics"]), "target": summarize_metrics(target_metrics), "bins": len(mapping["accuracy"])}

        proposed = result["shared_msp_full"]["selection"]
        coverage = 0.0 if proposed is None else float(np.mean([row["route_fractions"][0] for row in proposed["source_metrics"]]))
        random_source = []
        random_target = []
        for repeats, records, destination in ((200, source, random_source), (200, target, random_target)):
            for record in records:
                metrics = []
                for repeat in range(repeats):
                    generator = np.random.default_rng(20_260_905 + record["seed"] * 1000 + repeat)
                    metrics.append(route_metrics(record, generator.random(len(record["labels"])) < coverage))
                averaged = {key: float(np.mean([row[key] for row in metrics])) for key in ("accuracy", "balanced_accuracy", "macro_f1", "accuracy_drop", "balanced_accuracy_drop", "worst_class_accuracy_drop", "cost_saving_fraction")}
                destination.append(averaged)
        result["random_routing_matched_coverage"] = {"selection": {"coverage": coverage}, "source": summarize_metrics(random_source), "target": summarize_metrics(random_target)}

        per_model = []
        for record in source:
            chosen = select_shared([record], "msp", full_budget)
            if chosen is not None:
                per_model.extend(chosen["source_metrics"])
        result["per_model_msp"] = {"selection": {"threshold_count": len(source)}, "source": summarize_metrics(per_model), "target": {"feasible": False, "reason": "target recalibration forbidden"}}

        pooled_losses = []
        ucb_choice = None
        for threshold in np.linspace(1.0, 0.0, 1001):
            losses = []
            metrics = []
            for record in source:
                early = softmax_confidence(record["exit8_logits"]) >= threshold
                metrics.append(route_metrics(record, early))
                labels = record["labels"]
                losses.append((record["exit8_logits"].argmax(axis=1) != labels).astype(float) * early - (record["final_logits"].argmax(axis=1) != labels).astype(float) * early)
            pooled = np.concatenate(losses)
            upper = float(pooled.mean() + math.sqrt(math.log(20.0) / (2 * len(pooled))))
            pooled_losses.append((float(threshold), upper))
            if upper <= full_budget["overall_drop"] + 1e-12:
                ucb_choice = {"threshold": float(threshold), "upper_bound": upper, "source_metrics": metrics}
        ucb_target = [] if ucb_choice is None else evaluate_selected(ucb_choice, target, "msp")
        result["ucb_hoeffding_empirical_risk_adapter"] = {"selection": ucb_choice, "source": summarize_metrics([] if ucb_choice is None else ucb_choice["source_metrics"]), "target": summarize_metrics(ucb_target), "delta": 0.05}

        for method, values in result.items():
            values["source_risk_feasible"] = risk_feasible(values["source"], full_budget)
            values["target_risk_feasible"] = risk_feasible(values["target"], full_budget)
            flat_rows.append({"design": design, "method": method, "threshold": None if values.get("selection") is None else values["selection"].get("threshold"), "source_risk_feasible": values["source_risk_feasible"], "target_risk_feasible": values["target_risk_feasible"], **{f"source_{key}": value for key, value in values["source"].items()}, **{f"target_{key}": value for key, value in values["target"].items()}})
    write_csv(tables / "strategy_comparison.csv", flat_rows)
    latex_from_csv(tables / "strategy_comparison.csv", tables / "strategy_comparison.tex")
    return comparisons, flat_rows


def _frontier_task(payload: tuple) -> tuple[list[dict], list[dict]]:
    design, source_key, target_key, budget, method, score_name = payload
    source = _WORKER_COHORTS[source_key]
    target = _WORKER_COHORTS[target_key]
    candidates = thresholds_for(source, score_name)
    source_scores = [score_values(score_name, row["exit8_logits"]) for row in source]
    target_scores = [score_values(score_name, row["exit8_logits"]) for row in target]
    method_frontier = []
    matched_rows = []
    for threshold in candidates:
        source_metrics = [route_metrics(row, score >= threshold) for row, score in zip(source, source_scores)]
        source_summary = summarize_metrics(source_metrics)
        method_frontier.append({
            "design": design,
            "method": method,
            "threshold": float(threshold),
            "source_accuracy_drop_mean": source_summary["accuracy_drop_mean"],
            "source_accuracy_drop_max": source_summary["accuracy_drop_max"],
            "source_balanced_drop_max": source_summary["balanced_accuracy_drop_max"],
            "source_worst_class_drop_max": source_summary["worst_class_accuracy_drop_max"],
            "source_min_saving": source_summary["cost_saving_fraction_min"],
            "source_mean_saving": source_summary["cost_saving_fraction_mean"],
        })
    for target_saving in (0.10, 0.20, 0.30):
        eligible = [row for row in method_frontier if row["source_min_saving"] >= target_saving - 1e-12]
        if not eligible:
            matched_rows.append({"design": design, "method": method, "target_saving": target_saving, "status": "infeasible"})
            continue
        chosen = min(eligible, key=lambda row: (row["source_min_saving"] - target_saving, abs(row["source_accuracy_drop_mean"])))
        target_metrics = [route_metrics(record, score >= chosen["threshold"]) for record, score in zip(target, target_scores)]
        summary = summarize_metrics(target_metrics)
        source_risk_feasible = (
            chosen["source_accuracy_drop_max"] <= budget["overall_drop"] + 1e-12
            and chosen["source_balanced_drop_max"] <= budget["balanced_drop"] + 1e-12
            and chosen["source_worst_class_drop_max"] <= budget["worst_class_drop"] + 1e-12
        )
        target_risk_feasible = risk_feasible(summary, budget)
        status = "feasible" if source_risk_feasible and target_risk_feasible else "risk_violation"
        matched_rows.append({"design": design, "method": method, "target_saving": target_saving, "status": status, "source_risk_feasible": source_risk_feasible, "target_risk_feasible": target_risk_feasible, "threshold": chosen["threshold"], **{f"target_{key}": value for key, value in summary.items()}})
    return method_frontier, matched_rows


def frontier_and_matched_compute(cohorts: dict[str, list[dict]], protocol: dict, tables: Path, pool) -> tuple[list[dict], list[dict]]:
    designs = (
        ("cifar10", "cifar10_source", "cifar10_target", protocol["risk_budgets"]["cifar10"]),
        ("cifar100_strict", "cifar100_source", "cifar100_target", protocol["risk_budgets"]["cifar100_strict"]),
        ("cifar100_relaxed", "cifar100_source", "cifar100_confirmation", protocol["risk_budgets"]["cifar100_relaxed_boundary"]),
    )
    methods = (("shared_msp", "msp"), ("shared_entropy", "entropy"), ("shared_logit_margin", "margin"))
    tasks = [(*design, method, score_name) for design in designs for method, score_name in methods]
    results = pool.map(_frontier_task, tasks)
    frontier_rows = list(itertools.chain.from_iterable(result[0] for result in results))
    matched_rows = list(itertools.chain.from_iterable(result[1] for result in results))
    write_csv(tables / "development_frontier.csv", frontier_rows)
    write_csv(tables / "matched_compute.csv", matched_rows)
    latex_from_csv(tables / "matched_compute.csv", tables / "matched_compute.tex")
    return frontier_rows, matched_rows


def _sensitivity_task(payload: tuple[str, str, dict, int, int]) -> dict:
    design, cohort, budget, size, repeat = payload
    records = _WORKER_COHORTS[cohort]
    classes = np.unique(records[0]["labels"])
    sampled = []
    for record in records:
        generator = np.random.default_rng(20_260_905 + repeat * 100 + record["seed"])
        indices = []
        quotient, remainder = divmod(size, len(classes))
        for offset, class_id in enumerate(classes):
            choices = np.flatnonzero(record["labels"] == class_id)
            take = quotient + int(offset < remainder)
            indices.extend(generator.choice(choices, size=take, replace=take > len(choices)).tolist())
        indices = np.asarray(indices)
        sampled.append({key: value[indices] if isinstance(value, np.ndarray) and len(value) == len(record["labels"]) else value for key, value in record.items()})
    selected = select_shared(sampled, "msp", budget, thresholds=np.linspace(0, 1, 101))
    return {"analysis": "calibration_size", "design": design, "size": size, "replicate": repeat, "threshold": None if selected is None else selected["threshold"], "minimum_saving": None if selected is None else selected["min_saving"], "feasible": selected is not None}


def sensitivity(cohorts: dict[str, list[dict]], protocol: dict, tables: Path, pool) -> list[dict]:
    rows = []
    designs = (
        ("cifar10", "cifar10_source", protocol["risk_budgets"]["cifar10"]),
        ("cifar100_relaxed", "cifar100_source", protocol["risk_budgets"]["cifar100_relaxed_boundary"]),
    )
    for design, cohort, budget in designs:
        records = cohorts[cohort]
        for count in (1, 2, 3):
            for subset in itertools.combinations(records, count):
                selected = select_shared(list(subset), "msp", budget)
                rows.append({"analysis": "source_seed_subset", "design": design, "size": count, "replicate": "-".join(str(row["seed"]) for row in subset), "threshold": None if selected is None else selected["threshold"], "minimum_saving": None if selected is None else selected["min_saving"], "feasible": selected is not None})
    tasks = [
        (design, cohort, budget, size, repeat)
        for design, cohort, budget in designs
        for size in protocol["calibration_resampling"]["sizes_per_seed"]
        for repeat in range(protocol["calibration_resampling"]["replicates"])
    ]
    rows.extend(pool.map(_sensitivity_task, tasks, chunksize=2))
    write_csv(tables / "threshold_sensitivity.csv", rows)
    latex_from_csv(tables / "threshold_sensitivity.csv", tables / "threshold_sensitivity.tex")
    return rows


def bootstrap_locked_points(cohorts: dict[str, list[dict]], protocol: dict, tables: Path) -> list[dict]:
    rows = []
    for cohort in ("cifar10_source", "cifar10_target", "cifar100_source", "cifar100_target", "cifar100_confirmation"):
        threshold = protocol["frozen_thresholds"][cohorts[cohort][0]["dataset"]]
        for record in cohorts[cohort]:
            generator = np.random.default_rng(20_260_905 + record["seed"])
            labels = record["labels"]
            early = softmax_confidence(record["exit8_logits"]) >= threshold
            exit_correct = record["exit8_logits"].argmax(axis=1) == labels
            final_correct = record["final_logits"].argmax(axis=1) == labels
            differences = np.where(early, exit_correct.astype(float) - final_correct.astype(float), 0.0)
            samples = np.empty(protocol["paired_bootstrap_replicates"])
            for index in range(len(samples)):
                selected = generator.integers(0, len(labels), len(labels))
                samples[index] = differences[selected].mean()
            rows.append({"cohort": cohort, "seed": record["seed"], "policy_gain_mean": float(differences.mean()), "ci95_low": float(np.quantile(samples, 0.025)), "ci95_high": float(np.quantile(samples, 0.975)), "replicates": len(samples)})
    write_csv(tables / "paired_bootstrap.csv", rows)
    latex_from_csv(tables / "paired_bootstrap.csv", tables / "paired_bootstrap.tex")
    return rows


def figures(cohorts: dict[str, list[dict]], strategy_rows: list[dict], sensitivity_rows: list[dict], output: Path) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    created = []

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, cohort in zip(axes, ("cifar10_source", "cifar100_confirmation")):
        for record in cohorts[cohort]:
            run = ROOT / "artifacts/runs" / record["multi_exit_experiment_id"] / "logs/training.csv"
            values = np.genfromtxt(run, delimiter=",", names=True)
            axis.plot(values["epoch"], 100 * values["val_acc"], label=f"seed {record['seed']}")
        axis.set(title=cohort, xlabel="epoch", ylabel="validation accuracy (%)")
        axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / "training_convergence.pdf"); plt.close(fig); created.append("training_convergence.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, cohort in zip(axes, ("cifar10_source", "cifar100_confirmation")):
        record = cohorts[cohort][0]
        labels = record["labels"]
        early = softmax_confidence(record["exit8_logits"]) >= (0.984 if record["dataset"] == "cifar10" else 0.903)
        axis.bar(np.unique(labels), [early[labels == value].mean() for value in np.unique(labels)])
        axis.set(title=f"{cohort}: per-class routing", xlabel="class", ylabel="early fraction")
    fig.tight_layout(); fig.savefig(output / "per_class_route_risk.pdf"); plt.close(fig); created.append("per_class_route_risk.pdf")

    fig, axis = plt.subplots(figsize=(7, 4))
    all_records = list(itertools.chain.from_iterable(cohorts.values()))
    labels = [row["cohort"] + f"/{row['seed']}" for row in all_records]
    premature, rescue = [], []
    for record in all_records:
        threshold = 0.984 if record["dataset"] == "cifar10" else 0.903
        values = route_metrics(record, softmax_confidence(record["exit8_logits"]) >= threshold)
        premature.append(values["premature_count"]); rescue.append(values["rescue_count"])
    x = np.arange(len(labels)); axis.bar(x - .2, premature, .4, label="premature"); axis.bar(x + .2, rescue, .4, label="rescue")
    axis.set_xticks(x, labels, rotation=80, fontsize=6); axis.legend(); axis.set_ylabel("samples")
    fig.tight_layout(); fig.savefig(output / "rescue_harm.pdf"); plt.close(fig); created.append("rescue_harm.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, cohort in zip(axes, ("cifar10_source", "cifar100_confirmation")):
        for record in cohorts[cohort]:
            metrics = calibration_metrics(record["exit8_logits"], record["labels"])
            axis.plot([row["confidence"] for row in metrics["reliability"]], [row["accuracy"] for row in metrics["reliability"]], marker="o", label=f"seed {record['seed']}")
        axis.plot([0, 1], [0, 1], "k--", linewidth=.8); axis.set(title=cohort, xlabel="confidence", ylabel="accuracy"); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / "reliability_confidence.pdf"); plt.close(fig); created.append("reliability_confidence.pdf")

    fig, axis = plt.subplots(figsize=(7, 4))
    for design in ("cifar10", "cifar100_relaxed"):
        rows = [row for row in strategy_rows if row["design"] == design and row.get("target_feasible")]
        axis.scatter([100 * row["target_cost_saving_fraction_mean"] for row in rows], [100 * row["target_accuracy_drop_mean"] for row in rows], label=design)
        for row in rows: axis.annotate(row["method"].replace("shared_", ""), (100 * row["target_cost_saving_fraction_mean"], 100 * row["target_accuracy_drop_mean"]), fontsize=6)
    axis.axhline(0, color="k", linewidth=.8); axis.set(xlabel="target MAC saving (%)", ylabel="target accuracy drop (pp)"); axis.legend()
    fig.tight_layout(); fig.savefig(output / "policy_pareto.pdf"); plt.close(fig); created.append("policy_pareto.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, design in zip(axes, ("cifar10", "cifar100_relaxed")):
        rows = [row for row in sensitivity_rows if row["analysis"] == "calibration_size" and row["design"] == design]
        sizes = sorted({row["size"] for row in rows})
        rates = [np.mean([bool(row["feasible"]) for row in rows if row["size"] == size]) for size in sizes]
        axis.plot(sizes, rates, marker="o"); axis.set(title=design, xlabel="calibration samples per seed", ylabel="feasible fraction", ylim=(-.05, 1.05))
    fig.tight_layout(); fig.savefig(output / "seed_calibration_sensitivity.pdf"); plt.close(fig); created.append("seed_calibration_sensitivity.pdf")
    return created


def _pareto_dominates(other: dict, proposed: dict) -> bool:
    """Compare all preregistered risk dimensions plus mean compute saving."""
    objectives = (
        ("cost_saving_fraction_mean", 1.0),
        ("accuracy_drop_mean", -1.0),
        ("balanced_accuracy_drop_max", -1.0),
        ("worst_class_accuracy_drop_max", -1.0),
    )
    weak = all(direction * other[key] >= direction * proposed[key] - 1e-12 for key, direction in objectives)
    strict = any(direction * other[key] > direction * proposed[key] + 1e-12 for key, direction in objectives)
    return weak and strict


def _matched_compute_dominates(rows: list[dict], design: str, competitor: str) -> bool:
    proposed_rows = {
        float(row["target_saving"]): row
        for row in rows
        if row["design"] == design and row["method"] == "shared_msp" and row["status"] == "feasible"
    }
    competitor_rows = {
        float(row["target_saving"]): row
        for row in rows
        if row["design"] == design and row["method"] == competitor and row["status"] == "feasible"
    }
    if not proposed_rows or any(level not in competitor_rows for level in proposed_rows):
        return False
    metric_keys = (
        "cost_saving_fraction_mean",
        "accuracy_drop_mean",
        "balanced_accuracy_drop_max",
        "worst_class_accuracy_drop_max",
    )
    comparisons = [
        _pareto_dominates(
            {key: competitor_rows[level][f"target_{key}"] for key in metric_keys},
            {key: proposed[f"target_{key}"] for key in metric_keys},
        )
        for level, proposed in proposed_rows.items()
    ]
    return bool(comparisons) and all(comparisons)


def gate_a(checks: dict, comparisons: dict, matched_compute_rows: list[dict], protocol: dict) -> dict:
    nondominated = True
    audit = {}
    competitor_methods = {
        "shared_msp_overall": "shared_msp",
        "shared_entropy_full": "shared_entropy",
    }
    budgets = {
        "cifar10": protocol["risk_budgets"]["cifar10"],
        "cifar100_relaxed": protocol["risk_budgets"]["cifar100_relaxed_boundary"],
    }
    for design in ("cifar10", "cifar100_relaxed"):
        proposed = comparisons[design]["shared_msp_full"]
        proposed_risk_feasible = risk_feasible(proposed["source"], budgets[design]) and risk_feasible(proposed["target"], budgets[design])
        design_audit = {"proposed_risk_feasible": proposed_risk_feasible, "competitors": {}}
        if not proposed_risk_feasible:
            nondominated = False
        for locked_method, matched_method in competitor_methods.items():
            competitor = comparisons[design][locked_method]
            competitor_risk_feasible = risk_feasible(competitor["source"], budgets[design]) and risk_feasible(competitor["target"], budgets[design])
            locked_dominates = competitor_risk_feasible and _pareto_dominates(competitor["target"], proposed["target"])
            matched_dominates = _matched_compute_dominates(matched_compute_rows, design, matched_method)
            fully_dominates = locked_dominates and matched_dominates
            design_audit["competitors"][locked_method] = {
                "risk_feasible_at_locked_point": competitor_risk_feasible,
                "dominates_matched_risk_locked_point": locked_dominates,
                "dominates_all_feasible_matched_compute_rows": matched_dominates,
                "fully_dominates_both_views": fully_dominates,
            }
            if fully_dominates:
                nondominated = False
        audit[design] = design_audit
    tradeoff = any(comparisons[design]["shared_msp_full"]["source"].get("cost_saving_fraction_mean", 0.0) > 0 for design in comparisons)
    gates = {
        "frozen_points_reproduced": checks["all_passed"],
        "proposed_not_pareto_dominated_on_primary_designs": nondominated,
        "nonempty_risk_compute_tradeoff": tradeoff,
        "p3_stop_without_test_preserved": checks["p3_stop_without_test_preserved"],
    }
    return {"gates": gates, "status": "go_p5c" if all(gates.values()) else "simplify_or_redesign_before_training", "pareto_audit": audit}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-logits", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    torch.set_num_threads(1)
    protocol = validate_protocol()
    if args.verify_only:
        print(json.dumps({"status": "ready", "protocol_sha256": sha256(PROTOCOL), "model_inference_performed": False}, indent=2))
        return 0
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite P5 output: {output}")
    output.mkdir(parents=True)
    tables = output / "tables"; figures_dir = output / "figures"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P5 1/7] Loading development logits; workers={args.workers}", flush=True)
    if args.reuse_logits is None:
        cohorts, logits_manifest = collect_all(protocol, output, device)
    else:
        cohorts, logits_manifest = reuse_logits(protocol, (ROOT / args.reuse_logits).resolve(), output, device)
    print("[P5 2/7] Checking frozen-point reproduction", flush=True)
    checks = frozen_point_checks(cohorts, protocol["replay_amendment"])
    if not checks["all_passed"]:
        (output / "reproduction_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
        raise RuntimeError("Frozen point reproduction failed; P5 stopped before policy comparison")
    print("[P5 3/7] Building diagnostic tables", flush=True)
    diagnostic_counts = diagnostics(cohorts, tables)
    context = mp.get_context("fork")
    with context.Pool(
        processes=args.workers,
        initializer=_initialize_worker,
        initargs=(cohorts,),
    ) as pool:
        print("[P5 4/7] Comparing policy strategies in parallel", flush=True)
        comparisons, strategy_rows = strategy_comparison(cohorts, protocol, tables, pool)
        print("[P5 5/7] Computing development frontiers in parallel", flush=True)
        frontier_rows, matched_compute_rows = frontier_and_matched_compute(cohorts, protocol, tables, pool)
        print("[P5 6/7] Running calibration-size sensitivity in parallel", flush=True)
        sensitivity_rows = sensitivity(cohorts, protocol, tables, pool)
    print("[P5 7/7] Bootstrap, figures, and Gate A", flush=True)
    bootstrap_rows = bootstrap_locked_points(cohorts, protocol, tables)
    figure_files = figures(cohorts, strategy_rows, sensitivity_rows, figures_dir)
    decision = gate_a(checks, comparisons, matched_compute_rows, protocol)
    result = {
        "schema_version": 2,
        "status": decision["status"],
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(PROTOCOL),
        "device": str(device),
        "cpu_workers": args.workers,
        "reproduction_checks": checks,
        "diagnostics": diagnostic_counts,
        "strategy_comparison": comparisons,
        "development_frontier_rows": len(frontier_rows),
        "matched_compute_rows": len(matched_compute_rows),
        "sensitivity_rows": len(sensitivity_rows),
        "bootstrap_rows": len(bootstrap_rows),
        "figures": figure_files,
        "gate_a": decision["gates"],
        "gate_a_pareto_audit": decision["pareto_audit"],
        "test_boundary": "No official or external evaluator was executed; only existing versioned summaries may be cited.",
        "logits_manifest_sha256": sha256(output / "development_logits_manifest.json"),
    }
    (output / "p5ab_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text("# P5-A/B development-only batch\n\nStatus: **" + result["status"] + "**\n\n" + "\n".join(f"- [{'x' if passed else ' '}] `{name}`" for name, passed in result["gate_a"].items()) + "\n\nNo official/external evaluator was executed.\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output), "gate_a": result["gate_a"], "development_files": len(logits_manifest["files"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
