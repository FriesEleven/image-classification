"""Explore a source-fitted CIFAR-100 class guard after the frozen P3 failure."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from image_classification.selection.early_exit import softmax_confidence
from scripts.analysis.analyze_early_exit_p0 import _collect_logits, _cost_profile, _sha256
from scripts.analysis.analyze_early_exit_p3 import load_manifest
from scripts.launch_early_exit_p3 import SEEDS, SOURCE_SEEDS, TARGET_SEEDS

THRESHOLDS = tuple(index * 0.005 for index in range(201))
MINIMUM_MAC_SAVING = 0.15


def _prepare(
    labels: np.ndarray,
    exit_logits: np.ndarray,
    final_logits: np.ndarray,
) -> dict[str, np.ndarray]:
    labels = np.asarray(labels)
    exit_predictions = np.asarray(exit_logits).argmax(axis=1)
    final_predictions = np.asarray(final_logits).argmax(axis=1)
    return {
        "labels": labels,
        "confidence": softmax_confidence(exit_logits),
        "exit_predictions": exit_predictions,
        "final_predictions": final_predictions,
        "correctness_difference": (
            (final_predictions == labels).astype(np.int64) - (exit_predictions == labels).astype(np.int64)
        ),
    }


def evaluate_guarded_threshold(
    prepared: dict[str, np.ndarray],
    threshold: float,
    allowed_predicted_classes: tuple[int, ...],
    path_costs: list[float],
) -> dict:
    """Evaluate a confidence threshold plus a predicted-class allow list."""

    allowed = np.asarray(allowed_predicted_classes, dtype=np.int64)
    early = prepared["confidence"] >= threshold
    early &= np.isin(prepared["exit_predictions"], allowed)
    final_predictions = prepared["final_predictions"]
    predictions = np.where(early, prepared["exit_predictions"], final_predictions)
    labels = prepared["labels"]
    final_correct = final_predictions == labels
    policy_correct = predictions == labels
    changed = predictions != final_predictions
    class_counts = np.bincount(labels)
    present = class_counts > 0
    policy_class_correct = np.bincount(labels, weights=policy_correct, minlength=len(class_counts))
    final_class_correct = np.bincount(labels, weights=final_correct, minlength=len(class_counts))
    policy_class_accuracy = policy_class_correct[present] / class_counts[present]
    final_class_accuracy = final_class_correct[present] / class_counts[present]
    class_drops = final_class_accuracy - policy_class_accuracy
    early_fraction = float(np.mean(early))
    expected_cost = early_fraction * path_costs[0] + (1.0 - early_fraction) * path_costs[1]
    worst = float(class_drops.max())
    return {
        "accuracy": float(np.mean(policy_correct)),
        "reference_accuracy": float(np.mean(final_correct)),
        "accuracy_drop": float(np.mean(final_correct) - np.mean(policy_correct)),
        "balanced_accuracy": float(policy_class_accuracy.mean()),
        "reference_balanced_accuracy": float(final_class_accuracy.mean()),
        "balanced_accuracy_drop": float(final_class_accuracy.mean() - policy_class_accuracy.mean()),
        "worst_class_accuracy_drop": worst,
        "route_fractions": [early_fraction, 1.0 - early_fraction],
        "expected_cost_fraction": expected_cost,
        "cost_saving_fraction": 1.0 - expected_cost,
        "decision_changes": int(changed.sum()),
        "harmed": int(np.sum(changed & final_correct & ~policy_correct)),
        "rescued": int(np.sum(changed & ~final_correct & policy_correct)),
        "neutral_changes": int(np.sum(changed & (final_correct == policy_correct))),
        "worst_class_ids": [
            int(class_id) for class_id, drop in zip(np.flatnonzero(present), class_drops) if np.isclose(drop, worst)
        ],
    }


def _fixed_threshold_arrays(
    prepared: dict[int, dict[str, np.ndarray]],
    seeds: tuple[int, ...],
    threshold: float,
    num_classes: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    route_counts = []
    difference_matrices = []
    class_counts = []
    for seed in seeds:
        values = prepared[seed]
        eligible = values["confidence"] >= threshold
        predictions = values["exit_predictions"][eligible]
        labels = values["labels"][eligible]
        differences = values["correctness_difference"][eligible]
        route_counts.append(np.bincount(predictions, minlength=num_classes).astype(float))
        matrix = np.zeros((num_classes, num_classes), dtype=float)
        np.add.at(matrix, (labels, predictions), differences)
        difference_matrices.append(matrix)
        class_counts.append(np.bincount(values["labels"], minlength=num_classes).astype(float))
    return route_counts, difference_matrices, class_counts


def solve_guard(
    prepared: dict[int, dict[str, np.ndarray]],
    seeds: tuple[int, ...],
    threshold: float,
    path_costs: list[float],
    max_worst_class_drop: float = 0.0,
) -> dict | None:
    """Solve the exact fixed-threshold source class-guard problem as a MILP."""

    num_classes = max(int(values["labels"].max()) for values in prepared.values()) + 1
    samples = {seed: len(prepared[seed]["labels"]) for seed in seeds}
    required_early_fraction = MINIMUM_MAC_SAVING / (1.0 - path_costs[0])
    route_counts, difference_matrices, class_counts = _fixed_threshold_arrays(
        prepared,
        seeds,
        threshold,
        num_classes,
    )
    if any(route.sum() / samples[seed] < required_early_fraction for seed, route in zip(seeds, route_counts)):
        return None

    variable_count = num_classes + 1
    rows = []
    lower = []
    upper = []
    for seed_index, _seed in enumerate(seeds):
        for class_id in range(num_classes):
            row = np.zeros(variable_count)
            row[:num_classes] = difference_matrices[seed_index][class_id]
            rows.append(row)
            lower.append(-np.inf)
            upper.append(max_worst_class_drop * class_counts[seed_index][class_id])

        route_fraction = route_counts[seed_index] / samples[_seed]
        minimum_row = np.zeros(variable_count)
        minimum_row[:num_classes] = route_fraction
        minimum_row[-1] = -1.0
        rows.append(minimum_row)
        lower.append(0.0)
        upper.append(np.inf)

        maximum_row = np.zeros(variable_count)
        maximum_row[:num_classes] = route_fraction
        rows.append(maximum_row)
        lower.append(-np.inf)
        upper.append(0.95)

    mean_route = np.mean(
        [route / samples[seed] for seed, route in zip(seeds, route_counts)],
        axis=0,
    )
    objective = np.zeros(variable_count)
    objective[:num_classes] = -1e-5 * mean_route
    objective[-1] = -1.0
    bounds = Bounds(
        np.concatenate([np.zeros(num_classes), [required_early_fraction]]),
        np.concatenate([np.ones(num_classes), [0.95]]),
    )
    result = milp(
        objective,
        integrality=np.concatenate([np.ones(num_classes), [0]]),
        bounds=bounds,
        constraints=LinearConstraint(np.asarray(rows), np.asarray(lower), np.asarray(upper)),
        options={"presolve": True, "mip_rel_gap": 0.0, "time_limit": 10.0},
    )
    if result.status == 2:
        return None
    if not result.success or result.x is None:
        raise RuntimeError(
            f"MILP did not prove optimality or infeasibility at threshold {threshold}: "
            f"status={result.status}, message={result.message}"
        )
    allowed = tuple(int(value) for value in np.flatnonzero(result.x[:num_classes] > 0.5))
    metrics = {
        str(seed): evaluate_guarded_threshold(
            prepared[seed],
            threshold,
            allowed,
            path_costs,
        )
        for seed in seeds
    }
    tolerance = 1e-10
    if not all(
        value["worst_class_accuracy_drop"] <= max_worst_class_drop + tolerance
        and value["accuracy_drop"] <= max_worst_class_drop + tolerance
        and value["balanced_accuracy_drop"] <= max_worst_class_drop + tolerance
        and value["cost_saving_fraction"] >= MINIMUM_MAC_SAVING - tolerance
        and 0.15 <= value["route_fractions"][0] <= 0.95
        for value in metrics.values()
    ):
        raise RuntimeError("MILP class-guard solution failed exact metric verification")
    return {
        "threshold": threshold,
        "allowed_predicted_class_ids": list(allowed),
        "protected_predicted_class_ids": [value for value in range(num_classes) if value not in allowed],
        "source_metrics": metrics,
        "minimum_source_early_fraction": min(value["route_fractions"][0] for value in metrics.values()),
        "minimum_source_mac_saving": min(value["cost_saving_fraction"] for value in metrics.values()),
        "solver_minimum_early_fraction": float(result.x[-1]),
        "solver_objective": float(result.fun),
    }


def _target_record(
    candidate: dict,
    prepared: dict[int, dict[str, np.ndarray]],
    path_costs: list[float],
    max_worst_class_drop: float,
) -> dict:
    allowed = tuple(candidate["allowed_predicted_class_ids"])
    metrics = {
        str(seed): evaluate_guarded_threshold(
            prepared[seed],
            candidate["threshold"],
            allowed,
            path_costs,
        )
        for seed in TARGET_SEEDS
    }
    tolerance = 1e-12
    gates = {
        "each_accuracy_drop_at_most_budget": all(
            value["accuracy_drop"] <= max_worst_class_drop + tolerance for value in metrics.values()
        ),
        "each_balanced_drop_at_most_budget": all(
            value["balanced_accuracy_drop"] <= max_worst_class_drop + tolerance for value in metrics.values()
        ),
        "each_worst_class_drop_at_most_budget": all(
            value["worst_class_accuracy_drop"] <= max_worst_class_drop + tolerance for value in metrics.values()
        ),
        "each_mac_saving_at_least_0_15": all(
            value["cost_saving_fraction"] >= MINIMUM_MAC_SAVING - tolerance for value in metrics.values()
        ),
        "each_route_is_dynamic": all(0.15 <= value["route_fractions"][0] <= 0.95 for value in metrics.values()),
    }
    return {
        "metrics": metrics,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "minimum_target_mac_saving": min(value["cost_saving_fraction"] for value in metrics.values()),
        "maximum_target_worst_class_drop": max(value["worst_class_accuracy_drop"] for value in metrics.values()),
    }


def _best_candidate(candidates: list[dict]) -> dict | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda value: (
            value["minimum_source_mac_saving"],
            -len(value["protected_predicted_class_ids"]),
            value["threshold"],
        ),
    )


def diagnose(
    manifest_path: Path,
    audit_path: Path,
    selection_path: Path,
    boundary_path: Path,
) -> dict:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    manifest_sha256 = _sha256(manifest_path)
    if audit.get("issues") != {} or audit.get("manifest_sha256") != manifest_sha256:
        raise ValueError("P3 audit is not accepted for this manifest")
    if selection.get("status") != "stop_without_test":
        raise ValueError("P3 class-guard diagnosis requires the frozen failed selection")
    if selection.get("manifest_sha256") != manifest_sha256:
        raise ValueError("P3 selection references a different manifest")
    if boundary.get("status") != "p3_boundary_diagnostic_complete":
        raise ValueError("P3 boundary diagnosis is incomplete")
    if boundary.get("selection_sha256") != _sha256(selection_path):
        raise ValueError("P3 boundary diagnosis references a different selection")

    _manifest, runs, split_fingerprint = load_manifest(manifest_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cost_profile = _cost_profile("cifar100")
    path_costs = [cost_profile["path_cost_fractions"][0], 1.0]
    prepared = {}
    for seed in SEEDS:
        labels, values = _collect_logits(
            runs[("multi_exit", seed)],
            device,
            split="calibration",
        )
        final_logits, exit8_logits, _exit16_logits = values
        prepared[seed] = _prepare(labels, exit8_logits, final_logits)

    searches = {}
    for budget_name, max_worst_class_drop in (("strict_zero", 0.0), ("one_sample", 0.02)):
        candidates = []
        for threshold in THRESHOLDS:
            candidate = solve_guard(
                prepared,
                SOURCE_SEEDS,
                threshold,
                path_costs,
                max_worst_class_drop=max_worst_class_drop,
            )
            if candidate is not None:
                candidates.append(candidate)
        best = _best_candidate(candidates)
        target = None if best is None else _target_record(best, prepared, path_costs, max_worst_class_drop)
        searches[budget_name] = {
            "max_worst_class_drop": max_worst_class_drop,
            "threshold_candidates": len(THRESHOLDS),
            "source_feasible_candidate_count": len(candidates),
            "best_source_candidate": best,
            "target_transfer": target,
        }

    strict = searches["strict_zero"]
    if strict["best_source_candidate"] is None:
        status = "no_source_class_guard_with_budget"
    elif strict["target_transfer"]["all_gates_passed"]:
        status = "class_guard_transfer_feasible_exploratory"
    else:
        status = "class_guard_target_transfer_failed"
    return {
        "schema_version": 1,
        "status": status,
        "scope": (
            "Post-P3-failure source-only MILP search over a coarse 0.005 threshold grid and "
            "predicted-class guards, followed by unchanged target application. Calibration only; "
            "this cannot alter P3, authorize test access, or count as independent confirmation."
        ),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": manifest_sha256,
        "audit": str(audit_path.relative_to(ROOT)),
        "audit_sha256": _sha256(audit_path),
        "selection": str(selection_path.relative_to(ROOT)),
        "selection_sha256": _sha256(selection_path),
        "boundary_diagnostic": str(boundary_path.relative_to(ROOT)),
        "boundary_diagnostic_sha256": _sha256(boundary_path),
        "device": str(device),
        "split_fingerprint": split_fingerprint,
        "path_cost_fractions": path_costs,
        "minimum_mac_saving": MINIMUM_MAC_SAVING,
        "searches": searches,
        "official_test_accessed": False,
        "recommended_next_step": (
            "Pre-register the exact exploratory policy and confirm it on newly trained model "
            "versions before any CIFAR-100 test access."
            if status == "class_guard_transfer_feasible_exploratory"
            else "Do not add a class guard; archive the CIFAR-100 routing boundary."
        ),
    }


def _percent(value: float) -> str:
    return f"{100 * value:.3f}%"


def markdown(result: dict) -> str:
    lines = [
        "# CIFAR-100 P3 predicted-class guard diagnostic",
        "",
        f"Decision: **{result['status']}**",
        "",
        "This is a post-failure calibration-only diagnostic and does not unlock CIFAR-100 test.",
        "",
        "| risk budget | source candidates | threshold | protected classes | min source saving | target passed | min target saving | max target worst-class drop |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, search in result["searches"].items():
        candidate = search["best_source_candidate"]
        target = search["target_transfer"]
        if candidate is None:
            lines.append(f"| {name} | {search['source_feasible_candidate_count']} | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {name} | {search['source_feasible_candidate_count']} | "
            f"{candidate['threshold']} | {len(candidate['protected_predicted_class_ids'])} | "
            f"{_percent(candidate['minimum_source_mac_saving'])} | "
            f"{target['all_gates_passed']} | {_percent(target['minimum_target_mac_saving'])} | "
            f"{_percent(target['maximum_target_worst_class_drop'])} |"
        )
    lines.extend(["", f"Next: {result['recommended_next_step']}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite P3 class-guard diagnostic: {output}")
    result = diagnose(
        args.manifest.resolve(),
        args.audit.resolve(),
        args.selection.resolve(),
        args.boundary.resolve(),
    )
    output.mkdir(parents=True)
    (output / "diagnostic.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
