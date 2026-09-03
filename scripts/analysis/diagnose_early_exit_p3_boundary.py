"""Diagnose the failed P3 routing gate without accessing CIFAR-100 test data."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from image_classification.selection.early_exit import softmax_confidence
from scripts.analysis.analyze_early_exit_p0 import _collect_logits, _cost_profile, _sha256
from scripts.analysis.analyze_early_exit_p3 import (
    THRESHOLDS,
    load_manifest,
)
from scripts.launch_early_exit_p3 import SEEDS, SOURCE_SEEDS, TARGET_SEEDS


def evaluate_threshold(
    labels: np.ndarray,
    exit_logits: np.ndarray,
    final_logits: np.ndarray,
    threshold: float,
    path_costs: list[float],
) -> dict:
    """Return risk, routing, and decision-change counts for one frozen threshold."""

    prepared = {
        "labels": np.asarray(labels),
        "confidence": softmax_confidence(exit_logits),
        "exit_predictions": np.asarray(exit_logits).argmax(axis=1),
        "final_predictions": np.asarray(final_logits).argmax(axis=1),
    }
    return _evaluate_prepared(prepared, threshold, path_costs)


def _evaluate_prepared(
    prepared: dict[str, np.ndarray],
    threshold: float,
    path_costs: list[float],
) -> dict:
    labels = prepared["labels"]
    exit_predictions = prepared["exit_predictions"]
    final_predictions = prepared["final_predictions"]
    early = prepared["confidence"] >= threshold
    predictions = np.where(early, exit_predictions, final_predictions)
    paths = np.where(early, 0, 1)
    changed = predictions != final_predictions
    final_correct = final_predictions == labels
    policy_correct = predictions == labels
    class_counts = np.bincount(labels)
    present = class_counts > 0
    policy_class_correct = np.bincount(labels, weights=policy_correct, minlength=len(class_counts))
    final_class_correct = np.bincount(labels, weights=final_correct, minlength=len(class_counts))
    policy_class_accuracy = policy_class_correct[present] / class_counts[present]
    final_class_accuracy = final_class_correct[present] / class_counts[present]
    class_drops = final_class_accuracy - policy_class_accuracy
    worst = float(class_drops.max())
    route_fractions = [float(np.mean(paths == index)) for index in range(len(path_costs))]
    expected_cost = float(np.mean(np.asarray(path_costs)[paths]))
    return {
        "accuracy": float(np.mean(policy_correct)),
        "reference_accuracy": float(np.mean(final_correct)),
        "accuracy_drop": float(np.mean(final_correct) - np.mean(policy_correct)),
        "balanced_accuracy": float(policy_class_accuracy.mean()),
        "reference_balanced_accuracy": float(final_class_accuracy.mean()),
        "balanced_accuracy_drop": float(final_class_accuracy.mean() - policy_class_accuracy.mean()),
        "worst_class_accuracy_drop": worst,
        "route_fractions": route_fractions,
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


def _cohort_record(
    metric_table: dict[int, dict[float, dict]],
    seeds: tuple[int, ...],
    threshold: float,
) -> dict:
    metrics = {str(seed): metric_table[seed][threshold] for seed in seeds}
    return {
        "threshold": threshold,
        "metrics": metrics,
        "maximum_accuracy_drop": max(value["accuracy_drop"] for value in metrics.values()),
        "maximum_balanced_accuracy_drop": max(value["balanced_accuracy_drop"] for value in metrics.values()),
        "maximum_worst_class_accuracy_drop": max(value["worst_class_accuracy_drop"] for value in metrics.values()),
        "minimum_early_fraction": min(value["route_fractions"][0] for value in metrics.values()),
        "minimum_mac_saving": min(value["cost_saving_fraction"] for value in metrics.values()),
    }


def _selected_record(
    selected: dict | None,
    metric_table: dict[int, dict[float, dict]],
) -> dict:
    if selected is None:
        return {"found": False}
    threshold = selected["confidence_threshold"]
    return {
        "found": True,
        "threshold": threshold,
        "source": _cohort_record(metric_table, SOURCE_SEEDS, threshold),
        "target": _cohort_record(metric_table, TARGET_SEEDS, threshold),
    }


def _select(
    metric_table: dict[int, dict[float, dict]],
    seeds: tuple[int, ...],
    *,
    max_accuracy_drop: float,
    max_balanced_accuracy_drop: float,
    max_worst_class_drop: float,
    min_early_fraction: float,
) -> dict | None:
    best = None
    tolerance = 1e-12
    for threshold in THRESHOLDS:
        metrics = [metric_table[seed][threshold] for seed in seeds]
        if not all(
            value["accuracy_drop"] <= max_accuracy_drop + tolerance
            and value["balanced_accuracy_drop"] <= max_balanced_accuracy_drop + tolerance
            and value["worst_class_accuracy_drop"] <= max_worst_class_drop + tolerance
            and min_early_fraction <= value["route_fractions"][0] <= 0.95
            for value in metrics
        ):
            continue
        savings = [value["cost_saving_fraction"] for value in metrics]
        score = (
            min(savings),
            float(np.mean(savings)),
            -max(value["worst_class_accuracy_drop"] for value in metrics),
            -max(value["accuracy_drop"] for value in metrics),
            threshold,
        )
        if best is None or score > best[0]:
            best = (score, threshold)
    return None if best is None else {"confidence_threshold": best[1]}


def diagnose(
    manifest_path: Path,
    audit_path: Path,
    selection_path: Path,
) -> dict:
    """Run a calibration-only post-failure boundary analysis."""

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if audit.get("issues") != {}:
        raise ValueError("P3 audit contains issues")
    if selection.get("status") != "stop_without_test":
        raise ValueError("P3 boundary diagnosis requires the frozen failed selection")
    if audit.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("P3 audit references a different manifest")
    if selection.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("P3 selection references a different manifest")
    if selection.get("audit_sha256") != _sha256(audit_path):
        raise ValueError("P3 selection references a different audit")

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
        if len(values) != 3:
            raise ValueError(f"Unexpected P3 output count for seed {seed}")
        final_logits, exit8_logits, _exit16_logits = values
        prepared[seed] = {
            "labels": labels,
            "confidence": softmax_confidence(exit8_logits),
            "exit_predictions": exit8_logits.argmax(axis=1),
            "final_predictions": final_logits.argmax(axis=1),
        }
    metric_table = {
        seed: {threshold: _evaluate_prepared(values, threshold, path_costs) for threshold in THRESHOLDS}
        for seed, values in prepared.items()
    }

    strict_without_route_floor = _select(
        metric_table,
        SOURCE_SEEDS,
        max_accuracy_drop=0.0,
        max_balanced_accuracy_drop=0.0,
        max_worst_class_drop=0.0,
        min_early_fraction=0.0,
    )
    per_seed_strict = {}
    for seed in SOURCE_SEEDS:
        selected = _select(
            metric_table,
            (seed,),
            max_accuracy_drop=0.0,
            max_balanced_accuracy_drop=0.0,
            max_worst_class_drop=0.0,
            min_early_fraction=0.15,
        )
        per_seed_strict[str(seed)] = _selected_record(selected, metric_table)

    route_feasible = []
    for threshold in THRESHOLDS:
        record = _cohort_record(metric_table, SOURCE_SEEDS, threshold)
        fractions = [value["route_fractions"][0] for value in record["metrics"].values()]
        if all(0.15 <= value <= 0.95 for value in fractions):
            route_feasible.append(record)
    if not route_feasible:
        raise RuntimeError("No shared threshold satisfies the P3 dynamic-routing bounds")
    least_exposure = max(route_feasible, key=lambda value: value["threshold"])
    lowest_worst_class_risk = min(
        route_feasible,
        key=lambda value: (
            value["maximum_worst_class_accuracy_drop"],
            value["maximum_accuracy_drop"],
            value["maximum_balanced_accuracy_drop"],
            -value["minimum_mac_saving"],
            -value["threshold"],
        ),
    )
    saving_feasible = [value for value in route_feasible if value["minimum_mac_saving"] >= 0.15 - 1e-12]
    if not saving_feasible:
        raise RuntimeError("No shared threshold reaches the P3 15% MAC-saving gate")
    lowest_risk_with_required_saving = min(
        saving_feasible,
        key=lambda value: (
            value["maximum_worst_class_accuracy_drop"],
            value["maximum_accuracy_drop"],
            value["maximum_balanced_accuracy_drop"],
            -value["minimum_mac_saving"],
            -value["threshold"],
        ),
    )

    exploratory_budgets = {
        "one_class_sample": {
            "max_accuracy_drop": 0.0002,
            "max_balanced_accuracy_drop": 0.0002,
            "max_worst_class_drop": 0.02,
        },
        "preregistered_test_scale": {
            "max_accuracy_drop": 0.005,
            "max_balanced_accuracy_drop": 0.005,
            "max_worst_class_drop": 0.02,
        },
        "double_worst_class_tolerance": {
            "max_accuracy_drop": 0.005,
            "max_balanced_accuracy_drop": 0.005,
            "max_worst_class_drop": 0.04,
        },
    }
    exploratory = {}
    for name, budget in exploratory_budgets.items():
        selected = _select(
            metric_table,
            SOURCE_SEEDS,
            **budget,
            min_early_fraction=0.15,
        )
        exploratory[name] = {
            "post_hoc_budget": budget,
            **_selected_record(selected, metric_table),
        }
        if exploratory[name]["found"]:
            exploratory[name]["source_mac_saving_gate_passed"] = (
                exploratory[name]["source"]["minimum_mac_saving"] >= 0.15 - 1e-12
            )
            exploratory[name]["target_mac_saving_gate_passed"] = (
                exploratory[name]["target"]["minimum_mac_saving"] >= 0.15 - 1e-12
            )

    strict_record = _selected_record(strict_without_route_floor, metric_table)
    return {
        "schema_version": 1,
        "status": "p3_boundary_diagnostic_complete",
        "scope": (
            "Post-failure calibration-only diagnostic. It cannot change the frozen P3 "
            "stop_without_test decision, authorize test access, or serve as independent confirmation."
        ),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": _sha256(manifest_path),
        "audit": str(audit_path.relative_to(ROOT)),
        "audit_sha256": _sha256(audit_path),
        "selection": str(selection_path.relative_to(ROOT)),
        "selection_sha256": _sha256(selection_path),
        "device": str(device),
        "split_fingerprint": split_fingerprint,
        "threshold_candidate_count": len(THRESHOLDS),
        "path_cost_fractions": path_costs,
        "strict_zero_risk_without_15_percent_floor": strict_record,
        "per_source_seed_strict_zero_risk_with_15_percent_floor": per_seed_strict,
        "least_exposure_shared_dynamic_candidate": {
            "source": least_exposure,
            "target_at_same_threshold": _cohort_record(
                metric_table,
                TARGET_SEEDS,
                least_exposure["threshold"],
            ),
        },
        "lowest_worst_class_risk_shared_dynamic_candidate": {
            "source": lowest_worst_class_risk,
            "target_at_same_threshold": _cohort_record(
                metric_table,
                TARGET_SEEDS,
                lowest_worst_class_risk["threshold"],
            ),
        },
        "lowest_risk_shared_candidate_with_15_percent_mac_saving": {
            "source": lowest_risk_with_required_saving,
            "target_at_same_threshold": _cohort_record(
                metric_table,
                TARGET_SEEDS,
                lowest_risk_with_required_saving["threshold"],
            ),
        },
        "exploratory_relaxations": exploratory,
        "interpretation": {
            "strict_shared_policy_routes_at_least_15_percent": bool(
                strict_record.get("found") and strict_record["source"]["minimum_early_fraction"] >= 0.15
            ),
            "all_source_seeds_individually_have_strict_dynamic_policy": all(
                value.get("found", False) for value in per_seed_strict.values()
            ),
            "official_test_must_remain_unopened": True,
            "new_confirmation_required_for_any_relaxed_policy": True,
        },
    }


def _percent(value: float) -> str:
    return f"{100 * value:.3f}%"


def markdown(result: dict) -> str:
    strict = result["strict_zero_risk_without_15_percent_floor"]
    least = result["least_exposure_shared_dynamic_candidate"]["source"]
    lowest = result["lowest_worst_class_risk_shared_dynamic_candidate"]["source"]
    saving = result["lowest_risk_shared_candidate_with_15_percent_mac_saving"]["source"]
    lines = [
        "# CIFAR-100 P3 routing boundary diagnostic",
        "",
        "The frozen P3 decision remains `stop_without_test`; this report uses calibration data only.",
        "",
        "## Boundary summary",
        "",
        (f"- Strict zero-risk policy without the 15% route floor: `{'found' if strict['found'] else 'not found'}`."),
        (
            "- All source seeds individually admit a strict dynamic policy: "
            f"`{result['interpretation']['all_source_seeds_individually_have_strict_dynamic_policy']}`."
        ),
        (
            f"- Least-exposure shared dynamic threshold `{least['threshold']}` routes at least "
            f"`{_percent(least['minimum_early_fraction'])}` and requires worst-class drop "
            f"`{_percent(least['maximum_worst_class_accuracy_drop'])}`."
        ),
        (
            f"- Lowest-worst-class-risk dynamic threshold `{lowest['threshold']}` has minimum MAC "
            f"saving `{_percent(lowest['minimum_mac_saving'])}` and maximum worst-class drop "
            f"`{_percent(lowest['maximum_worst_class_accuracy_drop'])}`."
        ),
        (
            f"- With the required 15% MAC saving enforced, the lowest-risk threshold "
            f"`{saving['threshold']}` needs maximum worst-class drop "
            f"`{_percent(saving['maximum_worst_class_accuracy_drop'])}`."
        ),
        "",
        "Any relaxed policy below is post hoc and requires a new independent confirmation; it cannot unlock P3 test.",
        "",
        "| exploratory budget | found | threshold | min source saving | saving gate | max source overall drop | max source worst-class drop |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in result["exploratory_relaxations"].items():
        if not value["found"]:
            lines.append(f"| {name} | no | — | — | — | — | — |")
            continue
        source = value["source"]
        lines.append(
            f"| {name} | yes | {value['threshold']} | {_percent(source['minimum_mac_saving'])} | "
            f"{value['source_mac_saving_gate_passed']} | "
            f"{_percent(source['maximum_accuracy_drop'])} | "
            f"{_percent(source['maximum_worst_class_accuracy_drop'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite P3 boundary diagnostic: {output}")
    result = diagnose(
        args.manifest.resolve(),
        args.audit.resolve(),
        args.selection.resolve(),
    )
    output.mkdir(parents=True)
    (output / "diagnostic.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps(result["interpretation"], indent=2))
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
