"""Explore a shared predicted-class guard after the frozen P0 global policy failed."""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from image_classification.selection.early_exit import (
    policy_metrics,
    softmax_confidence,
    stratified_calibration_mask,
)
from scripts.analysis.analyze_early_exit_p0 import (
    EXPECTED_SEEDS,
    _collect_logits,
    _cost_profile,
    _load_manifest,
    _sha256,
)

DEFAULT_MANIFEST = ROOT / (
    "artifacts/sweeps/cifar10_early_exit_p0_serial_p0a_20260902_110000/manifest.json"
)
DEFAULT_OUTPUT = ROOT / "artifacts/analyses/early_exit_p0_p0a_class_guard_20260902_v1"
CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def _apply_guard(
    exit_logits: np.ndarray,
    final_logits: np.ndarray,
    threshold: float,
    protected_classes: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    exit_predictions = exit_logits.argmax(axis=1)
    final_predictions = final_logits.argmax(axis=1)
    early = softmax_confidence(exit_logits) >= threshold
    if protected_classes:
        early &= ~np.isin(exit_predictions, protected_classes)
    predictions = np.where(early, exit_predictions, final_predictions)
    paths = np.where(early, 0, 1)
    return predictions, paths


def _metrics(
    labels: np.ndarray,
    exit_logits: np.ndarray,
    final_logits: np.ndarray,
    threshold: float,
    protected_classes: tuple[int, ...],
    path_costs: list[float],
) -> dict:
    predictions, paths = _apply_guard(
        exit_logits, final_logits, threshold, protected_classes,
    )
    return policy_metrics(
        labels,
        predictions,
        final_logits.argmax(axis=1),
        paths,
        path_costs,
    )


def _class_rows(labels: np.ndarray, predictions: np.ndarray, final_predictions: np.ndarray) -> list[dict]:
    rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        selected = labels == class_id
        exit_accuracy = float(np.mean(predictions[selected] == labels[selected]))
        final_accuracy = float(np.mean(final_predictions[selected] == labels[selected]))
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "exit_accuracy": exit_accuracy,
                "final_accuracy": final_accuracy,
                "accuracy_drop": final_accuracy - exit_accuracy,
            }
        )
    return rows


def _preaggregate_thresholds(value: dict, thresholds: list[float]) -> dict[float, dict]:
    selected = value["calibration"]
    labels = value["labels"][selected]
    exit_logits = value["exit_logits"][selected]
    final_logits = value["final_logits"][selected]
    exit_predictions = exit_logits.argmax(axis=1)
    final_predictions = final_logits.argmax(axis=1)
    confidence = softmax_confidence(exit_logits)
    correctness_difference = (
        (final_predictions == labels).astype(np.int64)
        - (exit_predictions == labels).astype(np.int64)
    )
    class_counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    summaries = {}
    for threshold in thresholds:
        eligible = confidence >= threshold
        route_counts = np.bincount(
            exit_predictions[eligible], minlength=len(CLASS_NAMES),
        )
        difference_matrix = np.zeros(
            (len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64,
        )
        np.add.at(
            difference_matrix,
            (labels[eligible], exit_predictions[eligible]),
            correctness_difference[eligible],
        )
        summaries[threshold] = {
            "samples": len(labels),
            "class_counts": class_counts,
            "route_counts_by_predicted_class": route_counts,
            "correctness_difference_by_true_and_predicted_class": difference_matrix,
        }
    return summaries


def diagnose(manifest_path: Path) -> dict:
    _manifest, runs = _load_manifest(manifest_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cost_profile = _cost_profile()
    path_costs = [cost_profile["path_cost_fractions"][0], 1.0]
    data = {}
    pooled_confidence = []
    for seed in EXPECTED_SEEDS:
        labels, values = _collect_logits(runs[("multi_exit", seed)], device)
        final_logits, exit8_logits, _exit16_logits = values
        calibration = stratified_calibration_mask(labels, seed=20_260_902 + seed)
        data[seed] = {
            "labels": labels,
            "final_logits": final_logits,
            "exit_logits": exit8_logits,
            "calibration": calibration,
        }
        pooled_confidence.append(softmax_confidence(exit8_logits[calibration]))
    pooled_confidence = np.concatenate(pooled_confidence)
    thresholds = sorted(
        {
            0.0,
            *np.quantile(pooled_confidence, np.linspace(0, 1, 41)).tolist(),
            float(np.nextafter(pooled_confidence.max(), np.inf)),
        }
    )

    aggregated = {
        seed: _preaggregate_thresholds(data[seed], thresholds) for seed in EXPECTED_SEEDS
    }
    protected_options = [
        protected_classes
        for count in range(len(CLASS_NAMES) + 1)
        for protected_classes in itertools.combinations(range(len(CLASS_NAMES)), count)
    ]
    best = None
    tolerance = 1e-12
    for threshold in thresholds:
        for protected_classes in protected_options:
            active_classes = np.ones(len(CLASS_NAMES), dtype=bool)
            active_classes[list(protected_classes)] = False
            savings = []
            feasible = True
            for seed in EXPECTED_SEEDS:
                summary = aggregated[seed][threshold]
                early_count = summary["route_counts_by_predicted_class"][active_classes].sum()
                early_fraction = float(early_count / summary["samples"])
                class_difference = summary[
                    "correctness_difference_by_true_and_predicted_class"
                ][:, active_classes].sum(axis=1)
                class_drops = class_difference / summary["class_counts"]
                accuracy_drop = float(class_difference.sum() / summary["samples"])
                if (
                    accuracy_drop > tolerance
                    or float(class_drops.mean()) > tolerance
                    or float(class_drops.max()) > tolerance
                    or early_fraction < 0.15
                    or early_fraction > 0.95
                ):
                    feasible = False
                    break
                savings.append(early_fraction * (1 - path_costs[0]))
            if not feasible:
                continue
            score = (
                min(savings),
                float(np.mean(savings)),
                -len(protected_classes),
                threshold,
            )
            if best is None or score > best[0]:
                best = (score, protected_classes, threshold)
    if best is None:
        return {
            "schema_version": 1,
            "status": "no_shared_class_guard",
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "search_contract": {
                "shared_across_seeds": True,
                "calibration_max_drop": 0.0,
                "calibration_max_worst_class_drop": 0.0,
                "route_fraction_range": [0.15, 0.95],
            },
        }

    _score, protected_classes, threshold = best
    calibration_metrics = {}
    for seed in EXPECTED_SEEDS:
        value = data[seed]
        selected = value["calibration"]
        calibration_metrics[seed] = _metrics(
            value["labels"][selected],
            value["exit_logits"][selected],
            value["final_logits"][selected],
            threshold,
            protected_classes,
            path_costs,
        )
    holdout_metrics = {}
    class_risk = {}
    for seed in EXPECTED_SEEDS:
        value = data[seed]
        selected = ~value["calibration"]
        predictions, paths = _apply_guard(
            value["exit_logits"][selected],
            value["final_logits"][selected],
            threshold,
            protected_classes,
        )
        final_predictions = value["final_logits"][selected].argmax(axis=1)
        holdout_metrics[seed] = policy_metrics(
            value["labels"][selected],
            predictions,
            final_predictions,
            paths,
            path_costs,
        )
        class_risk[seed] = _class_rows(
            value["labels"][selected], predictions, final_predictions,
        )
    holdout_gates = {
        "each_saving_at_least_0_15": all(
            holdout_metrics[seed]["cost_saving_fraction"] >= 0.15
            for seed in EXPECTED_SEEDS
        ),
        "each_accuracy_drop_at_most_0_01": all(
            holdout_metrics[seed]["accuracy_drop"] <= 0.01 for seed in EXPECTED_SEEDS
        ),
        "each_worst_class_drop_at_most_0_03": all(
            holdout_metrics[seed]["worst_class_accuracy_drop"] <= 0.03
            for seed in EXPECTED_SEEDS
        ),
        "each_route_is_dynamic": all(
            0.05 <= holdout_metrics[seed]["route_fractions"][0] <= 0.95
            for seed in EXPECTED_SEEDS
        ),
    }
    return {
        "schema_version": 1,
        "status": (
            "class_guard_feasible_exploratory" if all(holdout_gates.values())
            else "class_guard_holdout_failed"
        ),
        "scope": "Post-failure P0 diagnostic only; not independent paper evidence.",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "device": str(device),
        "search_contract": {
            "shared_across_seeds": True,
            "calibration_max_accuracy_drop": 0.0,
            "calibration_max_balanced_accuracy_drop": 0.0,
            "calibration_max_worst_class_drop": 0.0,
            "calibration_route_fraction_range": [0.15, 0.95],
            "candidate_protected_sets": 2 ** len(CLASS_NAMES),
            "threshold_grid_size": len(thresholds),
        },
        "selected_policy": {
            "exit_position": 8,
            "confidence_threshold": threshold,
            "protected_predicted_class_ids": list(protected_classes),
            "protected_predicted_class_names": [CLASS_NAMES[index] for index in protected_classes],
            "fallback": "final",
            "path_cost_fractions": path_costs,
        },
        "calibration_metrics": calibration_metrics,
        "holdout_metrics": holdout_metrics,
        "holdout_class_risk": class_risk,
        "holdout_gates": holdout_gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output}")
    result = diagnose(args.manifest.resolve())
    output.mkdir(parents=True)
    (output / "diagnostic.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
