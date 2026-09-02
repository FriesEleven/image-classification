"""Calibration-only utilities for sequential early-exit policies."""

from collections.abc import Sequence

import numpy as np


def stratified_calibration_mask(
    labels: np.ndarray, calibration_fraction: float = 0.5, seed: int = 0,
) -> np.ndarray:
    """Split a labeled development set without changing class proportions."""

    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) < 2:
        raise ValueError("labels must be a nontrivial one-dimensional array")
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between 0 and 1")
    generator = np.random.default_rng(seed)
    mask = np.zeros(len(labels), dtype=bool)
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        if len(indices) < 2:
            raise ValueError("each class needs at least two samples")
        generator.shuffle(indices)
        count = min(len(indices) - 1, max(1, round(len(indices) * calibration_fraction)))
        mask[indices[:count]] = True
    return mask


def softmax_confidence(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError("logits must have shape [samples, classes]")
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.max(axis=1)


def apply_policy(
    exit_logits: Sequence[np.ndarray],
    final_logits: np.ndarray,
    thresholds: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ordered confidence thresholds and return predictions/path indices."""

    if len(exit_logits) != len(thresholds):
        raise ValueError("one threshold is required per early exit")
    final_logits = np.asarray(final_logits)
    if final_logits.ndim != 2:
        raise ValueError("final_logits must have shape [samples, classes]")
    predictions = final_logits.argmax(axis=1)
    paths = np.full(len(final_logits), len(exit_logits), dtype=np.int64)
    unresolved = np.ones(len(final_logits), dtype=bool)
    for path, (logits, threshold) in enumerate(zip(exit_logits, thresholds)):
        logits = np.asarray(logits)
        if logits.shape != final_logits.shape:
            raise ValueError("all exit and final logits must have the same shape")
        selected = unresolved & (softmax_confidence(logits) >= threshold)
        predictions[selected] = logits[selected].argmax(axis=1)
        paths[selected] = path
        unresolved[selected] = False
    return predictions, paths


def policy_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    reference_predictions: np.ndarray,
    paths: np.ndarray,
    path_costs: Sequence[float],
) -> dict:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    reference_predictions = np.asarray(reference_predictions)
    paths = np.asarray(paths)
    if not (labels.shape == predictions.shape == reference_predictions.shape == paths.shape):
        raise ValueError("policy metric arrays must have identical shapes")
    if not len(path_costs) or paths.min() < 0 or paths.max() >= len(path_costs):
        raise ValueError("path_costs must cover every exit and the final path")

    classes = np.unique(labels)
    accuracy = float(np.mean(predictions == labels))
    reference_accuracy = float(np.mean(reference_predictions == labels))
    class_accuracies = []
    reference_class_accuracies = []
    for class_id in classes:
        selected = labels == class_id
        class_accuracies.append(float(np.mean(predictions[selected] == labels[selected])))
        reference_class_accuracies.append(
            float(np.mean(reference_predictions[selected] == labels[selected]))
        )
    class_accuracies = np.asarray(class_accuracies)
    reference_class_accuracies = np.asarray(reference_class_accuracies)
    route_fractions = [float(np.mean(paths == path)) for path in range(len(path_costs))]
    expected_cost = float(np.mean(np.asarray(path_costs)[paths]))
    return {
        "accuracy": accuracy,
        "reference_accuracy": reference_accuracy,
        "accuracy_drop": reference_accuracy - accuracy,
        "balanced_accuracy": float(class_accuracies.mean()),
        "reference_balanced_accuracy": float(reference_class_accuracies.mean()),
        "balanced_accuracy_drop": float(
            reference_class_accuracies.mean() - class_accuracies.mean()
        ),
        "worst_class_accuracy_drop": float(
            np.max(reference_class_accuracies - class_accuracies)
        ),
        "route_fractions": route_fractions,
        "expected_cost_fraction": expected_cost,
        "cost_saving_fraction": 1.0 - expected_cost,
    }


def _threshold_grid(logits: np.ndarray, points: int) -> list[float]:
    confidence = softmax_confidence(logits)
    quantiles = np.linspace(0, 1, points)
    values = {float(value) for value in np.quantile(confidence, quantiles)}
    values.add(0.0)
    values.add(float(np.nextafter(confidence.max(), np.inf)))
    return sorted(values)


def select_policy(
    labels: np.ndarray,
    exit_logits: Sequence[np.ndarray],
    final_logits: np.ndarray,
    path_costs: Sequence[float],
    max_accuracy_drop: float = 0.005,
    max_balanced_accuracy_drop: float = 0.005,
    max_worst_class_drop: float = 0.02,
    grid_points: int = 41,
) -> dict:
    """Maximize MAC saving subject to frozen calibration-set risk constraints."""

    if len(exit_logits) != 2:
        raise ValueError("P0 policy selection expects exactly two ordered exits")
    if len(path_costs) != 3 or path_costs[-1] != 1.0:
        raise ValueError("P0 path costs must contain two exits followed by final cost 1.0")
    reference_predictions = np.asarray(final_logits).argmax(axis=1)
    grids = [_threshold_grid(logits, grid_points) for logits in exit_logits]
    best = None
    for first_threshold in grids[0]:
        for second_threshold in grids[1]:
            thresholds = [first_threshold, second_threshold]
            predictions, paths = apply_policy(exit_logits, final_logits, thresholds)
            metrics = policy_metrics(
                labels, predictions, reference_predictions, paths, path_costs,
            )
            feasible = (
                metrics["accuracy_drop"] <= max_accuracy_drop
                and metrics["balanced_accuracy_drop"] <= max_balanced_accuracy_drop
                and metrics["worst_class_accuracy_drop"] <= max_worst_class_drop
            )
            if not feasible:
                continue
            score = (
                metrics["cost_saving_fraction"],
                metrics["accuracy"],
                -metrics["worst_class_accuracy_drop"],
                first_threshold,
                second_threshold,
            )
            if best is None or score > best[0]:
                best = (score, thresholds, metrics)
    if best is None:
        raise RuntimeError("The final-only policy should always satisfy nonnegative drop constraints")
    return {
        "thresholds": best[1],
        "calibration_metrics": best[2],
        "constraints": {
            "max_accuracy_drop": max_accuracy_drop,
            "max_balanced_accuracy_drop": max_balanced_accuracy_drop,
            "max_worst_class_drop": max_worst_class_drop,
        },
        "grid_points": grid_points,
    }


def select_shared_single_exit_policy(
    datasets: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    path_costs: Sequence[float],
    thresholds: Sequence[float],
    max_accuracy_drop: float = 0.0,
    max_balanced_accuracy_drop: float = 0.0,
    max_worst_class_drop: float = 0.0,
    min_early_fraction: float = 0.15,
    max_early_fraction: float = 0.95,
) -> dict | None:
    """Select one threshold that satisfies empirical risk constraints on every dataset.

    Each dataset is ``(labels, exit_logits, final_logits)``. The same confidence
    threshold is applied to every dataset/model, which makes the selection
    deliberately robust to the individual training seed rather than tuning one
    threshold per checkpoint.
    """

    if not datasets:
        raise ValueError("at least one calibration dataset is required")
    if len(path_costs) != 2 or not 0 <= path_costs[0] < path_costs[1] or path_costs[1] != 1.0:
        raise ValueError("path_costs must contain one early path followed by final cost 1.0")
    if not 0 <= min_early_fraction <= max_early_fraction <= 1:
        raise ValueError("early route fraction bounds must satisfy 0 <= min <= max <= 1")
    candidates = sorted({float(value) for value in thresholds})
    if not candidates or not all(np.isfinite(candidates)):
        raise ValueError("thresholds must contain finite values")

    prepared = []
    for labels, exit_logits, final_logits in datasets:
        labels = np.asarray(labels)
        exit_logits = np.asarray(exit_logits)
        final_logits = np.asarray(final_logits)
        if labels.ndim != 1 or exit_logits.shape != final_logits.shape:
            raise ValueError("labels must be one-dimensional and logits must have identical shapes")
        if exit_logits.ndim != 2 or len(labels) != len(exit_logits):
            raise ValueError("logits must have shape [samples, classes] matching labels")
        prepared.append(
            {
                "labels": labels,
                "confidence": softmax_confidence(exit_logits),
                "exit_predictions": exit_logits.argmax(axis=1),
                "final_predictions": final_logits.argmax(axis=1),
            }
        )

    best = None
    tolerance = 1e-12
    for threshold in candidates:
        metrics_by_dataset = []
        for values in prepared:
            early = values["confidence"] >= threshold
            predictions = np.where(
                early, values["exit_predictions"], values["final_predictions"],
            )
            paths = np.where(early, 0, 1)
            metrics = policy_metrics(
                values["labels"],
                predictions,
                values["final_predictions"],
                paths,
                path_costs,
            )
            early_fraction = metrics["route_fractions"][0]
            feasible = (
                metrics["accuracy_drop"] <= max_accuracy_drop + tolerance
                and metrics["balanced_accuracy_drop"] <= max_balanced_accuracy_drop + tolerance
                and metrics["worst_class_accuracy_drop"] <= max_worst_class_drop + tolerance
                and min_early_fraction <= early_fraction <= max_early_fraction
            )
            if not feasible:
                break
            metrics_by_dataset.append(metrics)
        if len(metrics_by_dataset) != len(prepared):
            continue
        savings = [metrics["cost_saving_fraction"] for metrics in metrics_by_dataset]
        score = (
            min(savings),
            float(np.mean(savings)),
            -max(metrics["worst_class_accuracy_drop"] for metrics in metrics_by_dataset),
            -max(metrics["accuracy_drop"] for metrics in metrics_by_dataset),
            threshold,
        )
        if best is None or score > best[0]:
            best = (score, threshold, metrics_by_dataset)

    if best is None:
        return None
    return {
        "confidence_threshold": best[1],
        "calibration_metrics": best[2],
        "constraints": {
            "max_accuracy_drop": max_accuracy_drop,
            "max_balanced_accuracy_drop": max_balanced_accuracy_drop,
            "max_worst_class_accuracy_drop": max_worst_class_drop,
            "min_early_fraction": min_early_fraction,
            "max_early_fraction": max_early_fraction,
        },
        "candidate_threshold_count": len(candidates),
        "objective": "maximize minimum per-dataset cost saving, then mean saving",
    }
