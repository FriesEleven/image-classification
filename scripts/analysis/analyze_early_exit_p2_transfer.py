"""Evaluate the frozen P1b policy on unseen P2 model versions without reselection."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from image_classification.selection.early_exit import apply_policy, policy_metrics
from scripts.analysis.analyze_early_exit_p0 import (
    _accuracy,
    _collect_logits,
    _cost_profile,
    _sample_std,
    _sha256,
)
from scripts.launch_early_exit_p2 import (
    EXPECTED_SOURCE_SELECTION_SHA256,
    LOCKED_THRESHOLD,
    SEEDS,
    SOURCE_SEEDS,
    SOURCE_SELECTION,
    SPLIT_SEED,
    validate_source_policy,
)

EXPECTED_TYPES = ("mobilenetv2", "multi_exit")
EXPECTED_SWEEP_NAME = "cifar10_early_exit_p2_transfer_serial_p2a"
EXPECTED_PROTOCOL = {
    "dataset": "cifar10",
    "validation_size": 5000,
    "calibration_size": 5000,
    "split_seed": SPLIT_SEED,
    "evaluate_test": False,
    "epochs": 200,
    "batch_size": 128,
    "lr": 0.01,
    "amp": True,
    "cuda_graph": True,
    "torch_num_threads": 1,
    "measure_inference": False,
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


def _split_fingerprint(run: dict) -> str:
    path = ROOT / "artifacts/runs" / run["experiment_id"] / "split_indices.json"
    split = json.loads(path.read_text(encoding="utf-8"))
    partitions = [
        split.get("train_indices", []),
        split.get("validation_indices", []),
        split.get("calibration_indices", []),
    ]
    if [len(values) for values in partitions] != [40_000, 5000, 5000]:
        raise ValueError(f"Unexpected P2 data split: {run['experiment_id']}")
    sets = [set(values) for values in partitions]
    if any(sets[left] & sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError(f"Overlapping P2 split: {run['experiment_id']}")
    if set.union(*sets) != set(range(50_000)):
        raise ValueError(f"Incomplete P2 split coverage: {run['experiment_id']}")
    if split.get("split_seed") != SPLIT_SEED or split.get("training_seed") != run["seed"]:
        raise ValueError(f"P2 split metadata mismatch: {run['experiment_id']}")
    canonical = json.dumps(
        {"train": partitions[0], "validation": partitions[1], "calibration": partitions[2]},
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _load_manifest(path: Path) -> tuple[dict, dict[tuple[str, int], dict], str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("sweep_name") != EXPECTED_SWEEP_NAME:
        raise ValueError("Select the unique P2a transfer manifest")
    if manifest.get("status") != "completed" or manifest.get("concurrent_jobs") != 1:
        raise ValueError("P2a manifest must be completed and serial")
    indexed = {}
    for run in manifest.get("runs", []):
        config = run.get("resolved_config", {})
        key = (config.get("model_type"), int(run.get("seed", -1)))
        if key in indexed:
            raise ValueError(f"Duplicate P2a run: {key}")
        indexed[key] = run
    expected = {(model_type, seed) for model_type in EXPECTED_TYPES for seed in SEEDS}
    if set(indexed) != expected:
        raise ValueError("Expected exactly baseline/multi-exit x target seeds 57/58/59")

    fingerprints = set()
    for key, run in indexed.items():
        config = run["resolved_config"]
        if run.get("status") != "completed" or run.get("return_code") != 0:
            raise ValueError(f"Incomplete P2a run: {key}")
        if run.get("termination_signal") is not None:
            raise ValueError(f"Unexpected P2a termination signal: {key}")
        for name, expected_value in EXPECTED_PROTOCOL.items():
            if config.get(name) != expected_value:
                raise ValueError(f"P2a protocol mismatch for {name}: {key}")
        if key[0] == "multi_exit":
            for name, expected_value in EXPECTED_MULTI_EXIT.items():
                if config.get(name) != expected_value:
                    raise ValueError(f"P2a multi-exit mismatch for {name}: {key}")
        fingerprints.add(_split_fingerprint(run))
    if len(fingerprints) != 1:
        raise ValueError("All P2a runs must use one fixed data split")
    return manifest, indexed, fingerprints.pop()


def evaluate_frozen_policy(
    labels: np.ndarray,
    exit8_logits: np.ndarray,
    final_logits: np.ndarray,
    *,
    threshold: float,
    path_costs: list[float],
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Apply one supplied threshold; this function performs no fitting or search."""

    predictions, paths = apply_policy([exit8_logits], final_logits, [threshold])
    final_predictions = final_logits.argmax(axis=1)
    metrics = policy_metrics(labels, predictions, final_predictions, paths, path_costs)
    metrics["decision_changes_vs_final"] = int(np.sum(predictions != final_predictions))
    metrics["harmed_vs_final"] = int(np.sum((final_predictions == labels) & (predictions != labels)))
    metrics["rescued_vs_final"] = int(np.sum((final_predictions != labels) & (predictions == labels)))
    return metrics, predictions, paths


def frozen_gates(gains: list[float], policies: list[dict]) -> dict[str, bool]:
    """Evaluate the preregistered P2 gates without changing the frozen policy."""

    if len(gains) != len(SEEDS) or len(policies) != len(SEEDS):
        raise ValueError("P2 gates require exactly three paired target-seed results")
    tolerance = 1e-12
    return {
        "mean_final_validation_gain_at_least_minus_0_003": float(np.mean(gains)) >= -0.003,
        "each_final_validation_gain_at_least_minus_0_0075": min(gains) >= -0.0075,
        "each_transfer_accuracy_drop_at_most_0": all(value["accuracy_drop"] <= tolerance for value in policies),
        "each_transfer_balanced_drop_at_most_0": all(
            value["balanced_accuracy_drop"] <= tolerance for value in policies
        ),
        "each_transfer_worst_class_drop_at_most_0": all(
            value["worst_class_accuracy_drop"] <= tolerance for value in policies
        ),
        "each_transfer_route_is_dynamic": all(0.15 <= value["route_fractions"][0] <= 0.95 for value in policies),
        "each_transfer_mac_saving_at_least_0_15": all(value["cost_saving_fraction"] >= 0.15 for value in policies),
    }


def _validate_audit(audit_path: Path, manifest_path: Path) -> dict:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("issues") != {}:
        raise ValueError("The P2a audit contains issues")
    if audit.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("The P2a audit does not reference this manifest")
    return audit


def analyze(manifest_path: Path, audit_path: Path) -> dict:
    source_selection = validate_source_policy()
    audit = _validate_audit(audit_path, manifest_path)
    _manifest, runs, split_fingerprint = _load_manifest(manifest_path)
    source_fingerprint = source_selection["data_protocol"]["split_fingerprint"]
    if split_fingerprint != source_fingerprint:
        raise ValueError("P2a data split differs from the P1b source-policy split")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cost_profile = _cost_profile()
    path_costs = [cost_profile["path_cost_fractions"][0], 1.0]
    seed_results = []
    checkpoint_hashes = {}
    for seed in SEEDS:
        baseline_run = runs[("mobilenetv2", seed)]
        multi_run = runs[("multi_exit", seed)]
        validation_labels, baseline_values = _collect_logits(
            baseline_run,
            device,
            split="validation",
        )
        multi_validation_labels, multi_validation_values = _collect_logits(
            multi_run,
            device,
            split="validation",
        )
        if not np.array_equal(validation_labels, multi_validation_labels):
            raise ValueError(f"Matched validation order differs for target seed {seed}")
        if len(baseline_values) != 1 or len(multi_validation_values) != 3:
            raise ValueError(f"Unexpected P2a model output count for target seed {seed}")
        final_validation, exit8_validation, exit16_validation = multi_validation_values

        transfer_labels, transfer_values = _collect_logits(
            multi_run,
            device,
            split="calibration",
        )
        if len(transfer_values) != 3:
            raise ValueError(f"Unexpected P2a transfer output count for target seed {seed}")
        transfer_final, transfer_exit8, _transfer_exit16 = transfer_values
        policy, _predictions, _paths = evaluate_frozen_policy(
            transfer_labels,
            transfer_exit8,
            transfer_final,
            threshold=LOCKED_THRESHOLD,
            path_costs=path_costs,
        )
        baseline_accuracy = _accuracy(baseline_values[0], validation_labels)
        final_accuracy = _accuracy(final_validation, validation_labels)
        seed_results.append(
            {
                "seed": seed,
                "baseline_experiment_id": baseline_run["experiment_id"],
                "multi_exit_experiment_id": multi_run["experiment_id"],
                "baseline_validation_accuracy": baseline_accuracy,
                "multi_exit_final_validation_accuracy": final_accuracy,
                "paired_final_validation_gain": final_accuracy - baseline_accuracy,
                "exit8_validation_accuracy": _accuracy(exit8_validation, validation_labels),
                "exit16_validation_accuracy": _accuracy(exit16_validation, validation_labels),
                "frozen_policy_transfer_metrics": policy,
            }
        )
        for run in (baseline_run, multi_run):
            checkpoint = ROOT / "artifacts/runs" / run["experiment_id"] / "checkpoints/model_best.pth"
            checkpoint_hashes[run["experiment_id"]] = _sha256(checkpoint)

    gains = [value["paired_final_validation_gain"] for value in seed_results]
    policies = [value["frozen_policy_transfer_metrics"] for value in seed_results]
    gates = frozen_gates(gains, policies)
    status = "ready_for_external_shift_test" if all(gates.values()) else "stop_archive_transfer_failure"
    return {
        "schema_version": 1,
        "status": status,
        "scope": (
            "P2 cross-model transfer evaluation on unseen training seeds using the P1b-frozen "
            "threshold; no threshold selection and no official CIFAR-10 test iteration"
        ),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "audit": str(audit_path),
        "audit_sha256": _sha256(audit_path),
        "runtime_commit": audit.get("runtime_commit"),
        "device": str(device),
        "source_policy": {
            "selection": str(SOURCE_SELECTION.relative_to(ROOT)),
            "selection_sha256": EXPECTED_SOURCE_SELECTION_SHA256,
            "source_training_seeds": list(SOURCE_SEEDS),
            "target_training_seeds": list(SEEDS),
            "exit_position": 8,
            "confidence": "maximum softmax probability",
            "confidence_threshold": LOCKED_THRESHOLD,
            "protected_predicted_classes": [],
            "fallback": "final head",
            "threshold_candidates_considered_on_p2": 0,
            "per_target_model_recalibration": False,
        },
        "data_protocol": {
            "train_samples": 40_000,
            "model_selection_validation_samples": 5000,
            "cross_model_transfer_samples": 5000,
            "split_seed": SPLIT_SEED,
            "split_fingerprint": split_fingerprint,
            "caveat": (
                "The transfer images are the same P1b calibration indices. They test unseen model "
                "versions, not an independent data distribution; external evidence remains required."
            ),
        },
        "cost_profile": cost_profile,
        "best_checkpoint_sha256": checkpoint_hashes,
        "seed_results": seed_results,
        "aggregate": {
            "paired_final_validation_gain_mean": float(np.mean(gains)),
            "paired_final_validation_gain_sample_std": _sample_std(gains),
            "transfer_early_fraction_mean": float(np.mean([value["route_fractions"][0] for value in policies])),
            "transfer_early_fraction_sample_std": _sample_std([value["route_fractions"][0] for value in policies]),
            "transfer_mac_saving_mean": float(np.mean([value["cost_saving_fraction"] for value in policies])),
            "transfer_mac_saving_sample_std": _sample_std([value["cost_saving_fraction"] for value in policies]),
            "transfer_accuracy_drop_max": max(value["accuracy_drop"] for value in policies),
            "transfer_balanced_drop_max": max(value["balanced_accuracy_drop"] for value in policies),
            "transfer_worst_class_drop_max": max(value["worst_class_accuracy_drop"] for value in policies),
        },
        "gates": gates,
        "recommended_next_step": (
            "Freeze and hash this transfer result, then build a one-shot CIFAR-10.1 v6 "
            "distribution-shift evaluation. Never reopen the original CIFAR-10 test."
            if status == "ready_for_external_shift_test"
            else "Archive the cross-model transfer failure; do not tune on target seeds or open external test data."
        ),
    }


def _markdown(result: dict) -> str:
    rows = []
    for value in result["seed_results"]:
        policy = value["frozen_policy_transfer_metrics"]
        rows.append(
            "| {seed} | {baseline:.2f} | {final:.2f} | {gain:+.2f} | {early:.2f} | "
            "{saving:.2f} | {drop:+.2f} | {balanced:+.2f} | {worst:+.2f} |".format(
                seed=value["seed"],
                baseline=100 * value["baseline_validation_accuracy"],
                final=100 * value["multi_exit_final_validation_accuracy"],
                gain=100 * value["paired_final_validation_gain"],
                early=100 * policy["route_fractions"][0],
                saving=100 * policy["cost_saving_fraction"],
                drop=100 * policy["accuracy_drop"],
                balanced=100 * policy["balanced_accuracy_drop"],
                worst=100 * policy["worst_class_accuracy_drop"],
            )
        )
    return "\n".join(
        [
            "# Early-exit P2 frozen-policy transfer",
            "",
            f"Decision: **{result['status']}**",
            "",
            "P1b threshold `0.984` is applied unchanged to unseen target seeds 57/58/59.",
            "No target-seed threshold candidate is evaluated or selected.",
            "",
            (
                "| seed | baseline val % | final val % | gain pp | transfer early % | "
                "MAC saving % | overall drop pp | balanced drop pp | worst-class drop pp |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## Frozen gates",
            "",
            *(f"- [{'x' if passed else ' '}] `{name}`" for name, passed in result["gates"].items()),
            "",
            result["data_protocol"]["caveat"],
            "The official CIFAR-10 test loader is not iterated by this analysis.",
            "",
            f"Next: {result['recommended_next_step']}",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    audit_path = args.audit.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite P2 transfer output: {output}")
    result = analyze(manifest_path, audit_path)
    output.mkdir(parents=True)
    (output / "transfer_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(_markdown(result), encoding="utf-8")
    print(f"Decision: {result['status']}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
