"""Select on CIFAR-100 source models and test transfer to unseen model versions."""

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

from image_classification.selection.early_exit import select_shared_single_exit_policy
from scripts.analysis.analyze_early_exit_p0 import (
    _accuracy,
    _collect_logits,
    _cost_profile,
    _sample_std,
    _sha256,
)
from scripts.analysis.analyze_early_exit_p2_transfer import evaluate_frozen_policy
from scripts.launch_early_exit_p3 import SEEDS, SOURCE_SEEDS, SPLIT_SEED, TARGET_SEEDS

EXPECTED_TYPES = ("mobilenetv2", "multi_exit")
EXPECTED_SWEEP_NAME = "cifar100_early_exit_p3_serial_p3a"
THRESHOLD_STEP = 0.001
THRESHOLDS = tuple(index * THRESHOLD_STEP for index in range(1001)) + (float(np.nextafter(1.0, np.inf)),)
EXPECTED_PROTOCOL = {
    "dataset": "cifar100",
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
        raise ValueError(f"Unexpected P3 data split: {run['experiment_id']}")
    sets = [set(values) for values in partitions]
    if any(sets[left] & sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError(f"Overlapping P3 split: {run['experiment_id']}")
    if set.union(*sets) != set(range(50_000)):
        raise ValueError(f"Incomplete P3 split coverage: {run['experiment_id']}")
    if split.get("split_seed") != SPLIT_SEED or split.get("training_seed") != run["seed"]:
        raise ValueError(f"P3 split metadata mismatch: {run['experiment_id']}")
    canonical = json.dumps(
        {"train": partitions[0], "validation": partitions[1], "calibration": partitions[2]},
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def load_manifest(path: Path) -> tuple[dict, dict[tuple[str, int], dict], str]:
    """Load only the exact completed P3a matrix and verify its frozen protocol."""

    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("sweep_name") != EXPECTED_SWEEP_NAME:
        raise ValueError("Select the unique CIFAR-100 P3a manifest")
    if manifest.get("status") != "completed" or manifest.get("concurrent_jobs") != 1:
        raise ValueError("P3a manifest must be completed and serial")
    indexed = {}
    for run in manifest.get("runs", []):
        config = run.get("resolved_config", {})
        key = (config.get("model_type"), int(run.get("seed", -1)))
        if key in indexed:
            raise ValueError(f"Duplicate P3a run: {key}")
        indexed[key] = run
    expected = {(model_type, seed) for model_type in EXPECTED_TYPES for seed in SEEDS}
    if set(indexed) != expected:
        raise ValueError("Expected baseline/multi-exit x seeds 60 through 65")

    fingerprints = set()
    for key, run in indexed.items():
        config = run["resolved_config"]
        if run.get("status") != "completed" or run.get("return_code") != 0:
            raise ValueError(f"Incomplete P3a run: {key}")
        if run.get("termination_signal") is not None:
            raise ValueError(f"Unexpected P3a termination signal: {key}")
        for name, expected_value in EXPECTED_PROTOCOL.items():
            if config.get(name) != expected_value:
                raise ValueError(f"P3a protocol mismatch for {name}: {key}")
        if key[0] == "multi_exit":
            for name, expected_value in EXPECTED_MULTI_EXIT.items():
                if config.get(name) != expected_value:
                    raise ValueError(f"P3a multi-exit mismatch for {name}: {key}")
        fingerprints.add(_split_fingerprint(run))
    if len(fingerprints) != 1:
        raise ValueError("All P3a runs must use one fixed CIFAR-100 split")
    return manifest, indexed, fingerprints.pop()


def frozen_gates(
    final_gains: list[float],
    source_policies: list[dict],
    target_policies: list[dict],
) -> dict[str, bool]:
    """Apply the preregistered source-selection and no-recalibration transfer gates."""

    tolerance = 1e-12
    source_complete = len(source_policies) == len(SOURCE_SEEDS)
    target_complete = len(target_policies) == len(TARGET_SEEDS)
    return {
        "mean_final_validation_gain_at_least_minus_0_003": (
            len(final_gains) == len(SEEDS) and float(np.mean(final_gains)) >= -0.003
        ),
        "each_final_validation_gain_at_least_minus_0_0075": (
            len(final_gains) == len(SEEDS) and min(final_gains) >= -0.0075
        ),
        "shared_source_dynamic_policy_found": source_complete,
        "each_source_accuracy_drop_at_most_0": source_complete
        and all(value["accuracy_drop"] <= tolerance for value in source_policies),
        "each_source_balanced_drop_at_most_0": source_complete
        and all(value["balanced_accuracy_drop"] <= tolerance for value in source_policies),
        "each_source_worst_class_drop_at_most_0": source_complete
        and all(value["worst_class_accuracy_drop"] <= tolerance for value in source_policies),
        "each_source_route_is_dynamic": source_complete
        and all(0.15 <= value["route_fractions"][0] <= 0.95 for value in source_policies),
        "each_source_mac_saving_at_least_0_15": source_complete
        and all(value["cost_saving_fraction"] >= 0.15 for value in source_policies),
        "each_target_accuracy_drop_at_most_0": target_complete
        and all(value["accuracy_drop"] <= tolerance for value in target_policies),
        "each_target_balanced_drop_at_most_0": target_complete
        and all(value["balanced_accuracy_drop"] <= tolerance for value in target_policies),
        "each_target_worst_class_drop_at_most_0": target_complete
        and all(value["worst_class_accuracy_drop"] <= tolerance for value in target_policies),
        "each_target_route_is_dynamic": target_complete
        and all(0.15 <= value["route_fractions"][0] <= 0.95 for value in target_policies),
        "each_target_mac_saving_at_least_0_15": target_complete
        and all(value["cost_saving_fraction"] >= 0.15 for value in target_policies),
    }


def _validate_audit(audit_path: Path, manifest_path: Path) -> dict:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("issues") != {}:
        raise ValueError("The P3a audit contains issues")
    if audit.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("The P3a audit does not reference this manifest")
    return audit


def analyze(manifest_path: Path, audit_path: Path) -> dict:
    audit = _validate_audit(audit_path, manifest_path)
    _manifest, runs, split_fingerprint = load_manifest(manifest_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cost_profile = _cost_profile("cifar100")
    path_costs = [cost_profile["path_cost_fractions"][0], 1.0]
    seed_results = []
    calibration_data = {}
    checkpoint_hashes = {}
    for seed in SEEDS:
        baseline_run = runs[("mobilenetv2", seed)]
        multi_run = runs[("multi_exit", seed)]
        validation_labels, baseline_values = _collect_logits(
            baseline_run,
            device,
            split="validation",
        )
        multi_validation_labels, multi_values = _collect_logits(
            multi_run,
            device,
            split="validation",
        )
        if not np.array_equal(validation_labels, multi_validation_labels):
            raise ValueError(f"Matched validation order differs for P3 seed {seed}")
        if len(baseline_values) != 1 or len(multi_values) != 3:
            raise ValueError(f"Unexpected P3 model output count for seed {seed}")
        final_validation, exit8_validation, exit16_validation = multi_values
        calibration_labels, calibration_values = _collect_logits(
            multi_run,
            device,
            split="calibration",
        )
        if len(calibration_values) != 3:
            raise ValueError(f"Unexpected P3 calibration output count for seed {seed}")
        calibration_final, calibration_exit8, _calibration_exit16 = calibration_values
        calibration_data[seed] = (
            calibration_labels,
            calibration_exit8,
            calibration_final,
        )
        baseline_accuracy = _accuracy(baseline_values[0], validation_labels)
        final_accuracy = _accuracy(final_validation, validation_labels)
        seed_results.append(
            {
                "cohort": "source" if seed in SOURCE_SEEDS else "target",
                "seed": seed,
                "baseline_experiment_id": baseline_run["experiment_id"],
                "multi_exit_experiment_id": multi_run["experiment_id"],
                "baseline_validation_accuracy": baseline_accuracy,
                "multi_exit_final_validation_accuracy": final_accuracy,
                "paired_final_validation_gain": final_accuracy - baseline_accuracy,
                "exit8_validation_accuracy": _accuracy(exit8_validation, validation_labels),
                "exit16_validation_accuracy": _accuracy(exit16_validation, validation_labels),
            }
        )
        for run in (baseline_run, multi_run):
            checkpoint = ROOT / "artifacts/runs" / run["experiment_id"] / "checkpoints/model_best.pth"
            checkpoint_hashes[run["experiment_id"]] = _sha256(checkpoint)

    selected = select_shared_single_exit_policy(
        [calibration_data[seed] for seed in SOURCE_SEEDS],
        path_costs=path_costs,
        thresholds=THRESHOLDS,
        max_accuracy_drop=0.0,
        max_balanced_accuracy_drop=0.0,
        max_worst_class_drop=0.0,
        min_early_fraction=0.15,
        max_early_fraction=0.95,
    )
    source_policies = [] if selected is None else selected["calibration_metrics"]
    target_policies = []
    if selected is not None:
        for seed in TARGET_SEEDS:
            labels, exit8_logits, final_logits = calibration_data[seed]
            metrics, _predictions, _paths = evaluate_frozen_policy(
                labels,
                exit8_logits,
                final_logits,
                threshold=selected["confidence_threshold"],
                path_costs=path_costs,
            )
            target_policies.append(metrics)

    source_by_seed = {seed: metrics for seed, metrics in zip(SOURCE_SEEDS, source_policies)}
    target_by_seed = {seed: metrics for seed, metrics in zip(TARGET_SEEDS, target_policies)}
    for row in seed_results:
        row["policy_calibration_metrics"] = source_by_seed.get(row["seed"]) or target_by_seed.get(row["seed"])

    final_gains = [row["paired_final_validation_gain"] for row in seed_results]
    gates = frozen_gates(final_gains, source_policies, target_policies)
    status = "ready_for_locked_cifar100_test" if all(gates.values()) else "stop_without_test"
    locked_policy = None
    if selected is not None:
        locked_policy = {
            "policy_version": "shared_exit8_softmax_threshold_v1",
            "exit_position": 8,
            "confidence": "maximum softmax probability",
            "confidence_threshold": selected["confidence_threshold"],
            "protected_predicted_classes": [],
            "fallback": "final head",
            "source_training_seeds": list(SOURCE_SEEDS),
            "target_training_seeds": list(TARGET_SEEDS),
            "path_cost_fractions": path_costs,
            "source_calibration_metrics": {str(seed): metrics for seed, metrics in source_by_seed.items()},
            "target_transfer_metrics": {str(seed): metrics for seed, metrics in target_by_seed.items()},
            "selection_constraints": selected["constraints"],
            "selection_objective": selected["objective"],
            "threshold_candidates_considered_on_source": len(THRESHOLDS),
            "threshold_candidates_considered_on_target": 0,
            "per_target_model_recalibration": False,
        }
    return {
        "schema_version": 1,
        "status": status,
        "scope": (
            "CIFAR-100 P3: source-only shared-threshold selection on seeds 60/61/62, "
            "followed by no-recalibration transfer to unseen seeds 63/64/65; the official "
            "CIFAR-100 test loader is not iterated"
        ),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "audit": str(audit_path),
        "audit_sha256": _sha256(audit_path),
        "runtime_commit": audit.get("runtime_commit"),
        "device": str(device),
        "data_protocol": {
            "dataset": "cifar100",
            "train_samples": 40_000,
            "model_selection_validation_samples": 5000,
            "policy_calibration_or_transfer_samples": 5000,
            "samples_per_class_in_each_development_partition": 50,
            "split_seed": SPLIT_SEED,
            "split_fingerprint": split_fingerprint,
            "source_training_seeds": list(SOURCE_SEEDS),
            "target_training_seeds": list(TARGET_SEEDS),
        },
        "threshold_grid": {
            "frozen_before_training": True,
            "start": 0.0,
            "stop": 1.0,
            "step": THRESHOLD_STEP,
            "includes_final_only_sentinel": True,
            "candidate_count": len(THRESHOLDS),
            "target_candidate_count": 0,
        },
        "cost_profile": cost_profile,
        "best_checkpoint_sha256": checkpoint_hashes,
        "seed_results": seed_results,
        "aggregate": {
            "paired_final_validation_gain_mean": float(np.mean(final_gains)),
            "paired_final_validation_gain_sample_std": _sample_std(final_gains),
            "source_calibration_mac_saving_mean": (
                float(np.mean([value["cost_saving_fraction"] for value in source_policies]))
                if source_policies
                else None
            ),
            "target_transfer_mac_saving_mean": (
                float(np.mean([value["cost_saving_fraction"] for value in target_policies]))
                if target_policies
                else None
            ),
        },
        "locked_policy": locked_policy,
        "gates": gates,
        "historical_test_disclosure": (
            "Legacy baseline-only CIFAR-100 runs viewed official-test metrics before P3. P3 does "
            "not use those runs, and no P3 checkpoint or early-exit policy has been evaluated on "
            "the official test. Describe a later pass as method-locked, not globally blind."
        ),
        "recommended_next_step": (
            "Freeze this selection and its exact hashes, then evaluate all twelve best "
            "checkpoints and the locked policy on official CIFAR-100 test once."
            if status == "ready_for_locked_cifar100_test"
            else "Archive P3 as a second-dataset boundary result; do not inspect official CIFAR-100 test."
        ),
    }


def _markdown(result: dict) -> str:
    rows = []
    for value in result["seed_results"]:
        policy = value["policy_calibration_metrics"]
        if policy is None:
            early = saving = drop = worst = "—"
        else:
            early = f"{100 * policy['route_fractions'][0]:.2f}"
            saving = f"{100 * policy['cost_saving_fraction']:.2f}"
            drop = f"{100 * policy['accuracy_drop']:+.2f}"
            worst = f"{100 * policy['worst_class_accuracy_drop']:+.2f}"
        rows.append(
            "| {cohort} | {seed} | {baseline:.2f} | {final:.2f} | {gain:+.2f} | "
            "{early} | {saving} | {drop} | {worst} |".format(
                cohort=value["cohort"],
                seed=value["seed"],
                baseline=100 * value["baseline_validation_accuracy"],
                final=100 * value["multi_exit_final_validation_accuracy"],
                gain=100 * value["paired_final_validation_gain"],
                early=early,
                saving=saving,
                drop=drop,
                worst=worst,
            )
        )
    lines = [
        "# CIFAR-100 early-exit P3 source selection and target transfer",
        "",
        f"Decision: **{result['status']}**",
        "",
        (
            "| cohort | seed | baseline val % | final val % | gain pp | early % | "
            "MAC saving % | overall drop pp | worst-class drop pp |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Frozen gates",
        "",
        *(f"- [{'x' if passed else ' '}] `{name}`" for name, passed in result["gates"].items()),
        "",
    ]
    if result["locked_policy"] is not None:
        lines.extend(
            [
                f"Shared source-selected threshold: `{result['locked_policy']['confidence_threshold']}`.",
                "Target seeds considered zero threshold candidates and received no recalibration.",
                "",
            ]
        )
    lines.extend(
        [
            result["historical_test_disclosure"],
            "The official CIFAR-100 test loader is not iterated by this analysis.",
            "",
            f"Next: {result['recommended_next_step']}",
            "",
        ]
    )
    return "\n".join(lines)


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
        raise FileExistsError(f"Refusing to overwrite P3 selection output: {output}")
    result = analyze(manifest_path, audit_path)
    output.mkdir(parents=True)
    (output / "selection.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(_markdown(result), encoding="utf-8")
    print(f"Decision: {result['status']}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
