"""Lock one cross-seed exit-8 policy from a completed P1 calibration-only batch."""

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

EXPECTED_SEEDS = (54, 55, 56)
EXPECTED_TYPES = ("mobilenetv2", "multi_exit")
SPLIT_SEED = 20_260_902
THRESHOLD_STEP = 0.001
THRESHOLDS = tuple(index * THRESHOLD_STEP for index in range(1001)) + (
    float(np.nextafter(1.0, np.inf)),
)
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


def _split_fingerprint(run: dict) -> tuple[str, dict]:
    path = ROOT / "artifacts/runs" / run["experiment_id"] / "split_indices.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    train = record.get("train_indices", [])
    validation = record.get("validation_indices", [])
    calibration = record.get("calibration_indices", [])
    if [len(train), len(validation), len(calibration)] != [40_000, 5000, 5000]:
        raise ValueError(f"Unexpected 40k/5k/5k split: {run['experiment_id']}")
    sets = [set(values) for values in (train, validation, calibration)]
    if any(sets[left] & sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError(f"Overlapping P1 split: {run['experiment_id']}")
    if set.union(*sets) != set(range(50_000)):
        raise ValueError(f"Incomplete P1 split coverage: {run['experiment_id']}")
    if record.get("split_seed") != SPLIT_SEED or record.get("training_seed") != run["seed"]:
        raise ValueError(f"P1 split/training seed mismatch: {run['experiment_id']}")
    canonical = json.dumps(
        {"train": train, "validation": validation, "calibration": calibration},
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest(), record


def _load_manifest(path: Path) -> tuple[dict, dict[tuple[str, int], dict], str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    sweep_name = manifest.get("sweep_name", "")
    prefix = "cifar10_early_exit_p1_serial_"
    if not sweep_name.startswith(prefix) or not sweep_name.removeprefix(prefix):
        raise ValueError("Select an explicit early-exit P1 serial manifest")
    if manifest.get("status") != "completed" or manifest.get("concurrent_jobs") != 1:
        raise ValueError("P1 manifest must be completed and serial")
    indexed = {}
    for run in manifest.get("runs", []):
        config = run.get("resolved_config", {})
        key = (config.get("model_type"), int(run.get("seed", -1)))
        if key in indexed:
            raise ValueError(f"Duplicate P1 run: {key}")
        indexed[key] = run
    expected = {(model_type, seed) for model_type in EXPECTED_TYPES for seed in EXPECTED_SEEDS}
    if set(indexed) != expected:
        raise ValueError("Expected exactly baseline/multi-exit x seeds 54/55/56")

    common_split_fingerprints = set()
    for key, run in indexed.items():
        config = run["resolved_config"]
        if run.get("status") != "completed" or run.get("return_code") != 0:
            raise ValueError(f"Incomplete P1 run: {key}")
        if run.get("termination_signal") is not None:
            raise ValueError(f"Unexpected P1 termination signal: {key}")
        for name, expected_value in EXPECTED_PROTOCOL.items():
            if config.get(name) != expected_value:
                raise ValueError(f"P1 protocol mismatch for {name}: {key}")
        if key[0] == "multi_exit":
            for name, expected_value in EXPECTED_MULTI_EXIT.items():
                if config.get(name) != expected_value:
                    raise ValueError(f"P1 multi-exit mismatch for {name}: {key}")
        split_fingerprint, _record = _split_fingerprint(run)
        common_split_fingerprints.add(split_fingerprint)
    if len(common_split_fingerprints) != 1:
        raise ValueError("All P1 runs must use the same fixed data split")
    return manifest, indexed, common_split_fingerprints.pop()


def analyze(manifest_path: Path) -> dict:
    _manifest, runs, split_fingerprint = _load_manifest(manifest_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cost_profile = _cost_profile()
    path_costs = [cost_profile["path_cost_fractions"][0], 1.0]
    seed_results = []
    calibration_datasets = []
    checkpoint_hashes = {}
    for seed in EXPECTED_SEEDS:
        baseline_run = runs[("mobilenetv2", seed)]
        multi_run = runs[("multi_exit", seed)]
        validation_labels, baseline_values = _collect_logits(
            baseline_run, device, split="validation",
        )
        multi_labels, multi_values = _collect_logits(multi_run, device, split="validation")
        if not np.array_equal(validation_labels, multi_labels):
            raise ValueError(f"Matched validation order differs for seed {seed}")
        if len(baseline_values) != 1 or len(multi_values) != 3:
            raise ValueError(f"Unexpected P1 model output count for seed {seed}")
        final_logits, exit8_logits, exit16_logits = multi_values
        calibration_labels, calibration_values = _collect_logits(
            multi_run, device, split="calibration",
        )
        if len(calibration_values) != 3:
            raise ValueError(f"Unexpected P1 calibration output count for seed {seed}")
        calibration_final, calibration_exit8, _calibration_exit16 = calibration_values
        calibration_datasets.append(
            (calibration_labels, calibration_exit8, calibration_final)
        )
        baseline_accuracy = _accuracy(baseline_values[0], validation_labels)
        final_accuracy = _accuracy(final_logits, validation_labels)
        seed_results.append(
            {
                "seed": seed,
                "baseline_experiment_id": baseline_run["experiment_id"],
                "multi_exit_experiment_id": multi_run["experiment_id"],
                "baseline_validation_accuracy": baseline_accuracy,
                "multi_exit_final_validation_accuracy": final_accuracy,
                "paired_final_validation_gain": final_accuracy - baseline_accuracy,
                "exit8_validation_accuracy": _accuracy(exit8_logits, validation_labels),
                "exit16_validation_accuracy": _accuracy(exit16_logits, validation_labels),
            }
        )
        for run in (baseline_run, multi_run):
            checkpoint = (
                ROOT / "artifacts/runs" / run["experiment_id"] / "checkpoints/model_best.pth"
            )
            checkpoint_hashes[run["experiment_id"]] = _sha256(checkpoint)

    selected = select_shared_single_exit_policy(
        calibration_datasets,
        path_costs=path_costs,
        thresholds=THRESHOLDS,
        max_accuracy_drop=0.0,
        max_balanced_accuracy_drop=0.0,
        max_worst_class_drop=0.0,
        min_early_fraction=0.15,
        max_early_fraction=0.95,
    )
    gains = [value["paired_final_validation_gain"] for value in seed_results]
    gates = {
        "mean_final_validation_gain_at_least_minus_0_003": float(np.mean(gains)) >= -0.003,
        "each_final_validation_gain_at_least_minus_0_0075": min(gains) >= -0.0075,
        "shared_dynamic_policy_found": selected is not None,
    }
    locked_policy = None
    if selected is not None:
        calibration_metrics = {
            str(seed): metrics
            for seed, metrics in zip(EXPECTED_SEEDS, selected["calibration_metrics"])
        }
        gates.update(
            {
                "each_calibration_mac_saving_at_least_0_15": all(
                    metrics["cost_saving_fraction"] >= 0.15
                    for metrics in selected["calibration_metrics"]
                ),
                "each_calibration_accuracy_drop_at_most_0": all(
                    metrics["accuracy_drop"] <= 1e-12
                    for metrics in selected["calibration_metrics"]
                ),
                "each_calibration_balanced_drop_at_most_0": all(
                    metrics["balanced_accuracy_drop"] <= 1e-12
                    for metrics in selected["calibration_metrics"]
                ),
                "each_calibration_worst_class_drop_at_most_0": all(
                    metrics["worst_class_accuracy_drop"] <= 1e-12
                    for metrics in selected["calibration_metrics"]
                ),
                "each_calibration_route_is_dynamic": all(
                    0.15 <= metrics["route_fractions"][0] <= 0.95
                    for metrics in selected["calibration_metrics"]
                ),
            }
        )
        locked_policy = {
            "policy_version": "shared_exit8_softmax_threshold_v1",
            "exit_position": 8,
            "confidence": "maximum softmax probability",
            "confidence_threshold": selected["confidence_threshold"],
            "protected_predicted_classes": [],
            "fallback": "final head",
            "shared_across_training_seeds": list(EXPECTED_SEEDS),
            "path_cost_fractions": path_costs,
            "calibration_metrics": calibration_metrics,
            "selection_constraints": selected["constraints"],
            "selection_objective": selected["objective"],
        }
    status = "ready_for_locked_test" if all(gates.values()) else "stop_or_redesign"
    return {
        "schema_version": 1,
        "status": status,
        "scope": (
            "P1 model-selection validation plus disjoint policy calibration; official CIFAR-10 "
            "test is not evaluated by this analysis"
        ),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "device": str(device),
        "data_protocol": {
            "train_samples": 40_000,
            "model_selection_validation_samples": 5000,
            "policy_calibration_samples": 5000,
            "split_seed": SPLIT_SEED,
            "training_seeds": list(EXPECTED_SEEDS),
            "split_fingerprint": split_fingerprint,
        },
        "threshold_grid": {
            "frozen_before_training": True,
            "start": 0.0,
            "stop": 1.0,
            "step": THRESHOLD_STEP,
            "includes_final_only_sentinel": True,
            "candidate_count": len(THRESHOLDS),
        },
        "cost_profile": cost_profile,
        "best_checkpoint_sha256": checkpoint_hashes,
        "seed_results": seed_results,
        "aggregate": {
            "paired_final_validation_gain_mean": float(np.mean(gains)),
            "paired_final_validation_gain_sample_std": _sample_std(gains),
        },
        "locked_policy": locked_policy,
        "gates": gates,
        "recommended_next_step": (
            "Hash and freeze this selection, then evaluate the six best checkpoints and the one "
            "locked policy on official CIFAR-10 test exactly once."
            if status == "ready_for_locked_test"
            else "Do not open the official test set or expand the experiment matrix; stop/redesign."
        ),
    }


def _markdown(result: dict) -> str:
    rows = []
    for value in result["seed_results"]:
        rows.append(
            "| {seed} | {baseline:.2f} | {final:.2f} | {gain:+.2f} | {exit8:.2f} | {exit16:.2f} |".format(
                seed=value["seed"],
                baseline=100 * value["baseline_validation_accuracy"],
                final=100 * value["multi_exit_final_validation_accuracy"],
                gain=100 * value["paired_final_validation_gain"],
                exit8=100 * value["exit8_validation_accuracy"],
                exit16=100 * value["exit16_validation_accuracy"],
            )
        )
    lines = [
        "# Early-exit P1 policy lock",
        "",
        f"Decision: **{result['status']}**",
        "",
        "| seed | baseline validation % | final validation % | gain pp | exit8 % | exit16 % |",
        "|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Frozen gates",
        "",
        *(f"- [{'x' if passed else ' '}] `{name}`" for name, passed in result["gates"].items()),
        "",
    ]
    if result["locked_policy"] is not None:
        policy = result["locked_policy"]
        lines.extend(
            [
                "## Locked policy",
                "",
                f"Exit-8 confidence threshold: `{policy['confidence_threshold']}`; fallback: final head.",
                "The threshold is shared across all three training seeds and no class-specific guard is used.",
                "",
            ]
        )
    lines.extend(
        [
            "This script never iterates the official CIFAR-10 test loader.",
            "",
            f"Next: {result['recommended_next_step']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite P1 selection output: {output}")
    result = analyze(manifest_path)
    output.mkdir(parents=True)
    (output / "selection.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    (output / "selection.md").write_text(_markdown(result), encoding="utf-8")
    print(f"Decision: {result['status']}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
