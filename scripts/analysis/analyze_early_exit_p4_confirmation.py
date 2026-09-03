"""Apply the frozen P4 threshold to independent CIFAR-100 model versions."""

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

from scripts.analysis.analyze_early_exit_p0 import (
    _accuracy,
    _collect_logits,
    _cost_profile,
    _sample_std,
    _sha256,
)
from scripts.analysis.analyze_early_exit_p2_transfer import evaluate_frozen_policy
from scripts.launch_early_exit_p4 import (
    LOCKED_THRESHOLD,
    POLICY_LOCK,
    SEEDS,
    SPLIT_SEED,
    validate_policy_lock,
)

EXPECTED_TYPES = ("mobilenetv2", "multi_exit")
EXPECTED_SWEEP_NAME = "cifar100_early_exit_p4_confirmation_serial_p4a"
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
        raise ValueError(f"Unexpected P4 data split: {run['experiment_id']}")
    sets = [set(values) for values in partitions]
    if any(sets[left] & sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError(f"Overlapping P4 split: {run['experiment_id']}")
    if set.union(*sets) != set(range(50_000)):
        raise ValueError(f"Incomplete P4 split coverage: {run['experiment_id']}")
    if split.get("split_seed") != SPLIT_SEED or split.get("training_seed") != run["seed"]:
        raise ValueError(f"P4 split metadata mismatch: {run['experiment_id']}")
    canonical = json.dumps(
        {"train": partitions[0], "validation": partitions[1], "calibration": partitions[2]},
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def load_manifest(path: Path) -> tuple[dict, dict[tuple[str, int], dict], str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("sweep_name") != EXPECTED_SWEEP_NAME:
        raise ValueError("Select the unique CIFAR-100 P4a manifest")
    if manifest.get("status") != "completed" or manifest.get("concurrent_jobs") != 1:
        raise ValueError("P4a manifest must be completed and serial")
    indexed = {}
    for run in manifest.get("runs", []):
        config = run.get("resolved_config", {})
        key = (config.get("model_type"), int(run.get("seed", -1)))
        if key in indexed:
            raise ValueError(f"Duplicate P4a run: {key}")
        indexed[key] = run
    expected = {(model_type, seed) for model_type in EXPECTED_TYPES for seed in SEEDS}
    if set(indexed) != expected:
        raise ValueError("Expected baseline/multi-exit x seeds 66 through 68")
    fingerprints = set()
    for key, run in indexed.items():
        config = run["resolved_config"]
        if run.get("status") != "completed" or run.get("return_code") != 0:
            raise ValueError(f"Incomplete P4a run: {key}")
        if run.get("termination_signal") is not None:
            raise ValueError(f"Unexpected P4a termination signal: {key}")
        for name, expected_value in EXPECTED_PROTOCOL.items():
            if config.get(name) != expected_value:
                raise ValueError(f"P4a protocol mismatch for {name}: {key}")
        if key[0] == "multi_exit":
            for name, expected_value in EXPECTED_MULTI_EXIT.items():
                if config.get(name) != expected_value:
                    raise ValueError(f"P4a multi-exit mismatch for {name}: {key}")
        fingerprints.add(_split_fingerprint(run))
    if len(fingerprints) != 1:
        raise ValueError("All P4a runs must use one fixed CIFAR-100 split")
    return manifest, indexed, fingerprints.pop()


def frozen_gates(final_gains: list[float], policies: list[dict]) -> dict[str, bool]:
    tolerance = 1e-12
    complete = len(policies) == len(SEEDS)
    return {
        "mean_final_validation_gain_at_least_minus_0_003": (
            len(final_gains) == len(SEEDS) and float(np.mean(final_gains)) >= -0.003
        ),
        "each_final_validation_gain_at_least_minus_0_0075": (
            len(final_gains) == len(SEEDS) and min(final_gains) >= -0.0075
        ),
        "each_accuracy_drop_at_most_0": complete and all(value["accuracy_drop"] <= tolerance for value in policies),
        "each_balanced_drop_at_most_0": complete
        and all(value["balanced_accuracy_drop"] <= tolerance for value in policies),
        "each_worst_class_drop_at_most_0_04": complete
        and all(value["worst_class_accuracy_drop"] <= 0.04 + tolerance for value in policies),
        "each_route_is_dynamic": complete and all(0.15 <= value["route_fractions"][0] <= 0.95 for value in policies),
        "each_mac_saving_at_least_0_15": complete
        and all(value["cost_saving_fraction"] >= 0.15 - tolerance for value in policies),
    }


def _validate_inputs(audit_path: Path, manifest_path: Path, policy_path: Path) -> tuple[dict, dict]:
    if policy_path != POLICY_LOCK.resolve():
        raise ValueError("P4 confirmation requires the canonical policy lock")
    validate_policy_lock()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if audit.get("issues") != {}:
        raise ValueError("The P4a audit contains issues")
    if audit.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("The P4a audit does not reference this manifest")
    if audit.get("policy_lock", {}).get("sha256") != _sha256(policy_path):
        raise ValueError("The P4a audit does not reference this policy lock")
    if policy.get("status") != "ready_for_independent_p4_confirmation":
        raise ValueError("P4 policy lock is not ready")
    frozen = policy.get("frozen_policy", {})
    if frozen.get("confidence_threshold") != LOCKED_THRESHOLD:
        raise ValueError("P4 threshold differs from the frozen value")
    if frozen.get("protected_predicted_classes") != []:
        raise ValueError("P4 unexpectedly enables a class guard")
    if frozen.get("p4_threshold_candidates") != 0:
        raise ValueError("P4 confirmation considered threshold candidates")
    return audit, policy


def analyze(manifest_path: Path, audit_path: Path, policy_path: Path) -> dict:
    audit, policy_lock = _validate_inputs(audit_path, manifest_path, policy_path)
    _manifest, runs, split_fingerprint = load_manifest(manifest_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cost_profile = _cost_profile("cifar100")
    path_costs = [cost_profile["path_cost_fractions"][0], 1.0]
    if policy_lock["frozen_policy"].get("path_cost_fractions") != path_costs:
        raise ValueError("P4 path-cost profile differs from the frozen policy")
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
        multi_validation_labels, multi_values = _collect_logits(
            multi_run,
            device,
            split="validation",
        )
        if not np.array_equal(validation_labels, multi_validation_labels):
            raise ValueError(f"Matched validation order differs for P4 seed {seed}")
        if len(baseline_values) != 1 or len(multi_values) != 3:
            raise ValueError(f"Unexpected P4 model output count for seed {seed}")
        final_validation, exit8_validation, exit16_validation = multi_values
        confirmation_labels, confirmation_values = _collect_logits(
            multi_run,
            device,
            split="calibration",
        )
        if len(confirmation_values) != 3:
            raise ValueError(f"Unexpected P4 confirmation output count for seed {seed}")
        confirmation_final, confirmation_exit8, _confirmation_exit16 = confirmation_values
        policy_metrics, _predictions, _paths = evaluate_frozen_policy(
            confirmation_labels,
            confirmation_exit8,
            confirmation_final,
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
                "policy_confirmation_metrics": policy_metrics,
            }
        )
        for run in (baseline_run, multi_run):
            checkpoint = ROOT / "artifacts/runs" / run["experiment_id"] / "checkpoints/model_best.pth"
            checkpoint_hashes[run["experiment_id"]] = _sha256(checkpoint)

    final_gains = [row["paired_final_validation_gain"] for row in seed_results]
    policies = [row["policy_confirmation_metrics"] for row in seed_results]
    gates = frozen_gates(final_gains, policies)
    status = "ready_for_method_locked_cifar100_test" if all(gates.values()) else "stop_without_test"
    return {
        "schema_version": 1,
        "status": status,
        "scope": (
            "Independent P4 confirmation on new seeds 66/67/68 and split seed 20260904; "
            "threshold 0.903 is applied unchanged with zero candidates; official test is not iterated."
        ),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": _sha256(manifest_path),
        "audit": str(audit_path.relative_to(ROOT)),
        "audit_sha256": _sha256(audit_path),
        "policy_lock": str(policy_path.relative_to(ROOT)),
        "policy_lock_sha256": _sha256(policy_path),
        "runtime_commit": audit.get("runtime_commit"),
        "device": str(device),
        "data_protocol": {
            "dataset": "cifar100",
            "train_samples": 40_000,
            "model_selection_validation_samples": 5000,
            "policy_confirmation_samples": 5000,
            "samples_per_class_in_policy_confirmation": 50,
            "split_seed": SPLIT_SEED,
            "split_fingerprint": split_fingerprint,
            "training_seeds": list(SEEDS),
        },
        "locked_policy": {
            **policy_lock["frozen_policy"],
            "threshold_candidates_considered_on_p4": 0,
        },
        "cost_profile": cost_profile,
        "best_checkpoint_sha256": checkpoint_hashes,
        "seed_results": seed_results,
        "aggregate": {
            "paired_final_validation_gain_mean": float(np.mean(final_gains)),
            "paired_final_validation_gain_sample_std": _sample_std(final_gains),
            "policy_mac_saving_mean": float(np.mean([value["cost_saving_fraction"] for value in policies])),
            "policy_accuracy_gain_mean": float(-np.mean([value["accuracy_drop"] for value in policies])),
        },
        "gates": gates,
        "historical_test_disclosure": (
            "Legacy baseline-only CIFAR-100 runs exposed test metrics before P4. P3 stopped "
            "without test. No P4 checkpoint or policy has seen official test data; describe a "
            "later pass as method-locked rather than globally blind."
        ),
        "recommended_next_step": (
            "Freeze exact P4 evidence and evaluator hashes, then run one method-locked "
            "CIFAR-100 official-test evaluation."
            if status == "ready_for_method_locked_cifar100_test"
            else "Archive P4 without official-test access; do not tune or rerun replacement seeds."
        ),
    }


def _markdown(result: dict) -> str:
    rows = []
    for value in result["seed_results"]:
        policy = value["policy_confirmation_metrics"]
        rows.append(
            "| {seed} | {baseline:.2f} | {final:.2f} | {gain:+.2f} | {early:.2f} | "
            "{saving:.2f} | {drop:+.2f} | {worst:+.2f} |".format(
                seed=value["seed"],
                baseline=100 * value["baseline_validation_accuracy"],
                final=100 * value["multi_exit_final_validation_accuracy"],
                gain=100 * value["paired_final_validation_gain"],
                early=100 * policy["route_fractions"][0],
                saving=100 * policy["cost_saving_fraction"],
                drop=100 * policy["accuracy_drop"],
                worst=100 * policy["worst_class_accuracy_drop"],
            )
        )
    return "\n".join(
        [
            "# CIFAR-100 early-exit P4 independent confirmation",
            "",
            f"Decision: **{result['status']}**",
            "",
            "| seed | baseline val % | final val % | gain pp | early % | MAC saving % | overall drop pp | worst-class drop pp |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## Frozen gates",
            "",
            *(f"- [{'x' if passed else ' '}] `{name}`" for name, passed in result["gates"].items()),
            "",
            "P4 considered zero threshold candidates and performed no per-model recalibration.",
            "The official CIFAR-100 test loader is not iterated by this analysis.",
            "",
            f"Next: {result['recommended_next_step']}",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--policy-lock", type=Path, default=POLICY_LOCK)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite P4 confirmation output: {output}")
    result = analyze(
        args.manifest.resolve(),
        args.audit.resolve(),
        args.policy_lock.resolve(),
    )
    output.mkdir(parents=True)
    (output / "confirmation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(_markdown(result), encoding="utf-8")
    print(f"Decision: {result['status']}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
