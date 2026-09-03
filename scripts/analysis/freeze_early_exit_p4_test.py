"""Freeze exact P4 evidence before any P4 model sees CIFAR-100 test data."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.analysis.analyze_early_exit_p0 import _sha256
from scripts.analysis.analyze_early_exit_p4_confirmation import load_manifest
from scripts.launch_early_exit_p4 import LOCKED_THRESHOLD, POLICY_LOCK, SEEDS

EVALUATOR = ROOT / "scripts/analysis/evaluate_early_exit_p4_locked_test.py"
TEST_GATES = {
    "each_accuracy_drop_at_most": 0.005,
    "each_balanced_accuracy_drop_at_most": 0.005,
    "each_worst_class_accuracy_drop_at_most": 0.04,
    "mean_accuracy_drop_at_most": 0.002,
    "minimum_early_fraction": 0.15,
    "maximum_early_fraction": 0.95,
    "minimum_mac_saving_fraction": 0.15,
}


def build_lock(
    manifest_path: Path,
    audit_path: Path,
    confirmation_path: Path,
    policy_path: Path,
) -> dict:
    """Verify and describe P4 inputs without constructing a test loader."""

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    policy_lock = json.loads(policy_path.read_text(encoding="utf-8"))
    if audit.get("issues") != {}:
        raise ValueError("P4 audit contains issues")
    if confirmation.get("status") != "ready_for_method_locked_cifar100_test":
        raise ValueError("P4 confirmation did not authorize method-locked test evaluation")
    if not confirmation.get("gates") or not all(confirmation["gates"].values()):
        raise ValueError("P4 confirmation has a failed gate")
    manifest_sha256 = _sha256(manifest_path)
    audit_sha256 = _sha256(audit_path)
    confirmation_sha256 = _sha256(confirmation_path)
    policy_sha256 = _sha256(policy_path)
    if audit.get("manifest_sha256") != manifest_sha256:
        raise ValueError("P4 audit references a different manifest")
    if confirmation.get("manifest_sha256") != manifest_sha256:
        raise ValueError("P4 confirmation references a different manifest")
    if confirmation.get("audit_sha256") != audit_sha256:
        raise ValueError("P4 confirmation references a different audit")
    if confirmation.get("policy_lock_sha256") != policy_sha256:
        raise ValueError("P4 confirmation references a different policy lock")

    _manifest, runs, split_fingerprint = load_manifest(manifest_path)
    if confirmation.get("data_protocol", {}).get("split_fingerprint") != split_fingerprint:
        raise ValueError("P4 confirmation split differs from the manifest")
    checkpoint_hashes = confirmation.get("best_checkpoint_sha256", {})
    if len(checkpoint_hashes) != 6:
        raise ValueError("P4 confirmation must pin six best checkpoints")
    for run in runs.values():
        experiment_id = run["experiment_id"]
        checkpoint = ROOT / "artifacts/runs" / experiment_id / "checkpoints/model_best.pth"
        if _sha256(checkpoint) != checkpoint_hashes.get(experiment_id):
            raise ValueError(f"P4 checkpoint differs from confirmation: {experiment_id}")

    policy = confirmation.get("locked_policy") or {}
    expected_policy = {
        "policy_version": "shared_exit8_softmax_threshold_p4_v1",
        "exit_position": 8,
        "confidence": "maximum softmax probability",
        "confidence_threshold": LOCKED_THRESHOLD,
        "protected_predicted_classes": [],
        "fallback": "final head",
        "p4_threshold_candidates": 0,
        "per_model_recalibration": False,
        "threshold_candidates_considered_on_p4": 0,
    }
    for key, expected in expected_policy.items():
        if policy.get(key) != expected:
            raise ValueError(f"Unexpected P4 locked-policy field: {key}")
    if not EVALUATOR.is_file():
        raise FileNotFoundError(EVALUATOR)
    return {
        "schema_version": 1,
        "status": "ready_for_one_shot_method_locked_cifar100_test",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Exact completed P4a manifest, accepted audit, independently confirmed frozen "
            "policy and six best checkpoints; lock creation does not iterate a test loader."
        ),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": manifest_sha256,
        "audit": str(audit_path.relative_to(ROOT)),
        "audit_sha256": audit_sha256,
        "confirmation": str(confirmation_path.relative_to(ROOT)),
        "confirmation_sha256": confirmation_sha256,
        "policy_lock": str(policy_path.relative_to(ROOT)),
        "policy_lock_sha256": policy_sha256,
        "evaluator": str(EVALUATOR.relative_to(ROOT)),
        "evaluator_sha256": _sha256(EVALUATOR),
        "runtime_training_commit": audit.get("runtime_commit"),
        "dataset": "cifar100",
        "training_seeds": list(SEEDS),
        "split_seed": confirmation["data_protocol"]["split_seed"],
        "locked_policy": {
            "policy_version": policy["policy_version"],
            "exit_position": policy["exit_position"],
            "confidence": policy["confidence"],
            "confidence_threshold": policy["confidence_threshold"],
            "protected_predicted_classes": policy["protected_predicted_classes"],
            "fallback": policy["fallback"],
            "path_cost_fractions": policy["path_cost_fractions"],
            "selection_performed_on_test": False,
            "test_threshold_candidates": 0,
        },
        "best_checkpoint_sha256": checkpoint_hashes,
        "test_confirmation_gates": TEST_GATES,
        "historical_test_disclosure": confirmation["historical_test_disclosure"],
        "one_shot_rule": (
            "Create a permanent started marker before loading test samples; preserve any result "
            "or failure and never rerun with changed thresholds, seeds or class guards."
        ),
        "p3_boundary_preserved": {
            "p3_status": policy_lock["p3_frozen_result"]["status"],
            "p3_official_test_accessed": policy_lock["p3_frozen_result"]["official_test_accessed"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--policy-lock", type=Path, default=POLICY_LOCK)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"P4 official-test lock is immutable: {output}")
    lock = build_lock(
        args.manifest.resolve(),
        args.audit.resolve(),
        args.confirmation.resolve(),
        args.policy_lock.resolve(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"status": lock["status"], "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
