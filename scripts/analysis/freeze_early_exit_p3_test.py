"""Freeze exact P3 evidence before any P3 model sees CIFAR-100 test data."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.analysis.analyze_early_exit_p0 import _sha256
from scripts.analysis.analyze_early_exit_p3 import load_manifest
from scripts.launch_early_exit_p3 import SEEDS, SOURCE_SEEDS, TARGET_SEEDS

EVALUATOR = ROOT / "scripts/analysis/evaluate_early_exit_p3_locked_test.py"
TEST_GATES = {
    "each_accuracy_drop_at_most": 0.005,
    "each_balanced_accuracy_drop_at_most": 0.005,
    "each_worst_class_accuracy_drop_at_most": 0.02,
    "mean_accuracy_drop_at_most": 0.002,
    "minimum_early_fraction": 0.15,
    "maximum_early_fraction": 0.95,
    "minimum_mac_saving_fraction": 0.15,
}


def build_lock(manifest_path: Path, audit_path: Path, selection_path: Path) -> dict:
    """Verify and describe the exact inputs without constructing a test loader."""

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if audit.get("issues") != {}:
        raise ValueError("P3 audit contains issues")
    if selection.get("status") != "ready_for_locked_cifar100_test":
        raise ValueError("P3 selection did not authorize method-locked test evaluation")
    if not selection.get("gates") or not all(selection["gates"].values()):
        raise ValueError("P3 selection has a failed gate")
    manifest_sha256 = _sha256(manifest_path)
    audit_sha256 = _sha256(audit_path)
    selection_sha256 = _sha256(selection_path)
    if audit.get("manifest_sha256") != manifest_sha256:
        raise ValueError("P3 audit references a different manifest")
    if selection.get("manifest_sha256") != manifest_sha256:
        raise ValueError("P3 selection references a different manifest")
    if selection.get("audit_sha256") != audit_sha256:
        raise ValueError("P3 selection references a different audit")

    _manifest, runs, split_fingerprint = load_manifest(manifest_path)
    if selection.get("data_protocol", {}).get("split_fingerprint") != split_fingerprint:
        raise ValueError("P3 selection split differs from the manifest")
    checkpoint_hashes = selection.get("best_checkpoint_sha256", {})
    if len(checkpoint_hashes) != 12:
        raise ValueError("P3 selection must pin twelve best checkpoints")
    for run in runs.values():
        experiment_id = run["experiment_id"]
        checkpoint = ROOT / "artifacts/runs" / experiment_id / "checkpoints/model_best.pth"
        if _sha256(checkpoint) != checkpoint_hashes.get(experiment_id):
            raise ValueError(f"P3 checkpoint differs from the selection: {experiment_id}")

    policy = selection.get("locked_policy") or {}
    expected_policy = {
        "policy_version": "shared_exit8_softmax_threshold_v1",
        "exit_position": 8,
        "confidence": "maximum softmax probability",
        "protected_predicted_classes": [],
        "fallback": "final head",
        "source_training_seeds": list(SOURCE_SEEDS),
        "target_training_seeds": list(TARGET_SEEDS),
        "threshold_candidates_considered_on_target": 0,
        "per_target_model_recalibration": False,
    }
    for key, expected in expected_policy.items():
        if policy.get(key) != expected:
            raise ValueError(f"Unexpected P3 locked-policy field: {key}")
    threshold = policy.get("confidence_threshold")
    if not isinstance(threshold, (int, float)):
        raise TypeError("P3 selection omits a numeric threshold")
    if not EVALUATOR.is_file():
        raise FileNotFoundError(EVALUATOR)
    return {
        "schema_version": 1,
        "status": "ready_for_one_shot_method_locked_cifar100_test",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Exact completed P3a manifest, accepted audit, source-selected policy and twelve "
            "best checkpoints; this lock operation does not construct or iterate a test loader."
        ),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": manifest_sha256,
        "audit": str(audit_path.relative_to(ROOT)),
        "audit_sha256": audit_sha256,
        "selection": str(selection_path.relative_to(ROOT)),
        "selection_sha256": selection_sha256,
        "evaluator": str(EVALUATOR.relative_to(ROOT)),
        "evaluator_sha256": _sha256(EVALUATOR),
        "runtime_training_commit": audit.get("runtime_commit"),
        "dataset": "cifar100",
        "source_training_seeds": list(SOURCE_SEEDS),
        "target_training_seeds": list(TARGET_SEEDS),
        "all_training_seeds": list(SEEDS),
        "locked_policy": {
            "policy_version": policy["policy_version"],
            "exit_position": policy["exit_position"],
            "confidence": policy["confidence"],
            "confidence_threshold": threshold,
            "protected_predicted_classes": policy["protected_predicted_classes"],
            "fallback": policy["fallback"],
            "path_cost_fractions": policy["path_cost_fractions"],
            "selection_performed_on_test": False,
            "test_threshold_candidates": 0,
        },
        "best_checkpoint_sha256": checkpoint_hashes,
        "test_confirmation_gates": TEST_GATES,
        "historical_test_disclosure": (
            "Legacy baseline-only seeds42/43/44 exposed CIFAR-100 test metrics before P3. "
            "No P3 checkpoint or P3 policy has seen test data; describe the next evaluation as "
            "method-locked rather than globally blind."
        ),
        "one_shot_rule": (
            "Create a permanent started marker before loading test samples; preserve any result "
            "or failure and never rerun with changed thresholds, seeds or class guards."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"P3 official-test lock is immutable: {output}")
    lock = build_lock(
        args.manifest.resolve(),
        args.audit.resolve(),
        args.selection.resolve(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"status": lock["status"], "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
