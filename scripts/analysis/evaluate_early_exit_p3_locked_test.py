"""Evaluate the hash-locked P3 policy on CIFAR-100 test exactly once."""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from image_classification.data import build_dataloaders
from image_classification.training.provenance import runtime_provenance
from scripts.analysis.analyze_early_exit_p0 import _collect_logits, _config, _sha256
from scripts.analysis.analyze_early_exit_p3 import load_manifest
from scripts.analysis.evaluate_early_exit_p1_locked_test import (
    _aggregate,
    array_sha256,
    summarize_seed,
)
from scripts.analysis.freeze_early_exit_p3_test import EVALUATOR, TEST_GATES
from scripts.launch_early_exit_p3 import SEEDS, SOURCE_SEEDS, TARGET_SEEDS

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
DEFAULT_LOCK = Path("reports/experiments/2026-09-03-early-exit-p3-cifar100/official_test_lock.json")
DEFAULT_OUTPUT = Path("artifacts/official_tests/early_exit_p3_cifar100_20260903")
DEFAULT_REPORT = Path("reports/experiments/2026-09-03-early-exit-p3-cifar100-test")


def locked_test_gates(seed_results: list[dict]) -> dict[str, bool]:
    """Apply the confirmation boundaries frozen before P3 training."""

    if [row["seed"] for row in seed_results] != list(SEEDS):
        raise ValueError("P3 test gates require ordered seeds 60 through 65")
    policies = [row["locked_policy"] for row in seed_results]
    drops = [value["accuracy_drop"] for value in policies]
    tolerance = 1e-12
    return {
        "each_accuracy_drop_at_most_0_005": all(
            value <= TEST_GATES["each_accuracy_drop_at_most"] + tolerance for value in drops
        ),
        "each_balanced_accuracy_drop_at_most_0_005": all(
            value["balanced_accuracy_drop"] <= TEST_GATES["each_balanced_accuracy_drop_at_most"] + tolerance
            for value in policies
        ),
        "each_worst_class_accuracy_drop_at_most_0_02": all(
            value["worst_class_accuracy_drop"] <= TEST_GATES["each_worst_class_accuracy_drop_at_most"] + tolerance
            for value in policies
        ),
        "mean_accuracy_drop_at_most_0_002": (
            float(np.mean(drops)) <= TEST_GATES["mean_accuracy_drop_at_most"] + tolerance
        ),
        "each_route_is_dynamic": all(
            TEST_GATES["minimum_early_fraction"] <= value["route_fractions"][0] <= TEST_GATES["maximum_early_fraction"]
            for value in policies
        ),
        "each_mac_saving_at_least_0_15": all(
            value["cost_saving_fraction"] >= TEST_GATES["minimum_mac_saving_fraction"] - tolerance for value in policies
        ),
    }


def _require_versioned_clean_inputs(paths: tuple[Path, ...]) -> dict:
    runtime = runtime_provenance()
    if runtime.get("git_status") != "":
        raise RuntimeError("P3 method-locked test requires a clean Git working tree")
    if runtime.get("tracked_source_diff_sha256") != EMPTY_SHA256:
        raise RuntimeError("P3 method-locked test requires no tracked source diff")
    for path in paths:
        relative = str(path.relative_to(ROOT))
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"P3 lock input must be committed before test access: {relative}")
    return runtime


def verify_lock(lock_path: Path) -> tuple[dict, dict, dict, dict, dict]:
    """Verify the committed lock and all evidence without iterating test data."""

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "ready_for_one_shot_method_locked_cifar100_test":
        raise ValueError("P3 lock did not authorize test evaluation")
    if lock.get("dataset") != "cifar100":
        raise ValueError("P3 lock references a different dataset")
    if lock.get("source_training_seeds") != list(SOURCE_SEEDS):
        raise ValueError("P3 lock source seeds differ from the frozen cohort")
    if lock.get("target_training_seeds") != list(TARGET_SEEDS):
        raise ValueError("P3 lock target seeds differ from the frozen cohort")
    if lock.get("all_training_seeds") != list(SEEDS):
        raise ValueError("P3 lock full seed order differs from the frozen matrix")
    if lock.get("test_confirmation_gates") != TEST_GATES:
        raise ValueError("P3 test gates differ from the preregistered evaluator")
    if lock.get("evaluator") != str(EVALUATOR.relative_to(ROOT)):
        raise ValueError("P3 lock references a different evaluator")
    if lock.get("evaluator_sha256") != _sha256(EVALUATOR):
        raise ValueError("P3 evaluator differs from the frozen lock")

    manifest_path = ROOT / lock["manifest"]
    audit_path = ROOT / lock["audit"]
    selection_path = ROOT / lock["selection"]
    expected_hashes = {
        manifest_path: lock["manifest_sha256"],
        audit_path: lock["audit_sha256"],
        selection_path: lock["selection_sha256"],
    }
    for path, expected in expected_hashes.items():
        if _sha256(path) != expected:
            raise ValueError(f"P3 locked input hash mismatch: {path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if audit.get("issues") != {}:
        raise ValueError("P3 locked audit contains issues")
    if selection.get("status") != "ready_for_locked_cifar100_test":
        raise ValueError("P3 locked selection is not accepted")
    if not selection.get("gates") or not all(selection["gates"].values()):
        raise ValueError("P3 locked selection has a failed gate")
    if selection.get("manifest_sha256") != lock["manifest_sha256"]:
        raise ValueError("P3 selection references a different manifest")
    if selection.get("audit_sha256") != lock["audit_sha256"]:
        raise ValueError("P3 selection references a different audit")
    policy = selection.get("locked_policy")
    if not isinstance(policy, dict):
        raise TypeError("P3 selection omits the locked policy")
    lock_policy = lock["locked_policy"]
    for key in (
        "policy_version",
        "exit_position",
        "confidence",
        "confidence_threshold",
        "protected_predicted_classes",
        "fallback",
        "path_cost_fractions",
    ):
        if policy.get(key) != lock_policy.get(key):
            raise ValueError(f"P3 selection/lock policy mismatch: {key}")
    if policy.get("threshold_candidates_considered_on_target") != 0:
        raise ValueError("P3 target models considered threshold candidates")
    if policy.get("per_target_model_recalibration") is not False:
        raise ValueError("P3 target models were recalibrated")

    _manifest, runs, _split_fingerprint = load_manifest(manifest_path)
    checkpoint_hashes = lock.get("best_checkpoint_sha256", {})
    if checkpoint_hashes != selection.get("best_checkpoint_sha256"):
        raise ValueError("P3 lock checkpoint map differs from selection")
    for run in runs.values():
        experiment_id = run["experiment_id"]
        checkpoint = ROOT / "artifacts/runs" / experiment_id / "checkpoints/model_best.pth"
        if _sha256(checkpoint) != checkpoint_hashes.get(experiment_id):
            raise ValueError(f"P3 locked checkpoint mismatch: {experiment_id}")
    for name in ("train", "test", "meta"):
        path = ROOT / "data/cifar-100-python" / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing local CIFAR-100 file before one-shot access: {path}")
    return (
        lock,
        audit,
        selection,
        runs,
        {
            "manifest": manifest_path,
            "audit": audit_path,
            "selection": selection_path,
        },
    )


def _class_names(run: dict) -> tuple[str, ...]:
    config = _config(run["resolved_config"])
    loaders = build_dataloaders(
        dataset=config.dataset,
        batch_size=config.batch_size,
        num_workers=0,
        prefetch_factor=config.prefetch_factor,
        validation_size=config.validation_size,
        split_seed=config.split_seed,
        calibration_size=config.calibration_size,
        shuffle_seed=config.seed,
    )
    return loaders.class_names


def _markdown(result: dict) -> str:
    rows = []
    for value in result["seed_results"]:
        policy = value["locked_policy"]
        decisions = value["decision_changes_vs_final"]
        rows.append(
            "| {cohort} | {seed} | {baseline:.2f} | {final:.2f} | {locked:.2f} | "
            "{early:.2f} | {saving:.2f} | {drop:+.2f} | {worst:+.2f} | "
            "{changed}/{harmed}/{rescued} |".format(
                cohort=value["cohort"],
                seed=value["seed"],
                baseline=100 * value["baseline_test_accuracy"],
                final=100 * value["multi_exit_final_test_accuracy"],
                locked=100 * policy["accuracy"],
                early=100 * policy["route_fractions"][0],
                saving=100 * policy["cost_saving_fraction"],
                drop=100 * policy["accuracy_drop"],
                worst=100 * policy["worst_class_accuracy_drop"],
                changed=decisions["changed"],
                harmed=decisions["harmed"],
                rescued=decisions["rescued"],
            )
        )
    return "\n".join(
        [
            "# CIFAR-100 P3 method-locked official-test evaluation",
            "",
            f"Decision: **{result['status']}**",
            "",
            (
                "| cohort | seed | baseline % | final % | locked % | early % | MAC saving % | "
                "overall drop pp | worst-class drop pp | changed/harmed/rescued |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## Preregistered confirmation gates",
            "",
            *(f"- [{'x' if passed else ' '}] `{name}`" for name, passed in result["gates"].items()),
            "",
            result["historical_test_disclosure"],
            "No test-time threshold, class guard, seed or checkpoint selection was performed.",
            "",
            f"Next: {result['recommended_next_step']}",
            "",
        ]
    )


def evaluate_once(
    lock_path: Path,
    output: Path,
    report: Path,
    access_registry: Path | None,
) -> dict:
    lock, _audit, selection, runs, paths = verify_lock(lock_path)
    lock_sha256 = _sha256(lock_path)
    if access_registry is None:
        access_registry = (
            ROOT / "artifacts/official_test_access_registry" / f"early_exit_p3_cifar100_{lock_sha256[:16]}"
        )
    start_marker = access_registry.with_suffix(".started.json")
    complete_marker = access_registry.with_suffix(".completed.json")
    for path in (output, report, start_marker, complete_marker):
        if path.exists():
            raise FileExistsError(f"P3 CIFAR-100 one-shot guard already exists: {path}")

    runtime = _require_versioned_clean_inputs((lock_path, paths["audit"], paths["selection"], EVALUATOR))
    started_at = datetime.now(timezone.utc).isoformat()
    start_record = {
        "schema_version": 1,
        "status": "p3_method_locked_cifar100_test_started",
        "started_at_utc": started_at,
        "lock": str(lock_path.relative_to(ROOT)),
        "lock_sha256": lock_sha256,
        "manifest_sha256": lock["manifest_sha256"],
        "audit_sha256": lock["audit_sha256"],
        "selection_sha256": lock["selection_sha256"],
        "evaluator_sha256": lock["evaluator_sha256"],
        "runtime_commit": runtime["git_commit"],
        "output": str(output),
        "report": str(report),
        "rule": "This marker permanently forbids a second P3 CIFAR-100 test evaluation.",
    }
    start_marker.parent.mkdir(parents=True, exist_ok=True)
    with start_marker.open("x", encoding="utf-8") as handle:
        json.dump(start_record, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    output.mkdir(parents=True, exist_ok=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = _class_names(runs[("mobilenetv2", SEEDS[0])])
    policy = selection["locked_policy"]
    threshold = policy["confidence_threshold"]
    path_costs = list(policy["path_cost_fractions"])
    seed_results = []
    prediction_files = {}
    labels_fingerprint = None
    for seed in SEEDS:
        baseline_labels, baseline_values = _collect_logits(
            runs[("mobilenetv2", seed)],
            device,
            split="test",
        )
        multi_labels, multi_values = _collect_logits(
            runs[("multi_exit", seed)],
            device,
            split="test",
        )
        if not np.array_equal(baseline_labels, multi_labels):
            raise ValueError(f"Matched CIFAR-100 test order differs for seed {seed}")
        if len(baseline_values) != 1 or len(multi_values) != 3:
            raise ValueError(f"Unexpected P3 test model output count for seed {seed}")
        fingerprint = array_sha256(multi_labels)
        if labels_fingerprint is None:
            labels_fingerprint = fingerprint
        elif fingerprint != labels_fingerprint:
            raise ValueError("CIFAR-100 test label order differs across P3 models")
        seed_result, locked_predictions, routes = summarize_seed(
            multi_labels,
            baseline_values[0],
            multi_values,
            seed=seed,
            threshold=threshold,
            path_costs=path_costs,
            class_names=class_names,
        )
        seed_result["cohort"] = "source" if seed in SOURCE_SEEDS else "target"
        seed_result["baseline_experiment_id"] = runs[("mobilenetv2", seed)]["experiment_id"]
        seed_result["multi_exit_experiment_id"] = runs[("multi_exit", seed)]["experiment_id"]
        seed_results.append(seed_result)
        prediction_path = output / f"cifar100_test_logits_and_routes_seed{seed}.npz"
        np.savez_compressed(
            prediction_path,
            labels=multi_labels,
            baseline_logits=baseline_values[0],
            multi_exit_final_logits=multi_values[0],
            exit8_logits=multi_values[1],
            exit16_logits=multi_values[2],
            locked_predictions=locked_predictions,
            locked_routes=routes,
        )
        prediction_files[str(seed)] = {
            "path": str(prediction_path.relative_to(ROOT)),
            "size_bytes": prediction_path.stat().st_size,
            "sha256": _sha256(prediction_path),
        }

    gates = locked_test_gates(seed_results)
    status = "paper_evidence_complete" if all(gates.values()) else "p3_test_boundary_failure"
    raw_test = ROOT / "data/cifar-100-python/test"
    result = {
        "schema_version": 1,
        "status": status,
        "scope": (
            "One method-locked CIFAR-100 official-test evaluation of twelve preselected best "
            "checkpoints and one source-calibrated threshold; no test-time tuning."
        ),
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "runtime": runtime,
        "lock_sha256": lock_sha256,
        "manifest_sha256": lock["manifest_sha256"],
        "audit_sha256": lock["audit_sha256"],
        "selection_sha256": lock["selection_sha256"],
        "evaluator_sha256": lock["evaluator_sha256"],
        "test_set": {
            "dataset": "cifar100",
            "samples": 10_000,
            "classes": len(class_names),
            "samples_per_class": 100,
            "labels_evaluation_order_sha256": labels_fingerprint,
            "raw_test_file_sha256": _sha256(raw_test) if raw_test.is_file() else None,
        },
        "locked_policy": {
            **lock["locked_policy"],
            "threshold_candidates_considered_on_test": 0,
        },
        "seed_results": seed_results,
        "aggregate": {
            "source_seeds": _aggregate([row for row in seed_results if row["cohort"] == "source"]),
            "target_seeds": _aggregate([row for row in seed_results if row["cohort"] == "target"]),
            "all_seeds": _aggregate(seed_results),
        },
        "gates": gates,
        "prediction_files": prediction_files,
        "historical_test_disclosure": lock["historical_test_disclosure"],
        "limitations": [
            "CIFAR-100 test was historically exposed by unrelated legacy baseline runs.",
            "Worst-class evidence has only 100 test examples per class.",
            "MAC saving is an operation-count proxy, not measured latency.",
            "Both training datasets still use MobileNetV2.",
        ],
        "recommended_next_step": (
            "Stop new training and proceed to frozen paper tables, figures and writing."
            if status == "paper_evidence_complete"
            else "Archive this boundary result without retuning or rerunning; reassess the paper claim."
        ),
    }
    result_path = output / "test_results.json"
    markdown_path = output / "README.md"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    markdown_path.write_text(_markdown(result))
    source_index = {
        "lock": {"path": str(lock_path.relative_to(ROOT)), "sha256": lock_sha256},
        "manifest": {"path": lock["manifest"], "sha256": lock["manifest_sha256"]},
        "audit": {"path": lock["audit"], "sha256": lock["audit_sha256"]},
        "selection": {"path": lock["selection"], "sha256": lock["selection_sha256"]},
        "evaluator": {"path": lock["evaluator"], "sha256": lock["evaluator_sha256"]},
        "test_results": {
            "path": str(result_path.relative_to(ROOT)),
            "sha256": _sha256(result_path),
        },
        "prediction_files": prediction_files,
        "access_started": {
            "path": str(start_marker.relative_to(ROOT)),
            "sha256": _sha256(start_marker),
        },
    }
    source_index_path = output / "source_index.json"
    source_index_path.write_text(json.dumps(source_index, indent=2, ensure_ascii=False) + "\n")
    complete_record = {
        **start_record,
        "status": "p3_method_locked_cifar100_test_completed",
        "finished_at_utc": result["finished_at_utc"],
        "test_results_sha256": _sha256(result_path),
        "source_index_sha256": _sha256(source_index_path),
    }
    with complete_marker.open("x", encoding="utf-8") as handle:
        json.dump(complete_record, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    report.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(result_path, report / "test_results.json")
    shutil.copyfile(markdown_path, report / "README.md")
    shutil.copyfile(source_index_path, report / "source_index.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--access-registry", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    lock_path = (ROOT / args.lock).resolve()
    if args.verify_only:
        lock, _audit, selection, _runs, _paths = verify_lock(lock_path)
        print(
            json.dumps(
                {
                    "status": lock["status"],
                    "selection_status": selection["status"],
                    "threshold": lock["locked_policy"]["confidence_threshold"],
                    "model_or_test_inference_performed": False,
                },
                indent=2,
            )
        )
        return 0
    access_registry = (ROOT / args.access_registry).resolve() if args.access_registry is not None else None
    result = evaluate_once(
        lock_path,
        (ROOT / args.output).resolve(),
        (ROOT / args.report).resolve(),
        access_registry,
    )
    print(f"Status: {result['status']}")
    print(_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
