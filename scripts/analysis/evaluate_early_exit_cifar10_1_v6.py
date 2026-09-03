"""Evaluate the immutable P1b policy on CIFAR-10.1 v6 exactly once."""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from image_classification.data.cifar10_1 import (
    CIFAR10_1_REPOSITORY_COMMIT,
    CIFAR10_1_V6_DATA_FILE,
    CIFAR10_1_V6_DATA_SHA256,
    CIFAR10_1_V6_LABELS_FILE,
    CIFAR10_1_V6_LABELS_SHA256,
    CIFAR10_1_V6_ROOT,
    CIFAR10_CLASS_NAMES,
    build_cifar10_1_v6_loader,
    load_cifar10_1_v6,
)
from image_classification.models import build_model
from image_classification.training.provenance import runtime_provenance
from scripts.analysis.analyze_early_exit_p0 import _config, _cost_profile
from scripts.analysis.analyze_early_exit_p1 import _load_manifest as _load_p1_manifest
from scripts.analysis.analyze_early_exit_p2_transfer import _load_manifest as _load_p2_manifest
from scripts.analysis.evaluate_early_exit_p1_locked_test import (
    _aggregate,
    array_sha256,
    sha256,
    summarize_seed,
)

SOURCE_SEEDS = (54, 55, 56)
TARGET_SEEDS = (57, 58, 59)
ALL_SEEDS = (*SOURCE_SEEDS, *TARGET_SEEDS)
EXPECTED_THRESHOLD = 0.984
EXPECTED_SOURCE_SELECTION_SHA256 = "8dce5d938e06ae9d7432bd6baafd0c0ed9f978678fb53767ab02eeafecc723ab"
EXPECTED_P1_MANIFEST_SHA256 = "1545e85651103400c425bf3eaae0e70e4d1a4d0e413203158d2f6b6ff16c8c88"
EXPECTED_P2_MANIFEST_SHA256 = "c262be6b7ab6c2d84e0a14247504323833b966a0d314fe8bc0876267dbd50531"
EXPECTED_P2_AUDIT_SHA256 = "376d08415ff892bd041efa1c3e12e6e7678ef2208f33045ea2bdf1bcfd7e8bab"
EXPECTED_P2_TRANSFER_SHA256 = "ef86f749fba03255b5d31e9ae8ff7d7e4e1b78d3fc12bdf6ea1bec7a5eb79706"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

DEFAULT_SELECTION = Path("reports/experiments/2026-09-02-early-exit-p1b/locked_selection.json")
DEFAULT_P2_AUDIT = Path("reports/audits/2026-09-03-early-exit-p2a/audit_results.json")
DEFAULT_P2_TRANSFER = Path("reports/experiments/2026-09-03-early-exit-p2a-transfer/transfer_results.json")
DEFAULT_OUTPUT = Path("artifacts/external_tests/early_exit_p2_cifar10_1_v6_20260903")
DEFAULT_REPORT = Path("reports/experiments/2026-09-03-early-exit-p2-cifar10-1-v6")
ACCESS_STEM = f"early_exit_p2_cifar10_1_v6_{EXPECTED_P2_TRANSFER_SHA256[:16]}"
DEFAULT_ACCESS_REGISTRY = Path("artifacts/external_test_access_registry") / ACCESS_STEM


def _load_hashed_json(path: Path, expected_hash: str, label: str) -> dict:
    if sha256(path) != expected_hash:
        raise ValueError(f"{label} hash differs from the frozen input")
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_path(run: dict) -> Path:
    return ROOT / "artifacts/runs" / run["experiment_id"] / "checkpoints/model_best.pth"


def verify_frozen_inputs(
    selection_path: Path,
    p2_audit_path: Path,
    p2_transfer_path: Path,
    data_root: Path,
) -> dict:
    """Verify every evidence and dataset hash without running model inference."""

    selection = _load_hashed_json(
        selection_path,
        EXPECTED_SOURCE_SELECTION_SHA256,
        "P1b source selection",
    )
    p2_audit = _load_hashed_json(p2_audit_path, EXPECTED_P2_AUDIT_SHA256, "P2a audit")
    transfer = _load_hashed_json(
        p2_transfer_path,
        EXPECTED_P2_TRANSFER_SHA256,
        "P2a transfer result",
    )
    if selection.get("status") != "ready_for_locked_test":
        raise ValueError("P1b source selection is not accepted")
    if not selection.get("gates") or not all(selection["gates"].values()):
        raise ValueError("P1b source selection has a failed gate")
    policy = selection.get("locked_policy") or {}
    expected_policy = {
        "policy_version": "shared_exit8_softmax_threshold_v1",
        "exit_position": 8,
        "confidence": "maximum softmax probability",
        "confidence_threshold": EXPECTED_THRESHOLD,
        "protected_predicted_classes": [],
        "fallback": "final head",
        "shared_across_training_seeds": list(SOURCE_SEEDS),
    }
    for key, expected in expected_policy.items():
        if policy.get(key) != expected:
            raise ValueError(f"P1b policy mismatch: {key}")

    if p2_audit.get("issues") != {}:
        raise ValueError("P2a audit contains issues")
    if p2_audit.get("manifest_sha256") != EXPECTED_P2_MANIFEST_SHA256:
        raise ValueError("P2a audit references an unexpected manifest")
    if transfer.get("status") != "ready_for_external_shift_test":
        raise ValueError("P2a transfer result did not authorize external evaluation")
    if not transfer.get("gates") or not all(transfer["gates"].values()):
        raise ValueError("P2a transfer result has a failed gate")
    if transfer.get("manifest_sha256") != EXPECTED_P2_MANIFEST_SHA256:
        raise ValueError("P2a transfer result references an unexpected manifest")
    if transfer.get("audit_sha256") != EXPECTED_P2_AUDIT_SHA256:
        raise ValueError("P2a transfer result references an unexpected audit")
    transfer_policy = transfer.get("source_policy", {})
    if transfer_policy.get("selection_sha256") != EXPECTED_SOURCE_SELECTION_SHA256:
        raise ValueError("P2a transfer result references an unexpected source selection")
    if transfer_policy.get("confidence_threshold") != EXPECTED_THRESHOLD:
        raise ValueError("P2a transfer result changed the source threshold")
    if transfer_policy.get("threshold_candidates_considered_on_p2") != 0:
        raise ValueError("P2a unexpectedly considered target-model threshold candidates")
    if transfer_policy.get("per_target_model_recalibration") is not False:
        raise ValueError("P2a unexpectedly recalibrated target models")

    p1_manifest_path = Path(selection["manifest"])
    if sha256(p1_manifest_path) != EXPECTED_P1_MANIFEST_SHA256:
        raise ValueError("Current P1b manifest differs from the frozen source manifest")
    _p1_manifest, p1_runs, _p1_split = _load_p1_manifest(p1_manifest_path)
    p2_manifest_path = Path(transfer["manifest"])
    if sha256(p2_manifest_path) != EXPECTED_P2_MANIFEST_SHA256:
        raise ValueError("Current P2a manifest differs from the accepted transfer manifest")
    _p2_manifest, p2_runs, _p2_split = _load_p2_manifest(p2_manifest_path)
    for run in p1_runs.values():
        expected = selection["best_checkpoint_sha256"].get(run["experiment_id"])
        if sha256(_checkpoint_path(run)) != expected:
            raise ValueError(f"P1b source checkpoint mismatch: {run['experiment_id']}")
    for run in p2_runs.values():
        expected = transfer["best_checkpoint_sha256"].get(run["experiment_id"])
        if sha256(_checkpoint_path(run)) != expected:
            raise ValueError(f"P2a target checkpoint mismatch: {run['experiment_id']}")

    images, labels = load_cifar10_1_v6(data_root)
    return {
        "selection": selection,
        "p2_audit": p2_audit,
        "transfer": transfer,
        "p1_manifest_path": p1_manifest_path,
        "p2_manifest_path": p2_manifest_path,
        "p1_runs": p1_runs,
        "p2_runs": p2_runs,
        "dataset": {
            "samples": len(labels),
            "image_shape": list(images.shape),
            "image_dtype": str(images.dtype),
            "label_shape": list(labels.shape),
            "label_dtype": str(labels.dtype),
            "class_counts": np.bincount(labels, minlength=10).tolist(),
            "labels_array_sha256": array_sha256(labels),
        },
    }


def _collect_external_logits(
    run: dict,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, list[np.ndarray]]:
    config = _config(run["resolved_config"])
    model = build_model(config).to(device)
    model.load_state_dict(
        torch.load(_checkpoint_path(run), map_location=device, weights_only=True),
        strict=True,
    )
    model.eval()
    labels = []
    logits = defaultdict(list)
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs.to(device, non_blocking=True))
            values = outputs if isinstance(outputs, tuple) else (outputs,)
            labels.append(targets.numpy())
            for index, value in enumerate(values):
                logits[index].append(value.float().cpu().numpy())
    ordered = [np.concatenate(logits[index]) for index in sorted(logits)]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(labels), ordered


def frozen_external_gates(seed_results: list[dict]) -> dict[str, bool]:
    """Apply the preregistered policy gates to all six external evaluations."""

    if [row["seed"] for row in seed_results] != list(ALL_SEEDS):
        raise ValueError("External gates require ordered source and target seeds 54 through 59")
    tolerance = 1e-12
    return {
        "each_external_accuracy_drop_at_most_0": all(
            row["locked_policy"]["accuracy_drop"] <= tolerance for row in seed_results
        ),
        "each_external_balanced_drop_at_most_0": all(
            row["locked_policy"]["balanced_accuracy_drop"] <= tolerance for row in seed_results
        ),
        "each_external_worst_class_drop_at_most_0": all(
            row["locked_policy"]["worst_class_accuracy_drop"] <= tolerance for row in seed_results
        ),
        "each_external_route_is_dynamic": all(
            0.15 <= row["locked_policy"]["route_fractions"][0] <= 0.95 for row in seed_results
        ),
        "each_external_mac_saving_at_least_0_15": all(
            row["locked_policy"]["cost_saving_fraction"] >= 0.15 for row in seed_results
        ),
    }


def _markdown(result: dict) -> str:
    rows = []
    for value in result["seed_results"]:
        policy = value["locked_policy"]
        decisions = value["decision_changes_vs_final"]
        rows.append(
            "| {cohort} | {seed} | {baseline:.2f} | {final:.2f} | {locked:.2f} | "
            "{early:.2f} | {saving:.2f} | {worst:+.2f} | {changed}/{harmed}/{rescued} |".format(
                cohort=value["cohort"],
                seed=value["seed"],
                baseline=100 * value["baseline_test_accuracy"],
                final=100 * value["multi_exit_final_test_accuracy"],
                locked=100 * policy["accuracy"],
                early=100 * policy["route_fractions"][0],
                saving=100 * policy["cost_saving_fraction"],
                worst=100 * policy["worst_class_accuracy_drop"],
                changed=decisions["changed"],
                harmed=decisions["harmed"],
                rescued=decisions["rescued"],
            )
        )
    lines = [
        "# Frozen early-exit policy on CIFAR-10.1 v6",
        "",
        f"Decision: **{result['status']}**",
        "",
        "Threshold `0.984` was frozen on P1b before this external evaluation.",
        "No external-data calibration, threshold search, class guard or model selection was performed.",
        "",
        (
            "| cohort | seed | baseline % | final % | locked % | early % | MAC saving % | "
            "worst-class drop pp | changed/harmed/rescued |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Frozen external gates",
        "",
        *(f"- [{'x' if passed else ' '}] `{name}`" for name, passed in result["gates"].items()),
        "",
        "CIFAR-10.1 v6 contains 2,000 examples, exactly 200 per class.",
        "MAC saving is an operation-count proxy; the existing RTX4090D profile remains the latency evidence.",
        "",
        f"Next: {result['recommended_next_step']}",
        "",
    ]
    return "\n".join(lines)


def evaluate_once(
    selection_path: Path,
    p2_audit_path: Path,
    p2_transfer_path: Path,
    data_root: Path,
    output: Path,
    report: Path,
    access_registry: Path,
) -> dict:
    locked = verify_frozen_inputs(
        selection_path,
        p2_audit_path,
        p2_transfer_path,
        data_root,
    )
    start_marker = access_registry.with_suffix(".started.json")
    complete_marker = access_registry.with_suffix(".completed.json")
    for path in (output, report, start_marker, complete_marker):
        if path.exists():
            raise FileExistsError(f"CIFAR-10.1 one-shot guard already exists: {path}")

    runtime = runtime_provenance()
    if runtime.get("git_status") != "":
        raise RuntimeError("External evaluation requires a clean Git working tree")
    if runtime.get("tracked_source_diff_sha256") != EMPTY_SHA256:
        raise RuntimeError("External evaluation requires no tracked source diff")
    evaluator_path = Path(__file__).resolve()
    started_at = datetime.now(timezone.utc).isoformat()
    start_record = {
        "schema_version": 1,
        "status": "cifar10_1_v6_model_evaluation_started",
        "started_at_utc": started_at,
        "dataset_repository_commit": CIFAR10_1_REPOSITORY_COMMIT,
        "dataset_data_sha256": CIFAR10_1_V6_DATA_SHA256,
        "dataset_labels_sha256": CIFAR10_1_V6_LABELS_SHA256,
        "source_selection_sha256": EXPECTED_SOURCE_SELECTION_SHA256,
        "p2_manifest_sha256": EXPECTED_P2_MANIFEST_SHA256,
        "p2_audit_sha256": EXPECTED_P2_AUDIT_SHA256,
        "p2_transfer_sha256": EXPECTED_P2_TRANSFER_SHA256,
        "evaluator_sha256": sha256(evaluator_path),
        "runtime_commit": runtime["git_commit"],
        "output": str(output),
        "report": str(report),
        "rule": "This marker forbids a second model evaluation on CIFAR-10.1 v6.",
    }
    start_marker.parent.mkdir(parents=True, exist_ok=True)
    with start_marker.open("x", encoding="utf-8") as handle:
        json.dump(start_record, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    output.mkdir(parents=True, exist_ok=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_cifar10_1_v6_loader(data_root, batch_size=128)
    path_costs = [_cost_profile()["path_cost_fractions"][0], 1.0]
    seed_results = []
    prediction_files = {}
    labels_fingerprint = None
    for seed in ALL_SEEDS:
        cohort = "source" if seed in SOURCE_SEEDS else "target"
        runs = locked["p1_runs"] if cohort == "source" else locked["p2_runs"]
        baseline_labels, baseline_values = _collect_external_logits(
            runs[("mobilenetv2", seed)],
            loader,
            device,
        )
        multi_labels, multi_values = _collect_external_logits(
            runs[("multi_exit", seed)],
            loader,
            device,
        )
        if not np.array_equal(baseline_labels, multi_labels):
            raise ValueError(f"Matched CIFAR-10.1 order differs for seed {seed}")
        if len(baseline_values) != 1 or len(multi_values) != 3:
            raise ValueError(f"Unexpected CIFAR-10.1 model output count for seed {seed}")
        current_fingerprint = array_sha256(multi_labels)
        if labels_fingerprint is None:
            labels_fingerprint = current_fingerprint
        elif labels_fingerprint != current_fingerprint:
            raise ValueError("CIFAR-10.1 label order differs across model versions")
        seed_result, locked_predictions, paths = summarize_seed(
            multi_labels,
            baseline_values[0],
            multi_values,
            seed=seed,
            threshold=EXPECTED_THRESHOLD,
            path_costs=path_costs,
            class_names=CIFAR10_CLASS_NAMES,
        )
        seed_result["cohort"] = cohort
        seed_result["baseline_experiment_id"] = runs[("mobilenetv2", seed)]["experiment_id"]
        seed_result["multi_exit_experiment_id"] = runs[("multi_exit", seed)]["experiment_id"]
        seed_results.append(seed_result)
        prediction_path = output / f"cifar10_1_v6_logits_and_routes_seed{seed}.npz"
        np.savez_compressed(
            prediction_path,
            labels=multi_labels,
            baseline_logits=baseline_values[0],
            multi_exit_final_logits=multi_values[0],
            exit8_logits=multi_values[1],
            exit16_logits=multi_values[2],
            locked_predictions=locked_predictions,
            locked_paths=paths,
        )
        prediction_files[str(seed)] = {
            "path": str(prediction_path.relative_to(ROOT)),
            "size_bytes": prediction_path.stat().st_size,
            "sha256": sha256(prediction_path),
        }

    gates = frozen_external_gates(seed_results)
    status = "external_shift_confirmed" if all(gates.values()) else "stop_external_shift_failure"
    result = {
        "schema_version": 1,
        "status": status,
        "scope": (
            "One model-inference pass over hash-pinned CIFAR-10.1 v6 for six preselected "
            "baseline/multi-exit checkpoint pairs and the P1b-frozen threshold; no tuning."
        ),
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "runtime": runtime,
        "evaluator_sha256": sha256(evaluator_path),
        "source_selection_sha256": EXPECTED_SOURCE_SELECTION_SHA256,
        "p1_manifest_sha256": EXPECTED_P1_MANIFEST_SHA256,
        "p2_manifest_sha256": EXPECTED_P2_MANIFEST_SHA256,
        "p2_audit_sha256": EXPECTED_P2_AUDIT_SHA256,
        "p2_transfer_sha256": EXPECTED_P2_TRANSFER_SHA256,
        "external_test_set": {
            "dataset": "cifar10.1_v6",
            "official_repository": "https://github.com/modestyachts/CIFAR-10.1",
            "repository_commit": CIFAR10_1_REPOSITORY_COMMIT,
            "data_file": CIFAR10_1_V6_DATA_FILE,
            "data_sha256": CIFAR10_1_V6_DATA_SHA256,
            "labels_file": CIFAR10_1_V6_LABELS_FILE,
            "labels_sha256": CIFAR10_1_V6_LABELS_SHA256,
            **locked["dataset"],
            "labels_evaluation_order_sha256": labels_fingerprint,
            "metadata_integrity_checked_before_model_evaluation": True,
        },
        "locked_policy": {
            "exit_position": 8,
            "confidence": "maximum softmax probability",
            "threshold": EXPECTED_THRESHOLD,
            "fallback": "final head",
            "protected_predicted_classes": [],
            "path_cost_fractions": path_costs,
            "selection_performed_on_external_data": False,
            "external_threshold_candidates_considered": 0,
        },
        "seed_results": seed_results,
        "aggregate": {
            "source_seeds": _aggregate([row for row in seed_results if row["cohort"] == "source"]),
            "target_seeds": _aggregate([row for row in seed_results if row["cohort"] == "target"]),
            "all_seeds": _aggregate(seed_results),
        },
        "gates": gates,
        "prediction_files": prediction_files,
        "limitations": [
            "CIFAR-10.1 is an independent natural distribution shift but remains CIFAR-10-like.",
            "Worst-class and zero-drop statements are empirical on 200 examples per class.",
            "MAC saving is not latency; use the separate synchronized RTX4090D profile.",
            "The architecture evidence still covers only MobileNetV2.",
        ],
        "recommended_next_step": (
            "Freeze this result and run one minimal second-dataset or second-backbone confirmation "
            "before declaring the full paper evidence complete."
            if status == "external_shift_confirmed"
            else "Archive the external-shift failure and stop the current no-recalibration claim."
        ),
    }
    result_path = output / "external_results.json"
    markdown_path = output / "README.md"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    markdown_path.write_text(_markdown(result))
    source_index = {
        "dataset": result["external_test_set"],
        "source_selection": {
            "path": str(selection_path.relative_to(ROOT)),
            "sha256": EXPECTED_SOURCE_SELECTION_SHA256,
        },
        "p1_manifest": {
            "path": str(locked["p1_manifest_path"].relative_to(ROOT)),
            "sha256": EXPECTED_P1_MANIFEST_SHA256,
        },
        "p2_manifest": {
            "path": str(locked["p2_manifest_path"].relative_to(ROOT)),
            "sha256": EXPECTED_P2_MANIFEST_SHA256,
        },
        "p2_audit": {
            "path": str(p2_audit_path.relative_to(ROOT)),
            "sha256": EXPECTED_P2_AUDIT_SHA256,
        },
        "p2_transfer": {
            "path": str(p2_transfer_path.relative_to(ROOT)),
            "sha256": EXPECTED_P2_TRANSFER_SHA256,
        },
        "evaluator": {
            "path": str(evaluator_path.relative_to(ROOT)),
            "sha256": result["evaluator_sha256"],
        },
        "external_results": {
            "path": str(result_path.relative_to(ROOT)),
            "sha256": sha256(result_path),
        },
        "prediction_files": prediction_files,
        "access_started": {
            "path": str(start_marker.relative_to(ROOT)),
            "sha256": sha256(start_marker),
        },
    }
    source_index_path = output / "source_index.json"
    source_index_path.write_text(json.dumps(source_index, indent=2, ensure_ascii=False) + "\n")
    complete_record = {
        **start_record,
        "status": "cifar10_1_v6_model_evaluation_completed",
        "finished_at_utc": result["finished_at_utc"],
        "external_results_sha256": sha256(result_path),
        "source_index_sha256": sha256(source_index_path),
    }
    with complete_marker.open("x", encoding="utf-8") as handle:
        json.dump(complete_record, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    report.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(result_path, report / "external_results.json")
    shutil.copyfile(markdown_path, report / "README.md")
    shutil.copyfile(source_index_path, report / "source_index.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--p2-audit", type=Path, default=DEFAULT_P2_AUDIT)
    parser.add_argument("--p2-transfer", type=Path, default=DEFAULT_P2_TRANSFER)
    parser.add_argument("--data-root", type=Path, default=CIFAR10_1_V6_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--access-registry", type=Path, default=DEFAULT_ACCESS_REGISTRY)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    selection_path = (ROOT / args.selection).resolve()
    p2_audit_path = (ROOT / args.p2_audit).resolve()
    p2_transfer_path = (ROOT / args.p2_transfer).resolve()
    data_root = args.data_root.resolve()
    if args.verify_only:
        locked = verify_frozen_inputs(
            selection_path,
            p2_audit_path,
            p2_transfer_path,
            data_root,
        )
        print(
            json.dumps(
                {
                    "status": locked["transfer"]["status"],
                    "threshold": EXPECTED_THRESHOLD,
                    "source_seeds": list(SOURCE_SEEDS),
                    "target_seeds": list(TARGET_SEEDS),
                    "dataset": locked["dataset"],
                    "model_inference_performed": False,
                },
                indent=2,
            )
        )
        return 0
    result = evaluate_once(
        selection_path,
        p2_audit_path,
        p2_transfer_path,
        data_root,
        (ROOT / args.output).resolve(),
        (ROOT / args.report).resolve(),
        (ROOT / args.access_registry).resolve(),
    )
    print(f"Status: {result['status']}")
    print(_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
