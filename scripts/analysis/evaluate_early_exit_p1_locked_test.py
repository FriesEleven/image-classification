"""Evaluate the immutable P1b policy on official CIFAR-10 test exactly once."""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from image_classification.selection.early_exit import apply_policy, policy_metrics
from scripts.analysis.analyze_early_exit_p0 import _accuracy, _collect_logits, _sample_std
from scripts.analysis.analyze_early_exit_p1 import EXPECTED_SEEDS, _load_manifest

EXPECTED_SELECTION_SHA256 = "8dce5d938e06ae9d7432bd6baafd0c0ed9f978678fb53767ab02eeafecc723ab"
EXPECTED_MANIFEST_SHA256 = "1545e85651103400c425bf3eaae0e70e4d1a4d0e413203158d2f6b6ff16c8c88"
EXPECTED_AUDIT_SHA256 = "5681878b3b3fcc97d2f667e2832b1875aaa00b40ba82edf06e8356c8fb7fa486"
EXPECTED_THRESHOLD = 0.984
DEFAULT_SELECTION = Path(
    "artifacts/policy_selections/early_exit_p1b_20260902_165900/selection.json"
)
DEFAULT_AUDIT = Path("reports/audits/2026-09-02-early-exit-p1b/audit_results.json")
DEFAULT_OUTPUT = Path("artifacts/locked_tests/early_exit_p1b_20260902")
DEFAULT_REPORT = Path("reports/experiments/2026-09-02-early-exit-p1b")
ACCESS_STEM = f"early_exit_p1b_{EXPECTED_SELECTION_SHA256[:16]}"
DEFAULT_ACCESS_REGISTRY = Path("artifacts/test_access_registry") / ACCESS_STEM
CIFAR10_CLASS_NAMES = (
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(json.dumps(values.shape).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _accuracy_for_class(predictions: np.ndarray, labels: np.ndarray, class_id: int) -> float:
    selected = labels == class_id
    if not np.any(selected):
        raise ValueError(f"Test labels omit class {class_id}")
    return float(np.mean(predictions[selected] == labels[selected]))


def summarize_seed(
    labels: np.ndarray,
    baseline_logits: np.ndarray,
    multi_logits: list[np.ndarray],
    *,
    seed: int,
    threshold: float,
    path_costs: list[float],
    class_names: tuple[str, ...] = CIFAR10_CLASS_NAMES,
) -> tuple[dict, np.ndarray, np.ndarray]:
    if len(multi_logits) != 3:
        raise ValueError("P1b multi-exit output must be final, exit8 and exit16")
    final_logits, exit8_logits, exit16_logits = multi_logits
    baseline_predictions = baseline_logits.argmax(axis=1)
    final_predictions = final_logits.argmax(axis=1)
    locked_predictions, paths = apply_policy(
        [exit8_logits],
        final_logits,
        [threshold],
    )
    locked_metrics = policy_metrics(
        labels,
        locked_predictions,
        final_predictions,
        paths,
        path_costs,
    )
    harmed = int(np.sum((final_predictions == labels) & (locked_predictions != labels)))
    rescued = int(np.sum((final_predictions != labels) & (locked_predictions == labels)))
    if set(np.unique(labels)) != set(range(len(class_names))):
        raise ValueError("Test labels do not cover the expected classes")
    class_metrics = []
    for class_id, class_name in enumerate(class_names):
        baseline_accuracy = _accuracy_for_class(baseline_predictions, labels, class_id)
        final_accuracy = _accuracy_for_class(final_predictions, labels, class_id)
        locked_accuracy = _accuracy_for_class(locked_predictions, labels, class_id)
        class_metrics.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "support": int(np.sum(labels == class_id)),
                "baseline_accuracy": baseline_accuracy,
                "multi_exit_final_accuracy": final_accuracy,
                "locked_policy_accuracy": locked_accuracy,
                "locked_policy_drop_vs_final": final_accuracy - locked_accuracy,
                "locked_policy_gain_vs_baseline": locked_accuracy - baseline_accuracy,
            }
        )
    result = {
        "seed": seed,
        "samples": len(labels),
        "baseline_test_accuracy": _accuracy(baseline_logits, labels),
        "multi_exit_final_test_accuracy": _accuracy(final_logits, labels),
        "exit8_test_accuracy": _accuracy(exit8_logits, labels),
        "exit16_test_accuracy": _accuracy(exit16_logits, labels),
        "paired_final_gain_vs_baseline": (
            _accuracy(final_logits, labels) - _accuracy(baseline_logits, labels)
        ),
        "locked_policy": locked_metrics,
        "locked_policy_gain_vs_baseline": (
            locked_metrics["accuracy"] - _accuracy(baseline_logits, labels)
        ),
        "locked_policy_gain_vs_final": (
            locked_metrics["accuracy"] - _accuracy(final_logits, labels)
        ),
        "decision_changes_vs_final": {
            "changed": int(np.sum(locked_predictions != final_predictions)),
            "harmed": harmed,
            "rescued": rescued,
            "net_correct": rescued - harmed,
        },
        "early_route_count": int(np.sum(paths == 0)),
        "final_route_count": int(np.sum(paths == 1)),
        "class_metrics": class_metrics,
    }
    return result, locked_predictions, paths


def _aggregate(seed_results: list[dict]) -> dict:
    metrics = {
        "baseline_test_accuracy": [row["baseline_test_accuracy"] for row in seed_results],
        "multi_exit_final_test_accuracy": [
            row["multi_exit_final_test_accuracy"] for row in seed_results
        ],
        "paired_final_gain_vs_baseline": [
            row["paired_final_gain_vs_baseline"] for row in seed_results
        ],
        "locked_policy_accuracy": [row["locked_policy"]["accuracy"] for row in seed_results],
        "locked_policy_gain_vs_baseline": [
            row["locked_policy_gain_vs_baseline"] for row in seed_results
        ],
        "locked_policy_gain_vs_final": [
            row["locked_policy_gain_vs_final"] for row in seed_results
        ],
        "early_route_fraction": [
            row["locked_policy"]["route_fractions"][0] for row in seed_results
        ],
        "mac_saving_fraction": [
            row["locked_policy"]["cost_saving_fraction"] for row in seed_results
        ],
        "worst_class_accuracy_drop_vs_final": [
            row["locked_policy"]["worst_class_accuracy_drop"] for row in seed_results
        ],
    }
    return {
        name: {
            "values": values,
            "mean": float(np.mean(values)),
            "sample_std": _sample_std(values),
        }
        for name, values in metrics.items()
    }


def verify_lock(selection_path: Path, audit_path: Path) -> tuple[dict, dict]:
    if sha256(selection_path) != EXPECTED_SELECTION_SHA256:
        raise ValueError("The policy selection hash differs from the frozen P1b selection")
    if sha256(audit_path) != EXPECTED_AUDIT_SHA256:
        raise ValueError("The P1b audit hash differs from the accepted audit")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if selection.get("status") != "ready_for_locked_test":
        raise ValueError("The P1b selector did not authorize official-test evaluation")
    if not selection.get("gates") or not all(selection["gates"].values()):
        raise ValueError("Not every frozen P1b gate passed")
    if audit.get("issues") != {}:
        raise ValueError("The accepted P1b audit contains issues")
    if selection.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Selection references an unexpected P1b manifest hash")
    if audit.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Audit references an unexpected P1b manifest hash")

    policy = selection.get("locked_policy") or {}
    expected_policy = {
        "policy_version": "shared_exit8_softmax_threshold_v1",
        "exit_position": 8,
        "confidence": "maximum softmax probability",
        "confidence_threshold": EXPECTED_THRESHOLD,
        "protected_predicted_classes": [],
        "fallback": "final head",
        "shared_across_training_seeds": list(EXPECTED_SEEDS),
    }
    for key, expected in expected_policy.items():
        if policy.get(key) != expected:
            raise ValueError(f"Locked policy mismatch: {key}")
    if sorted(policy.get("calibration_metrics", {})) != ["54", "55", "56"]:
        raise ValueError("Locked policy does not cover all three training seeds")

    manifest_path = Path(selection["manifest"])
    if sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Current P1b manifest differs from the frozen manifest")
    manifest, runs, _split_fingerprint = _load_manifest(manifest_path)
    for run in runs.values():
        experiment_id = run["experiment_id"]
        checkpoint = ROOT / "artifacts/runs" / experiment_id / "checkpoints/model_best.pth"
        if sha256(checkpoint) != selection["best_checkpoint_sha256"].get(experiment_id):
            raise ValueError(f"Frozen checkpoint hash mismatch: {experiment_id}")
    return selection, {"manifest": manifest, "runs": runs, "path": manifest_path}


def _markdown(result: dict) -> str:
    rows = []
    for value in result["seed_results"]:
        policy = value["locked_policy"]
        rows.append(
            "| {seed} | {baseline:.2f} | {final:.2f} | {final_gain:+.2f} | "
            "{locked:.2f} | {policy_gain:+.2f} | {early:.2f} | {saving:.2f} | "
            "{worst:.2f} |".format(
                seed=value["seed"],
                baseline=100 * value["baseline_test_accuracy"],
                final=100 * value["multi_exit_final_test_accuracy"],
                final_gain=100 * value["paired_final_gain_vs_baseline"],
                locked=100 * policy["accuracy"],
                policy_gain=100 * value["locked_policy_gain_vs_baseline"],
                early=100 * policy["route_fractions"][0],
                saving=100 * policy["cost_saving_fraction"],
                worst=100 * policy["worst_class_accuracy_drop"],
            )
        )
    aggregate = result["aggregate"]

    def mean_sd(name: str) -> str:
        values = aggregate[name]
        return f"{100 * values['mean']:.2f} ± {100 * values['sample_std']:.2f}"

    rows.append(
        "| mean ± sample SD | {baseline} | {final} | {final_gain} | {locked} | "
        "{policy_gain} | {early} | {saving} | {worst} |".format(
            baseline=mean_sd("baseline_test_accuracy"),
            final=mean_sd("multi_exit_final_test_accuracy"),
            final_gain=mean_sd("paired_final_gain_vs_baseline"),
            locked=mean_sd("locked_policy_accuracy"),
            policy_gain=mean_sd("locked_policy_gain_vs_baseline"),
            early=mean_sd("early_route_fraction"),
            saving=mean_sd("mac_saving_fraction"),
            worst=mean_sd("worst_class_accuracy_drop_vs_final"),
        )
    )
    return "\n".join(
        [
            "# Early-exit P1b locked official-test evaluation",
            "",
            "The official CIFAR-10 test set was evaluated once after the shared policy was frozen.",
            "No threshold, class guard or model choice was tuned on test.",
            "",
            f"Frozen exit-8 maximum-softmax threshold: `{result['locked_policy']['threshold']}`.",
            "",
            (
                "| seed | baseline % | final % | final−base pp | locked % | locked−base pp | "
                "early % | MAC saving % | worst-class drop vs final pp |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "MAC saving is an architecture-level conv/linear operation proxy; it is not measured latency.",
            "Per-class metrics and retained logits are indexed in `source_index.json`.",
            "",
            f"Selection SHA-256: `{result['selection_sha256']}`.",
            f"Manifest SHA-256: `{result['manifest_sha256']}`.",
            "",
        ]
    )


def evaluate_once(
    selection_path: Path,
    audit_path: Path,
    output: Path,
    report: Path,
    access_registry: Path,
) -> dict:
    selection, locked_inputs = verify_lock(selection_path, audit_path)
    start_marker = access_registry.with_suffix(".started.json")
    complete_marker = access_registry.with_suffix(".completed.json")
    for path in (output, report, start_marker, complete_marker):
        if path.exists():
            raise FileExistsError(f"Official-test one-shot guard already exists: {path}")

    evaluator_path = Path(__file__).resolve()
    started_at = datetime.now(timezone.utc).isoformat()
    start_record = {
        "schema_version": 1,
        "status": "official_test_access_started",
        "started_at_utc": started_at,
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "audit_sha256": EXPECTED_AUDIT_SHA256,
        "evaluator_sha256": sha256(evaluator_path),
        "output": str(output),
        "report": str(report),
        "rule": "This marker forbids rerunning the locked official-test evaluation.",
    }
    start_marker.parent.mkdir(parents=True, exist_ok=True)
    with start_marker.open("x", encoding="utf-8") as handle:
        json.dump(start_record, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    output.mkdir(parents=True, exist_ok=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = locked_inputs["runs"]
    policy = selection["locked_policy"]
    path_costs = list(policy["path_cost_fractions"])
    seed_results = []
    prediction_files = {}
    labels_fingerprint = None
    for seed in EXPECTED_SEEDS:
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
            raise ValueError(f"Matched test order differs for seed {seed}")
        if len(baseline_values) != 1:
            raise ValueError(f"Unexpected baseline output count for seed {seed}")
        current_labels_fingerprint = array_sha256(multi_labels)
        if labels_fingerprint is None:
            labels_fingerprint = current_labels_fingerprint
        elif labels_fingerprint != current_labels_fingerprint:
            raise ValueError("Official test label order differs across seeds")

        seed_result, locked_predictions, paths = summarize_seed(
            multi_labels,
            baseline_values[0],
            multi_values,
            seed=seed,
            threshold=policy["confidence_threshold"],
            path_costs=path_costs,
        )
        seed_result["baseline_experiment_id"] = runs[("mobilenetv2", seed)]["experiment_id"]
        seed_result["multi_exit_experiment_id"] = runs[("multi_exit", seed)]["experiment_id"]
        seed_results.append(seed_result)
        prediction_path = output / f"test_logits_and_routes_seed{seed}.npz"
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

    raw_test_batch = ROOT / "data/cifar-10-batches-py/test_batch"
    result = {
        "schema_version": 1,
        "status": "locked_official_test_complete",
        "scope": (
            "One official CIFAR-10 test evaluation of six preselected best checkpoints "
            "and one calibration-locked shared exit8 policy; no test-time tuning."
        ),
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "audit_sha256": EXPECTED_AUDIT_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "evaluator_sha256": sha256(evaluator_path),
        "test_set": {
            "dataset": "cifar10",
            "samples": 10_000,
            "class_names": list(CIFAR10_CLASS_NAMES),
            "labels_sha256": labels_fingerprint,
            "raw_test_batch_sha256": sha256(raw_test_batch) if raw_test_batch.is_file() else None,
        },
        "locked_policy": {
            "policy_version": policy["policy_version"],
            "exit_position": policy["exit_position"],
            "confidence": policy["confidence"],
            "threshold": policy["confidence_threshold"],
            "fallback": policy["fallback"],
            "protected_predicted_classes": policy["protected_predicted_classes"],
            "path_cost_fractions": path_costs,
            "selection_performed_on_test": False,
        },
        "seed_results": seed_results,
        "aggregate": _aggregate(seed_results),
        "prediction_files": prediction_files,
        "limitations": [
            "MAC saving counts convolution and linear operations only and is not measured latency.",
            "Risk constraints are empirical on a finite 5k calibration set, not formal guarantees.",
            "The current evidence covers CIFAR-10, MobileNetV2 and three training seeds only.",
        ],
    }
    results_path = output / "test_results.json"
    markdown_path = output / "test_results.md"
    results_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    markdown_path.write_text(_markdown(result))
    source_index = {
        "selection": {
            "path": str(selection_path.relative_to(ROOT)),
            "sha256": EXPECTED_SELECTION_SHA256,
        },
        "audit": {
            "path": str(audit_path.relative_to(ROOT)),
            "sha256": EXPECTED_AUDIT_SHA256,
        },
        "manifest": {
            "path": str(locked_inputs["path"].relative_to(ROOT)),
            "sha256": EXPECTED_MANIFEST_SHA256,
        },
        "evaluator": {
            "path": str(evaluator_path.relative_to(ROOT)),
            "sha256": result["evaluator_sha256"],
        },
        "test_results": {
            "path": str(results_path.relative_to(ROOT)),
            "sha256": sha256(results_path),
        },
        "test_markdown": {
            "path": str(markdown_path.relative_to(ROOT)),
            "sha256": sha256(markdown_path),
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
        "status": "official_test_access_completed",
        "finished_at_utc": result["finished_at_utc"],
        "test_results_sha256": sha256(results_path),
        "source_index_sha256": sha256(source_index_path),
    }
    with complete_marker.open("x", encoding="utf-8") as handle:
        json.dump(complete_record, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    report.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(selection_path, report / "locked_selection.json")
    shutil.copyfile(results_path, report / "test_results.json")
    shutil.copyfile(markdown_path, report / "README.md")
    shutil.copyfile(source_index_path, report / "source_index.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--access-registry", type=Path, default=DEFAULT_ACCESS_REGISTRY)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    selection_path = (ROOT / args.selection).resolve()
    audit_path = (ROOT / args.audit).resolve()
    if args.verify_only:
        selection, _locked_inputs = verify_lock(selection_path, audit_path)
        print(
            json.dumps(
                {
                    "status": selection["status"],
                    "selection_sha256": sha256(selection_path),
                    "threshold": selection["locked_policy"]["confidence_threshold"],
                },
                indent=2,
            )
        )
        return 0
    result = evaluate_once(
        selection_path,
        audit_path,
        (ROOT / args.output).resolve(),
        (ROOT / args.report).resolve(),
        (ROOT / args.access_registry).resolve(),
    )
    print("Status: locked_official_test_complete")
    print(_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
