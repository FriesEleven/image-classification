"""Apply the frozen P7 ImageNet-100 source gate and lock one shared threshold."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.analysis.analyze_early_exit_p0 import _collect_logits
from scripts.analysis.analyze_early_exit_p6_source import (
    best_head_metrics,
    configure_numerics,
    read_json,
    select_shared_threshold,
    sha256,
    write_csv,
)

PROTOCOL = ROOT / "reports/experiments/2026-09-07-early-exit-p7-source-analysis-design/protocol_manifest.json"


def p7_risk_budget(training_protocol: dict) -> dict[str, float]:
    """Translate the P7 training-protocol field names without changing values."""

    budget = training_protocol["risk_budget"]
    return {
        "overall_drop": float(budget["overall_accuracy_drop"]),
        "balanced_drop": float(budget["balanced_accuracy_drop"]),
        "worst_class_drop": float(budget["worst_class_accuracy_drop"]),
    }


def corrected_cost_fractions(profile: dict, num_classes: int) -> dict[str, float | int]:
    """Return costs for exit-8 routing with exit-15 excluded at deployment."""

    deployable = profile["selected"]["deployable"]
    auxiliary = profile["selected"]["training_only_auxiliary"]
    if deployable["position"] != 8 or auxiliary["position"] != 15:
        raise ValueError("P7 architecture profile no longer selects exits 8 and 15")
    final_macs = int(profile["final_path_macs"])
    exit_path_macs = int(deployable["path_macs_including_exit_head"])
    exit_head_macs = int(deployable["channels"]) * int(num_classes)
    return {
        "final_path_macs": final_macs,
        "exit_path_macs": exit_path_macs,
        "exit_head_macs": exit_head_macs,
        "fallback_path_macs": final_macs + exit_head_macs,
        "exit_cost_fraction": exit_path_macs / final_macs,
        "fallback_cost_fraction": (final_macs + exit_head_macs) / final_macs,
    }


def unpack_p7_logits(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bind model output order to final, deployable exit 8, and auxiliary exit 15."""

    if len(values) != 3:
        raise ValueError("P7 A3 must return final, exit8, and exit15 logits")
    final_logits, deployable_logits, auxiliary_logits = values
    if final_logits.shape != deployable_logits.shape or final_logits.shape != auxiliary_logits.shape:
        raise ValueError("P7 logit heads have inconsistent shapes")
    return final_logits, deployable_logits, auxiliary_logits


def _validate_receipt(item: dict, label: str) -> Path:
    path = ROOT / item["path"]
    if not path.is_file() or sha256(path) != item["sha256"]:
        raise ValueError(f"P7 {label} changed: {item['path']}")
    return path


def validate_inputs() -> tuple[dict, dict, dict, dict[tuple[str, int], dict]]:
    protocol = read_json(PROTOCOL)
    if protocol.get("status") != "frozen_before_p7_source_analysis":
        raise ValueError("P7 source analysis protocol is not frozen")

    training_path = _validate_receipt(protocol["training_protocol"], "training protocol")
    architecture_path = _validate_receipt(protocol["architecture_profile"], "architecture profile")
    source_path = _validate_receipt(protocol["source_manifest"], "source manifest")
    dataset_path = _validate_receipt(protocol["prepared_dataset_manifest"], "dataset manifest")
    for dependency in protocol["analysis_dependencies"]:
        _validate_receipt(dependency, "analysis dependency")

    training_protocol = read_json(training_path)
    profile = read_json(architecture_path)
    source_manifest = read_json(source_path)
    dataset_manifest = read_json(dataset_path)
    if training_protocol.get("status") != "frozen_before_p7_source_training":
        raise ValueError("P7 training protocol is not frozen")
    if p7_risk_budget(training_protocol) != protocol["risk_budget"]:
        raise ValueError("P7 source risk budget differs from the training freeze")
    costs = corrected_cost_fractions(profile, protocol["dataset"]["num_classes"])
    expected_costs = protocol["cost_model"]
    cost_keys = {
        "final_path_macs": "reference_final_path_macs",
        "exit_path_macs": "deployable_exit_path_macs_including_head",
        "exit_head_macs": "deployable_exit_head_macs",
        "fallback_path_macs": "fallback_path_macs",
    }
    if any(int(costs[key]) != int(expected_costs[frozen]) for key, frozen in cost_keys.items()):
        raise ValueError("P7 corrected cost model differs from the analysis freeze")
    if profile.get("data_or_labels_accessed") is not False:
        raise ValueError("P7 architecture selection accessed data or labels")
    if source_manifest.get("status") != "completed" or source_manifest.get("concurrent_jobs") != 1:
        raise ValueError("P7 source manifest is incomplete or nonserial")
    if source_manifest.get("runtime", {}).get("git_commit") != protocol["training_commit"]:
        raise ValueError("P7 source manifest training commit changed")
    if (
        dataset_manifest.get("status") != "prepared_train_pool"
        or dataset_manifest.get("official_test_prepared") is not False
        or dataset_manifest.get("official_test_accessed") is not False
        or dataset_manifest.get("protocol_sha256") != protocol["training_protocol"]["sha256"]
    ):
        raise ValueError("P7 dataset crossed the frozen official-test boundary")

    runs: dict[tuple[str, int], dict] = {}
    expected_seeds = set(protocol["dataset"]["source_seeds"])
    for run in source_manifest["runs"]:
        config = run["resolved_config"]
        variant = "A0" if config["model_type"] == "mobilenetv2" else "A3"
        seed = int(run["seed"])
        key = (variant, seed)
        if variant not in {"A0", "A3"} or seed not in expected_seeds or key in runs:
            raise ValueError(f"Unexpected P7 source run: {run.get('experiment_id')}")
        if run.get("status") != "completed" or run.get("return_code") != 0:
            raise ValueError(f"Incomplete P7 source run: {run['experiment_id']}")
        if (
            config.get("dataset") != "imagenet100"
            or config.get("evaluate_test") is not False
            or config.get("split_seed") != protocol["dataset"]["split_seed"]
            or config.get("calibration_size") != protocol["dataset"]["calibration_samples_each_seed"]
        ):
            raise ValueError(f"P7 source data boundary changed: {run['experiment_id']}")
        if variant == "A3" and (
            config.get("exit_positions") != [8, 15]
            or config.get("exit_distillation_alpha") != 0.0
        ):
            raise ValueError(f"P7 A3 architecture changed: {run['experiment_id']}")
        run_root = ROOT / "artifacts/runs" / run["experiment_id"]
        summary = read_json(run_root / "summary.json")
        checkpoint = run_root / "checkpoints/model_best.pth"
        if (
            summary.get("test_evaluated") is not False
            or summary.get("calibration_samples") != protocol["dataset"]["calibration_samples_each_seed"]
            or sha256(checkpoint) != summary.get("best_checkpoint_sha256")
        ):
            raise ValueError(f"P7 source checkpoint boundary failed: {run_root}")
        runs[key] = run

    expected = {(variant, seed) for variant in ("A0", "A3") for seed in expected_seeds}
    if set(runs) != expected:
        raise ValueError("P7 source run matrix must be A0/A3 x seeds 77-79")
    for seed in expected_seeds:
        a0_summary = runs[("A0", seed)]["summary"]
        a3_summary = runs[("A3", seed)]["summary"]
        if a0_summary["split_indices_sha256"] != a3_summary["split_indices_sha256"]:
            raise ValueError(f"P7 paired split mismatch for seed {seed}")
    return protocol, training_protocol, profile, runs


def execute(output: Path) -> dict:
    protocol, _training_protocol, profile, runs = validate_inputs()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    logits_dir = output / "calibration_logits"
    logits_dir.mkdir()
    configure_numerics()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or torch.cuda.get_device_name(0) != protocol["numerics"]["device"]:
        raise RuntimeError("P7 source analysis requires the frozen RTX 3080 Ti")

    costs = corrected_cost_fractions(profile, protocol["dataset"]["num_classes"])
    seed_rows: list[dict] = []
    records: list[dict] = []
    final_gains: list[float] = []
    checkpoint_receipts: dict[str, str] = {}
    logits_receipts: dict[str, dict] = {}
    for seed in protocol["dataset"]["source_seeds"]:
        a0 = runs[("A0", seed)]
        a3 = runs[("A3", seed)]
        a0_root = ROOT / "artifacts/runs" / a0["experiment_id"]
        a3_root = ROOT / "artifacts/runs" / a3["experiment_id"]
        a0_summary = read_json(a0_root / "summary.json")
        a3_summary = read_json(a3_root / "summary.json")
        gain = float(a3_summary["best_validation_accuracy"] - a0_summary["best_validation_accuracy"])
        final_gains.append(gain)

        labels, values = _collect_logits(a3, device, split="calibration")
        final_logits, exit8_logits, exit15_logits = unpack_p7_logits(values)
        expected_shape = (protocol["dataset"]["calibration_samples_each_seed"], protocol["dataset"]["num_classes"])
        if final_logits.shape != expected_shape or labels.shape != (expected_shape[0],):
            raise ValueError(f"Unexpected P7 calibration shape for seed {seed}")
        if not all(np.isfinite(values).all() for values in (final_logits, exit8_logits, exit15_logits)):
            raise ValueError(f"Non-finite P7 calibration logits for seed {seed}")

        logits_path = logits_dir / f"imagenet100_a3_seed{seed}.npz"
        np.savez_compressed(
            logits_path,
            labels=labels,
            final_logits=final_logits,
            exit8_logits=exit8_logits,
            exit15_logits=exit15_logits,
        )
        records.append({
            "seed": seed,
            "labels": labels,
            "final_logits": final_logits,
            "exit_logits": exit8_logits,
            "num_classes": protocol["dataset"]["num_classes"],
            "exit_cost_fraction": costs["exit_cost_fraction"],
            "fallback_cost_fraction": costs["fallback_cost_fraction"],
        })
        seed_rows.append({
            "dataset": "imagenet100",
            "seed": seed,
            "a0_experiment_id": a0["experiment_id"],
            "a3_experiment_id": a3["experiment_id"],
            "a0_final_validation_accuracy": a0_summary["best_validation_accuracy"],
            "a3_final_validation_accuracy": a3_summary["best_validation_accuracy"],
            "paired_final_gain": gain,
            "calibration_final_accuracy": float(np.mean(final_logits.argmax(1) == labels)),
            "calibration_exit8_accuracy": float(np.mean(exit8_logits.argmax(1) == labels)),
            "calibration_exit15_accuracy": float(np.mean(exit15_logits.argmax(1) == labels)),
            **best_head_metrics(a3_root, int(a3_summary["best_epoch"])),
        })
        checkpoint_receipts[f"imagenet100/A0/seed{seed}"] = a0_summary["best_checkpoint_sha256"]
        checkpoint_receipts[f"imagenet100/A3/seed{seed}"] = a3_summary["best_checkpoint_sha256"]
        logits_receipts[f"imagenet100/A3/seed{seed}"] = {
            "path": str(logits_path.relative_to(ROOT)),
            "sha256": sha256(logits_path),
            "samples": len(labels),
            "checkpoint_sha256": a3_summary["best_checkpoint_sha256"],
            "split_indices_sha256": a3_summary["split_indices_sha256"],
        }

    selected, frontier = select_shared_threshold(records, protocol["risk_budget"], protocol["selection"])
    final_gate = (
        float(np.mean(final_gains)) >= protocol["final_head_gate"]["paired_a3_minus_a0_mean_minimum"] - 1e-12
        and min(final_gains) >= protocol["final_head_gate"]["paired_a3_minus_a0_each_seed_minimum"] - 1e-12
    )
    policy_gate = selected is not None
    status = "ready_for_p7_target_training" if final_gate and policy_gate else "stop_without_target"
    if selected:
        for row, metrics in zip(seed_rows, selected["source_metrics"]):
            row.update({
                "selected_threshold": selected["threshold"],
                **{f"policy_{key}": value for key, value in metrics.items()},
            })

    write_csv(output / "source_seed_metrics.csv", seed_rows)
    write_csv(output / "threshold_frontier.csv", [{"dataset": "imagenet100", **row} for row in frontier])
    result = {
        "schema_version": 1,
        "status": status,
        "official_test_accessed": False,
        "target_data_accessed": False,
        "training_commit": protocol["training_commit"],
        "analysis_code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "protocol_sha256": sha256(PROTOCOL),
        "source_manifest": protocol["source_manifest"],
        "prepared_dataset_manifest": protocol["prepared_dataset_manifest"],
        "checkpoint_sha256": checkpoint_receipts,
        "calibration_logits": logits_receipts,
        "cost_model": protocol["cost_model"],
        "threshold_candidates_evaluated": len(frontier),
        "decision": {
            "status": status,
            "final_head_gate_passed": final_gate,
            "policy_gate_passed": policy_gate,
            "paired_final_gain_mean": float(np.mean(final_gains)),
            "paired_final_gain_minimum": min(final_gains),
            "selected_policy": selected,
        },
    }
    (output / "source_lock.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# P7 ImageNet-100 source gate",
        "",
        f"Overall status: **{status}**.",
        "",
        (
            f"Mean paired A3-A0 final-validation gain: {100 * np.mean(final_gains):.2f} pp; "
            f"minimum seed gain: {100 * min(final_gains):.2f} pp."
        ),
        "",
    ]
    if selected:
        lines.extend([
            (
                f"Selected shared threshold: {selected['threshold']:.3f}; minimum source-seed MAC saving: "
                f"{100 * selected['minimum_saving']:.2f}%."
            ),
            "",
        ])
    else:
        lines.extend(["No feasible shared threshold was found.", ""])
    lines.append(
        "Selection used only the fixed 10,000-sample source calibration split for seeds 77-79. "
        "No target seed or official ImageNet validation data was accessed.\n"
    )
    (output / "README.md").write_text("\n".join(lines))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.output.resolve()), indent=2))
