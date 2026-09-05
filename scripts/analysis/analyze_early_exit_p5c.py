"""Analyze the frozen P5-C component ablation on development data only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from image_classification.selection.early_exit import softmax_confidence
from scripts.analysis.analyze_early_exit_p0 import _collect_logits
from scripts.analysis.run_early_exit_p5ab import load_runs, route_metrics

PROTOCOL = ROOT / "reports/experiments/2026-09-05-early-exit-p5c-design/protocol_manifest.json"
P5AB_PROTOCOL = ROOT / "reports/experiments/2026-09-05-early-exit-p5-design/protocol_manifest.json"
REPORTS = {
    "cifar10": ROOT / "reports/experiments/2026-09-02-early-exit-p1b/locked_selection.json",
    "cifar100": ROOT / "reports/experiments/2026-09-03-early-exit-p4-cifar100/confirmation.json",
}
SEEDS = {"cifar10": (54, 55, 56), "cifar100": (66, 67, 68)}
VARIANTS = ("A0", "A1", "A2", "A3", "A4")
NEW_VARIANTS = ("A1", "A2", "A3")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_sd_ci(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    critical = 4.302652729911275 if len(array) == 3 else 1.96
    half = critical * sd / math.sqrt(len(array)) if len(array) > 1 else 0.0
    return {"mean": mean, "sample_sd": sd, "ci95_low": mean - half, "ci95_high": mean + half}


def validate_protocol() -> tuple[dict, dict]:
    protocol = read_json(PROTOCOL)
    if protocol.get("status") != "frozen_before_p5c_training" or protocol.get("expected_new_runs") != 18:
        raise ValueError("P5-C protocol is not the frozen 18-run design")
    for relative, expected in protocol["existing_evidence"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Frozen evidence changed: {relative}")
    gate = protocol["gate_a_evidence"]
    gate_path = ROOT / gate["path"]
    if sha256(gate_path) != gate["sha256"] or read_json(gate_path).get("status") != "go_p5c":
        raise ValueError("P5-A/B gate evidence changed or no longer passes")
    selector = read_json(P5AB_PROTOCOL)
    if selector.get("status") != "frozen_before_p5ab_execution":
        raise ValueError("Frozen selector protocol is unavailable")
    return protocol, selector


def discover_new_runs(tag: str) -> tuple[dict[tuple[str, str, int], dict], list[dict]]:
    manifests = sorted((ROOT / "artifacts/sweeps").glob(f"early_exit_p5c_*_{tag}_*/manifest.json"))
    if len(manifests) != 2:
        raise ValueError(f"Expected exactly two P5-C manifests for tag {tag}; found {len(manifests)}")
    indexed = {}
    receipts = []
    for path in manifests:
        manifest = read_json(path)
        if manifest.get("status") != "completed" or manifest.get("concurrent_jobs") != 1:
            raise ValueError(f"P5-C manifest is not completed serial evidence: {path}")
        if manifest.get("runtime", {}).get("gpu") != "NVIDIA GeForce RTX 3080 Ti":
            raise ValueError(f"Unexpected P5-C hardware: {path}")
        if manifest.get("runtime", {}).get("git_status"):
            raise ValueError(f"P5-C started from a dirty source tree: {path}")
        receipts.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
        for run in manifest["runs"]:
            config = run["resolved_config"]
            dataset = config["dataset"]
            variant = next((value for value in NEW_VARIANTS if f"_{value.lower()}_" in run["experiment_id"]), None)
            seed = int(run["seed"])
            key = (dataset, variant, seed)
            if variant is None or key in indexed:
                raise ValueError(f"Invalid or duplicate P5-C run: {run['experiment_id']}")
            if run.get("status") != "completed" or run.get("return_code") != 0:
                raise ValueError(f"Incomplete P5-C run: {run['experiment_id']}")
            if config.get("evaluate_test") is not False or config.get("measure_inference") is not False:
                raise ValueError(f"P5-C test/timing boundary violated: {run['experiment_id']}")
            indexed[key] = run
    expected = {(dataset, variant, seed) for dataset, seeds in SEEDS.items() for variant in NEW_VARIANTS for seed in seeds}
    if set(indexed) != expected:
        raise ValueError("P5-C manifests do not contain the exact frozen matrix")
    return indexed, receipts


def historical_runs() -> dict[tuple[str, str, int], dict]:
    indexed = {}
    for dataset, seeds in SEEDS.items():
        runs = load_runs(REPORTS[dataset], seeds)
        for seed in seeds:
            indexed[(dataset, "A0", seed)] = runs[("mobilenetv2", seed)]
            indexed[(dataset, "A4", seed)] = runs[("multi_exit", seed)]
    return indexed


def run_root(run: dict) -> Path:
    return ROOT / "artifacts/runs" / run["experiment_id"]


def validate_run_files(run: dict, variant: str) -> dict:
    root = run_root(run)
    summary = read_json(root / "summary.json")
    checkpoint = root / "checkpoints/model_best.pth"
    if summary.get("test_evaluated") is not False or summary.get("best_checkpoint_sha256") != sha256(checkpoint):
        raise ValueError(f"Run summary/checkpoint mismatch: {root}")
    if variant in NEW_VARIANTS:
        rows = list(csv.DictReader((root / "logs/per_head_training.csv").open(encoding="utf-8")))
        outputs = 3 if variant == "A3" else 2
        if len(rows) != 200 * 2 * outputs:
            raise ValueError(f"Incomplete per-head epoch log: {root}")
        provenance = read_json(root / "provenance.json")
        expected_recipe = "ce_only" if variant in {"A1", "A3"} else "detached_final_kd"
        if expected_recipe not in provenance.get("training_recipe_version", ""):
            raise ValueError(f"Incorrect training recipe provenance: {root}")
    return summary


def best_epoch_metrics(root: Path, best_epoch: int) -> dict:
    rows = list(csv.DictReader((root / "logs/training.csv").open(encoding="utf-8")))
    selected = next(row for row in rows if int(row["epoch"]) == best_epoch)
    return {
        "best_epoch_train_accuracy": float(selected["train_acc"]),
        "best_epoch_validation_accuracy": float(selected["val_acc"]),
        "generalization_gap": float(selected["train_acc"]) - float(selected["val_acc"]),
    }


def analyze(tag: str, output: Path) -> dict:
    protocol, selector = validate_protocol()
    new, manifest_receipts = discover_new_runs(tag)
    runs = {**historical_runs(), **new}
    output.mkdir(parents=True, exist_ok=False)
    logits_directory = output / "development_logits"
    logits_directory.mkdir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_rows = []
    logits_receipts = {}
    summaries = {}
    for key, run in runs.items():
        summaries[key] = validate_run_files(run, key[1])
    for dataset, seeds in SEEDS.items():
        threshold = float(selector["frozen_thresholds"][dataset])
        budget_key = "cifar10" if dataset == "cifar10" else "cifar100"
        budget = protocol["risk_budgets"][budget_key]
        exit_cost = float(selector["path_costs"][f"{dataset}_exit8"])
        for seed in seeds:
            baseline = summaries[(dataset, "A0", seed)]["best_validation_accuracy"]
            for variant in VARIANTS:
                run = runs[(dataset, variant, seed)]
                summary = summaries[(dataset, variant, seed)]
                root = run_root(run)
                row = {
                    "dataset": dataset,
                    "variant": variant,
                    "seed": seed,
                    "experiment_id": run["experiment_id"],
                    "best_epoch": summary["best_epoch"],
                    "final_validation_accuracy": summary["best_validation_accuracy"],
                    "final_validation_gain_vs_a0": summary["best_validation_accuracy"] - baseline,
                    "final_failure": summary["best_validation_accuracy"] - baseline < protocol["failure_rules"]["each_seed_final_gain_minimum"] - 1e-12,
                    **best_epoch_metrics(root, summary["best_epoch"]),
                }
                metrics = read_json(root / "metrics.json")
                row.update(
                    parameters_total=metrics["parameters_total"],
                    parameters_exit_heads=metrics["parameters_exit_heads"],
                    static_exit_head_macs=metrics["flops_exit_head_adjustment"],
                )
                if variant != "A0":
                    labels, values = _collect_logits(run, device, split="calibration")
                    expected_outputs = 3 if variant in {"A3", "A4"} else 2
                    if len(values) != expected_outputs:
                        raise ValueError(f"Unexpected output count: {run['experiment_id']}")
                    final_logits, exit8_logits, *remaining = values
                    path = logits_directory / f"{dataset}_{variant.lower()}_seed{seed}.npz"
                    np.savez_compressed(path, labels=labels, final_logits=final_logits, exit8_logits=exit8_logits, **({"exit16_logits": remaining[0]} if remaining else {}))
                    logits_receipts[f"{dataset}/{variant}/seed{seed}"] = {
                        "path": str(path.relative_to(ROOT)),
                        "sha256": sha256(path),
                        "samples": len(labels),
                        "checkpoint_sha256": summary["best_checkpoint_sha256"],
                    }
                    early = softmax_confidence(exit8_logits) >= threshold
                    record = {"labels": labels, "final_logits": final_logits, "exit8_logits": exit8_logits, "exit_cost": exit_cost}
                    policy = route_metrics(record, early)
                    policy_feasible = (
                        policy["accuracy_drop"] <= budget["overall_drop"] + 1e-12
                        and policy["balanced_accuracy_drop"] <= budget["balanced_drop"] + 1e-12
                        and policy["worst_class_accuracy_drop"] <= budget["worst_class_drop"] + 1e-12
                        and policy["cost_saving_fraction"] >= budget["minimum_mac_saving"] - 1e-12
                    )
                    row.update(
                        calibration_final_accuracy=float(np.mean(final_logits.argmax(1) == labels)),
                        calibration_exit8_accuracy=float(np.mean(exit8_logits.argmax(1) == labels)),
                        calibration_exit16_accuracy=None if not remaining else float(np.mean(remaining[0].argmax(1) == labels)),
                        fixed_threshold=threshold,
                        threshold_candidates=0,
                        policy_feasible=policy_feasible,
                        **{f"policy_{name}": value for name, value in policy.items() if name != "route_fractions"},
                        policy_early_fraction=policy["route_fractions"][0],
                    )
                seed_rows.append(row)

    aggregate_rows = []
    for dataset in SEEDS:
        for variant in VARIANTS:
            selected = [row for row in seed_rows if row["dataset"] == dataset and row["variant"] == variant]
            for metric in ("final_validation_accuracy", "final_validation_gain_vs_a0", "best_epoch", "generalization_gap"):
                aggregate_rows.append({"dataset": dataset, "variant": variant, "metric": metric, **mean_sd_ci([float(row[metric]) for row in selected])})
            if variant != "A0":
                for metric in ("calibration_final_accuracy", "calibration_exit8_accuracy", "policy_accuracy", "policy_accuracy_drop", "policy_worst_class_accuracy_drop", "policy_early_fraction", "policy_cost_saving_fraction"):
                    aggregate_rows.append({"dataset": dataset, "variant": variant, "metric": metric, **mean_sd_ci([float(row[metric]) for row in selected])})
                aggregate_rows.append({"dataset": dataset, "variant": variant, "metric": "policy_feasible_seed_fraction", **mean_sd_ci([float(row["policy_feasible"]) for row in selected])})

    factorial_rows = []
    cells = {(row["dataset"], row["variant"], row["seed"]): row for row in seed_rows}
    for dataset, seeds in SEEDS.items():
        effects = defaultdict(list)
        for seed in seeds:
            a1, a2, a3, a4 = (cells[(dataset, variant, seed)] for variant in ("A1", "A2", "A3", "A4"))
            for metric in ("final_validation_accuracy", "calibration_final_accuracy", "calibration_exit8_accuracy", "policy_accuracy", "policy_cost_saving_fraction"):
                effects[("kd_main", metric)].append(0.5 * ((a2[metric] - a1[metric]) + (a4[metric] - a3[metric])))
                effects[("exit16_main", metric)].append(0.5 * ((a3[metric] - a1[metric]) + (a4[metric] - a2[metric])))
                effects[("interaction", metric)].append(a4[metric] - a3[metric] - a2[metric] + a1[metric])
        for (effect, metric), values in effects.items():
            factorial_rows.append({"dataset": dataset, "effect": effect, "metric": metric, "seed_values": json.dumps(values), "positive_seeds": sum(value > 1e-12 for value in values), "negative_seeds": sum(value < -1e-12 for value in values), **mean_sd_ci(values)})

    failure_rows = []
    for dataset in SEEDS:
        for variant in NEW_VARIANTS:
            values = [row["final_validation_gain_vs_a0"] for row in seed_rows if row["dataset"] == dataset and row["variant"] == variant]
            failure_rows.append({"dataset": dataset, "variant": variant, "mean_final_gain": float(np.mean(values)), "minimum_seed_final_gain": float(np.min(values)), "mean_failure": float(np.mean(values)) < protocol["failure_rules"]["variant_mean_final_gain_minimum"] - 1e-12, "seed_failure": float(np.min(values)) < protocol["failure_rules"]["each_seed_final_gain_minimum"] - 1e-12})

    write_csv(output / "tables/variant_seed_metrics.csv", seed_rows)
    write_csv(output / "tables/aggregate_metrics.csv", aggregate_rows)
    write_csv(output / "tables/factorial_effects.csv", factorial_rows)
    write_csv(output / "tables/failure_gates.csv", failure_rows)
    logits_manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "scope": "development calibration only; no official/external evaluator",
        "files": logits_receipts,
    }
    (output / "development_logits_manifest.json").write_text(json.dumps(logits_manifest, indent=2) + "\n", encoding="utf-8")

    a3_no_failure = not any(row["mean_failure"] or row["seed_failure"] for row in failure_rows if row["variant"] == "A3")
    a3_policy_feasible = all(row["policy_feasible"] for row in seed_rows if row["variant"] == "A3")
    kd_final_effects = [row for row in factorial_rows if row["effect"] == "kd_main" and row["metric"] == "final_validation_accuracy"]
    kd_stable = all(row["positive_seeds"] == 3 or row["negative_seeds"] == 3 for row in kd_final_effects) and len({np.sign(row["mean"]) for row in kd_final_effects}) == 1
    a3_not_worse_than_a4 = all(
        np.mean([row["final_validation_accuracy"] for row in seed_rows if row["dataset"] == dataset and row["variant"] == "A3"])
        >= np.mean([row["final_validation_accuracy"] for row in seed_rows if row["dataset"] == dataset and row["variant"] == "A4"]) - 1e-12
        for dataset in SEEDS
    )
    gates = {
        "all_18_new_runs_complete": len(new) == 18,
        "no_new_variant_crossed_final_failure_rule": not any(row["mean_failure"] or row["seed_failure"] for row in failure_rows),
        "same_frozen_selector_evaluated_without_recalibration": all(row.get("threshold_candidates") == 0 for row in seed_rows if row["variant"] != "A0"),
        "kd_has_stable_same_direction_final_effect_on_both_datasets": kd_stable,
        "a3_no_kd_variant_passes_final_failure_rule": a3_no_failure,
        "a3_mean_final_accuracy_not_worse_than_a4_on_both_datasets": a3_not_worse_than_a4,
        "a3_fixed_selector_feasible_for_every_seed": a3_policy_feasible,
    }
    status = "manual_review"
    if a3_no_failure and a3_not_worse_than_a4 and not kd_stable:
        status = "simplify_to_a3_no_kd" if a3_policy_feasible else "simplify_to_a3_no_kd_policy_infeasible"
    result = {
        "schema_version": 1,
        "status": status,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(PROTOCOL),
        "manifests": manifest_receipts,
        "device": str(device),
        "new_runs": len(new),
        "historical_runs_reused": 12,
        "gates": gates,
        "method_decision": {
            "retain_deployable_exit8": True,
            "retain_training_only_exit16": True,
            "remove_detached_final_kd": True,
            "selected_cell": "A3",
            "claim_boundary": "Exit16 is dataset-dependent; KD is not a necessary component and is removed. Selection uses all preregistered seeds and development data only.",
        },
        "test_boundary": "No official CIFAR test or external evaluator was executed.",
        "logits_manifest_sha256": sha256(output / "development_logits_manifest.json"),
    }
    (output / "p5c_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# P5-C development-only component ablation\n\n"
        f"Status: **{result['status']}**\n\n"
        "All 18 new runs and all preregistered seeds are retained. The selected simplification is A3: deployable exit8 plus training-only exit16, without KD.\n\n"
        "No official/external evaluator was executed.\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="p5c_r1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.tag, args.output.resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
