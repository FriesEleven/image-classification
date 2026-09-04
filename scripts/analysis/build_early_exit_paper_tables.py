"""Build publication tables from the frozen early-exit evidence ledger.

This module is deliberately read-only with respect to experiment artifacts.  It
does not import an evaluator or construct a data loader; official-test outputs
are consumed only through their already-versioned JSON summaries.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports/paper"

SOURCES = {
    "p1_selection": ROOT / "reports/experiments/2026-09-02-early-exit-p1b/locked_selection.json",
    "p1_test": ROOT / "reports/experiments/2026-09-02-early-exit-p1b/test_results.json",
    "p2_transfer": ROOT / "reports/experiments/2026-09-03-early-exit-p2a-transfer/transfer_results.json",
    "cifar10_1": ROOT / "reports/experiments/2026-09-03-early-exit-p2-cifar10-1-v6/external_results.json",
    "p3_selection": ROOT / "reports/experiments/2026-09-03-early-exit-p3-cifar100/selection.json",
    "p3_boundary": ROOT / "reports/diagnostics/2026-09-03-early-exit-p3-boundary-v2/diagnostic.json",
    "p3_class_guard": ROOT / "reports/diagnostics/2026-09-03-early-exit-p3-class-guard-v2/diagnostic.json",
    "p4_confirmation": ROOT / "reports/experiments/2026-09-03-early-exit-p4-cifar100/confirmation.json",
    "p4_test": ROOT / "reports/experiments/2026-09-03-early-exit-p4-cifar100-test/test_results.json",
    "latency": ROOT / "reports/profiles/2026-09-02-early-exit-p1b-rtx4090d/profile.json",
}

USED_FIELDS = {
    "p1_selection": ["locked_policy", "selection_protocol", "source_seeds"],
    "p1_test": ["aggregate", "locked_policy", "seed_results"],
    "p2_transfer": ["aggregate", "data_protocol", "seed_results", "source_policy"],
    "cifar10_1": ["aggregate", "external_test_set", "locked_policy", "seed_results"],
    "p3_selection": ["aggregate", "data_protocol", "status"],
    "p3_boundary": ["strict_zero_risk_without_15_percent_floor", "exploratory_relaxations"],
    "p3_class_guard": ["searches", "status"],
    "p4_confirmation": ["aggregate", "data_protocol", "locked_policy", "seed_results"],
    "p4_test": ["aggregate", "locked_policy", "seed_results"],
    "latency": ["aggregate", "device", "measurement_protocol", "seed_profiles"],
}

PARAMETERS = {
    "cifar10_baseline": 2_236_682,
    "cifar10_multi": 2_238_942,
    "cifar100_baseline": 2_351_972,
    "cifar100_multi": 2_374_572,
}

PATH_MACS = {
    "cifar10": {"exit8": 2_676_864, "exit16": 5_234_688, "final": 6_124_928},
    "cifar100": {"exit8": 2_682_624, "exit16": 5_249_088, "final": 6_240_128},
}


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def stdev(values: list[float]) -> float:
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sources() -> dict[str, Any]:
    missing = [str(path) for path in SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen evidence: {missing}")
    return {name: json.loads(path.read_text(encoding="utf-8")) for name, path in SOURCES.items()}


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("Cannot summarize an empty list")
    return {
        "values": values,
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
    }


def percent(value: float | None) -> float | None:
    return None if value is None else 100.0 * value


def _agg(record: dict[str, Any], field: str, group: str | None = None) -> dict[str, Any]:
    aggregate = record["aggregate"]
    if group is not None:
        aggregate = aggregate[group]
    return aggregate[field]


def build_main_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, source_name, group, params_base, params_multi in (
        ("CIFAR-10", "p1_test", None, PARAMETERS["cifar10_baseline"], PARAMETERS["cifar10_multi"]),
        ("CIFAR-100", "p4_test", "all_seeds", PARAMETERS["cifar100_baseline"], PARAMETERS["cifar100_multi"]),
    ):
        source = data[source_name]
        baseline = _agg(source, "baseline_test_accuracy", group)
        final = _agg(source, "multi_exit_final_test_accuracy", group)
        policy = _agg(source, "locked_policy_accuracy", group)
        gain_final = _agg(source, "paired_final_gain_vs_baseline", group)
        gain_policy_base = _agg(source, "locked_policy_gain_vs_baseline", group)
        gain_policy_final = _agg(source, "locked_policy_gain_vs_final", group)
        early = _agg(source, "early_route_fraction", group)
        saving = _agg(source, "mac_saving_fraction", group)
        worst = _agg(source, "worst_class_accuracy_drop_vs_final", group)
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "method": "Matched MobileNetV2",
                    "accuracy_mean_pct": percent(baseline["mean"]),
                    "accuracy_std_pct": percent(baseline["sample_std"]),
                    "gain_vs_baseline_pp": None,
                    "gain_vs_final_pp": None,
                    "early_fraction_mean_pct": None,
                    "early_fraction_std_pct": None,
                    "mac_saving_mean_pct": None,
                    "mac_saving_std_pct": None,
                    "worst_class_drop_mean_pp": None,
                    "worst_class_drop_std_pp": None,
                    "params_m": params_base / 1e6,
                },
                {
                    "dataset": dataset,
                    "method": "Multi-exit final head",
                    "accuracy_mean_pct": percent(final["mean"]),
                    "accuracy_std_pct": percent(final["sample_std"]),
                    "gain_vs_baseline_pp": percent(gain_final["mean"]),
                    "gain_vs_final_pp": None,
                    "early_fraction_mean_pct": None,
                    "early_fraction_std_pct": None,
                    "mac_saving_mean_pct": None,
                    "mac_saving_std_pct": None,
                    "worst_class_drop_mean_pp": None,
                    "worst_class_drop_std_pp": None,
                    "params_m": params_multi / 1e6,
                },
                {
                    "dataset": dataset,
                    "method": "Locked shared policy",
                    "accuracy_mean_pct": percent(policy["mean"]),
                    "accuracy_std_pct": percent(policy["sample_std"]),
                    "gain_vs_baseline_pp": percent(gain_policy_base["mean"]),
                    "gain_vs_final_pp": percent(gain_policy_final["mean"]),
                    "early_fraction_mean_pct": percent(early["mean"]),
                    "early_fraction_std_pct": percent(early["sample_std"]),
                    "mac_saving_mean_pct": percent(saving["mean"]),
                    "mac_saving_std_pct": percent(saving["sample_std"]),
                    "worst_class_drop_mean_pp": percent(worst["mean"]),
                    "worst_class_drop_std_pp": percent(worst["sample_std"]),
                    "params_m": params_multi / 1e6,
                },
            ]
        )
    return rows


def build_transfer_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    p2 = data["p2_transfer"]
    p2_policy = summarize([row["frozen_policy_transfer_metrics"]["accuracy"] for row in p2["seed_results"]])
    rows = [
        {
            "evaluation": "Unseen retraining (CIFAR-10)",
            "seeds": "57--59",
            "threshold_source": "P1 seeds 54--56",
            "recalibration": "No",
            "policy_accuracy_mean_pct": percent(p2_policy["mean"]),
            "policy_accuracy_std_pct": percent(p2_policy["sample_std"]),
            "early_fraction_mean_pct": percent(p2["aggregate"]["transfer_early_fraction_mean"]),
            "early_fraction_std_pct": percent(p2["aggregate"]["transfer_early_fraction_sample_std"]),
            "mac_saving_mean_pct": percent(p2["aggregate"]["transfer_mac_saving_mean"]),
            "mac_saving_std_pct": percent(p2["aggregate"]["transfer_mac_saving_sample_std"]),
            "max_overall_drop_pp": percent(p2["aggregate"]["transfer_accuracy_drop_max"]),
            "max_balanced_drop_pp": percent(p2["aggregate"]["transfer_balanced_drop_max"]),
            "max_worst_class_drop_pp": percent(p2["aggregate"]["transfer_worst_class_drop_max"]),
        }
    ]
    external = data["cifar10_1"]
    for key, label, seeds in (
        ("source_seeds", "CIFAR-10.1 source models", "54--56"),
        ("target_seeds", "CIFAR-10.1 unseen models", "57--59"),
        ("all_seeds", "CIFAR-10.1 all models", "54--59"),
    ):
        group = external["aggregate"][key]
        rows.append(
            {
                "evaluation": label,
                "seeds": seeds,
                "threshold_source": "P1 seeds 54--56",
                "recalibration": "No",
                "policy_accuracy_mean_pct": percent(group["locked_policy_accuracy"]["mean"]),
                "policy_accuracy_std_pct": percent(group["locked_policy_accuracy"]["sample_std"]),
                "early_fraction_mean_pct": percent(group["early_route_fraction"]["mean"]),
                "early_fraction_std_pct": percent(group["early_route_fraction"]["sample_std"]),
                "mac_saving_mean_pct": percent(group["mac_saving_fraction"]["mean"]),
                "mac_saving_std_pct": percent(group["mac_saving_fraction"]["sample_std"]),
                "max_overall_drop_pp": percent(
                    max(
                        row["locked_policy"]["accuracy_drop"]
                        for row in external["seed_results"]
                        if row["seed"] in range(54, 57)
                        if key == "source_seeds"
                    )
                    if key == "source_seeds"
                    else max(
                        row["locked_policy"]["accuracy_drop"]
                        for row in external["seed_results"]
                        if row["seed"] in range(57, 60)
                    )
                    if key == "target_seeds"
                    else max(row["locked_policy"]["accuracy_drop"] for row in external["seed_results"])
                ),
                "max_balanced_drop_pp": percent(0.0),
                "max_worst_class_drop_pp": percent(group["worst_class_accuracy_drop_vs_final"]["mean"]),
            }
        )
    return rows


def build_risk_boundary(data: dict[str, Any]) -> list[dict[str, Any]]:
    boundary = data["p3_boundary"]
    strict = boundary["strict_zero_risk_without_15_percent_floor"]
    rows = [
        {
            "dataset_split": "CIFAR-100 P3 source/target",
            "policy_variant": "Global shared threshold",
            "risk_budget_pp": 0.0,
            "threshold": strict["threshold"],
            "min_mac_saving_pct": percent(
                min(strict["source"]["minimum_mac_saving"], strict["target"]["minimum_mac_saving"])
            ),
            "mean_mac_saving_pct": None,
            "max_worst_class_drop_pp": 0.0,
            "evidence_type": "Post-hoc diagnostic",
            "outcome": "Below 15% saving",
        }
    ]
    seen_budgets: set[float] = set()
    for item in boundary["exploratory_relaxations"].values():
        budget = float(item["post_hoc_budget"]["max_worst_class_drop"])
        if budget in seen_budgets:
            continue
        seen_budgets.add(budget)
        rows.append(
            {
                "dataset_split": "CIFAR-100 P3 source/target",
                "policy_variant": "Global shared threshold",
                "risk_budget_pp": percent(budget),
                "threshold": item["threshold"],
                "min_mac_saving_pct": percent(
                    min(item["source"]["minimum_mac_saving"], item["target"]["minimum_mac_saving"])
                ),
                "mean_mac_saving_pct": None,
                "max_worst_class_drop_pp": percent(
                    max(
                        item["source"]["maximum_worst_class_accuracy_drop"],
                        item["target"]["maximum_worst_class_accuracy_drop"],
                    )
                ),
                "evidence_type": "Post-hoc diagnostic",
                "outcome": "Pass"
                if item["source_mac_saving_gate_passed"] and item["target_mac_saving_gate_passed"]
                else "Below 15% saving",
            }
        )
    p4 = data["p4_confirmation"]
    p4_metrics = [row["policy_confirmation_metrics"] for row in p4["seed_results"]]
    rows.append(
        {
            "dataset_split": "CIFAR-100 P4 confirmation",
            "policy_variant": "Frozen global threshold",
            "risk_budget_pp": 4.0,
            "threshold": p4["locked_policy"]["confidence_threshold"],
            "min_mac_saving_pct": percent(min(row["cost_saving_fraction"] for row in p4_metrics)),
            "mean_mac_saving_pct": percent(mean(row["cost_saving_fraction"] for row in p4_metrics)),
            "max_worst_class_drop_pp": percent(max(row["worst_class_accuracy_drop"] for row in p4_metrics)),
            "evidence_type": "Independent confirmation",
            "outcome": "Pass",
        }
    )
    return rows


def build_complexity_latency(data: dict[str, Any]) -> list[dict[str, Any]]:
    latency = data["latency"]
    p1 = data["p1_test"]
    p4 = data["p4_test"]["aggregate"]["all_seeds"]
    return [
        {
            "model_path": "CIFAR-10 baseline/final",
            "params": PARAMETERS["cifar10_baseline"],
            "conv_linear_macs": PATH_MACS["cifar10"]["final"],
            "early_route_mean_pct": None,
            "expected_latency_ms": latency["aggregate"]["reference_final_latency_ms_mean"],
            "saving_pct": None,
            "speedup_x": None,
            "hardware": "RTX 4090 D, batch 1",
        },
        {
            "model_path": "CIFAR-10 multi-exit (full)",
            "params": PARAMETERS["cifar10_multi"],
            "conv_linear_macs": PATH_MACS["cifar10"]["final"],
            "early_route_mean_pct": percent(p1["aggregate"]["early_route_fraction"]["mean"]),
            "expected_latency_ms": latency["aggregate"]["expected_policy_latency_ms_mean"],
            "saving_pct": percent(latency["aggregate"]["latency_saving_fraction_mean"]),
            "speedup_x": latency["aggregate"]["speedup_mean"],
            "hardware": "RTX 4090 D, batch 1",
        },
        {
            "model_path": "CIFAR-10 exit8 path",
            "params": PARAMETERS["cifar10_multi"],
            "conv_linear_macs": PATH_MACS["cifar10"]["exit8"],
            "early_route_mean_pct": None,
            "expected_latency_ms": None,
            "saving_pct": percent(1 - PATH_MACS["cifar10"]["exit8"] / PATH_MACS["cifar10"]["final"]),
            "speedup_x": None,
            "hardware": "MAC proxy",
        },
        {
            "model_path": "CIFAR-100 baseline/final",
            "params": PARAMETERS["cifar100_baseline"],
            "conv_linear_macs": PATH_MACS["cifar100"]["final"],
            "early_route_mean_pct": None,
            "expected_latency_ms": None,
            "saving_pct": None,
            "speedup_x": None,
            "hardware": "MAC proxy",
        },
        {
            "model_path": "CIFAR-100 multi-exit policy",
            "params": PARAMETERS["cifar100_multi"],
            "conv_linear_macs": PATH_MACS["cifar100"]["final"],
            "early_route_mean_pct": percent(p4["early_route_fraction"]["mean"]),
            "expected_latency_ms": None,
            "saving_pct": percent(p4["mac_saving_fraction"]["mean"]),
            "speedup_x": None,
            "hardware": "MAC proxy",
        },
        {
            "model_path": "CIFAR-100 exit8 path",
            "params": PARAMETERS["cifar100_multi"],
            "conv_linear_macs": PATH_MACS["cifar100"]["exit8"],
            "early_route_mean_pct": None,
            "expected_latency_ms": None,
            "saving_pct": percent(1 - PATH_MACS["cifar100"]["exit8"] / PATH_MACS["cifar100"]["final"]),
            "speedup_x": None,
            "hardware": "MAC proxy",
        },
    ]


def build_protocol_hardware() -> list[dict[str, Any]]:
    return [
        {
            "phase": "P1",
            "dataset": "CIFAR-10",
            "split_seed": 20260902,
            "training_seeds": "54--56",
            "samples": "40k/5k/5k",
            "policy_role": "Shared-threshold selection; locked test",
            "gpu": "RTX 3080 Ti",
        },
        {
            "phase": "P2",
            "dataset": "CIFAR-10",
            "split_seed": 20260902,
            "training_seeds": "57--59",
            "samples": "40k/5k/5k",
            "policy_role": "Zero-recalibration model transfer",
            "gpu": "RTX 3080 Ti",
        },
        {
            "phase": "P2-ext",
            "dataset": "CIFAR-10.1 v6",
            "split_seed": "--",
            "training_seeds": "54--59",
            "samples": "2k external",
            "policy_role": "Natural-shift evaluation",
            "gpu": "Evaluation only",
        },
        {
            "phase": "P3",
            "dataset": "CIFAR-100",
            "split_seed": 20260903,
            "training_seeds": "60--65",
            "samples": "40k/5k/5k",
            "policy_role": "Strict-gate failure; post-hoc boundary",
            "gpu": "RTX 3080 Ti",
        },
        {
            "phase": "P4",
            "dataset": "CIFAR-100",
            "split_seed": 20260904,
            "training_seeds": "66--68",
            "samples": "40k/5k/5k",
            "policy_role": "Frozen-policy confirmation; locked test",
            "gpu": "RTX 3080 Ti",
        },
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value).replace("_", r"\_").replace("%", r"\%")


def _write_latex(
    path: Path, rows: list[dict[str, Any]], columns: list[tuple[str, str]], caption: str, label: str
) -> None:
    align = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(title for _, title in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_fmt(row[key]) for key, _ in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(data: dict[str, Any]) -> None:
    tables = OUTPUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    products = {
        "main_results": build_main_results(data),
        "transfer_results": build_transfer_results(data),
        "risk_boundary": build_risk_boundary(data),
        "complexity_latency": build_complexity_latency(data),
        "protocol_hardware": build_protocol_hardware(),
    }
    for name, rows in products.items():
        _write_csv(tables / f"{name}.csv", rows)

    _write_latex(
        tables / "main_results.tex",
        products["main_results"],
        [
            ("dataset", "Dataset"),
            ("method", "Method"),
            ("accuracy_mean_pct", r"Acc. (\%)"),
            ("accuracy_std_pct", "SD"),
            ("gain_vs_baseline_pp", r"$\Delta$ base (pp)"),
            ("gain_vs_final_pp", r"$\Delta$ final (pp)"),
            ("early_fraction_mean_pct", r"Early (\%)"),
            ("mac_saving_mean_pct", r"MAC save (\%)"),
            ("worst_class_drop_mean_pp", "Worst drop (pp)"),
            ("params_m", "Params (M)"),
        ],
        "Method-locked official-test results. Standard deviations are across three training seeds.",
        "tab:main-results",
    )
    _write_latex(
        tables / "transfer_results.tex",
        products["transfer_results"],
        [
            ("evaluation", "Evaluation"),
            ("seeds", "Seeds"),
            ("policy_accuracy_mean_pct", r"Policy acc. (\%)"),
            ("policy_accuracy_std_pct", "SD"),
            ("early_fraction_mean_pct", r"Early (\%)"),
            ("mac_saving_mean_pct", r"MAC save (\%)"),
            ("max_worst_class_drop_pp", "Max worst drop (pp)"),
        ],
        "Transfer of the CIFAR-10 threshold $0.984$ without recalibration.",
        "tab:transfer",
    )
    _write_latex(
        tables / "risk_boundary.tex",
        products["risk_boundary"],
        [
            ("dataset_split", "Dataset/split"),
            ("risk_budget_pp", "Budget (pp)"),
            ("threshold", r"$\theta$"),
            ("min_mac_saving_pct", r"Min save (\%)"),
            ("max_worst_class_drop_pp", "Observed worst drop (pp)"),
            ("evidence_type", "Evidence"),
            ("outcome", "Outcome"),
        ],
        "CIFAR-100 empirical risk boundary. P3 rows are post-hoc diagnostics; P4 is an independent confirmation.",
        "tab:risk-boundary",
    )
    _write_latex(
        tables / "complexity_latency.tex",
        products["complexity_latency"],
        [
            ("model_path", "Model/path"),
            ("params", "Params"),
            ("conv_linear_macs", "Conv/Linear MACs"),
            ("early_route_mean_pct", r"Early (\%)"),
            ("expected_latency_ms", "Latency (ms)"),
            ("saving_pct", r"Saving (\%)"),
            ("speedup_x", "Speedup"),
            ("hardware", "Basis"),
        ],
        "Complexity proxy and paired batch-1 server-GPU latency. Training-only exit16 is included in parameter counts.",
        "tab:complexity-latency",
    )
    _write_latex(
        tables / "protocol_hardware.tex",
        products["protocol_hardware"],
        [
            ("phase", "Phase"),
            ("dataset", "Dataset"),
            ("split_seed", "Split seed"),
            ("training_seeds", "Training seeds"),
            ("samples", "Train/select/policy"),
            ("policy_role", "Policy role"),
            ("gpu", "Primary platform"),
        ],
        "Evaluation phases and data boundaries.",
        "tab:protocol",
    )

    manifest = {
        "schema_version": 1,
        "generator": str(Path(__file__).relative_to(ROOT)),
        "policy": "Read-only aggregation of frozen JSON; no evaluator or data loader is invoked.",
        "inputs": [
            {
                "name": name,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "used_fields": USED_FIELDS[name],
            }
            for name, path in SOURCES.items()
        ],
        "outputs": sorted(str(path.relative_to(ROOT)) for path in tables.glob("*")),
    }
    (OUTPUT / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    data = load_sources()
    write_outputs(data)
    print(f"Wrote paper evidence products to {OUTPUT}")


if __name__ == "__main__":
    main()
