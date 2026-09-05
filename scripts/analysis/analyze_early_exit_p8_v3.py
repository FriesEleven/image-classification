"""Audit and aggregate the completed P8 v3 dynamic-routing benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
T_CRITICAL_DF2_975 = 4.302652729696142


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def summarize(values: list[float]) -> dict[str, float]:
    if len(values) != 3:
        raise ValueError(f"Expected three seed units, found {len(values)}")
    center = mean(values)
    sample_sd = float(np.std(np.asarray(values, dtype=np.float64), ddof=1))
    half_width = T_CRITICAL_DF2_975 * sample_sd / math.sqrt(len(values))
    return {
        "mean": center,
        "sample_sd": sample_sd,
        "ci95_low": center - half_width,
        "ci95_high": center + half_width,
    }


def close(a: float, b: float, *, atol: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=atol)


def audit(input_dir: Path, rows: list[dict[str, str]]) -> dict:
    errors: list[str] = []
    cells: dict[tuple, dict[str, dict]] = defaultdict(dict)
    raw_paths: set[Path] = set()
    workload_hashes: dict[Path, str] = {}
    latency_observations = route_observations = 0
    for row in rows:
        key = (row["device"], row["dataset"], int(row["seed"]),
               int(row["batch_size"]), int(row["round"]))
        mode = row["mode"]
        if mode in cells[key]:
            errors.append(f"duplicate mode: {key} {mode}")
        cells[key][mode] = row
        raw = ROOT / row["raw_path"]
        workload = ROOT / row["workload_path"]
        if raw in raw_paths:
            errors.append(f"duplicate raw path: {raw}")
        raw_paths.add(raw)
        if not raw.is_file() or sha256(raw) != row["raw_sha256"]:
            errors.append(f"raw hash mismatch: {raw}")
            continue
        if workload not in workload_hashes:
            workload_hashes[workload] = sha256(workload) if workload.is_file() else "missing"
        if workload_hashes[workload] != row["workload_sha256"]:
            errors.append(f"workload hash mismatch: {workload}")
        with np.load(raw) as payload:
            latency = payload["latency_ms"]
            if latency.shape != (1000,) or not np.isfinite(latency).all() or np.any(latency <= 0):
                errors.append(f"invalid latency array: {raw}")
            latency_observations += latency.size
            batch_size = int(row["batch_size"])
            calculated = {
                "latency_mean_ms": float(latency.mean()),
                "latency_median_ms": float(np.median(latency)),
                "latency_p95_ms": float(np.percentile(latency, 95)),
                "latency_sample_sd_ms": float(latency.std(ddof=1)),
                "throughput_samples_per_second": 1000.0 * batch_size / float(latency.mean()),
            }
            for field, value in calculated.items():
                if not close(value, float(row[field])):
                    errors.append(f"summary mismatch: {raw} {field}")
            if mode in {"actual_dynamic", "forced_fallback"}:
                counts = payload["early_counts"]
                route_observations += counts.size
                if counts.shape != (1000,) or np.any(counts < 0) or np.any(counts > batch_size):
                    errors.append(f"invalid route counts: {raw}")
                if int(counts.sum()) != int(row["early_count"]):
                    errors.append(f"route sum mismatch: {raw}")
                if mode == "forced_fallback" and np.any(counts):
                    errors.append(f"forced fallback exited: {raw}")
        with np.load(workload) as payload:
            ids = payload["timed_sample_ids"]
            if ids.shape != (1000, int(row["batch_size"])):
                errors.append(f"workload shape mismatch: {workload}")
            if ids.size != int(row["timed_samples"]):
                errors.append(f"timed sample mismatch: {workload}")
            if len(np.unique(ids)) != int(row["unique_timed_samples"]):
                errors.append(f"unique sample mismatch: {workload}")

    expected_modes = {"final_only", "actual_dynamic", "exit8_only", "forced_fallback"}
    if len(rows) != 1200 or len(cells) != 300 or len(raw_paths) != 1200 or len(workload_hashes) != 300:
        errors.append("global row/file/cell count mismatch")
    for key, modes in cells.items():
        if set(modes) != expected_modes or sorted(int(row["mode_order"]) for row in modes.values()) != [0, 1, 2, 3]:
            errors.append(f"mode coverage/order mismatch: {key}")

    check_totals = Counter()
    check_maxima = Counter()
    check_paths = sorted((input_dir / "checks").glob("*.json"))
    for path in check_paths:
        check = json.loads(path.read_text())
        for field in ("samples_checked", "route_errors", "prediction_errors", "logit_tolerance_errors"):
            check_totals[field] += check[field]
        for field in ("max_abs_logit_difference", "max_tolerance_ratio"):
            check_maxima[field] = max(check_maxima[field], check[field])
    if len(check_paths) != 300 or check_totals["route_errors"] or check_totals["prediction_errors"] or check_totals["logit_tolerance_errors"]:
        errors.append("dynamic correctness receipts failed")

    result_path = input_dir / "p8_v3_results.json"
    result = json.loads(result_path.read_text())
    protocol = ROOT / "reports/experiments/2026-09-05-early-exit-p8-v3-design/protocol_manifest.json"
    if result.get("status") != "completed" or result.get("raw_files") != 1200 or not result.get("source_unchanged"):
        errors.append("invalid completion receipt")
    if sha256(protocol) != result.get("protocol_sha256"):
        errors.append("protocol hash mismatch")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "round_rows": len(rows),
        "cells": len(cells),
        "raw_files": len(raw_paths),
        "workloads": len(workload_hashes),
        "check_files": len(check_paths),
        "latency_observations": latency_observations,
        "route_count_observations": route_observations,
        "check_totals": dict(check_totals),
        "check_maxima": dict(check_maxima),
        "input_hashes": {
            "protocol_manifest.json": sha256(protocol),
            "p8_v3_results.json": sha256(result_path),
            "round_metrics.csv": sha256(input_dir / "tables/round_metrics.csv"),
            "seed_metrics.csv": sha256(input_dir / "tables/seed_metrics.csv"),
        },
    }


def aggregate(rows: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    cells: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (row["device"], row["dataset"], int(row["seed"]),
               int(row["batch_size"]), int(row["round"]))
        cells[key][row["mode"]] = row
    seed_groups: dict[tuple, list[dict]] = defaultdict(list)
    for (device, dataset, seed, batch_size, round_index), modes in cells.items():
        dynamic = modes["actual_dynamic"]
        final_ms = float(modes["final_only"]["latency_mean_ms"])
        fallback_ms = float(modes["forced_fallback"]["latency_mean_ms"])
        expected = dynamic["singleton_isolated_expected_ms"]
        seed_groups[(device, dataset, batch_size, seed)].append({
            "actual_saving_percent": 100 * float(dynamic["actual_dynamic_saving_fraction"]),
            "early_fraction_percent": 100 * float(dynamic["timed_early_fraction"]),
            "mac_saving_percent": 100 * float(dynamic["mac_saving_fraction"]),
            "final_latency_ms": final_ms,
            "dynamic_latency_ms": float(dynamic["latency_mean_ms"]),
            "exit8_latency_ms": float(modes["exit8_only"]["latency_mean_ms"]),
            "forced_fallback_latency_ms": fallback_ms,
            "forced_fallback_overhead_percent": 100 * (fallback_ms / final_ms - 1),
            "isolated_expected_saving_percent": None if not expected else 100 * (1 - float(expected) / final_ms),
        })
    seed_rows = []
    metric_names = (
        "actual_saving_percent", "early_fraction_percent", "mac_saving_percent",
        "final_latency_ms", "dynamic_latency_ms", "exit8_latency_ms",
        "forced_fallback_latency_ms", "forced_fallback_overhead_percent",
        "isolated_expected_saving_percent",
    )
    for (device, dataset, batch_size, seed), values in sorted(seed_groups.items()):
        if len(values) != 5:
            raise ValueError("Each seed must have five paired rounds")
        result = {"device": device, "dataset": dataset, "batch_size": batch_size, "seed": seed, "rounds": 5}
        for metric in metric_names:
            present = [value[metric] for value in values if value[metric] is not None]
            result[metric] = "" if not present else mean(present)
        seed_rows.append(result)

    across_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in seed_rows:
        across_groups[(row["device"], row["dataset"], row["batch_size"])].append(row)
    aggregate_rows = []
    for (device, dataset, batch_size), values in sorted(across_groups.items()):
        result = {"device": device, "dataset": dataset, "batch_size": batch_size,
                  "seeds": 3, "rounds_per_seed": 5,
                  "positive_speedup_seeds": sum(float(value["actual_saving_percent"]) > 0 for value in values)}
        for metric in metric_names:
            present = [float(value[metric]) for value in values if value[metric] != ""]
            if not present:
                continue
            for suffix, number in summarize(present).items():
                result[f"{metric}_{suffix}"] = number
        aggregate_rows.append(result)
    return seed_rows, aggregate_rows


def write_latex(path: Path, rows: list[dict]) -> None:
    labels = {"cuda": "RTX 3080 Ti", "cpu": "12-thread CPU", "cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Device & Dataset & Batch & Early (\\%) & MAC saving (\\%) & Actual saving (\\%) \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{labels[row['device']]} & {labels[row['dataset']]} & {row['batch_size']} & "
            f"{row['early_fraction_percent_mean']:.2f} & {row['mac_saving_percent_mean']:.2f} & "
            f"{row['actual_saving_percent_mean']:.2f} $\\pm$ {row['actual_saving_percent_sample_sd']:.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def write_figure(output: Path, rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.6), sharex=True, sharey=True)
    batches = [1, 4, 8, 16, 32]
    for axis, device, dataset in zip(axes.flat, ("cuda", "cuda", "cpu", "cpu"),
                                      ("cifar10", "cifar100", "cifar10", "cifar100")):
        selected = sorted((row for row in rows if row["device"] == device and row["dataset"] == dataset),
                          key=lambda row: row["batch_size"])
        y = [row["actual_saving_percent_mean"] for row in selected]
        err = [row["actual_saving_percent_sample_sd"] for row in selected]
        axis.axhline(0, color="black", linewidth=0.8)
        axis.errorbar(batches, y, yerr=err, marker="o", capsize=3, linewidth=1.5)
        axis.set_xscale("log", base=2)
        axis.set_xticks(batches, labels=[str(value) for value in batches])
        axis.grid(alpha=0.25)
        axis.set_title(f"{'RTX 3080 Ti' if device == 'cuda' else '12-thread CPU'} / {dataset.upper().replace('CIFAR', 'CIFAR-')}")
    for axis in axes[:, 0]:
        axis.set_ylabel("Actual latency saving (%)")
    for axis in axes[-1, :]:
        axis.set_xlabel("Batch size")
    fig.tight_layout()
    fig.savefig(output / "actual_latency_saving_by_batch.pdf", bbox_inches="tight")
    fig.savefig(output / "actual_latency_saving_by_batch.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_readme(path: Path, receipt: dict, rows: list[dict], input_dir: Path) -> None:
    indexed = {(row["device"], row["dataset"], row["batch_size"]): row for row in rows}
    def saving(device, dataset, batch):
        row = indexed[(device, dataset, batch)]
        return f"{row['actual_saving_percent_mean']:.2f}% ± {row['actual_saving_percent_sample_sd']:.2f}%"
    lines = [
        "# P8 v3 completed-run audit and analysis",
        "",
        f"The completed output `{input_dir.relative_to(ROOT)}` passed the full integrity audit: "
        f"{receipt['raw_files']} raw arrays, {receipt['latency_observations']:,} latency observations, "
        f"{receipt['workloads']} exact workload receipts, and {receipt['check_files']} correctness receipts. "
        f"Across {receipt['check_totals']['samples_checked']:,} checked sample executions there were zero route, "
        "prediction, or logit-tolerance errors. No official test set was accessed.",
        "",
        "Latency saving is `1 - actual_dynamic / final_only` within each paired round. Five rounds are averaged "
        "within each training seed; the table reports mean ± sample SD across the three seeds. Negative values "
        "are slowdowns. Three seeds are too few for broad significance claims.",
        "",
        "| Device / dataset | b=1 | b=4 | b=8 | b=16 | b=32 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for device, dataset, label in (
        ("cuda", "cifar10", "RTX 3080 Ti / CIFAR-10"),
        ("cuda", "cifar100", "RTX 3080 Ti / CIFAR-100"),
        ("cpu", "cifar10", "12-thread CPU / CIFAR-10"),
        ("cpu", "cifar100", "12-thread CPU / CIFAR-100"),
    ):
        lines.append(f"| {label} | " + " | ".join(saving(device, dataset, batch) for batch in (1, 4, 8, 16, 32)) + " |")
    c10 = indexed[("cuda", "cifar10", 1)]
    c100 = indexed[("cuda", "cifar100", 1)]
    lines.extend([
        "",
        "The deployable claim is conditional. On the RTX 3080 Ti, dynamic routing accelerated singleton "
        f"CIFAR-10 inference by {saving('cuda', 'cifar10', 1)}, versus an isolated-path estimate of "
        f"{c10['isolated_expected_saving_percent_mean']:.2f}%. CIFAR-100 singleton inference was effectively "
        f"neutral ({saving('cuda', 'cifar100', 1)}; isolated estimate "
        f"{c100['isolated_expected_saving_percent_mean']:.2f}%). Every GPU batch-size setting from 4 to 32 "
        "slowed down despite positive MAC savings, exposing compaction, branching, and kernel-launch overhead. "
        "CPU behavior was hardware- and dataset-dependent: CIFAR-10 accelerated at all measured batch sizes, "
        "whereas CIFAR-100 had a clear gain only at batch 1 and mixed/negative larger-batch behavior.",
        "",
        "The reference is the same A3 checkpoint's final-only path, not an independently trained A0 model. "
        "CUDA memory numbers are allocated-memory scopes; CPU RSS is process-lifetime high-water only and does "
        "not support per-mode peak comparisons. Telemetry contains boundary snapshots, not integrated energy. "
        "The failed TF32 P8 v2 run remains separate and none of its timings are included here.",
        "",
        "Machine-readable outputs: `seed_summary.csv`, `aggregate_summary.csv`, `aggregate_summary.tex`, and "
        "`analysis_receipt.json`. The figure `actual_latency_saving_by_batch.pdf` visualizes the complete result, "
        "including slowdowns.",
    ])
    path.write_text("\n".join(lines) + "\n")


def execute(input_dir: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    rows = read_csv(input_dir / "tables/round_metrics.csv")
    receipt = audit(input_dir, rows)
    if receipt["status"] != "passed":
        raise RuntimeError(json.dumps(receipt, indent=2))
    seed_rows, aggregate_rows = aggregate(rows)
    write_csv(output / "seed_summary.csv", seed_rows)
    write_csv(output / "aggregate_summary.csv", aggregate_rows)
    write_latex(output / "aggregate_summary.tex", aggregate_rows)
    write_figure(output, aggregate_rows)
    receipt.update({
        "schema_version": 1,
        "statistical_unit": "training seed after averaging five paired rounds",
        "seed_rows": len(seed_rows),
        "aggregate_rows": len(aggregate_rows),
    })
    (output / "analysis_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    write_readme(output / "README.md", receipt, aggregate_rows, input_dir)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(execute(arguments.input.resolve(), arguments.output.resolve()), indent=2))
