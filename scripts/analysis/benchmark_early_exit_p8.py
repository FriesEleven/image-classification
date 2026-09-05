"""Benchmark actual A3 dynamic execution on RTX 3080 Ti and same-server CPU."""

from __future__ import annotations

import argparse
import csv
import json
import random
import resource
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import Subset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from image_classification.data.cifar import DATASET_SPECS, _transforms
from image_classification.models import build_model
from scripts.analysis.analyze_early_exit_p0 import _config
from scripts.analysis.analyze_early_exit_p5c import discover_new_runs, read_json, sha256

PROTOCOL = ROOT / "reports/experiments/2026-09-05-early-exit-p8-design/protocol_manifest.json"
P5AB_PROTOCOL = ROOT / "reports/experiments/2026-09-05-early-exit-p5-design/protocol_manifest.json"


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def latency_statistics(values: np.ndarray, batch_size: int) -> dict:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    return {
        "latency_mean_ms": mean,
        "latency_median_ms": float(np.median(values)),
        "latency_p95_ms": float(np.quantile(values, 0.95)),
        "latency_sample_sd_ms": float(values.std(ddof=1)),
        "throughput_samples_per_second": float(batch_size * 1000.0 / mean),
    }


def expected_latency(early_fraction: float, exit8_ms: float, final_ms: float) -> float:
    return early_fraction * exit8_ms + (1.0 - early_fraction) * final_ms


def telemetry() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw",
        "--format=csv,noheader,nounits",
        "--id=0",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        return {"available": False, "error": result.stderr.strip()}
    values = [value.strip() for value in result.stdout.strip().split(",")]
    return {
        "available": True,
        "temperature_c": float(values[0]),
        "sm_clock_mhz": float(values[1]),
        "memory_clock_mhz": float(values[2]),
        "power_w": float(values[3]),
    }


def load_calibration_inputs(dataset: str, run: dict) -> torch.Tensor:
    root = ROOT / "artifacts/runs" / run["experiment_id"]
    split = read_json(root / "split_indices.json")
    config = run["resolved_config"]
    if split["split_seed"] != config["split_seed"] or split.get("calibration_indices") is None:
        raise ValueError(f"Invalid calibration split receipt: {root}")
    spec = DATASET_SPECS[dataset]
    _train_transform, evaluation_transform = _transforms(spec)
    data = spec.dataset_class(root=ROOT / "data", train=True, download=False, transform=evaluation_transform)
    subset = Subset(data, split["calibration_indices"])
    return torch.stack([subset[index][0] for index in range(len(subset))])


def load_model(run: dict, device: torch.device):
    config = _config(run["resolved_config"])
    if config.exit_positions != (8, 16) or config.exit_distillation_alpha != 0.0:
        raise ValueError(f"P8 requires frozen A3 checkpoint: {run['experiment_id']}")
    root = ROOT / "artifacts/runs" / run["experiment_id"]
    summary = read_json(root / "summary.json")
    checkpoint = root / "checkpoints/model_best.pth"
    if summary["best_checkpoint_sha256"] != sha256(checkpoint) or summary["test_evaluated"] is not False:
        raise ValueError(f"Invalid A3 checkpoint receipt: {root}")
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    return model.eval(), summary


def route_fraction(model, inputs: torch.Tensor, threshold: float, device: torch.device) -> float:
    selected = 0
    count = 0
    with torch.inference_mode():
        for start in range(0, len(inputs), 128):
            batch = inputs[start:start + 128].to(device)
            confidence = torch.softmax(model.forward_to_exit(batch, 8).float(), dim=1).amax(dim=1)
            selected += int((confidence >= threshold).sum().item())
            count += len(batch)
    return selected / count


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_mode(model, batches: list[torch.Tensor], mode: str, threshold: float, warmups: int, iterations: int, device: torch.device) -> tuple[np.ndarray, int]:
    if mode == "final_only":
        function = lambda batch: model.forward_to_exit(batch, None)
    elif mode == "exit8_only":
        function = lambda batch: model.forward_to_exit(batch, 8)
    elif mode == "actual_dynamic":
        function = lambda batch: model.forward_with_policy(batch, threshold, exit_position=8)
    else:
        raise ValueError(mode)
    with torch.inference_mode():
        for index in range(warmups):
            function(batches[index % len(batches)])
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        latencies = np.empty(iterations, dtype=np.float64)
        for index in range(iterations):
            batch = batches[index % len(batches)]
            synchronize(device)
            started = time.perf_counter_ns()
            function(batch)
            synchronize(device)
            latencies[index] = (time.perf_counter_ns() - started) / 1e6
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return latencies, int(peak)


def benchmark(output: Path, tag: str) -> dict:
    protocol = read_json(PROTOCOL)
    if protocol.get("status") != "frozen_before_p8_execution":
        raise ValueError("P8 protocol is not frozen")
    receipt = protocol["method_decision_receipt"]
    if sha256(ROOT / receipt["path"]) != receipt["sha256"]:
        raise ValueError("P5-C method decision receipt changed")
    p5ab = read_json(P5AB_PROTOCOL)
    new, manifests = discover_new_runs(tag)
    runs = {(dataset, seed): new[(dataset, "A3", seed)] for dataset, values in protocol["matrix"].items() if dataset in {"cifar10", "cifar100"} for seed in values["seeds"]}
    if len(runs) != protocol["expected_model_checkpoints"]:
        raise ValueError("Expected six A3 checkpoints")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3080 Ti":
        raise RuntimeError("P8 requires the current RTX 3080 Ti")
    output.mkdir(parents=True, exist_ok=False)
    raw_directory = output / "raw_latency"
    raw_directory.mkdir()
    input_cache = {dataset: load_calibration_inputs(dataset, runs[(dataset, values["seeds"][0])]) for dataset, values in protocol["matrix"].items() if dataset in {"cifar10", "cifar100"}}
    round_rows = []
    path_costs = p5ab["path_costs"]
    for device_name in ("cuda", "cpu"):
        device = torch.device(device_name)
        torch.set_num_threads(1 if device_name == "cuda" else 12)
        print(f"[P8] device={device_name} threads={torch.get_num_threads()}", flush=True)
        for dataset in ("cifar10", "cifar100"):
            matrix = protocol["matrix"][dataset]
            threshold = float(matrix["threshold"])
            for seed in matrix["seeds"]:
                model, summary = load_model(runs[(dataset, seed)], device)
                inputs = input_cache[dataset]
                early_fraction = route_fraction(model, inputs, threshold, device)
                mac_saving = early_fraction * (1.0 - float(path_costs[f"{dataset}_exit8"]))
                print(
                    f"[P8] device={device_name} dataset={dataset} seed={seed} "
                    f"early={early_fraction:.4f} mac_saving={mac_saving:.4f}",
                    flush=True,
                )
                for batch_size in protocol["matrix"]["batch_sizes"]:
                    usable = len(inputs) - len(inputs) % batch_size
                    resident = inputs[:usable].to(device)
                    batches = list(resident.split(batch_size))
                    for round_index in range(protocol["timing"]["paired_rounds"]):
                        order = list(protocol["matrix"]["modes"])
                        random.Random(20_260_905 + seed * 1000 + batch_size * 10 + round_index).shuffle(order)
                        measured = {}
                        before = telemetry() if device_name == "cuda" else {"available": False, "reason": "CPU"}
                        for order_index, mode in enumerate(order):
                            values, peak = time_mode(model, batches, mode, threshold, protocol["timing"]["warmup_batches_per_mode_round"], protocol["timing"]["timed_batches_per_mode_round"], device)
                            raw_path = raw_directory / f"{device_name}_{dataset}_seed{seed}_b{batch_size}_r{round_index}_{mode}.npz"
                            np.savez_compressed(raw_path, latency_ms=values)
                            measured[mode] = latency_statistics(values, batch_size)
                            round_rows.append({
                                "device": device_name,
                                "dataset": dataset,
                                "seed": seed,
                                "checkpoint_sha256": summary["best_checkpoint_sha256"],
                                "batch_size": batch_size,
                                "round": round_index,
                                "mode_order": order_index,
                                "mode": mode,
                                "threshold": threshold,
                                "threshold_candidates": 0,
                                "early_fraction": early_fraction,
                                "mac_saving_fraction": mac_saving,
                                "peak_memory_bytes": peak,
                                "raw_path": str(raw_path.relative_to(ROOT)),
                                "raw_sha256": sha256(raw_path),
                                **measured[mode],
                            })
                        expected = expected_latency(early_fraction, measured["exit8_only"]["latency_mean_ms"], measured["final_only"]["latency_mean_ms"])
                        actual = measured["actual_dynamic"]["latency_mean_ms"]
                        final = measured["final_only"]["latency_mean_ms"]
                        after = telemetry() if device_name == "cuda" else {"available": False, "reason": "CPU"}
                        for row in round_rows[-3:]:
                            row["isolated_expected_latency_ms"] = expected
                            row["isolated_expected_saving_fraction"] = 1.0 - expected / final
                            row["actual_dynamic_saving_fraction"] = 1.0 - actual / final
                            row["telemetry_before"] = json.dumps(before, sort_keys=True)
                            row["telemetry_after"] = json.dumps(after, sort_keys=True)
                        print(
                            f"[P8] done device={device_name} dataset={dataset} seed={seed} "
                            f"batch={batch_size} round={round_index + 1}/5 "
                            f"actual_saving={1.0 - actual / final:.4f}",
                            flush=True,
                        )
                    del resident, batches
                del model
                if device_name == "cuda":
                    torch.cuda.empty_cache()
    write_csv(output / "tables/round_metrics.csv", round_rows)
    groups = defaultdict(list)
    for row in round_rows:
        groups[(row["device"], row["dataset"], row["batch_size"], row["mode"])].append(row)
    summary_rows = []
    for (device, dataset, batch_size, mode), rows in groups.items():
        summary_rows.append({
            "device": device,
            "dataset": dataset,
            "batch_size": batch_size,
            "mode": mode,
            "seed_round_count": len(rows),
            **{f"{metric}_mean": float(np.mean([row[metric] for row in rows])) for metric in ("latency_mean_ms", "latency_median_ms", "latency_p95_ms", "throughput_samples_per_second", "early_fraction", "mac_saving_fraction", "isolated_expected_saving_fraction", "actual_dynamic_saving_fraction")},
        })
    write_csv(output / "tables/summary_metrics.csv", summary_rows)
    figures = output / "figures"
    figures.mkdir()
    for device in ("cuda", "cpu"):
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        for axis, dataset in zip(axes, ("cifar10", "cifar100")):
            for mode in ("final_only", "actual_dynamic", "exit8_only"):
                rows = sorted((row for row in summary_rows if row["device"] == device and row["dataset"] == dataset and row["mode"] == mode), key=lambda row: row["batch_size"])
                axis.plot([row["batch_size"] for row in rows], [row["latency_mean_ms_mean"] for row in rows], marker="o", label=mode)
            axis.set(title=dataset, xlabel="batch size", ylabel="latency (ms)")
            axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(figures / f"batch_scaling_{device}.pdf")
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(6, 4))
    dynamic = [row for row in round_rows if row["mode"] == "actual_dynamic"]
    axis.scatter([100 * row["mac_saving_fraction"] for row in dynamic], [100 * row["actual_dynamic_saving_fraction"] for row in dynamic], alpha=0.35)
    axis.set(xlabel="MAC saving (%)", ylabel="actual latency saving (%)")
    figure.tight_layout()
    figure.savefig(figures / "mac_vs_actual_latency_saving.pdf")
    plt.close(figure)
    result = {
        "schema_version": 1,
        "status": "completed",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(PROTOCOL),
        "p5c_manifests": manifests,
        "round_rows": len(round_rows),
        "summary_rows": len(summary_rows),
        "raw_latency_files": len(list(raw_directory.glob("*.npz"))),
        "official_or_external_test_accessed": False,
    }
    (output / "p8_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default="p5c_r1")
    args = parser.parse_args()
    print(json.dumps(benchmark(args.output.resolve(), args.tag), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
