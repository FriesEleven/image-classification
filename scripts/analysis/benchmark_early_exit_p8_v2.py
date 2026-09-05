"""P8 v2: measured routes, forced fallback and explicit memory scope."""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import socket
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from image_classification.training.provenance import runtime_provenance, source_fingerprint
from scripts.analysis.analyze_early_exit_p5c import discover_new_runs, read_json, sha256
from scripts.analysis.benchmark_early_exit_p8 import (
    latency_statistics,
    load_calibration_inputs,
    load_model,
    synchronize,
    telemetry,
    write_csv,
)
from scripts.launch_early_exit_p8 import validate as validate_original_protocol

PROTOCOL = ROOT / "reports/experiments/2026-09-05-early-exit-p8-v2-design/protocol_manifest.json"


def validate_protocol() -> dict:
    validate_original_protocol()
    p = read_json(PROTOCOL)
    if p["status"] != "frozen_before_p8_v2_execution":
        raise ValueError("P8 v2 protocol not frozen")
    original = ROOT / p["original_output"] / "p8_results.json"
    if sha256(original) != p["original_results_sha256"]:
        raise ValueError("Original P8 evidence changed")
    assert p["expected_raw_files"] == 1200
    return p


def sample_order(count: int, batch_size: int, round_index: int, seed: int) -> np.ndarray:
    generator = np.random.Generator(np.random.PCG64(20260905 + seed * 100 + round_index))
    return generator.permutation(count)[:count - count % batch_size]


def timed_sample_ids(ids: np.ndarray, batch_size: int, iterations: int) -> np.ndarray:
    batches = np.asarray(ids).reshape(-1, batch_size)
    return batches[np.arange(iterations) % len(batches)]


def mac_saving(early_fraction: float, early_macs: int, final_macs: int, exit_head_macs: int) -> float:
    return 1.0 - (early_fraction * early_macs + (1.0 - early_fraction) * (final_macs + exit_head_macs)) / final_macs


def mode_forward(model, batch, mode, threshold):
    if mode == "final_only":
        return model.forward_to_exit(batch, None)
    if mode == "exit8_only":
        return model.forward_to_exit(batch, 8)
    if mode == "actual_dynamic":
        return model.forward_with_policy(batch, threshold, exit_position=8)
    if mode == "forced_fallback":
        return model.forward_with_policy(batch, 2.0, exit_position=8)
    raise ValueError(mode)


def check_routing(model, batches, threshold) -> dict:
    checked = 0
    route_errors = prediction_errors = 0
    with torch.inference_mode():
        for batch in batches[:32]:
            early_logits = model.forward_to_exit(batch, 8)
            final_logits = model.forward_to_exit(batch, None)
            early = early_logits.float().softmax(1).amax(1) >= threshold
            logits, paths = model.forward_with_policy(batch, threshold, exit_position=8)
            expected = torch.where(early, early_logits.argmax(1), final_logits.argmax(1))
            route_errors += int(((paths == 0) != early).sum())
            prediction_errors += int((logits.argmax(1) != expected).sum())
            checked += len(batch)
    return {"samples_checked": checked, "route_errors": route_errors, "prediction_errors": prediction_errors}


def time_mode(model, batches, mode, threshold, warmups, iterations, device):
    paths_buffer = []
    with torch.inference_mode():
        for i in range(warmups):
            mode_forward(model, batches[i % len(batches)], mode, threshold)
        synchronize(device)
        baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        latencies = np.empty(iterations, dtype=np.float64)
        for i in range(iterations):
            batch = batches[i % len(batches)]
            synchronize(device)
            started = time.perf_counter_ns()
            result = mode_forward(model, batch, mode, threshold)
            synchronize(device)
            latencies[i] = (time.perf_counter_ns() - started) / 1e6
            if isinstance(result, tuple):
                # Transfer after the timing boundary: no .item() inside timing.
                paths_buffer.append(result[1].detach().cpu())
            del result
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    early_counts = None
    if paths_buffer:
        early_counts = (torch.stack(paths_buffer) == 0).sum(1).numpy()
    memory = {
        "cuda_allocated_baseline_bytes": baseline,
        "cuda_allocated_peak_bytes": peak,
        "cuda_allocated_incremental_peak_bytes": None if peak is None else peak - baseline,
        "cpu_process_lifetime_rss_high_water_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "cpu_mode_peak_bytes": None,
    }
    return latencies, early_counts, memory


def execute(output: Path) -> dict:
    p = validate_protocol()
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != p["gpu"]:
        raise RuntimeError("Expected RTX 3080 Ti")
    new, manifests = discover_new_runs("p5c_r1")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("raw", "workloads", "tables", "checks"):
        (output / name).mkdir()
    provenance = runtime_provenance()
    source_hashes = source_fingerprint()
    provenance.update(hostname=socket.gethostname(), protocol_sha256=sha256(PROTOCOL),
                      started_at_utc=datetime.now(timezone.utc).isoformat(), p5c_manifests=manifests,
                      cpu_memory_scope="process lifetime high-water only", cleanup_during_run="not authorized")
    provenance.update(
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        cudnn_version=torch.backends.cudnn.version(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        gpu_properties=str(torch.cuda.get_device_properties(0)),
        route_copy_scope="Paths copied to host after each timed call; copy excluded from latency",
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    rows = []
    for device_name in p["devices"]:
        device = torch.device(device_name)
        torch.set_num_threads(p["cpu_threads"] if device_name == "cpu" else 1)
        for dataset, matrix in p["datasets"].items():
            for seed in matrix["seeds"]:
                run = new[(dataset, "A3", seed)]
                split_path = ROOT / "artifacts/runs" / run["experiment_id"] / "split_indices.json"
                original_ids = np.asarray(read_json(split_path)["calibration_indices"])
                inputs = load_calibration_inputs(dataset, run)
                model, summary = load_model(run, device)
                threshold = matrix["threshold"]
                early_macs, final_macs = ((2676864, 6124928) if dataset == "cifar10" else (2682624, 6240128))
                head = model.exit_heads["8"].classifier
                exit_head_macs = head.in_features * head.out_features
                for batch_size in p["batch_sizes"]:
                    for round_index in range(p["rounds"]):
                        if source_fingerprint() != source_hashes or sha256(PROTOCOL) != provenance["protocol_sha256"]:
                            raise RuntimeError("Source/protocol changed during benchmark")
                        key = f"{device_name}_{dataset}_s{seed}_b{batch_size}_r{round_index}"
                        order = sample_order(len(inputs), batch_size, round_index, seed)
                        resident = inputs[torch.from_numpy(order)].to(device)
                        batches = list(resident.split(batch_size))
                        actual_ids = timed_sample_ids(original_ids[order], batch_size, p["timed_batches"])
                        workload = output / "workloads" / f"{key}.npz"
                        np.savez_compressed(workload, timed_sample_ids=actual_ids)
                        checks = check_routing(model, batches, threshold)
                        (output / "checks" / f"{key}.json").write_text(json.dumps(checks) + "\n")
                        if checks["route_errors"] or checks["prediction_errors"]:
                            raise RuntimeError(f"Dynamic correctness check failed: {key}")
                        modes = list(p["modes"])
                        random.Random(20260905 + seed * 1000 + batch_size * 10 + round_index).shuffle(modes)
                        before = telemetry() if device_name == "cuda" else {}
                        cell_rows = []
                        for mode_index, mode in enumerate(modes):
                            timings, counts, memory = time_mode(model, batches, mode, threshold,
                                                               p["warmup_batches"], p["timed_batches"], device)
                            raw = output / "raw" / f"{key}_{mode}.npz"
                            arrays = {"latency_ms": timings}
                            if counts is not None:
                                arrays["early_counts"] = counts
                            np.savez_compressed(raw, **arrays)
                            if mode == "forced_fallback" and np.any(counts):
                                raise RuntimeError("Forced-fallback unexpectedly exited early")
                            cell_rows.append({"device": device_name, "dataset": dataset, "seed": seed,
                                              "batch_size": batch_size, "round": round_index, "mode": mode,
                                              "mode_order": mode_index, "threshold": threshold,
                                              "checkpoint_sha256": summary["best_checkpoint_sha256"],
                                              "split_sha256": sha256(split_path),
                                              "workload_path": str(workload.relative_to(ROOT)), "workload_sha256": sha256(workload),
                                              "raw_path": str(raw.relative_to(ROOT)), "raw_sha256": sha256(raw),
                                              "early_count": None if counts is None else int(counts.sum()),
                                              "timed_samples": actual_ids.size, "unique_timed_samples": len(np.unique(actual_ids)),
                                              "torch_num_threads": torch.get_num_threads(),
                                              **memory, **latency_statistics(timings, batch_size)})
                        by_mode = {r["mode"]: r for r in cell_rows}
                        early_fraction = by_mode["actual_dynamic"]["early_count"] / actual_ids.size
                        baseline = by_mode["final_only"]["latency_mean_ms"]
                        saving = 1 - by_mode["actual_dynamic"]["latency_mean_ms"] / baseline
                        expected = None if batch_size != 1 else (
                            early_fraction * by_mode["exit8_only"]["latency_mean_ms"]
                            + (1 - early_fraction) * by_mode["forced_fallback"]["latency_mean_ms"])
                        after = telemetry() if device_name == "cuda" else {}
                        for row in cell_rows:
                            row.update(timed_early_fraction=early_fraction,
                                       actual_dynamic_saving_fraction=saving,
                                       mac_saving_fraction=mac_saving(early_fraction, early_macs, final_macs, exit_head_macs),
                                       historical_mac_proxy=early_fraction * (1 - early_macs / final_macs),
                                       singleton_isolated_expected_ms=expected,
                                       telemetry_before=json.dumps(before), telemetry_after=json.dumps(after))
                        rows.extend(cell_rows)
                        write_csv(output / "tables/round_metrics.csv", rows)
                        print(f"[P8 v2] {key} routes={early_fraction:.4f} actual_saving={saving:.4f}", flush=True)
                        del batches, resident
                del model, inputs
                if device_name == "cuda":
                    torch.cuda.empty_cache()
    # Seeds, not the 15 seed-round combinations, are the statistical units.
    groups = defaultdict(list)
    for row in rows:
        if row["mode"] == "actual_dynamic":
            groups[(row["device"], row["dataset"], row["batch_size"], row["seed"])].append(row)
    seed_rows = []
    for (device, dataset, batch_size, seed), values in groups.items():
        seed_rows.append({"device": device, "dataset": dataset, "batch_size": batch_size, "seed": seed,
                          **{k: float(np.mean([r[k] for r in values])) for k in
                             ("actual_dynamic_saving_fraction", "latency_mean_ms", "mac_saving_fraction")}})
    write_csv(output / "tables/seed_metrics.csv", seed_rows)
    result = {"status": "completed", "schema_version": 2, "raw_files": len(rows),
              "protocol_sha256": sha256(PROTOCOL), "official_test_accessed": False,
              "finished_at_utc": datetime.now(timezone.utc).isoformat(), "source_unchanged": source_fingerprint() == source_hashes}
    if len(rows) != p["expected_raw_files"] or not result["source_unchanged"]:
        raise RuntimeError("Incomplete or mixed-source P8 v2")
    (output / "p8_v2_results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.output.resolve()), indent=2))
