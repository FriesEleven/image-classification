"""Profile all 64 stage-sparse attention candidates without training them."""

import argparse
import json
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from image_classification.config import ExperimentConfig
from image_classification.models import build_model
from image_classification.selection import attention_operation_profile, enumerate_stage_candidates
from image_classification.training.benchmark import benchmark_inference, model_metrics
from image_classification.training.provenance import runtime_provenance

PROFILE_FAMILY = "mobilenetv2_stage_sparse_se_eca_cbam_v1"


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def _config(candidate: dict) -> ExperimentConfig:
    positions = candidate["positions"]
    return ExperimentConfig(
        experiment_name=f"profile_{candidate['candidate_id']}",
        model_type="stage_sparse",
        dataset="cifar10",
        evaluate_test=False,
        measure_inference=False,
        eca_positions=tuple(positions["eca"]),
        se_positions=tuple(positions["se"]),
        cbam_positions=tuple(positions["cbam"]),
    )


def _build(candidate: dict, device: torch.device):
    torch.manual_seed(20260901)
    config = _config(candidate)
    model = build_model(config).to(device)
    return config, model


def _static_profiles(candidates: list[dict], device: torch.device) -> list[dict]:
    profiles = []
    for candidate in candidates:
        config, model = _build(candidate, device)
        metrics = model_metrics(model, config)
        operations = attention_operation_profile(model)
        profiles.append(
            {
                **candidate,
                "parameters_total": metrics["parameters_total"],
                "attention_parameters": (
                    metrics["parameters_eca"] + metrics["parameters_se"] + metrics["parameters_cbam"]
                ),
                "attention_operations_estimate": operations["operations_estimate"],
                "attention_operation_breakdown": operations,
            }
        )
        del model
    baseline = next(profile for profile in profiles if profile["active_stages"] == 0)
    for profile in profiles:
        profile["parameter_delta"] = profile["parameters_total"] - baseline["parameters_total"]
    return profiles


def _latency_rounds(
    candidates: list[dict],
    device: torch.device,
    rounds: int,
    runs: int,
) -> dict[str, list[dict]]:
    measurements = {candidate["candidate_id"]: [] for candidate in candidates}
    randomizer = random.Random(20260901)
    baseline = next(candidate for candidate in candidates if candidate["active_stages"] == 0)
    non_baseline = [candidate for candidate in candidates if candidate["active_stages"] > 0]
    for round_index in range(rounds):
        order = list(non_baseline)
        randomizer.shuffle(order)
        _baseline_config, baseline_model = _build(baseline, device)
        for candidate in order:
            _config_value, model = _build(candidate, device)
            baseline_first = randomizer.random() < 0.5
            if baseline_first:
                baseline_result = _benchmark_latency(baseline_model, device, runs)
                result = _benchmark_latency(model, device, runs)
            else:
                result = _benchmark_latency(model, device, runs)
                baseline_result = _benchmark_latency(baseline_model, device, runs)
            pair_order = "baseline_then_candidate" if baseline_first else "candidate_then_baseline"
            baseline_result.update(
                round=round_index + 1,
                paired_candidate_id=candidate["candidate_id"],
                pair_order=pair_order,
            )
            result.update(
                round=round_index + 1,
                pair_order=pair_order,
                paired_baseline_latency_mean=baseline_result["inference_latency_mean"],
                paired_latency_overhead_percent=100.0
                * (
                    result["inference_latency_mean"]
                    - baseline_result["inference_latency_mean"]
                )
                / baseline_result["inference_latency_mean"],
            )
            measurements[baseline["candidate_id"]].append(baseline_result)
            measurements[candidate["candidate_id"]].append(result)
            del model
        del baseline_model
    return measurements


def _benchmark_latency(model, device: torch.device, runs: int) -> dict:
    if device.type != "cuda":
        result = benchmark_inference(model, device, runs=runs)
        result["timing_backend"] = "synchronized_wall_clock"
        return result

    was_training = model.training
    model.eval()
    sample = torch.randn((1, 3, 32, 32), device=device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    with torch.inference_mode():
        for _ in range(20):
            model(sample)
        torch.cuda.synchronize()
        for start, end in zip(starts, ends, strict=True):
            start.record()
            model(sample)
            end.record()
        torch.cuda.synchronize()
    latencies = [float(start.elapsed_time(end)) for start, end in zip(starts, ends, strict=True)]
    model.train(was_training)
    mean = statistics.mean(latencies)
    return {
        "device_name": torch.cuda.get_device_name(0),
        "batch_size": 1,
        "input_resolution": "32x32",
        "inference_latency_mean": mean,
        "inference_latency_std": statistics.pstdev(latencies),
        "throughput_fps": 1000.0 / mean,
        "num_runs": runs,
        "timing_backend": "cuda_events_single_stream",
    }


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling was requested but CUDA is unavailable")
    return device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or args.runs < 1:
        raise ValueError("rounds and runs must be positive")

    torch.set_num_threads(1)
    device = _resolve_device(args.device)
    candidates = enumerate_stage_candidates()
    if len(candidates) != 64 or len({row["candidate_id"] for row in candidates}) != 64:
        raise RuntimeError("Expected exactly 4^3 unique stage-sparse candidates")
    profiles = _static_profiles(candidates, device)
    if args.dry_run:
        print(
            f"Validated {len(profiles)} candidates on {device}; "
            "no latency benchmark run and no artifact written."
        )
        return 0

    measurements = _latency_rounds(candidates, device, args.rounds, args.runs)
    for profile in profiles:
        rounds = measurements[profile["candidate_id"]]
        means = [float(row["inference_latency_mean"]) for row in rounds]
        profile["latency_ms"] = statistics.median(means)
        profile["latency_round_mean_ms"] = statistics.mean(means)
        profile["latency_round_std_ms"] = statistics.stdev(means) if len(means) > 1 else 0.0
        profile["latency_rounds"] = rounds
    baseline = next(profile for profile in profiles if profile["active_stages"] == 0)
    baseline_latency = baseline["latency_ms"]
    for profile in profiles:
        if profile["active_stages"] == 0:
            profile["latency_overhead_percent"] = 0.0
            profile["paired_latency_overhead_rounds_percent"] = []
            continue
        paired_overheads = [
            float(row["paired_latency_overhead_percent"])
            for row in profile["latency_rounds"]
        ]
        profile["latency_overhead_percent"] = statistics.median(paired_overheads)
        profile["paired_latency_overhead_rounds_percent"] = paired_overheads
        profile["latency_normalized_ms"] = baseline_latency * (
            1.0 + profile["latency_overhead_percent"] / 100.0
        )

    provenance = runtime_provenance()
    created = datetime.now().astimezone().isoformat(timespec="seconds")
    payload = {
        "schema_version": 1,
        "profile_family": PROFILE_FAMILY,
        "created_at": created,
        "device": str(device),
        "measurement_protocol": {
            "input_shape": [1, 3, 32, 32],
            "warmup_runs_per_round": 20 if device.type == "cuda" else 2,
            "timed_runs_per_round": args.runs,
            "rounds": args.rounds,
            "round_order_seed": 20260901,
            "candidate_order_randomized_each_round": True,
            "latency_summary": "median paired candidate-versus-adjacent-baseline overhead",
            "paired_baseline": True,
            "pair_order_randomized": True,
            "baseline_model_reused_within_each_round": True,
            "timing_backend": (
                "cuda_events_single_stream" if device.type == "cuda" else "synchronized_wall_clock"
            ),
            "training_performed": False,
        },
        "runtime": provenance,
        "source_sha256": provenance["source_sha256"],
        "baseline_candidate_id": baseline["candidate_id"],
        "baseline_parameters_total": baseline["parameters_total"],
        "baseline_latency_ms": baseline_latency,
        "candidates": profiles,
    }
    output = args.output
    if output is None:
        output = ROOT / "artifacts/budget_selector/profiles" / f"stage_sparse_{_timestamp()}" / "profile.json"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output)
    print(f"Profiled {len(profiles)} candidates without training; saved {output}")
    print(f"Baseline latency: {baseline_latency:.4f} ms on {payload['runtime']['gpu'] or device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
