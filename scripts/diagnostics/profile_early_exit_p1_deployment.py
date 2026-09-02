"""Measure the locked P1b single-sample deployment paths without reading test data."""

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from image_classification.models import build_model
from image_classification.training.provenance import runtime_provenance
from scripts.analysis.analyze_early_exit_p0 import _config
from scripts.analysis.analyze_early_exit_p1 import EXPECTED_SEEDS, _load_manifest

EXPECTED_SELECTION_SHA256 = "8dce5d938e06ae9d7432bd6baafd0c0ed9f978678fb53767ab02eeafecc723ab"
EXPECTED_TEST_RESULTS_SHA256 = "239ef583cfd4352eb64b398ce1390706f0fa187aa3c6b6c96c2f90675c45d8f5"
EXPECTED_MANIFEST_SHA256 = "1545e85651103400c425bf3eaae0e70e4d1a4d0e413203158d2f6b6ff16c8c88"
EXPECTED_THRESHOLD = 0.984
DEFAULT_SELECTION = Path(
    "artifacts/policy_selections/early_exit_p1b_20260902_165900/selection.json"
)
DEFAULT_TEST_RESULTS = Path("artifacts/locked_tests/early_exit_p1b_20260902/test_results.json")
DEFAULT_OUTPUT = Path(
    "artifacts/deployment_profiles/early_exit_p1b_rtx4090d_batch1_20260902/profile.json"
)
DEFAULT_REPORT = Path("reports/profiles/2026-09-02-early-exit-p1b-rtx4090d")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _measure(function, device: torch.device, *, warmup: int, runs: int) -> dict:
    with torch.inference_mode():
        for _ in range(warmup):
            function()
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies = []
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter_ns()
            function()
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter_ns() - start) / 1_000_000)
    return {
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "population_std_ms": statistics.pstdev(latencies),
        "p05_ms": sorted(latencies)[max(0, round(0.05 * (runs - 1)))],
        "p95_ms": sorted(latencies)[round(0.95 * (runs - 1))],
        "runs": runs,
    }


def _paired_rounds(
    reference,
    routed,
    device: torch.device,
    *,
    rounds: int,
    runs: int,
    warmup: int,
    randomizer: random.Random,
) -> list[dict]:
    results = []
    for round_index in range(rounds):
        reference_first = randomizer.random() < 0.5
        if reference_first:
            reference_result = _measure(reference, device, warmup=warmup, runs=runs)
            routed_result = _measure(routed, device, warmup=warmup, runs=runs)
        else:
            routed_result = _measure(routed, device, warmup=warmup, runs=runs)
            reference_result = _measure(reference, device, warmup=warmup, runs=runs)
        results.append(
            {
                "round": round_index + 1,
                "order": "reference_then_routed" if reference_first else "routed_then_reference",
                "reference_final": reference_result,
                "routed_path": routed_result,
                "latency_saving_fraction": (
                    1.0 - routed_result["mean_ms"] / reference_result["mean_ms"]
                ),
            }
        )
    return results


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _profile_seed(
    run: dict,
    route_fraction: float,
    device: torch.device,
    *,
    rounds: int,
    runs: int,
    warmup: int,
    randomizer: random.Random,
) -> dict:
    config = _config(run["resolved_config"])
    model = build_model(config).to(device).eval()
    checkpoint = ROOT / "artifacts/runs" / run["experiment_id"] / "checkpoints/model_best.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    generator = torch.Generator().manual_seed(20_260_902 + config.seed)
    sample = torch.randn((1, 3, 32, 32), generator=generator).to(device)

    with torch.inference_mode():
        final_logits = model.forward_to_exit(sample, None)
        exit8_logits = model.forward_to_exit(sample, 8)
        early_logits, early_paths = model.forward_with_policy(sample, 0.0)
        fallback_logits, fallback_paths = model.forward_with_policy(sample, 2.0)
    torch.testing.assert_close(early_logits, exit8_logits)
    torch.testing.assert_close(fallback_logits, final_logits)
    if early_paths.tolist() != [0] or fallback_paths.tolist() != [1]:
        raise RuntimeError("Dynamic deployment path did not honor the isolated route")

    reference = lambda: model.forward_to_exit(sample, None)
    early = lambda: model.forward_with_policy(sample, 0.0)
    fallback = lambda: model.forward_with_policy(sample, 2.0)
    early_rounds = _paired_rounds(
        reference,
        early,
        device,
        rounds=rounds,
        runs=runs,
        warmup=warmup,
        randomizer=randomizer,
    )
    fallback_rounds = _paired_rounds(
        reference,
        fallback,
        device,
        rounds=rounds,
        runs=runs,
        warmup=warmup,
        randomizer=randomizer,
    )
    expected_rounds = []
    for early_round, fallback_round in zip(early_rounds, fallback_rounds, strict=True):
        early_latency = early_round["routed_path"]["mean_ms"]
        fallback_latency = fallback_round["routed_path"]["mean_ms"]
        reference_latency = (
            route_fraction * early_round["reference_final"]["mean_ms"]
            + (1.0 - route_fraction) * fallback_round["reference_final"]["mean_ms"]
        )
        expected_latency = (
            route_fraction * early_latency + (1.0 - route_fraction) * fallback_latency
        )
        expected_rounds.append(
            {
                "round": early_round["round"],
                "reference_final_ms": reference_latency,
                "expected_policy_ms": expected_latency,
                "latency_saving_fraction": 1.0 - expected_latency / reference_latency,
                "speedup": reference_latency / expected_latency,
            }
        )
    result = {
        "seed": config.seed,
        "experiment_id": run["experiment_id"],
        "checkpoint_sha256": sha256(checkpoint),
        "locked_test_early_route_fraction": route_fraction,
        "early_path_rounds": early_rounds,
        "fallback_path_rounds": fallback_rounds,
        "weighted_expected_rounds": expected_rounds,
        "summary": {
            "reference_final_latency_ms": _median(
                [row["reference_final_ms"] for row in expected_rounds]
            ),
            "expected_policy_latency_ms": _median(
                [row["expected_policy_ms"] for row in expected_rounds]
            ),
            "latency_saving_fraction": _median(
                [row["latency_saving_fraction"] for row in expected_rounds]
            ),
            "speedup": _median([row["speedup"] for row in expected_rounds]),
            "early_path_latency_ms": _median(
                [row["routed_path"]["mean_ms"] for row in early_rounds]
            ),
            "fallback_path_latency_ms": _median(
                [row["routed_path"]["mean_ms"] for row in fallback_rounds]
            ),
        },
    }
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _markdown(payload: dict) -> str:
    rows = []
    for value in payload["seed_profiles"]:
        summary = value["summary"]
        rows.append(
            f"| {value['seed']} | {100 * value['locked_test_early_route_fraction']:.2f}% | "
            f"{summary['reference_final_latency_ms']:.4f} | "
            f"{summary['expected_policy_latency_ms']:.4f} | "
            f"{100 * summary['latency_saving_fraction']:.2f}% | {summary['speedup']:.3f}× |"
        )
    aggregate = payload["aggregate"]
    rows.append(
        f"| mean | {100 * aggregate['early_route_fraction_mean']:.2f}% | "
        f"{aggregate['reference_final_latency_ms_mean']:.4f} | "
        f"{aggregate['expected_policy_latency_ms_mean']:.4f} | "
        f"{100 * aggregate['latency_saving_fraction_mean']:.2f}% | "
        f"{aggregate['speedup_mean']:.3f}× |"
    )
    return "\n".join(
        [
            "# Early-exit P1b RTX 4090D deployment profile",
            "",
            "Batch-1 end-to-end synchronized wall-clock timings on an otherwise idle GPU.",
            "The implementation computes exit8 once and continues only unresolved samples; exit16 is skipped.",
            "",
            "| seed | locked test early route | final-only ms | expected policy ms | saving | speedup |",
            "|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "Each path is isolated with thresholds 0/2 only during timing; the expected latency is weighted by",
            "the already frozen test route fraction at threshold 0.984. No dataset is loaded by this profiler.",
            "This server GPU result is deployment evidence for the implementation, not mobile-device evidence.",
            "",
        ]
    )


def profile(
    selection_path: Path,
    test_results_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    device: torch.device,
    rounds: int,
    runs: int,
    warmup: int,
) -> dict:
    if output_path.exists() or report_path.exists():
        raise FileExistsError("Deployment profile is immutable; choose fresh output paths")
    if sha256(selection_path) != EXPECTED_SELECTION_SHA256:
        raise ValueError("Unexpected locked selection")
    if sha256(test_results_path) != EXPECTED_TEST_RESULTS_SHA256:
        raise ValueError("Unexpected locked official-test result")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    test_results = json.loads(test_results_path.read_text(encoding="utf-8"))
    if selection["locked_policy"]["confidence_threshold"] != EXPECTED_THRESHOLD:
        raise ValueError("Locked threshold mismatch")
    if test_results.get("status") != "locked_official_test_complete":
        raise ValueError("Official-test result is incomplete")
    manifest_path = Path(selection["manifest"])
    if sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Unexpected P1b manifest")
    _manifest, indexed, _split_fingerprint = _load_manifest(manifest_path)
    route_fractions = {
        int(row["seed"]): float(row["locked_policy"]["route_fractions"][0])
        for row in test_results["seed_results"]
    }
    randomizer = random.Random(20_260_902)
    seed_profiles = []
    for seed in EXPECTED_SEEDS:
        seed_profiles.append(
            _profile_seed(
                indexed[("multi_exit", seed)],
                route_fractions[seed],
                device,
                rounds=rounds,
                runs=runs,
                warmup=warmup,
                randomizer=randomizer,
            )
        )
    summaries = [row["summary"] for row in seed_profiles]
    payload = {
        "schema_version": 1,
        "profile_family": "early_exit_p1b_real_staged_batch1_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "device": str(device),
        "runtime": runtime_provenance(),
        "inputs": {
            "selection_sha256": EXPECTED_SELECTION_SHA256,
            "test_results_sha256": EXPECTED_TEST_RESULTS_SHA256,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "locked_threshold": EXPECTED_THRESHOLD,
        },
        "measurement_protocol": {
            "input": "seeded synthetic tensor; no train/calibration/test dataset loaded",
            "input_shape": [1, 3, 32, 32],
            "timing_backend": "synchronized_end_to_end_wall_clock",
            "rounds": rounds,
            "warmup_runs_per_measurement": warmup,
            "timed_runs_per_measurement": runs,
            "paired_final_reference": True,
            "pair_order_randomized": True,
            "path_isolation_thresholds": {"early": 0.0, "fallback": 2.0},
            "expected_latency_weight": "locked official-test route fraction at threshold 0.984",
            "training_performed": False,
        },
        "seed_profiles": seed_profiles,
        "aggregate": {
            "early_route_fraction_mean": statistics.mean(
                row["locked_test_early_route_fraction"] for row in seed_profiles
            ),
            "reference_final_latency_ms_mean": statistics.mean(
                row["reference_final_latency_ms"] for row in summaries
            ),
            "expected_policy_latency_ms_mean": statistics.mean(
                row["expected_policy_latency_ms"] for row in summaries
            ),
            "latency_saving_fraction_mean": statistics.mean(
                row["latency_saving_fraction"] for row in summaries
            ),
            "speedup_mean": statistics.mean(row["speedup"] for row in summaries),
        },
        "limitations": [
            "RTX 4090D is a server screening device, not the target mobile device.",
            "Expected latency combines isolated batch-1 path measurements with locked test route fractions.",
            "Power and energy are not measured.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    report_path.mkdir(parents=True, exist_ok=False)
    (report_path / "profile.json").write_bytes(output_path.read_bytes())
    (report_path / "README.md").write_text(_markdown(payload))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--test-results", type=Path, default=DEFAULT_TEST_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if min(args.rounds, args.runs, args.warmup) < 1:
        raise ValueError("rounds, runs and warmup must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requested but CUDA is unavailable")
    torch.set_num_threads(1)
    selection_path = (ROOT / args.selection).resolve()
    test_results_path = (ROOT / args.test_results).resolve()
    if args.verify_only:
        if sha256(selection_path) != EXPECTED_SELECTION_SHA256:
            raise ValueError("Unexpected locked selection")
        if sha256(test_results_path) != EXPECTED_TEST_RESULTS_SHA256:
            raise ValueError("Unexpected official-test results")
        print("Deployment-profile inputs verified; no model or dataset was loaded.")
        return 0
    payload = profile(
        selection_path,
        test_results_path,
        (ROOT / args.output).resolve(),
        (ROOT / args.report).resolve(),
        device=torch.device(args.device),
        rounds=args.rounds,
        runs=args.runs,
        warmup=args.warmup,
    )
    print(_markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
