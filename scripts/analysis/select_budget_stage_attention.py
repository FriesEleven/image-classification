"""Select stage-sparse attention candidates from paired probe runs and hardware budgets."""

import argparse
import json
import sys
from datetime import datetime
from math import sqrt
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from image_classification.selection import (
    candidate_from_positions,
    enumerate_stage_candidates,
    score_candidates,
    select_candidates_for_budgets,
)

EXPECTED_SEEDS = (45, 46, 47)
EXPECTED_PROTOCOL = {
    "model_type": "stage_sparse",
    "dataset": "cifar10",
    "validation_size": 5000,
    "evaluate_test": False,
    "epochs": 200,
    "batch_size": 128,
    "lr": 0.01,
    "amp": True,
    "cuda_graph": True,
    "torch_num_threads": 1,
    "measure_inference": False,
    "accumulation_steps": 1,
    "num_workers": 8,
    "prefetch_factor": 4,
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _probe_candidates() -> list[dict]:
    return [candidate for candidate in enumerate_stage_candidates() if candidate["active_stages"] <= 1]


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return sum(values) / len(values)


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("Sample standard deviation requires at least two values")
    mean = _mean(values)
    return sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _validate_manifest(path: Path) -> tuple[dict, dict[str, dict[int, float]]]:
    manifest = _load_json(path)
    expected_candidates = {candidate["candidate_id"] for candidate in _probe_candidates()}
    if manifest.get("status") != "completed":
        raise ValueError(f"Probe manifest must be completed, got {manifest.get('status')}")
    runs = manifest.get("runs", [])
    if len(runs) != len(expected_candidates) * len(EXPECTED_SEEDS):
        raise ValueError("Expected exactly 10 candidates x 3 calibration seeds")

    accuracies = {candidate_id: {} for candidate_id in expected_candidates}
    for run in runs:
        config = run.get("resolved_config") or {}
        if run.get("status") not in {"completed", "skipped_complete"} or run.get("return_code") != 0:
            raise ValueError(f"Incomplete probe run: {run.get('experiment_id')}")
        for key, expected in EXPECTED_PROTOCOL.items():
            if config.get(key) != expected:
                raise ValueError(f"Unexpected {key} in {run.get('experiment_id')}: {config.get(key)!r}")
        seed = int(run["seed"])
        if seed not in EXPECTED_SEEDS or config.get("seed") != seed:
            raise ValueError(f"Unexpected seed in {run.get('experiment_id')}")
        candidate = candidate_from_positions(
            config.get("eca_positions", []),
            config.get("se_positions", []),
            config.get("cbam_positions", []),
        )
        candidate_id = candidate["candidate_id"]
        if candidate_id not in expected_candidates:
            raise ValueError(f"Probe run is not baseline or a singleton: {candidate_id}")
        summary = run.get("summary") or {}
        if summary.get("test_evaluated") is not False or "test_accuracy" in summary:
            raise ValueError(f"Probe run touched official test data: {run.get('experiment_id')}")
        accuracy = float(summary["best_validation_accuracy"]) * 100.0
        if seed in accuracies[candidate_id]:
            raise ValueError(f"Duplicate candidate/seed: {candidate_id} seed{seed}")
        accuracies[candidate_id][seed] = accuracy

    expected_seed_set = set(EXPECTED_SEEDS)
    for candidate_id, rows in accuracies.items():
        if set(rows) != expected_seed_set:
            raise ValueError(f"Missing paired seed for {candidate_id}: {sorted(rows)}")
    return manifest, accuracies


def _unit_evidence(accuracies: dict[str, dict[int, float]]) -> tuple[str, dict[str, dict]]:
    baseline = next(candidate for candidate in _probe_candidates() if candidate["active_stages"] == 0)
    baseline_id = baseline["candidate_id"]
    evidence = {}
    for candidate in _probe_candidates():
        if candidate["active_stages"] != 1:
            continue
        candidate_id = candidate["candidate_id"]
        stage, choice = next(
            (stage, choice) for stage, choice in candidate["choices"].items() if choice != "none"
        )
        unit = f"{stage}_{choice}"
        gains = [accuracies[candidate_id][seed] - accuracies[baseline_id][seed] for seed in EXPECTED_SEEDS]
        evidence[unit] = {
            "stage": stage,
            "attention": choice,
            "candidate_id": candidate_id,
            "paired_seeds": list(EXPECTED_SEEDS),
            "probe_validation_accuracy_percent": [accuracies[candidate_id][seed] for seed in EXPECTED_SEEDS],
            "baseline_validation_accuracy_percent": [accuracies[baseline_id][seed] for seed in EXPECTED_SEEDS],
            "paired_gain_pp": gains,
            "mean_gain_pp": _mean(gains),
            "sample_std_pp": _sample_std(gains),
            "wins": sum(gain > 0 for gain in gains),
        }
    return baseline_id, evidence


def _load_profiles(path: Path, expected_source: dict[str, str]) -> dict:
    profile = _load_json(path)
    if profile.get("schema_version") != 1 or len(profile.get("candidates", [])) != 64:
        raise ValueError("Expected a schema-v1 profile with all 64 candidates")
    if profile.get("source_sha256") != expected_source:
        raise ValueError("Hardware profile source snapshot differs from the probe sweep")
    expected_ids = {candidate["candidate_id"] for candidate in enumerate_stage_candidates()}
    observed_ids = {candidate["candidate_id"] for candidate in profile["candidates"]}
    if observed_ids != expected_ids:
        raise ValueError("Hardware profile candidate set is incomplete or unexpected")
    return profile


def _load_budget_config(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if value.get("schema_version") != 1 or not value.get("budgets"):
        raise ValueError("Budget config must be schema version 1 and contain budgets")
    risk_penalty = float(value.get("risk_penalty", 0.5))
    if risk_penalty < 0:
        raise ValueError("risk_penalty must be non-negative")
    value["risk_penalty"] = risk_penalty
    return value


def _markdown(result: dict) -> str:
    lines = [
        "# Budget-aware stage-sparse selection",
        "",
        (
            "This is a validation-only calibration result. Predicted multi-stage gains are additive proxies and must "
            "be confirmed by training the selected architectures from scratch."
        ),
        "",
        "## Paired singleton probes",
        "",
        "| Unit | Paired gains (pp) | Mean ± sample std (pp) | Wins |",
        "|---|---:|---:|---:|",
    ]
    for unit, row in sorted(result["unit_evidence"].items()):
        gains = "/".join(f"{value:+.2f}" for value in row["paired_gain_pp"])
        lines.append(
            f"| {unit} | {gains} | {row['mean_gain_pp']:+.3f} ± {row['sample_std_pp']:.3f} | "
            f"{row['wins']}/3 |"
        )
    lines.extend(
        [
            "",
            "## Selected candidates",
            "",
            "| Budget | Candidate | Risk-adjusted gain (pp) | Params Δ | Attention ops | Latency Δ |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in result["selections"]:
        selected = row["selected"]
        lines.append(
            f"| {row['budget']['name']} | `{selected['candidate_id']}` | "
            f"{selected['risk_adjusted_gain_pp']:+.3f} | {selected['parameter_delta']:,} | "
            f"{selected['attention_operations_estimate']:,} | "
            f"{selected['latency_overhead_percent']:+.2f}% |"
        )
    lines.extend(
        [
            "",
            (
                "No official test set was evaluated. If a budget selects the all-none candidate, retain that negative "
                "result instead of forcing an attention deployment."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument(
        "--budgets",
        type=Path,
        default=ROOT / "configs/budgets/stage_sparse_mobilenetv2.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    profile_path = args.profiles.resolve()
    manifest, accuracies = _validate_manifest(manifest_path)
    baseline_id, evidence = _unit_evidence(accuracies)
    profile = _load_profiles(profile_path, manifest["runtime"]["source_sha256"])
    budget_config = _load_budget_config(args.budgets.resolve())
    scored = score_candidates(profile["candidates"], evidence, budget_config["risk_penalty"])
    selections = select_candidates_for_budgets(scored, budget_config["budgets"])
    selected_ids = list(dict.fromkeys(row["selected"]["candidate_id"] for row in selections))
    result = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "probe_manifest": str(manifest_path),
        "probe_sweep_name": manifest["sweep_name"],
        "hardware_profile": str(profile_path),
        "profile_family": profile["profile_family"],
        "budget_config": str(args.budgets.resolve()),
        "calibration_seeds": list(EXPECTED_SEEDS),
        "official_test_evaluated": False,
        "baseline_candidate_id": baseline_id,
        "risk_penalty": budget_config["risk_penalty"],
        "unit_evidence": evidence,
        "selections": selections,
        "confirmation_plan": {
            "selected_candidate_ids": selected_ids,
            "matched_baseline_candidate_id": baseline_id,
            "recommended_confirmation_seeds": [48, 49, 50],
            "retrain_from_scratch": True,
            "evaluate_test": False,
            "note": "Freeze candidates before confirmation; additive probe scores are not final accuracy evidence.",
        },
    }
    if args.dry_run:
        print(_markdown(result))
        print("Dry run: evidence verified; no selection files written.")
        return 0

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "selection.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "selection.md").write_text(_markdown(result), encoding="utf-8")
    (output / "confirmation_plan.yaml").write_text(
        yaml.safe_dump(result["confirmation_plan"], sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Selected candidates for {len(selections)} budgets; saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
