"""Audit and analyze a completed six-run early-exit P0 manifest without test evaluation."""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from image_classification.config import ExperimentConfig
from image_classification.data import build_dataloaders
from image_classification.models import build_model
from image_classification.selection.early_exit import (
    apply_policy,
    policy_metrics,
    select_policy,
    stratified_calibration_mask,
)

EXPECTED_SEEDS = (51, 52, 53)
EXPECTED_TYPES = ("mobilenetv2", "multi_exit")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config(values: dict) -> ExperimentConfig:
    values = dict(values)
    for key in (
        "aux_positions", "eca_positions", "se_positions", "cbam_positions", "exit_positions",
    ):
        if key in values:
            values[key] = tuple(values[key])
    if "exit_loss_weights" in values:
        values["exit_loss_weights"] = tuple(values["exit_loss_weights"])
    return ExperimentConfig(**values)


def _load_manifest(path: Path) -> tuple[dict, dict[tuple[str, int], dict]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"P0 manifest is not completed: {manifest.get('status')}")
    indexed = {}
    for run in manifest.get("runs", []):
        config = run.get("resolved_config", {})
        key = (config.get("model_type"), int(run.get("seed", -1)))
        if key in indexed:
            raise ValueError(f"Duplicate P0 run: {key}")
        indexed[key] = run
    expected = {(model_type, seed) for model_type in EXPECTED_TYPES for seed in EXPECTED_SEEDS}
    if set(indexed) != expected:
        raise ValueError("Expected exactly mobilenetv2/multi_exit x seeds 51/52/53")
    for key, run in indexed.items():
        config = run["resolved_config"]
        if run.get("status") != "completed" or run.get("return_code") != 0:
            raise ValueError(f"Incomplete P0 run: {key}")
        if config.get("evaluate_test") is not False:
            raise ValueError(f"P0 run is not validation-only: {key}")
        if config.get("epochs") != 200:
            raise ValueError(f"Unexpected P0 epoch count: {key}")
    return manifest, indexed


def _run_paths(run: dict) -> tuple[Path, Path]:
    root = ROOT / "artifacts/runs" / run["experiment_id"]
    summary_path = root / "summary.json"
    checkpoint_path = root / "checkpoints/model_best.pth"
    if not summary_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing completed run evidence: {root}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("test_evaluated") is not False:
        raise ValueError(f"Official test data was evaluated in P0: {root}")
    if summary.get("best_checkpoint_sha256") != _sha256(checkpoint_path):
        raise ValueError(f"Best checkpoint hash mismatch: {root}")
    return root, checkpoint_path


def _collect_logits(run: dict, device: torch.device) -> tuple[np.ndarray, list[np.ndarray]]:
    config = _config(run["resolved_config"])
    _root, checkpoint_path = _run_paths(run)
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True), strict=True)
    model.eval()
    loaders = build_dataloaders(
        dataset=config.dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        validation_size=config.validation_size,
        split_seed=config.seed,
    )
    labels = []
    logits = defaultdict(list)
    with torch.no_grad():
        for inputs, targets in loaders.validation:
            outputs = model(inputs.to(device, non_blocking=True))
            values = outputs if isinstance(outputs, tuple) else (outputs,)
            labels.append(targets.numpy())
            for index, value in enumerate(values):
                logits[index].append(value.float().cpu().numpy())
    ordered = [np.concatenate(logits[index]) for index in sorted(logits)]
    return np.concatenate(labels), ordered


def _path_macs(model, exit_position: int | None) -> int:
    total = 0
    handles = []

    def count(module, inputs, output):
        nonlocal total
        if isinstance(module, nn.Conv2d):
            output_elements = output.numel()
            kernel = module.kernel_size[0] * module.kernel_size[1]
            total += output_elements * (module.in_channels // module.groups) * kernel
        elif isinstance(module, nn.Linear):
            total += output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(count))
    try:
        with torch.no_grad():
            model.forward_to_exit(torch.zeros(1, 3, 32, 32), exit_position)
    finally:
        for handle in handles:
            handle.remove()
    return total


def _cost_profile() -> dict:
    config = ExperimentConfig(
        model_type="multi_exit",
        exit_positions=(8, 16),
        exit_loss_weights=(0.2, 0.3),
    )
    model = build_model(config).eval()
    macs = [_path_macs(model, position) for position in (*config.exit_positions, None)]
    final = macs[-1]
    return {
        "method": "conv_linear_macs_per_sample_v1",
        "exit_positions": list(config.exit_positions),
        "path_macs": macs,
        "path_cost_fractions": [value / final for value in macs],
        "ignored_operations": ["batch_norm", "activations", "pooling", "memory_traffic"],
        "paper_evidence": False,
    }


def _accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(logits.argmax(axis=1) == labels))


def _sample_std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def analyze(manifest_path: Path) -> dict:
    _manifest, runs = _load_manifest(manifest_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cost_profile = _cost_profile()
    costs = cost_profile["path_cost_fractions"]
    seed_results = []
    for seed in EXPECTED_SEEDS:
        baseline_labels, baseline_values = _collect_logits(runs[("mobilenetv2", seed)], device)
        exit_labels, exit_values = _collect_logits(runs[("multi_exit", seed)], device)
        if not np.array_equal(baseline_labels, exit_labels):
            raise ValueError(f"Matched seed {seed} does not share validation order")
        if len(baseline_values) != 1 or len(exit_values) != 3:
            raise ValueError(f"Unexpected model output count for seed {seed}")
        labels = exit_labels
        final_logits, exit8_logits, exit16_logits = exit_values
        calibration = stratified_calibration_mask(labels, seed=20_260_902 + seed)
        evaluation = ~calibration
        selected = select_policy(
            labels[calibration],
            [exit8_logits[calibration], exit16_logits[calibration]],
            final_logits[calibration],
            costs,
        )
        predictions, paths = apply_policy(
            [exit8_logits[evaluation], exit16_logits[evaluation]],
            final_logits[evaluation],
            selected["thresholds"],
        )
        holdout = policy_metrics(
            labels[evaluation],
            predictions,
            final_logits[evaluation].argmax(axis=1),
            paths,
            costs,
        )
        baseline_holdout_accuracy = _accuracy(baseline_values[0][evaluation], labels[evaluation])
        holdout["matched_baseline_accuracy"] = baseline_holdout_accuracy
        holdout["accuracy_vs_matched_baseline"] = holdout["accuracy"] - baseline_holdout_accuracy
        baseline_accuracy = _accuracy(baseline_values[0], labels)
        final_accuracy = _accuracy(final_logits, labels)
        seed_results.append(
            {
                "seed": seed,
                "baseline_experiment_id": runs[("mobilenetv2", seed)]["experiment_id"],
                "multi_exit_experiment_id": runs[("multi_exit", seed)]["experiment_id"],
                "baseline_accuracy": baseline_accuracy,
                "multi_exit_final_accuracy": final_accuracy,
                "paired_final_gain": final_accuracy - baseline_accuracy,
                "exit8_accuracy": _accuracy(exit8_logits, labels),
                "exit16_accuracy": _accuracy(exit16_logits, labels),
                "calibration_samples": int(calibration.sum()),
                "holdout_samples": int(evaluation.sum()),
                "selected_policy": selected,
                "holdout_policy_metrics": holdout,
            }
        )

    final_gains = [value["paired_final_gain"] for value in seed_results]
    savings = [value["holdout_policy_metrics"]["cost_saving_fraction"] for value in seed_results]
    policy_drops = [value["holdout_policy_metrics"]["accuracy_drop"] for value in seed_results]
    worst_class_drops = [
        value["holdout_policy_metrics"]["worst_class_accuracy_drop"]
        for value in seed_results
    ]
    gates = {
        "mean_final_gain_at_least_minus_0_003": float(np.mean(final_gains)) >= -0.003,
        "each_final_gain_at_least_minus_0_0075": min(final_gains) >= -0.0075,
        "each_holdout_mac_saving_at_least_0_15": min(savings) >= 0.15,
        "each_holdout_accuracy_drop_at_most_0_01": max(policy_drops) <= 0.01,
        "each_holdout_worst_class_drop_at_most_0_03": max(worst_class_drops) <= 0.03,
    }
    proceed = all(gates.values())
    return {
        "schema_version": 1,
        "status": "go_formal_design" if proceed else "stop_or_redesign",
        "scope": "exploratory P0; validation split only; official test not evaluated",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "device": str(device),
        "cost_profile": cost_profile,
        "policy_calibration": {
            "split": "class-stratified 50/50 within each run's 5k validation set",
            "selection_constraints": {
                "overall_accuracy_drop": 0.005,
                "balanced_accuracy_drop": 0.005,
                "worst_class_accuracy_drop": 0.02,
            },
            "caveat": (
                "Best checkpoints were selected on the parent 5k validation set; this P0 holdout "
                "is exploratory and cannot be reported as independent paper evidence."
            ),
        },
        "seed_results": seed_results,
        "aggregate": {
            "paired_final_gain_mean": float(np.mean(final_gains)),
            "paired_final_gain_sample_std": _sample_std(final_gains),
            "holdout_mac_saving_mean": float(np.mean(savings)),
            "holdout_mac_saving_sample_std": _sample_std(savings),
            "holdout_policy_accuracy_drop_mean": float(np.mean(policy_drops)),
            "holdout_worst_class_drop_max": max(worst_class_drops),
        },
        "gates": gates,
        "recommended_next_step": (
            "Freeze a disjoint model-selection/calibration/evaluation protocol and fresh seeds "
            "before any paper-facing run."
            if proceed
            else "Do not expand early-exit training; inspect the failed gate before redesign."
        ),
    }


def _markdown(result: dict) -> str:
    rows = []
    for value in result["seed_results"]:
        holdout = value["holdout_policy_metrics"]
        rows.append(
            "| {seed} | {baseline:.2f} | {final:.2f} | {gain:+.2f} | {exit8:.2f} | "
            "{exit16:.2f} | {saving:.2f} | {drop:+.2f} | {worst:+.2f} |".format(
                seed=value["seed"],
                baseline=100 * value["baseline_accuracy"],
                final=100 * value["multi_exit_final_accuracy"],
                gain=100 * value["paired_final_gain"],
                exit8=100 * value["exit8_accuracy"],
                exit16=100 * value["exit16_accuracy"],
                saving=100 * holdout["cost_saving_fraction"],
                drop=100 * holdout["accuracy_drop"],
                worst=100 * holdout["worst_class_accuracy_drop"],
            )
        )
    aggregate = result["aggregate"]
    gates = "\n".join(
        f"- [{'x' if passed else ' '}] `{name}`" for name, passed in result["gates"].items()
    )
    return "\n".join(
        [
            "# Early-exit P0 analysis",
            "",
            f"Decision: **{result['status']}**",
            "",
            (
                "| seed | baseline % | final % | final gain pp | exit8 % | exit16 % | "
                "holdout MAC saving % | policy drop pp | worst-class drop pp |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            (
                f"Paired final gain: {100 * aggregate['paired_final_gain_mean']:+.3f} ± "
                f"{100 * aggregate['paired_final_gain_sample_std']:.3f} pp (sample std)."
            ),
            (
                f"Holdout MAC saving: {100 * aggregate['holdout_mac_saving_mean']:.2f} ± "
                f"{100 * aggregate['holdout_mac_saving_sample_std']:.2f}%."
            ),
            "",
            "## Frozen go/no-go gates",
            "",
            gates,
            "",
            (
                "This is exploratory validation-only evidence. The 5k parent validation set also "
                "selected the best checkpoint, so its 50/50 child split is not independent paper "
                "evidence. The official CIFAR-10 test set remains untouched."
            ),
            "",
            f"Next: {result['recommended_next_step']}",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite analysis output: {output}")
    result = analyze(manifest_path)
    output.mkdir(parents=True)
    (output / "analysis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (output / "analysis.md").write_text(_markdown(result), encoding="utf-8")
    print(f"Decision: {result['status']}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
