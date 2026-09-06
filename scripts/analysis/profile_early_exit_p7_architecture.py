"""Profile label-free MobileNetV2 224x224 MACs and freeze P7 exit positions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch

from image_classification.config import ExperimentConfig
from image_classification.models import build_model
from image_classification.training.benchmark import model_metrics


def canonical_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def select_boundary(candidates: list[dict], target: float, *, after: int = 0) -> dict:
    eligible = [candidate for candidate in candidates if candidate["position"] > after]
    if not eligible:
        raise ValueError("No eligible MobileNetV2 feature boundary")
    return min(eligible, key=lambda item: (abs(item["fraction"] - target), item["position"]))


def execute(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    all_positions = tuple(range(1, 18))
    config = ExperimentConfig(
        model_type="multi_exit",
        dataset="imagenet100",
        exit_positions=all_positions,
        exit_loss_weights=(1.0,) * len(all_positions),
        exit_distillation_alpha=0.0,
    )
    torch.manual_seed(0)
    model = build_model(config)
    metrics = model_metrics(model, config)
    final_macs = metrics["path_macs"]["final"]
    candidates = [
        {
            "position": position,
            "channels": model.exit_heads[str(position)].classifier.in_features,
            "path_macs_including_exit_head": metrics["path_macs"][f"exit{position}"],
            "fraction": metrics["path_macs"][f"exit{position}"] / final_macs,
        }
        for position in all_positions
    ]
    deploy = select_boundary(candidates, 0.425)
    auxiliary = select_boundary(candidates, 0.825, after=deploy["position"])
    selected_config = ExperimentConfig(
        model_type="multi_exit",
        dataset="imagenet100",
        exit_positions=(deploy["position"], auxiliary["position"]),
        exit_loss_weights=(0.1, 0.15),
        exit_distillation_alpha=0.0,
    )
    torch.manual_seed(0)
    selected_model = build_model(selected_config)
    selected_metrics = model_metrics(selected_model, selected_config)
    parameter_schema = [
        {"name": name, "shape": list(value.shape)}
        for name, value in selected_model.state_dict().items()
    ]
    report = {
        "schema_version": 1,
        "status": "profiled_before_p7_training",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_or_labels_accessed": False,
        "input_shape": [1, 3, 224, 224],
        "selection_rule": {
            "deployable_target_fraction": 0.425,
            "auxiliary_target_fraction": 0.825,
            "eligible_boundaries": "MobileNetV2 completed features 1 through 17",
            "tie_break": "lower feature position",
        },
        "final_path_macs": final_macs,
        "candidates": candidates,
        "selected": {
            "deployable": deploy,
            "training_only_auxiliary": auxiliary,
        },
        "selected_model": {
            "exit_positions": list(selected_config.exit_positions),
            "exit_loss_weights": list(selected_config.exit_loss_weights),
            "parameters_total": selected_metrics["parameters_total"],
            "parameters_exit_heads": selected_metrics["parameters_exit_heads"],
            "path_macs": selected_metrics["path_macs"],
            "parameter_schema_sha256": canonical_sha256(parameter_schema),
        },
    }
    if (deploy["position"], auxiliary["position"]) != (8, 15):
        raise RuntimeError("Unexpected P7 exit-position selection")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.output.resolve()), indent=2))
