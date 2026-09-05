"""Profile label-free ResNet-18 block-boundary MACs and freeze P6 exits."""

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
from torch import nn

from image_classification.config import ExperimentConfig
from image_classification.models import build_model
from image_classification.training.benchmark import model_metrics


def canonical_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def select_boundary(candidates: list[dict], target: float, *, after: int = -1) -> dict:
    eligible = [candidate for candidate in candidates if candidate["position"] > after]
    if not eligible:
        raise ValueError("No eligible residual-block boundary")
    return min(eligible, key=lambda candidate: (abs(candidate["fraction"] - target), candidate["position"]))


def module_signature(model: nn.Module) -> list[dict]:
    signature = []
    for name, module in model.named_modules():
        row = {"name": name, "type": type(module).__name__}
        if isinstance(module, nn.Conv2d):
            row.update(in_channels=module.in_channels, out_channels=module.out_channels,
                       kernel_size=list(module.kernel_size), stride=list(module.stride),
                       padding=list(module.padding), groups=module.groups, bias=module.bias is not None)
        elif isinstance(module, nn.Linear):
            row.update(in_features=module.in_features, out_features=module.out_features,
                       bias=module.bias is not None)
        signature.append(row)
    return signature


def profile(dataset: str) -> dict:
    config = ExperimentConfig(
        model_type="resnet18_multi_exit",
        dataset=dataset,
        exit_positions=tuple(range(7)),
        exit_loss_weights=(1.0,) * 7,
        exit_distillation_alpha=0.0,
    )
    torch.manual_seed(0)
    model = build_model(config)
    metrics = model_metrics(model, config)
    final_macs = metrics["path_macs"]["final"]
    candidates = [
        {
            "position": position,
            "boundary": f"after_residual_block_{position}",
            "channels": model.exit_heads[str(position)].classifier.in_features,
            "path_macs_including_exit_head": metrics["path_macs"][f"exit{position}"],
            "fraction": metrics["path_macs"][f"exit{position}"] / final_macs,
        }
        for position in range(7)
    ]
    deploy = select_boundary(candidates, 0.425)
    auxiliary = select_boundary(candidates, 0.825, after=deploy["position"])
    parameter_schema = [{"name": name, "shape": list(value.shape)} for name, value in model.state_dict().items()]
    graph = module_signature(model)
    selected_config = ExperimentConfig(
        model_type="resnet18_multi_exit",
        dataset=dataset,
        exit_positions=(deploy["position"], auxiliary["position"]),
        exit_loss_weights=(0.1, 0.15),
        exit_distillation_alpha=0.0,
    )
    torch.manual_seed(0)
    selected_model = build_model(selected_config)
    selected_metrics = model_metrics(selected_model, selected_config)
    selected_parameter_schema = [
        {"name": name, "shape": list(value.shape)}
        for name, value in selected_model.state_dict().items()
    ]
    selected_graph = module_signature(selected_model)
    return {
        "dataset": dataset,
        "num_classes": config.num_classes,
        "input_shape": [1, 3, 32, 32],
        "final_path_macs": final_macs,
        "candidates": candidates,
        "selected": {
            "deployable": deploy,
            "training_only_auxiliary": auxiliary,
        },
        "parameters_with_all_candidate_heads": metrics["parameters_total"],
        "parameter_schema_sha256": canonical_sha256(parameter_schema),
        "module_graph_sha256": canonical_sha256(graph),
        "selected_model": {
            "exit_positions": list(selected_config.exit_positions),
            "exit_loss_weights": list(selected_config.exit_loss_weights),
            "parameters_total": selected_metrics["parameters_total"],
            "parameters_exit_heads": selected_metrics["parameters_exit_heads"],
            "path_macs": selected_metrics["path_macs"],
            "parameter_schema_sha256": canonical_sha256(selected_parameter_schema),
            "module_graph_sha256": canonical_sha256(selected_graph),
        },
    }


def execute(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    report = {
        "schema_version": 1,
        "status": "profiled_before_p6_training",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_or_labels_accessed": False,
        "selection_rule": {
            "deployable_target_fraction": 0.425,
            "auxiliary_target_fraction": 0.825,
            "eligible_boundaries": "completed residual blocks 0 through 6",
            "tie_break": "lower block position",
        },
        "profiles": [profile("cifar10"), profile("cifar100")],
    }
    selections = {
        (profile["selected"]["deployable"]["position"],
         profile["selected"]["training_only_auxiliary"]["position"])
        for profile in report["profiles"]
    }
    if selections != {(2, 6)}:
        raise RuntimeError(f"Unexpected architecture selection: {selections}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.output.resolve()), indent=2))
