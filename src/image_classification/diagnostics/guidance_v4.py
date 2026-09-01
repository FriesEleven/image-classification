"""Version-matched checkpoint interventions for CSGHA-v4/v5/v6 and controls."""

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from image_classification.config import ExperimentConfig
from image_classification.data.cifar import DATASET_SPECS, _transforms
from image_classification.diagnostics.guidance import (
    array_sha256,
    derangement,
    paired_metrics,
    sha256,
    tensor_statistics,
)
from image_classification.models import build_model
from image_classification.models.attention import _rms_normalize_channels
from image_classification.paths import DATA_DIR, PROJECT_ROOT

MODEL_SOURCE_FILES = (
    "src/image_classification/config.py",
    "src/image_classification/models/attention.py",
    "src/image_classification/models/mobilenetv2.py",
    "src/image_classification/models/factory.py",
    "src/image_classification/data/cifar.py",
)


class _DescriptorReady(Exception):
    pass


def verify_training_sources(provenance: dict, source_snapshot: Path) -> dict:
    verified = {}
    recorded = provenance["source_sha256"]
    for relative in MODEL_SOURCE_FILES:
        expected = recorded[relative]
        archived = source_snapshot / relative
        current = PROJECT_ROOT / relative
        if not archived.is_file() or sha256(archived) != expected:
            raise ValueError(f"Training source snapshot mismatch: {relative}")
        if sha256(current) != expected:
            raise ValueError(f"Current model source differs from training snapshot: {relative}")
        verified[relative] = expected
    return verified


def loaders_from_saved_split(config: dict, run_directory: Path, batch_size: int, workers: int):
    spec = DATASET_SPECS[config["dataset"]]
    _, transform = _transforms(spec)
    data = spec.dataset_class(root=DATA_DIR, train=True, download=False, transform=transform)
    split_path = run_directory / "split_indices.json"
    split = json.loads(split_path.read_text())
    train_indices = split["train_indices"]
    validation_indices = split["validation_indices"]
    if len(train_indices) != 45000 or len(validation_indices) != 5000:
        raise ValueError("Saved split has unexpected sizes")
    options = {
        "batch_size": batch_size, "shuffle": False, "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(Subset(data, train_indices), **options),
        DataLoader(Subset(data, validation_indices), **options),
        train_indices,
        validation_indices,
        sha256(split_path),
    )


def channel_modules(model, model_type: str):
    guided = model_type in {"csgha_v4", "csgha_v5", "csgha_v6"}
    suffix = "guided_cbam.channel_attention" if guided else "cbam.channel_attention"
    modules = {name: module for name, module in model.named_modules() if name.endswith(suffix)}
    if len(modules) != 2:
        raise ValueError(f"Expected two target channel modules for {model_type}, found {list(modules)}")
    return modules


def collect_descriptors(model, loader, device):
    values = []

    def capture(_module, _inputs, output):
        values.append(output.mean((2, 3)).detach().cpu())
        raise _DescriptorReady

    source = model.model.features[model.guidance_position].se
    handle = source.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            for inputs, _ in loader:
                try:
                    model(inputs.to(device, non_blocking=True))
                except _DescriptorReady:
                    pass
    finally:
        handle.remove()
    descriptors = torch.cat(values)
    if len(descriptors) != len(loader.dataset):
        raise ValueError("Descriptor collection did not cover the dataset")
    return descriptors


def projected_guidance(module, descriptors):
    raw = module.guide_projection(module.guide_normalization(descriptors))
    guidance_for_bounding = raw
    if module.guidance_output_normalization == "rms":
        guidance_for_bounding = _rms_normalize_channels(raw)
    bounded = torch.tanh(guidance_for_bounding)
    contribution = module.guidance_scale_cap * torch.tanh(module.guidance_scale) * bounded
    return raw, bounded, contribution


def _statistics(storage: dict, guided: bool) -> dict:
    result = {}
    for name, entries in storage.items():
        values = {key: torch.cat(parts) for key, parts in entries.items()}
        result[name] = {key: tensor_statistics(value) for key, value in values.items()}
        gates = values["gate"]
        result[name]["sigmoid_saturation_fraction"] = (
            (gates < 0.05) | (gates > 0.95)
        ).float().mean().item()
        if guided:
            result[name]["tanh_saturation_fraction_abs_gt_099"] = (
                values["bounded"].abs() > 0.99
            ).float().mean().item()
    return result


def run_condition(model, loader, device, model_type, mode, descriptors=None, means=None, collect_stats=False):
    guided = model_type in {"csgha_v4", "csgha_v5", "csgha_v6"}
    modules = channel_modules(model, model_type)
    storage = {name: {} for name in modules}
    handles = []
    cursor = 0

    def pre_hook(_module, inputs):
        if mode == "descriptor":
            batch = inputs[0].shape[0]
            return inputs[0], descriptors[cursor:cursor + batch].to(device)

    def post_hook(name):
        def hook(module, inputs, output):
            features = inputs[0]
            avg_pre = module.fc[0](module.avg_pool(features))
            max_pre = module.fc[0](module.max_pool(features))
            avg_post = module.fc[1](avg_pre)
            max_post = module.fc[1](max_pre)
            deep = module.fc(module.avg_pool(features)) + module.fc(module.max_pool(features))
            tensors = {
                "deep": deep.flatten(1), "gate": output.flatten(1),
                "avg_hidden_pre_activation": avg_pre.flatten(1),
                "max_hidden_pre_activation": max_pre.flatten(1),
                "avg_hidden_post_activation": avg_post.flatten(1),
                "max_hidden_post_activation": max_post.flatten(1),
            }
            contribution = torch.zeros_like(deep.flatten(1))
            if guided:
                raw, bounded, contribution = projected_guidance(module, inputs[1])
                reconstructed = torch.sigmoid(deep + contribution[:, :, None, None])
                if not torch.allclose(output, reconstructed, atol=2e-6, rtol=2e-6):
                    raise ValueError("Guided deep+guidance decomposition disagrees with forward")
                tensors.update(raw=raw, bounded=bounded, contribution=contribution)
            elif not torch.allclose(output, torch.sigmoid(deep), atol=2e-6, rtol=2e-6):
                raise ValueError("Control deep decomposition disagrees with forward")
            if collect_stats:
                for key, tensor in tensors.items():
                    storage[name].setdefault(key, []).append(tensor.detach().cpu())
            if mode == "guidance_zero":
                return torch.sigmoid(deep)
            if mode == "mean_contribution":
                return torch.sigmoid(deep + means[name].to(device).view(1, -1, 1, 1))
            if mode == "deep_zero":
                return torch.sigmoid(contribution[:, :, None, None])
            return output
        return hook

    for name, module in modules.items():
        if guided:
            handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook(name)))
    logits, labels = [], []
    try:
        with torch.inference_mode():
            for inputs, targets in loader:
                logits.append(model(inputs.to(device, non_blocking=True)).cpu())
                labels.append(targets)
                cursor += len(targets)
    finally:
        for handle in handles:
            handle.remove()
    return torch.cat(logits), torch.cat(labels), _statistics(storage, guided) if collect_stats else {}


def preflight_retry1_run(row: dict, source_snapshot: Path):
    run_directory = PROJECT_ROOT / "artifacts/runs" / row["run_id"]
    config_data = yaml.safe_load((run_directory / "config.yaml").read_text())
    config_data.pop("runtime", None)
    if config_data != row["config"]:
        raise ValueError("Run config differs from P1 audit")
    provenance = json.loads((run_directory / "provenance.json").read_text())
    verified_sources = verify_training_sources(provenance, source_snapshot)
    checkpoint = run_directory / "checkpoints/model_best.pth"
    if sha256(checkpoint) != row["best_checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from P1 audit")
    config = ExperimentConfig(**{
        **config_data,
        "aux_positions": tuple(config_data["aux_positions"]),
        "se_positions": tuple(config_data["se_positions"]),
        "cbam_positions": tuple(config_data["cbam_positions"]),
    })
    model = build_model(config)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    split_path = run_directory / "split_indices.json"
    if sha256(split_path) != row["split_indices_sha256"]:
        raise ValueError("Saved split differs from P1 audit")
    return {
        "run_directory": run_directory,
        "config_data": config_data,
        "config": config,
        "model": model,
        "checkpoint": checkpoint,
        "verified_sources": verified_sources,
    }


def diagnose_retry1_run(row: dict, output: Path, source_snapshot: Path, batch_size=128, workers=4) -> dict:
    preflight = preflight_retry1_run(row, source_snapshot)
    run_directory = preflight["run_directory"]
    config_data = preflight["config_data"]
    config = preflight["config"]
    model = preflight["model"]
    checkpoint = preflight["checkpoint"]
    verified_sources = preflight["verified_sources"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    train_loader, validation_loader, train_indices, validation_indices, split_hash = loaders_from_saved_split(
        config_data, run_directory, batch_size, workers,
    )
    if split_hash != row["split_indices_sha256"]:
        raise ValueError("Saved split differs from P1 audit")
    original, labels, statistics = run_condition(
        model, validation_loader, device, config.model_type, "original", collect_stats=True,
    )
    conditions = {"original": paired_metrics(original, labels)}
    if abs(conditions["original"]["accuracy_percent"] - row["best_validation_percent"]) > 1e-8:
        raise ValueError("Full validation reproduction failed")
    arrays = {
        "train_indices": np.asarray(train_indices), "validation_indices": np.asarray(validation_indices),
        "labels": labels.numpy(), "original_logits": original.numpy(),
    }
    planned = [("deep_zero", "deep_zero", None)]
    means = {}
    if config.model_type in {"csgha_v4", "csgha_v5", "csgha_v6"}:
        train_descriptors = collect_descriptors(model, train_loader, device)
        validation_descriptors = collect_descriptors(model, validation_loader, device)
        with torch.inference_mode():
            for name, module in channel_modules(model, config.model_type).items():
                parts = [projected_guidance(module, part.to(device))[2].cpu()
                         for part in train_descriptors.split(batch_size)]
                means[name] = torch.cat(parts).double().mean(0).float()
        mean_descriptor = train_descriptors.mean(0).expand(len(labels), -1)
        arrays["train_mean_descriptor"] = train_descriptors.mean(0).numpy()
        planned += [
            ("guidance_zero", "guidance_zero", None),
            ("train_mean_descriptor", "descriptor", mean_descriptor),
            ("train_mean_contribution", "mean_contribution", None),
        ]
        for name, mean in means.items():
            arrays[f"train_mean_contribution__{name}"] = mean.numpy()
        for seed in (7301, 7302, 7303):
            permutation = derangement(len(labels), seed)
            arrays[f"permutation_{seed}"] = permutation.numpy()
            planned.append((f"permuted_{seed}", "descriptor", validation_descriptors[permutation]))
    for name, mode, descriptors in planned:
        logits, repeated_labels, _ = run_condition(
            model, validation_loader, device, config.model_type, mode, descriptors, means,
        )
        if not torch.equal(labels, repeated_labels):
            raise ValueError("Validation order changed")
        conditions[name] = paired_metrics(logits, labels, original)
        arrays[f"{name}_logits"] = logits.numpy()
    array_path = output / f"{config.model_type}_seed{config.seed}_paired_predictions.npz"
    np.savez_compressed(array_path, **arrays)
    report = {
        "run_id": row["run_id"], "model_type": config.model_type, "seed": config.seed,
        "architecture_version": config.architecture_version,
        "checkpoint_sha256": sha256(checkpoint), "split_indices_sha256": split_hash,
        "verified_training_sources": verified_sources, "strict_state_dict_load": True,
        "original_validation_matches_audit": True, "official_test_evaluated": False,
        "train_index_sha256": array_sha256(train_indices),
        "validation_index_sha256": array_sha256(validation_indices),
        "mean_source": "all 45000 saved training indices; evaluation transform; no labels",
        "permutation_seeds": [7301, 7302, 7303]
        if config.model_type in {"csgha_v4", "csgha_v5", "csgha_v6"} else [],
        "permutation_scope": "full saved validation split; no fixed points"
        if config.model_type in {"csgha_v4", "csgha_v5", "csgha_v6"} else None,
        "batch_size": batch_size, "precision": "float32; no autocast",
        "statistics": statistics, "conditions": conditions,
        "paired_arrays": str(array_path), "paired_arrays_sha256": sha256(array_path),
    }
    report_path = output / f"{config.model_type}_seed{config.seed}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
