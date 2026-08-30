"""Historical CSGHA loading and paired guidance interventions (no training)."""

import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from image_classification.data.cifar import DATASET_SPECS, _transforms, stratified_split_indices
from image_classification.paths import DATA_DIR, PROJECT_ROOT


def sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def array_sha256(array) -> str:
    return hashlib.sha256(np.asarray(array, dtype="<i8").tobytes()).hexdigest()


def load_historical_model(revision: str, config: dict):
    """Execute the actual two model source files from Git, not a new approximation."""
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"], cwd=PROJECT_ROOT, text=True,
    ).strip()
    package_name = f"_csgha_history_{revision}"
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package
    sources = {}
    for name in ("attention", "mobilenetv2"):
        path = f"src/image_classification/models/{name}.py"
        source = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=PROJECT_ROOT)
        sources[path] = hashlib.sha256(source).hexdigest()
        full_name = f"{package_name}.{name}"
        spec = importlib.util.spec_from_loader(full_name, loader=None)
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        # Intentionally load trusted repository code at an explicit resolved Git
        # commit; replacing this with the current import would defeat version matching.
        exec(compile(source, f"git:{revision}:{path}", "exec"), module.__dict__)  # noqa: S102
    architecture = sys.modules[f"{package_name}.mobilenetv2"].CSGHAMobileNetV2
    model = architecture(
        num_classes=DATASET_SPECS[config["dataset"]].num_classes,
        se_positions=tuple(config["se_positions"]), cbam_positions=tuple(config["cbam_positions"]),
        guidance_position=config["guidance_position"], guidance_reduction=config["guidance_reduction"],
    )
    # Evaluation transforms and split algorithm must agree with the historical source.
    data_path = "src/image_classification/data/cifar.py"
    historical_data = subprocess.check_output(["git", "show", f"{revision}:{data_path}"], cwd=PROJECT_ROOT)
    if historical_data != (PROJECT_ROOT / data_path).read_bytes():
        raise ValueError("Current CIFAR preprocessing differs from historical code")
    sources[data_path] = hashlib.sha256(historical_data).hexdigest()
    return model, {"reference_commit": revision, "source_sha256": sources}


def diagnostic_loaders(config: dict, batch_size: int, workers: int):
    """Use the official TRAIN data only; the official test dataset is never opened."""
    spec = DATASET_SPECS[config["dataset"]]
    _, transform = _transforms(spec)
    data = spec.dataset_class(root=DATA_DIR, train=True, download=False, transform=transform)
    train_indices, val_indices = stratified_split_indices(
        data.targets, config["validation_size"], config["seed"],
    )
    options = {"batch_size": batch_size, "shuffle": False, "num_workers": workers, "pin_memory": True}
    return (
        DataLoader(Subset(data, train_indices), **options),
        DataLoader(Subset(data, val_indices), **options),
        train_indices, val_indices,
    )


def channel_modules(model):
    return {name: module for name, module in model.named_modules()
            if name.endswith("guided_cbam.channel_attention")}


class _DescriptorReady(Exception):
    pass


def collect_descriptors(model, loader, device):
    """Stop immediately after source SE; do not repeat the downstream network."""
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


def projected_guidance(module, descriptors, version):
    normalized = descriptors if version == "v1" else module.guide_normalization(descriptors)
    raw = module.guide_projection(normalized)
    bounded = torch.tanh(raw) if version == "v3" else raw
    contribution = bounded if version == "v1" else torch.tanh(module.guidance_scale) * bounded
    return raw, bounded, contribution


def derangement(size: int, seed: int) -> torch.Tensor:
    if size < 2:
        raise ValueError("A permutation intervention needs at least two samples")
    generator = torch.Generator().manual_seed(seed)
    for _ in range(1000):
        permutation = torch.randperm(size, generator=generator)
        if not torch.any(permutation == torch.arange(size)):
            return permutation
    raise RuntimeError("Failed to construct a derangement")


def tensor_statistics(values: torch.Tensor) -> dict:
    values = values.double()
    return {
        "abs_mean": values.abs().mean().item(), "abs_max": values.abs().max().item(),
        "sample_std_per_channel_mean": values.std(dim=0, correction=0).mean().item(),
        "zero_fraction": (values == 0).double().mean().item(),
        "positive_fraction": (values > 0).double().mean().item(),
        "constant_channel_fraction": (values.std(dim=0, correction=0) < 1e-7).double().mean().item(),
    }


def run_intervention(model, loader, device, version, mode, descriptors, means, collect_stats=False):
    """Only alter the shallow contribution. Both targets use the same donor image."""
    modules = channel_modules(model)
    cursor = 0
    storage = {name: {} for name in modules}
    handles = []

    def pre_hook(_module, inputs):
        if mode == "descriptor":
            replacement = descriptors[cursor:cursor + inputs[0].shape[0]].to(device)
            return inputs[0], replacement

    def post_hook(name):
        def hook(module, inputs, output):
            features, shallow = inputs
            avg_hidden = module.fc[0](module.avg_pool(features))
            max_hidden = module.fc[0](module.max_pool(features))
            deep = module.fc(module.avg_pool(features)) + module.fc(module.max_pool(features))
            raw, bounded, contribution = projected_guidance(module, shallow, version)
            reconstructed = torch.sigmoid(deep + contribution[:, :, None, None])
            if not torch.allclose(output, reconstructed, atol=2e-6, rtol=2e-6):
                raise ValueError("Version-specific decomposition disagrees with the historical forward")
            if collect_stats:
                tensors = {"deep": deep.flatten(1), "raw": raw, "bounded": bounded,
                           "contribution": contribution, "gate": output.flatten(1),
                           "avg_hidden_pre_relu": avg_hidden.flatten(1),
                           "max_hidden_pre_relu": max_hidden.flatten(1)}
                for key, tensor in tensors.items():
                    storage[name].setdefault(key, []).append(tensor.detach().cpu())
            if mode == "zero":
                return torch.sigmoid(deep)
            if mode == "mean_contribution":
                return torch.sigmoid(deep + means[name].to(device).view(1, -1, 1, 1))
            return output
        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook(name)))
    logits, targets = [], []
    try:
        with torch.inference_mode():
            for inputs, labels in loader:
                logits.append(model(inputs.to(device, non_blocking=True)).cpu())
                targets.append(labels)
                cursor += len(labels)
    finally:
        for handle in handles:
            handle.remove()
    statistics = {}
    for name, entries in storage.items():
        if not entries:
            continue
        values = {key: torch.cat(parts) for key, parts in entries.items()}
        statistics[name] = {key: tensor_statistics(value) for key, value in values.items()}
        gates = values["gate"]
        statistics[name]["sigmoid_saturation_fraction"] = ((gates < 0.05) | (gates > 0.95)).float().mean().item()
        if version == "v3":
            statistics[name]["tanh_saturation_fraction_abs_gt_099"] = (
                values["bounded"].abs() > 0.99
            ).float().mean().item()
    return torch.cat(logits), torch.cat(targets), statistics


def paired_metrics(logits, labels, original=None):
    predictions = logits.argmax(1)
    correct = predictions == labels
    result = {"samples": len(labels), "correct": int(correct.sum()),
              "accuracy_percent": correct.double().mean().item() * 100,
              "nll": F.cross_entropy(logits.double(), labels).item()}
    if original is not None:
        original_correct = original.argmax(1) == labels
        result.update(
            delta_accuracy_pp=(correct.double().mean() - original_correct.double().mean()).item() * 100,
            prediction_changes=int((predictions != original.argmax(1)).sum()),
            originally_correct_now_wrong=int((original_correct & ~correct).sum()),
            originally_wrong_now_correct=int((~original_correct & correct).sum()),
            mean_absolute_logit_change=(logits - original).abs().mean().item(),
            mean_probability_l1=(logits.softmax(1) - original.softmax(1)).abs().sum(1).mean().item(),
        )
    return result


def diagnose_run(row: dict, output: Path, batch_size=128, workers=4) -> dict:
    import yaml

    version = row["variant"].split()[-1]
    references = {"v1": "6dc4c57", "v2": "82625b4", "v3": "f11d0af"}
    provenance = row["provenance"]
    revision = provenance[0]["git_commit"] if provenance else references[version]
    run_directory = PROJECT_ROOT / "artifacts/runs" / row["run_id"]
    config = yaml.safe_load((run_directory / "config.yaml").read_text())
    if config != row["config"]:
        raise ValueError("Run config changed since the audit")
    checkpoint = run_directory / "checkpoints/model_best.pth"
    if sha256(checkpoint) != row["best_checkpoint"]["sha256"]:
        raise ValueError("Checkpoint changed since the audit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, historical = load_historical_model(revision, config)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"), strict=True)
    model.to(device).eval()
    train_loader, validation_loader, train_indices, val_indices = diagnostic_loaders(config, batch_size, workers)
    original, labels, statistics = run_intervention(
        model, validation_loader, device, version, "original", None, None, collect_stats=True,
    )
    original_metrics = paired_metrics(original, labels)
    if abs(original_metrics["accuracy_percent"] - row["best_validation_percent"]) > 1e-8:
        raise ValueError(f"Historical validation reproduction failed: {original_metrics} vs audit {row['best_validation_percent']}")
    print(f"{row['id']} {version} seed={row['seed']}: original reproduced {original_metrics['accuracy_percent']:.2f}%", flush=True)
    train_descriptors = collect_descriptors(model, train_loader, device)
    val_descriptors = collect_descriptors(model, validation_loader, device)
    means = {}
    with torch.inference_mode():
        for name, module in channel_modules(model).items():
            contributions = [projected_guidance(module, part.to(device), version)[2].cpu()
                             for part in train_descriptors.split(batch_size)]
            means[name] = torch.cat(contributions).double().mean(0).float()
    conditions = [("zero_guidance", "zero", None),
                  ("train_mean_descriptor", "descriptor", train_descriptors.mean(0).expand(len(labels), -1)),
                  ("train_mean_contribution", "mean_contribution", None)]
    arrays = {"validation_indices": np.asarray(val_indices), "train_indices": np.asarray(train_indices),
              "labels": labels.numpy(), "original_logits": original.numpy(),
              "train_mean_descriptor": train_descriptors.mean(0).numpy()}
    for name, mean in means.items():
        arrays[f"train_mean_contribution__{name}"] = mean.numpy()
    for seed in (7301, 7302, 7303):
        permutation = derangement(len(labels), seed)
        arrays[f"permutation_{seed}"] = permutation.numpy()
        conditions.append((f"permuted_{seed}", "descriptor", val_descriptors[permutation]))
    results = {"original": original_metrics}
    for name, mode, descriptors in conditions:
        logits, repeated_labels, _ = run_intervention(
            model, validation_loader, device, version, mode, descriptors, means,
        )
        if not torch.equal(labels, repeated_labels):
            raise ValueError("Validation order changed between conditions")
        results[name] = paired_metrics(logits, labels, original)
        arrays[f"{name}_logits"] = logits.numpy()
        print(f"  {name}: {results[name]}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    array_path = output / f"{row['id']}_paired_predictions.npz"
    if array_path.exists():
        raise FileExistsError(array_path)
    np.savez_compressed(array_path, **arrays)
    report = {
        "run_id": row["run_id"], "audit_id": row["id"], "version": version, "seed": row["seed"],
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "historical_model": historical,
        "version_evidence": "manifest_commit" if provenance else "reference_implementation_not_recorded_runtime_commit",
        "strict_state_dict_load": True, "original_validation_matches_audit": True,
        "official_test_evaluated": False, "config": config,
        "train_index_sha256": array_sha256(train_indices), "validation_index_sha256": array_sha256(val_indices),
        "mean_source": "all 45000 train images, evaluation transforms, no labels used",
        "permutation_seeds": [7301, 7302, 7303], "permutation_scope": "all validation images; no fixed points",
        "batch_size": batch_size, "precision": "float32; no autocast", "statistics": statistics,
        "conditions": results, "paired_arrays": str(array_path), "paired_arrays_sha256": sha256(array_path),
    }
    (output / f"{row['id']}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
