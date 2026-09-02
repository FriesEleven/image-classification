"""End-to-end experiment orchestration."""

import json
import math
import sys
import time
from pathlib import Path

import torch
import yaml
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from image_classification.config import ExperimentConfig
from image_classification.data import build_dataloaders
from image_classification.models import build_model
from image_classification.paths import RunPaths
from image_classification.training.benchmark import benchmark_inference, model_metrics
from image_classification.training.checkpoint import append_training_log, save_checkpoint
from image_classification.training.cuda_graph import prepare_training_graph
from image_classification.training.evaluate import EpochAccumulator, save_evaluation_data, validate
from image_classification.training.objectives import primary_logits, training_objective
from image_classification.training.optimizer_step import OptimizerStepTracker
from image_classification.training.provenance import file_sha256, runtime_provenance
from image_classification.utils import seed_everything


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def _save_resolved_config(path: Path, config: ExperimentConfig, device: torch.device) -> None:
    value = config.to_dict()
    value["runtime"] = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else str(device).upper(),
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True)


def optimizer_updates_per_epoch(num_batches: int, accumulation_steps: int) -> int:
    """Return the number of optimizer updates made during one training epoch."""
    return math.ceil(num_batches / accumulation_steps)


def _step_optimizer_and_scheduler(optimizer, scheduler, scaler, tracker=None) -> bool:
    """Advance the scheduler only when AMP did not skip the optimizer update."""
    tracked = tracker is not None and tracker.supported
    if tracked:
        tracker.did_step = False
    else:
        scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    optimizer_was_run = tracker.did_step if tracked else scaler.get_scale() >= scale_before
    if optimizer_was_run:
        scheduler.step()
    return optimizer_was_run


def _train_epoch(model, loader, criterion, optimizer, scheduler, scaler, config, device) -> dict:
    model.train()
    optimizer.zero_grad()
    accumulator = EpochAccumulator()
    amp_enabled = config.amp and device.type == "cuda"
    interactive = sys.stderr.isatty()
    with OptimizerStepTracker(optimizer) as tracker, tqdm(
        loader, desc="Training", unit="batch", disable=not interactive,
    ) as progress:
        for step, (inputs, targets) in enumerate(progress, start=1):
            host_targets = targets
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            group_start = ((step - 1) // config.accumulation_steps) * config.accumulation_steps
            group_size = min(config.accumulation_steps, len(loader) - group_start)
            with autocast(device_type=device.type, enabled=amp_enabled, cache_enabled=not config.cuda_graph):
                outputs = model(inputs)
                batch_loss = training_objective(outputs, targets, criterion, config)
                loss = batch_loss / group_size
            scaler.scale(loss).backward()
            accumulator.update(primary_logits(outputs), host_targets, batch_loss)
            if interactive and step % 50 == 0:
                progress.set_postfix(loss=f"{batch_loss.item():.4f}")
            if step % config.accumulation_steps == 0 or step == len(loader):
                _step_optimizer_and_scheduler(optimizer, scheduler, scaler, tracker)
    metrics, _labels, _predictions, _probabilities = accumulator.finish()
    metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
    return metrics


def train(config: ExperimentConfig) -> dict:
    if config.torch_num_threads:
        torch.set_num_threads(config.torch_num_threads)
    seed_everything(config.seed)
    paths = RunPaths(config.experiment_id).create()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    _save_resolved_config(paths.root / "config.yaml", config, device)
    provenance = {
        **runtime_provenance(), "architecture_version": config.architecture_version,
        "command": sys.argv,
        "split_seed": config.seed if config.split_seed is None else config.split_seed,
        "training_seed": config.seed,
        "seed_protocol": (
            "split and training seeds are intentionally coupled; historical protocol unchanged"
            if config.split_seed is None
            else "fixed data split seed is independent of model/training seed"
        ),
        "training_implementation": "deferred_metrics_post_step_hook_v1",
        "execution_backend": "cuda_graph_training_v1" if config.cuda_graph else "eager",
        "amp_cache_enabled": not config.cuda_graph,
        "inference_benchmark_enabled": config.measure_inference,
    }
    _write_json(paths.root / "provenance.json", provenance)
    _write_json(paths.root / "metrics.json", model_metrics(model, config))
    if config.measure_inference:
        benchmark = benchmark_inference(model, device)
    else:
        # Shared-GPU training latency is not a valid isolated inference result.
        # Null values are intentionally distinct from a measured zero latency.
        benchmark = {
            "measurement_status": "skipped",
            "reason": "Inference timing disabled for this throughput/concurrent-training configuration; remeasure alone",
            "inference_latency_mean": None, "inference_latency_std": None,
            "throughput_fps": None, "num_runs": 0,
        }
    _write_json(paths.root / "benchmark.json", benchmark)
    if config.cuda_graph:
        capture = prepare_training_graph(model, config.batch_size, device, config.amp)
        provenance["graph_capture"] = capture
        _write_json(paths.root / "provenance.json", provenance)
        print(f"Training graph prepared in {capture['capture_seconds']:.2f}s", flush=True)

    resolved_split_seed = config.seed if config.split_seed is None else config.split_seed
    loaders = build_dataloaders(
        dataset=config.dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        validation_size=config.validation_size,
        split_seed=resolved_split_seed,
        calibration_size=config.calibration_size,
        shuffle_seed=config.seed,
    )
    split_path = paths.root / "split_indices.json"
    calibration_loader = getattr(loaders, "calibration", None)
    split_record = {
        "dataset": config.dataset,
        "split_seed": resolved_split_seed,
        "train_indices": getattr(loaders.train.dataset, "indices", None),
        "validation_indices": getattr(loaders.validation.dataset, "indices", None),
    }
    if config.calibration_size or config.split_seed is not None:
        split_record["training_seed"] = config.seed
        split_record["calibration_indices"] = (
            getattr(calibration_loader.dataset, "indices", None)
            if calibration_loader is not None
            else None
        )
    _write_json(split_path, split_record)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    updates_per_epoch = optimizer_updates_per_epoch(len(loaders.train), config.accumulation_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.lr, epochs=config.epochs, steps_per_epoch=updates_per_epoch,
        pct_start=0.3, anneal_strategy="cos", div_factor=25, final_div_factor=100,
    )
    best_accuracy = -1.0
    best_epoch = 0
    train_losses: list[float] = []
    val_accuracies: list[float] = []

    with SummaryWriter(paths.tensorboard) as writer:
        for epoch in range(config.epochs):
            started = time.perf_counter()
            train_metrics = _train_epoch(
                model, loaders.train, criterion, optimizer, scheduler, scaler, config, device,
            )
            train_metrics["learning_rate"] = scheduler.get_last_lr()[0]
            val_metrics, _labels, _predictions, _probabilities = validate(
                model, loaders.validation, criterion, device,
            )
            train_losses.append(train_metrics["loss"])
            val_accuracies.append(val_metrics["accuracy"])
            append_training_log(paths.training_log, epoch, train_metrics, val_metrics)
            for name, value in (("Loss/train", train_metrics["loss"]), ("Loss/val", val_metrics["loss"]),
                                ("Accuracy/train", train_metrics["accuracy"]), ("Accuracy/val", val_metrics["accuracy"]),
                                ("Learning_rate", train_metrics["learning_rate"])):
                writer.add_scalar(name, value, epoch)
            if (epoch + 1) % 10 == 0:
                save_checkpoint(paths.checkpoints / f"epoch_{epoch + 1}.pth", epoch, model, optimizer, scheduler,
                                best_accuracy, train_losses, val_accuracies)
            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy = val_metrics["accuracy"]
                best_epoch = epoch + 1
                torch.save(model.state_dict(), paths.checkpoints / "model_best.pth")
            torch.save(model.state_dict(), paths.checkpoints / "model_latest.pth")
            print(
                f"Epoch {epoch + 1}/{config.epochs} ({time.perf_counter() - started:.2f}s) | "
                f"train_acc={train_metrics['accuracy']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
                f"best={best_accuracy:.4f}",
                flush=True,
            )

    save_checkpoint(paths.checkpoints / "final.pth", config.epochs - 1, model, optimizer, scheduler,
                    best_accuracy, train_losses, val_accuracies)
    summary = {
        "experiment_id": config.experiment_id,
        "dataset": config.dataset,
        "num_classes": config.num_classes,
        "train_samples": len(loaders.train.dataset),
        "validation_samples": len(loaders.validation.dataset),
        "test_samples": len(loaders.test.dataset),
        "test_evaluated": config.evaluate_test,
        "best_validation_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "architecture_version": config.architecture_version,
        "best_checkpoint_sha256": file_sha256(paths.checkpoints / "model_best.pth"),
        "split_indices_sha256": file_sha256(split_path),
        "run_directory": str(paths.root),
    }
    if calibration_loader is not None:
        summary["calibration_samples"] = len(calibration_loader.dataset)
    if config.evaluate_test:
        model.load_state_dict(
            torch.load(paths.checkpoints / "model_best.pth", map_location=device, weights_only=True)
        )
        test_metrics, labels, predictions, probabilities = validate(
            model, loaders.test, criterion, device, description="Test",
        )
        roc_auc = save_evaluation_data(
            paths.predictions, labels, predictions, probabilities, loaders.class_names,
        )
        summary.update(
            {
                "test_accuracy": test_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "micro_auc": float(roc_auc["micro"]),
            }
        )
    _write_json(paths.root / "summary.json", summary)
    return summary
