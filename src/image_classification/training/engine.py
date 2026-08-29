"""End-to-end experiment orchestration."""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report
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
from image_classification.training.evaluate import save_evaluation_data, validate
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


def _step_optimizer_and_scheduler(optimizer, scheduler, scaler) -> bool:
    """Advance the scheduler only when AMP did not skip the optimizer update."""
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    optimizer_was_run = scaler.get_scale() >= scale_before
    if optimizer_was_run:
        scheduler.step()
    return optimizer_was_run


def _train_epoch(model, loader, criterion, optimizer, scheduler, scaler, config, device) -> dict:
    model.train()
    optimizer.zero_grad()
    accumulated_loss = 0.0
    predictions: list[int] = []
    labels: list[int] = []
    amp_enabled = config.amp and device.type == "cuda"
    with tqdm(loader, desc="Training", unit="batch", disable=not sys.stderr.isatty()) as progress:
        for step, (inputs, targets) in enumerate(progress, start=1):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            group_start = ((step - 1) // config.accumulation_steps) * config.accumulation_steps
            group_size = min(config.accumulation_steps, len(loader) - group_start)
            with autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)
                loss = batch_loss / group_size
            scaler.scale(loss).backward()
            accumulated_loss += batch_loss.item()
            predictions.extend(outputs.argmax(dim=1).detach().cpu().numpy())
            labels.extend(targets.cpu().numpy())
            progress.set_postfix(loss=f"{batch_loss.item():.4f}")
            if step % config.accumulation_steps == 0 or step == len(loader):
                _step_optimizer_and_scheduler(optimizer, scheduler, scaler)
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    return {
        "loss": accumulated_loss / len(loader),
        "accuracy": float(np.mean(np.asarray(predictions) == np.asarray(labels))),
        "precision": float(report["macro avg"]["precision"]),
        "recall": float(report["macro avg"]["recall"]),
        "f1": float(report["macro avg"]["f1-score"]),
        "learning_rate": optimizer.param_groups[0]["lr"],
    }


def train(config: ExperimentConfig) -> dict:
    seed_everything(config.seed)
    paths = RunPaths(config.experiment_id).create()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    _save_resolved_config(paths.root / "config.yaml", config, device)
    _write_json(paths.root / "metrics.json", model_metrics(model, config))
    _write_json(paths.root / "benchmark.json", benchmark_inference(model, device))

    loaders = build_dataloaders(
        dataset=config.dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        validation_size=config.validation_size,
        split_seed=config.seed,
    )
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
    model.load_state_dict(
        torch.load(paths.checkpoints / "model_best.pth", map_location=device, weights_only=True)
    )
    test_metrics, labels, predictions, probabilities = validate(
        model, loaders.test, criterion, device, description="Test",
    )
    roc_auc = save_evaluation_data(
        paths.predictions, labels, predictions, probabilities, loaders.class_names,
    )
    summary = {
        "experiment_id": config.experiment_id,
        "dataset": config.dataset,
        "num_classes": config.num_classes,
        "train_samples": len(loaders.train.dataset),
        "validation_samples": len(loaders.validation.dataset),
        "test_samples": len(loaders.test.dataset),
        "best_validation_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "micro_auc": float(roc_auc["micro"]),
        "run_directory": str(paths.root),
    }
    _write_json(paths.root / "summary.json", summary)
    return summary
