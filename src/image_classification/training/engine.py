"""End-to-end experiment orchestration."""

import json
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


def _train_epoch(model, loader, criterion, optimizer, scaler, config, device) -> dict:
    model.train()
    optimizer.zero_grad()
    accumulated_loss = 0.0
    predictions: list[int] = []
    labels: list[int] = []
    amp_enabled = config.amp and device.type == "cuda"
    with tqdm(loader, desc="Training", unit="batch") as progress:
        for step, (inputs, targets) in enumerate(progress, start=1):
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(inputs)
                loss = criterion(outputs, targets) / config.accumulation_steps
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()
            predictions.extend(outputs.argmax(dim=1).detach().cpu().numpy())
            labels.extend(targets.cpu().numpy())
            progress.set_postfix(loss=f"{loss.item() * config.accumulation_steps:.4f}")
            if step % config.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        if len(loader) % config.accumulation_steps:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    return {
        "loss": accumulated_loss / len(loader) * config.accumulation_steps,
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

    train_loader, test_loader = build_dataloaders(config.batch_size, config.num_workers)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.lr, epochs=config.epochs, steps_per_epoch=len(train_loader),
        pct_start=0.3, anneal_strategy="cos", div_factor=25, final_div_factor=100,
    )
    best_accuracy = 0.0
    best_epoch = 0
    best_outputs = None
    train_losses: list[float] = []
    val_accuracies: list[float] = []

    with SummaryWriter(paths.tensorboard) as writer:
        for epoch in range(config.epochs):
            started = time.perf_counter()
            train_metrics = _train_epoch(model, train_loader, criterion, optimizer, scaler, config, device)
            # Preserve the historical experiment schedule: one scheduler step per epoch.
            scheduler.step()
            train_metrics["learning_rate"] = scheduler.get_last_lr()[0]
            val_metrics, labels, predictions, probabilities = validate(model, test_loader, criterion, device)
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
                best_outputs = labels, predictions, probabilities
                torch.save(model.state_dict(), paths.checkpoints / "model_best.pth")
            torch.save(model.state_dict(), paths.checkpoints / "model_latest.pth")
            print(
                f"Epoch {epoch + 1}/{config.epochs} ({time.perf_counter() - started:.2f}s) | "
                f"train_acc={train_metrics['accuracy']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
                f"best={best_accuracy:.4f}"
            )

    roc_auc = save_evaluation_data(paths.predictions, *best_outputs) if best_outputs is not None else None
    save_checkpoint(paths.checkpoints / "final.pth", config.epochs - 1, model, optimizer, scheduler,
                    best_accuracy, train_losses, val_accuracies)
    summary = {
        "experiment_id": config.experiment_id,
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "micro_auc": float(roc_auc["micro"]) if roc_auc else None,
        "run_directory": str(paths.root),
    }
    _write_json(paths.root / "summary.json", summary)
    return summary
