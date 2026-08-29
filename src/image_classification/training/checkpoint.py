"""Checkpoint and tabular-log persistence."""

import csv
from pathlib import Path

import torch
from torch import nn


LOG_FIELDS = (
    "epoch", "learning_rate", "train_loss", "train_acc", "val_loss", "val_acc",
    "train_precision", "train_recall", "train_f1", "val_precision", "val_recall", "val_f1",
)


def append_training_log(path: Path, epoch: int, train_metrics: dict, val_metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "epoch": epoch + 1,
                "learning_rate": train_metrics.get("learning_rate", 0),
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
            }
        )


def save_checkpoint(path: Path, epoch: int, model: nn.Module, optimizer, scheduler,
                    best_accuracy: float, train_losses: list[float], val_accuracies: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_acc": best_accuracy,
            "train_loss_list": train_losses,
            "val_acc_list": val_accuracies,
        },
        path,
    )
