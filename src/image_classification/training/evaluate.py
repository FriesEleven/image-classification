"""Validation metrics and reusable prediction exports."""

import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
from tqdm import tqdm


def classification_metrics(labels, predictions, loss: float) -> dict:
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    return {
        "loss": float(loss),
        "accuracy": float(np.mean(np.asarray(predictions) == np.asarray(labels))),
        "precision": float(report["macro avg"]["precision"]),
        "recall": float(report["macro avg"]["recall"]),
        "f1": float(report["macro avg"]["f1-score"]),
    }


class EpochAccumulator:
    """Buffer tiny detached metrics and synchronize once at the epoch boundary."""

    def __init__(self, probabilities: bool = False):
        self.keep_probabilities = probabilities
        self.losses = []
        self.predictions = []
        self.labels = []
        self.probabilities = []

    def update(self, outputs, labels, loss) -> None:
        outputs = outputs.detach()
        self.losses.append(loss.detach())
        self.predictions.append(outputs.argmax(dim=1))
        self.labels.append(labels.detach())
        if self.keep_probabilities:
            self.probabilities.append(torch.softmax(outputs, dim=1))

    def finish(self):
        if not self.losses:
            raise ValueError("Cannot summarize an empty epoch")
        # Preserve the existing mean-of-batch-means loss definition, including
        # the smaller last batch. Float64 mirrors the old Python-float sum.
        loss = torch.stack(self.losses).double().mean().item()
        predictions = torch.cat(self.predictions).cpu().numpy()
        labels = torch.cat(self.labels).cpu().numpy()
        probabilities = torch.cat(self.probabilities).cpu().numpy() if self.keep_probabilities else None
        return classification_metrics(labels, predictions, loss), labels, predictions, probabilities


def validate(
    model, loader, criterion, device: torch.device, description: str = "Validation",
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    accumulator = EpochAccumulator(probabilities=True)
    with torch.no_grad(), tqdm(
        loader, desc=description, unit="batch", disable=not sys.stderr.isatty(),
    ) as progress:
        for inputs, targets in progress:
            host_targets = targets
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            accumulator.update(outputs, host_targets, criterion(outputs, targets))
    return accumulator.finish()


def save_evaluation_data(
    directory: Path,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    class_names = tuple(class_names)
    matrix = confusion_matrix(labels, predictions)
    np.savez(
        directory / "test.npz", true_labels=labels, predictions=predictions,
        probabilities=probabilities, class_names=class_names,
    )
    np.savez(
        directory / "confusion_matrix.npz", confusion_matrix=matrix,
        true_labels=labels, predictions=predictions, class_names=class_names,
    )
    binary_labels = label_binarize(labels, classes=range(len(class_names)))
    fpr, tpr, roc_auc = {}, {}, {}
    for index in range(len(class_names)):
        fpr[index], tpr[index], _ = roc_curve(binary_labels[:, index], probabilities[:, index])
        roc_auc[index] = auc(fpr[index], tpr[index])
    fpr["micro"], tpr["micro"], _ = roc_curve(binary_labels.ravel(), probabilities.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    np.savez(
        directory / "roc.npz", fpr=fpr, tpr=tpr, roc_auc=roc_auc, true_labels=labels,
        probabilities=probabilities, class_names=class_names,
    )
    return roc_auc
