"""Validation metrics and reusable prediction exports."""

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
from tqdm import tqdm


CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def classification_metrics(labels, predictions, loss: float) -> dict:
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    return {
        "loss": float(loss),
        "accuracy": float(np.mean(np.asarray(predictions) == np.asarray(labels))),
        "precision": float(report["macro avg"]["precision"]),
        "recall": float(report["macro avg"]["recall"]),
        "f1": float(report["macro avg"]["f1-score"]),
    }


def validate(model, loader, criterion, device: torch.device) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    probabilities: list[np.ndarray] = []
    total_loss = 0.0
    with torch.no_grad(), tqdm(loader, desc="Validation", unit="batch") as progress:
        for inputs, targets in progress:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item()
            probabilities.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            labels.extend(targets.cpu().numpy())
    metrics = classification_metrics(labels, predictions, total_loss / len(loader))
    return metrics, np.asarray(labels), np.asarray(predictions), np.asarray(probabilities)


def save_evaluation_data(directory: Path, labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(labels, predictions)
    np.savez(directory / "best.npz", true_labels=labels, predictions=predictions, probabilities=probabilities)
    np.savez(directory / "confusion_matrix.npz", confusion_matrix=matrix, true_labels=labels, predictions=predictions)
    binary_labels = label_binarize(labels, classes=range(len(CLASS_NAMES)))
    fpr, tpr, roc_auc = {}, {}, {}
    for index in range(len(CLASS_NAMES)):
        fpr[index], tpr[index], _ = roc_curve(binary_labels[:, index], probabilities[:, index])
        roc_auc[index] = auc(fpr[index], tpr[index])
    fpr["micro"], tpr["micro"], _ = roc_curve(binary_labels.ravel(), probabilities.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    np.savez(
        directory / "roc.npz", fpr=fpr, tpr=tpr, roc_auc=roc_auc, true_labels=labels,
        probabilities=probabilities, class_names=CLASS_NAMES,
    )
    return roc_auc
