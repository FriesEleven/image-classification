"""Render a confusion matrix from a run prediction archive."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="best.npz or confusion_matrix.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/figures/confusion_matrix.svg")
    args = parser.parse_args()
    archive = np.load(args.input, allow_pickle=True)
    matrix = archive["confusion_matrix"] if "confusion_matrix" in archive else confusion_matrix(archive["true_labels"], archive["predictions"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close()
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
