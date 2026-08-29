"""Render one-vs-rest ROC curves from a run prediction archive."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize


ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="best.npz containing true_labels and probabilities")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/figures/roc_curve.svg")
    args = parser.parse_args()
    archive = np.load(args.input, allow_pickle=True)
    labels = label_binarize(archive["true_labels"], classes=range(len(CLASS_NAMES)))
    probabilities = archive["probabilities"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 7))
    for index, class_name in enumerate(CLASS_NAMES):
        false_positive, true_positive, _ = roc_curve(labels[:, index], probabilities[:, index])
        plt.plot(false_positive, true_positive, label=f"{class_name} ({auc(false_positive, true_positive):.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close()
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
