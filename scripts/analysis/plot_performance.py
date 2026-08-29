"""Generate the standard paper performance plots."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]


def _scatter(table, x, output, title):
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=table, x=x, y="Best_Val_Acc (%)", hue="Model_Type", s=100)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "reports/tables/experiment_results_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/figures")
    args = parser.parse_args()
    table = pd.read_csv(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    _scatter(table, "Parameters_Total (M)", args.output_dir / "accuracy_vs_parameters.svg", "Accuracy vs Model Size")
    _scatter(table, "FLOPs_Total (M)", args.output_dir / "accuracy_vs_flops.svg", "Accuracy vs Computational Cost")
    _scatter(table, "Inference_Latency (ms)", args.output_dir / "accuracy_vs_latency.svg", "Accuracy vs Inference Latency")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
