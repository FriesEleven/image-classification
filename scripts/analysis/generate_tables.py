"""Generate LaTeX tables and a concise findings file."""

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "reports/tables/experiment_results_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/tables")
    args = parser.parse_args()
    table = pd.read_csv(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["Experiment", "Parameters_Total (M)", "FLOPs_Total (M)", "Best_Val_Acc (%)", "Inference_Latency (ms)"]
    latex = table[columns].round(3).to_latex(index=False, caption="Experimental Results Comparison", label="tab:results")
    (args.output_dir / "latex_tables.tex").write_text(latex, encoding="utf-8")
    baseline = table.loc[table["Model_Type"] == "mobilenetv2"].iloc[0]
    best = table.loc[table["Best_Val_Acc (%)"].idxmax()]
    findings = (
        f"Baseline accuracy: {baseline['Best_Val_Acc (%)']:.2f}%\n"
        f"Best experiment: {best['Experiment']} ({best['Best_Val_Acc (%)']:.2f}%)\n"
        f"Absolute improvement: {best['Best_Val_Acc (%)'] - baseline['Best_Val_Acc (%)']:.2f} percentage points\n"
    )
    (args.output_dir / "key_findings.txt").write_text(findings, encoding="utf-8")
    print(f"Wrote tables and findings to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
