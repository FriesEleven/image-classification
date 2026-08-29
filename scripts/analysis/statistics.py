"""Report accuracy improvements and CBAM/SE significance tests."""

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "reports/tables/experiment_results_summary.csv")
    args = parser.parse_args()
    table = pd.read_csv(args.input)
    baseline = table.loc[table["Model_Type"] == "mobilenetv2", "Best_Val_Acc (%)"].iloc[0]
    table["Accuracy_Improvement (%)"] = table["Best_Val_Acc (%)"] - baseline
    print(table[["Experiment", "Best_Val_Acc (%)", "Accuracy_Improvement (%)"]].to_string(index=False))
    cbam = table.loc[table["Model_Type"] == "cbam", "Best_Val_Acc (%)"]
    se = table.loc[table["Model_Type"] == "se", "Best_Val_Acc (%)"]
    if len(cbam) >= 2 and len(se) >= 2:
        result = stats.ttest_ind(cbam, se, equal_var=False)
        print(f"\nWelch t-test CBAM vs SE: t={result.statistic:.4f}, p={result.pvalue:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
