"""Build the versioned experiment table from per-run artifacts."""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "artifacts/runs"
DEFAULT_OUTPUT = ROOT / "reports/tables/experiment_results_summary.csv"


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _row(run: Path) -> dict | None:
    required = (run / "config.yaml", run / "metrics.json", run / "benchmark.json", run / "summary.json")
    if not all(path.exists() for path in required):
        return None
    with (run / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    metrics, benchmark, summary = _load(run / "metrics.json"), _load(run / "benchmark.json"), _load(run / "summary.json")
    positions = config.get("aux_positions") or []
    if config["model_type"] == "hybrid":
        positions = f"SE={config.get('se_positions', [])}; CBAM={config.get('cbam_positions', [])}"
    training_log = run / "logs/training.csv"
    best_validation_accuracy = summary.get("best_validation_accuracy", summary.get("best_accuracy"))
    final_accuracy = best_validation_accuracy
    if training_log.exists():
        final_accuracy = pd.read_csv(training_log).iloc[-1]["val_acc"]
    return {
        "Experiment": summary["experiment_id"],
        "Dataset": config.get("dataset", "cifar10"),
        "Model_Type": config["model_type"],
        "Positions": positions or "N/A",
        "Parameters_Total (M)": metrics["parameters_total"] / 1e6,
        "Parameters_Aux (K)": metrics["parameters_aux_attention"] / 1e3,
        "FLOPs_Total (M)": metrics["flops_total"] / 1e6,
        "FLOPs_Aux (M)": metrics["flops_attention_adjustment"] / 1e6,
        "Best_Val_Acc (%)": best_validation_accuracy * 100,
        "Final_Val_Acc (%)": float(final_accuracy) * 100,
        "Test_Acc (%)": summary.get("test_accuracy", float("nan")) * 100,
        "Inference_Latency (ms)": benchmark["inference_latency_mean"],
        "Throughput (FPS)": benchmark["throughput_fps"],
        "SE_Positions": config.get("se_positions") or "",
        "CBAM_Positions": config.get("cbam_positions") or "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--new-only", action="store_true", help="Do not merge the existing curated table")
    args = parser.parse_args()
    rows = [row for run in sorted(RUNS_DIR.glob("*")) if (row := _row(run)) is not None]
    frames = []
    if args.output.exists() and not args.new_only:
        frames.append(pd.read_csv(args.output))
    if rows:
        frames.append(pd.DataFrame(rows))
    if not frames:
        raise SystemExit("No complete experiment runs were found")
    table = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Experiment"], keep="last")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    table.to_excel(args.output.with_suffix(".xlsx"), index=False)
    print(f"Wrote {len(table)} experiments to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
