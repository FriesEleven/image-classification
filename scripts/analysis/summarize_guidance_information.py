"""Rebuild the curated guidance diagnostic tables from immutable JSON evidence."""

import argparse
import hashlib
import json
import math
import runpy
from pathlib import Path


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(directory: Path, output: Path):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "completed":
        raise ValueError("Diagnostics did not finish")
    entries = []
    for run in manifest["runs"]:
        arrays = directory / Path(run["paired_arrays"]).name
        if checksum(arrays) != run["paired_arrays_sha256"]:
            raise ValueError(f"Paired prediction checksum mismatch: {arrays}")
        if json.loads((directory / f"{run['audit_id']}.json").read_text()) != run:
            raise ValueError("Per-run JSON disagrees with manifest")
        conditions = run["conditions"]
        if not run["original_validation_matches_audit"] or run["official_test_evaluated"]:
            raise ValueError("Invalid diagnostic protocol")
        if any(value["samples"] != 5000 for value in conditions.values()):
            raise ValueError("Expected full validation coverage")
        permutation_values = [conditions[f"permuted_{seed}"]["accuracy_percent"] for seed in run["permutation_seeds"]]
        permutation_mean = math.fsum(permutation_values) / len(permutation_values)
        original = conditions["original"]["accuracy_percent"]
        entries.append({
            "id": run["audit_id"], "version": run["version"], "seed": run["seed"],
            "original": original, "zero": conditions["zero_guidance"]["accuracy_percent"],
            "mean_descriptor": conditions["train_mean_descriptor"]["accuracy_percent"],
            "mean_contribution": conditions["train_mean_contribution"]["accuracy_percent"],
            "permutations": permutation_values, "permutation_mean": permutation_mean,
            "permutation_delta_pp": permutation_mean - original,
        })
    for relative, expected in manifest["diagnostic_sources_sha256"].items():
        if checksum(directory / "source" / relative) != expected:
            raise ValueError(f"Archived diagnostic source differs: {relative}")
    report = {"manifest_sha256": checksum(manifest_path), "evidence": manifest, "table": entries}
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnostic_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# Guidance 输入信息诊断：可复算结果", "",
             "完整 validation，每个 checkpoint 5,000 张；所有数值为准确率百分比。", "",
             "| 版本 | Seed | 原始 | 置零 | 训练集平均描述 | 训练集平均引导加项 | 打乱均值 | 打乱差值（百分点） |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in entries:
        lines.append(f"| {row['version']} | {row['seed']} | {row['original']:.2f} | {row['zero']:.2f} | "
                     f"{row['mean_descriptor']:.2f} | {row['mean_contribution']:.2f} | {row['permutation_mean']:.2f} | "
                     f"{row['permutation_delta_pp']:+.2f} |")
    lines += ["", "打乱均值来自置换 seeds 7301/7302/7303，不是三次重新训练。它们不能当作额外训练 seeds。",
              "训练集均值仅使用对应 45,000 张训练图像的 evaluation transform，不使用标签或 validation 均值。",
              "", "## v3 三次置换明细", "", "| 训练 seed | 7301 | 7302 | 7303 |", "|---|---:|---:|---:|"]
    for row in entries:
        if row["version"] == "v3":
            lines.append(f"| {row['seed']} | " + " | ".join(f"{value:.2f}" for value in row["permutations"]) + " |")
    lines += ["", "## v3 通道分支统计", "",
              "零值比例针对全部 validation 的 deep-logit 元素；tanh 饱和定义为 |tanh(raw)| > 0.99。", "",
              "| Seed | 目标 block | Deep logits 零值比例 | Guidance tanh 饱和比例 | 通道门跨样本标准差均值 |",
              "|---|---:|---:|---:|---:|"]
    for run in manifest["runs"]:
        if run["version"] != "v3":
            continue
        for name, value in run["statistics"].items():
            block = name.split(".")[2]
            lines.append(f"| {run['seed']} | {block} | {100 * value['deep']['zero_fraction']:.2f}% | "
                         f"{100 * value['tanh_saturation_fraction_abs_gt_099']:.2f}% | "
                         f"{value['gate']['sample_std_per_channel_mean']:.4f} |")
    lines += ["", "跨样本标准差用于描述当前 5,000 张样本的变化，计算采用 correction=0，区别于训练 seed 汇总的样本标准差。",
              "", "## 版本与证据", "", f"原始 manifest SHA-256：`{report['manifest_sha256']}`。",
              "", "| ID | 历史实现引用 | 来源等级 | 原验证结果复现 |", "|---|---|---|---|"]
    for run in manifest["runs"]:
        grade = "运行时 manifest commit" if run["version_evidence"] == "manifest_commit" else "参考历史实现；运行时 commit 未记录"
        lines.append(f"| {run['audit_id']} | `{run['historical_model']['reference_commit']}` | {grade} | 通过 |")
    lines += ["", "每个 checkpoint 严格加载权重；每个 batch 还核对拆分后的 deep+guidance 公式与历史 forward 一致。",
              "复现准确率与结构匹配不补足缺失的原始运行 commit。源代码、checkpoint、划分索引、诊断代码及配对预测的校验信息见 `diagnostic_summary.json`。",
              "原始 JSON、配对 logits、索引和诊断源码备份在 Git 忽略目录 `artifacts/diagnostics/csgha_information_20260830_v1/`。"]
    (output / "results.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    root = Path(__file__).resolve().parents[2]
    root = runpy.run_path(str(root / "src/image_classification/paths.py"))["PROJECT_ROOT"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/diagnostics/csgha_information_20260830_v1"))
    parser.add_argument("--output", type=Path, default=Path("reports/diagnostics/2026-08-30-guidance"))
    args = parser.parse_args()
    report = summarize(root / args.source, root / args.output)
    print(f"Verified {len(report['table'])} checkpoint diagnostics; saved {root / args.output}")


if __name__ == "__main__":
    main()
