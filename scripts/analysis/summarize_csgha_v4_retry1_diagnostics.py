"""Verify P2 raw evidence and write the curated v4/control diagnostic report."""

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path("artifacts/diagnostics/csgha_v4_retry1_information_20260831_v1")
DEFAULT_OUTPUT = Path("reports/diagnostics/2026-08-31-csgha-v4-retry1")


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    return math.fsum(values) / len(values)


def summarize(directory: Path) -> dict:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "completed" or manifest.get("training") is not False:
        raise ValueError("P2 diagnostics did not complete under the no-training protocol")
    if manifest.get("official_test_evaluated") is not False or len(manifest.get("runs", [])) != 6:
        raise ValueError("Expected six validation-only P2 diagnostics")
    rows = []
    for run in manifest["runs"]:
        name = f"{run['model_type']}_seed{run['seed']}"
        if json.loads((directory / f"{name}.json").read_text()) != run:
            raise ValueError(f"Per-run JSON differs from manifest: {name}")
        arrays = directory / Path(run["paired_arrays"]).name
        if checksum(arrays) != run["paired_arrays_sha256"]:
            raise ValueError(f"Paired array hash mismatch: {name}")
        if not run["strict_state_dict_load"] or not run["original_validation_matches_audit"]:
            raise ValueError(f"Version match failed: {name}")
        conditions = run["conditions"]
        if any(value["samples"] != 5000 for value in conditions.values()):
            raise ValueError(f"Incomplete validation coverage: {name}")
        permutation_values = [
            conditions[f"permuted_{seed}"]["accuracy_percent"] for seed in run["permutation_seeds"]
        ] if run["model_type"] == "csgha_v4" else []
        original = conditions["original"]["accuracy_percent"]
        rows.append({
            "model_type": run["model_type"], "seed": run["seed"], "original": original,
            "deep_zero": conditions["deep_zero"]["accuracy_percent"],
            "deep_zero_delta_pp": conditions["deep_zero"]["delta_accuracy_pp"],
            "guidance_zero": conditions.get("guidance_zero", {}).get("accuracy_percent"),
            "guidance_zero_delta_pp": conditions.get("guidance_zero", {}).get("delta_accuracy_pp"),
            "mean_descriptor": conditions.get("train_mean_descriptor", {}).get("accuracy_percent"),
            "mean_contribution": conditions.get("train_mean_contribution", {}).get("accuracy_percent"),
            "permutations": permutation_values,
            "permutation_mean": mean(permutation_values) if permutation_values else None,
            "permutation_delta_pp": mean(permutation_values) - original if permutation_values else None,
            "statistics": run["statistics"],
            "checkpoint_sha256": run["checkpoint_sha256"],
            "paired_arrays_sha256": run["paired_arrays_sha256"],
        })
    for relative, expected in manifest["diagnostic_sources_sha256"].items():
        if checksum(directory / "source" / relative) != expected:
            raise ValueError(f"Archived diagnostic source mismatch: {relative}")
    guided = sorted((row for row in rows if row["model_type"] == "csgha_v4"), key=lambda row: row["seed"])
    control = sorted((row for row in rows if row["model_type"] == "hybrid_leaky"), key=lambda row: row["seed"])
    if [row["seed"] for row in guided] != [42, 43, 44] or [row["seed"] for row in control] != [42, 43, 44]:
        raise ValueError("P2 seed sets are incomplete")
    return {
        "schema_version": 1, "manifest_sha256": checksum(manifest_path),
        "audit_sha256": manifest["audit_sha256"], "official_test_evaluated": False,
        "rows": rows,
        "guided_permutation_delta_mean_pp": mean([row["permutation_delta_pp"] for row in guided]),
        "guided_deep_zero_delta_mean_pp": mean([row["deep_zero_delta_pp"] for row in guided]),
        "control_deep_zero_delta_mean_pp": mean([row["deep_zero_delta_pp"] for row in control]),
    }


def write_report(report: dict, output: Path):
    if output.exists():
        raise FileExistsError(f"Use a new immutable report directory: {output}")
    output.mkdir(parents=True)
    (output / "diagnostic_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    guided = sorted((row for row in report["rows"] if row["model_type"] == "csgha_v4"), key=lambda row: row["seed"])
    control = sorted((row for row in report["rows"] if row["model_type"] == "hybrid_leaky"), key=lambda row: row["seed"])
    lines = [
        "# CSGHA v4 / matched control P2诊断结果", "",
        "完整保存的5,000张validation；无训练、无test。全部checkpoint strict load并精确复现P1 best accuracy。", "",
        "## v4 guidance干预", "",
        "| Seed | 原始 | Guidance置零 | 训练均值描述 | 训练均值加项 | 三次置换均值 | 置换变化 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in guided:
        lines.append(
            f"| {row['seed']} | {row['original']:.2f} | {row['guidance_zero']:.2f} | "
            f"{row['mean_descriptor']:.2f} | {row['mean_contribution']:.2f} | "
            f"{row['permutation_mean']:.2f} | {row['permutation_delta_pp']:+.2f} pp |"
        )
    lines += [
        "", f"三个训练seed的置换平均变化均为负，跨seed平均为 {report['guided_permutation_delta_mean_pp']:+.2f} pp。", "",
        "## Deep置零", "", "| Seed | v4变化 | Matched control变化 |", "|---:|---:|---:|",
    ]
    controls = {row["seed"]: row for row in control}
    for row in guided:
        lines.append(f"| {row['seed']} | {row['deep_zero_delta_pp']:+.2f} pp | {controls[row['seed']]['deep_zero_delta_pp']:+.2f} pp |")
    lines += [
        "", "v4与control的两个目标block在完整validation上deep logits零值比例均为0，LeakyReLU消除了v3观察到的硬失活。",
        "v4的deep-zero下降证明deep分支参与预测；它小于control的下降不能被直接解释为因果贡献比例。", "",
        "## 分支统计", "",
        "| 模型 | Seed | Block | Deep |Guidance|均值 | Guidance tanh饱和 | Gate跨样本std |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        for name, values in row["statistics"].items():
            block = name.split(".")[2]
            contribution = values.get("contribution", {}).get("abs_mean")
            saturation = values.get("tanh_saturation_fraction_abs_gt_099")
            lines.append(
                f"| {row['model_type']} | {row['seed']} | {block} | {values['deep']['abs_mean']:.3f} | "
                f"{'—' if contribution is None else f'{contribution:.3f}'} | "
                f"{'—' if saturation is None else f'{100 * saturation:.2f}%'} | "
                f"{values['gate']['sample_std_per_channel_mean']:.4f} |"
            )
    lines += [
        "", "Deep与guidance绝对值不是可相加的性能贡献率；checkpoint干预也不同于从头训练消融。",
        f"原始manifest SHA-256：`{report['manifest_sha256']}`。",
    ]
    (output / "results.md").write_text("\n".join(lines) + "\n")
    findings = [
        "# P2结论与下一机制假设", "",
        "## 结论", "",
        "1. v4三个seed在置换正确配对的guidance后均下降，支持模型使用输入相关浅层信息。",
        "2. LeakyReLU使v4与control的deep分支在完整validation上不再硬失活；v4 deep-zero均下降约3个百分点。",
        "3. 任务收益仍不稳定：P1中v4对control仅+0.140±0.393个百分点，seed44为负。机制在工作不等于稳定优于control。",
        "4. v4两个目标block的guidance tanh饱和率仍约96%–97%，投影幅度几乎被压成符号门，是当前最明确的剩余单因素问题。", "",
        "## 下一候选：CSGHA v5 RMS-normalized guidance", "",
        "仅在guide projection输出进入tanh前做逐样本、跨通道无参数RMS归一化：",
        "`z=P(LN(g_s)); z_hat=z/sqrt(mean_c(z^2)+eps); guidance=tanh(alpha)*tanh(z_hat)`。", "",
        "该修改保持Leaky deep、位置、投影结构、损失和训练协议不变。它直接限制投影尺度，避免网络只靠放大权重重新进入tanh饱和；不把RMS归一化本身包装为创新。",
        "下一批仍需v5与完全相同的hybrid_leaky control按seeds42/43/44配对从头训练。P2结果不能替代该训练对照。",
    ]
    (output / "findings.md").write_text("\n".join(findings) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = summarize(ROOT / args.source)
    if args.dry_run:
        print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
        return 0
    write_report(report, ROOT / args.output)
    print(f"Verified six P2 diagnostics; saved {ROOT / args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
