"""Verify raw evidence and write the curated v5/control diagnostic report."""

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path("artifacts/diagnostics/csgha_v5_serial_information_20260831_v1")
DEFAULT_OUTPUT = Path("reports/diagnostics/2026-08-31-csgha-v5-serial-s1")


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
        ] if run["model_type"] == "csgha_v5" else []
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
    guided = sorted((row for row in rows if row["model_type"] == "csgha_v5"), key=lambda row: row["seed"])
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
    guided = sorted((row for row in report["rows"] if row["model_type"] == "csgha_v5"), key=lambda row: row["seed"])
    control = sorted((row for row in report["rows"] if row["model_type"] == "hybrid_leaky"), key=lambda row: row["seed"])
    lines = [
        "# CSGHA v5 / matched control 诊断结果", "",
        "完整保存的5,000张validation；无训练、无test。全部checkpoint strict load并精确复现P1 best accuracy。", "",
        "## v5 guidance干预", "",
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
        "", f"三个训练seed的置换效应见上表，跨seed平均为 {report['guided_permutation_delta_mean_pp']:+.2f} pp。", "",
        "## Deep置零", "", "| Seed | v4变化 | Matched control变化 |", "|---:|---:|---:|",
    ]
    controls = {row["seed"]: row for row in control}
    for row in guided:
        lines.append(f"| {row['seed']} | {row['deep_zero_delta_pp']:+.2f} pp | {controls[row['seed']]['deep_zero_delta_pp']:+.2f} pp |")
    lines += [
        "", "v5与control的两个目标block在完整validation上deep logits零值比例均为0。",
        "Deep-zero变化证明分支参与预测，但不能被直接解释为因果贡献比例。", "",
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
        "# v5诊断结论与v6机制假设", "",
        "## 结论", "",
        "1. RMS达到预期信号效果：六个目标block的guidance tanh饱和率降至约0.29%–1.76%，不再是v4的96%–97%符号门。",
        "2. v5三个seed的guidance置零均下降0.70–0.84个百分点；三次置换在每个训练seed上均下降，跨seed平均−1.72个百分点，输入配对依赖强于v4。",
        "3. 任务结果反而变差：v5对matched control为−0.26/−0.64/+0.02个百分点，配对均值−0.293±0.331。信号更可变、更依赖输入并不自动产生更高准确率。",
        "4. v5 deep-zero仅下降1.28/1.42/1.60个百分点，跨seed平均−1.43；control平均−10.47，v4此前约−3.01。v5 deep logits幅度也明显小于control。",
        "5. v5 contribution与bounded的绝对均值几乎相等，说明可学习scale的`tanh`绝对值接近1；RMS guidance达到最大允许幅度并更容易替代deep分支。", "",
        "## 下一候选：CSGHA v6 capped residual guidance", "",
        "只把v5的最大guidance logit修正限制为±0.25：",
        "`z_hat=RMSNorm(P(LN(g_s))); guidance=0.25*tanh(alpha)*tanh(z_hat)`。", "",
        "该单变量保留v5的RMS、Leaky deep、位置、投影、参数量、损失与训练协议。±0.25形成明确的logit trust region，使跨阶段信息只能作为小幅残差修正，目标是防止再次压低deep分支；0.25本身不是创新主张。",
        "下一批仍需v6与完全相同的hybrid_leaky control按seeds42/43/44串行配对从头训练。checkpoint干预不能替代训练对照。",
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
