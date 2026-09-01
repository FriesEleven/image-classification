"""Verify raw evidence and write the curated v6/control diagnostic report."""

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path("artifacts/diagnostics/csgha_v6_serial_information_20260901_v1")
DEFAULT_OUTPUT = Path("reports/diagnostics/2026-09-01-csgha-v6-serial-v6s1")


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
        ] if run["model_type"] == "csgha_v6" else []
        original = conditions["original"]["accuracy_percent"]
        rows.append({
            "model_type": run["model_type"], "seed": run["seed"], "original": original,
            "deep_zero": conditions["deep_zero"]["accuracy_percent"],
            "deep_zero_delta_pp": conditions["deep_zero"]["delta_accuracy_pp"],
            "guidance_zero": conditions.get("guidance_zero", {}).get("accuracy_percent"),
            "guidance_zero_delta_pp": conditions.get("guidance_zero", {}).get("delta_accuracy_pp"),
            "guidance_removal_prediction_changes": conditions.get("guidance_zero", {}).get("prediction_changes"),
            "guidance_removal_hurts": conditions.get("guidance_zero", {}).get("originally_correct_now_wrong"),
            "guidance_removal_helps": conditions.get("guidance_zero", {}).get("originally_wrong_now_correct"),
            "mean_descriptor": conditions.get("train_mean_descriptor", {}).get("accuracy_percent"),
            "mean_descriptor_delta_pp": conditions.get("train_mean_descriptor", {}).get("delta_accuracy_pp"),
            "mean_contribution": conditions.get("train_mean_contribution", {}).get("accuracy_percent"),
            "mean_contribution_delta_pp": conditions.get("train_mean_contribution", {}).get("delta_accuracy_pp"),
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
    guided = sorted((row for row in rows if row["model_type"] == "csgha_v6"), key=lambda row: row["seed"])
    control = sorted((row for row in rows if row["model_type"] == "hybrid_leaky"), key=lambda row: row["seed"])
    if [row["seed"] for row in guided] != [42, 43, 44] or [row["seed"] for row in control] != [42, 43, 44]:
        raise ValueError("P2 seed sets are incomplete")
    return {
        "schema_version": 1, "manifest_sha256": checksum(manifest_path),
        "audit_sha256": manifest["audit_sha256"], "official_test_evaluated": False,
        "rows": rows,
        "guided_guidance_zero_delta_mean_pp": mean([row["guidance_zero_delta_pp"] for row in guided]),
        "guided_permutation_delta_mean_pp": mean([row["permutation_delta_pp"] for row in guided]),
        "guided_deep_zero_delta_mean_pp": mean([row["deep_zero_delta_pp"] for row in guided]),
        "control_deep_zero_delta_mean_pp": mean([row["deep_zero_delta_pp"] for row in control]),
    }


def write_report(report: dict, output: Path):
    if output.exists():
        raise FileExistsError(f"Use a new immutable report directory: {output}")
    output.mkdir(parents=True)
    (output / "diagnostic_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    guided = sorted((row for row in report["rows"] if row["model_type"] == "csgha_v6"), key=lambda row: row["seed"])
    control = sorted((row for row in report["rows"] if row["model_type"] == "hybrid_leaky"), key=lambda row: row["seed"])
    lines = [
        "# CSGHA v6 / matched control 诊断结果", "",
        "完整保存的5,000张validation；无训练、无test。全部checkpoint strict load并精确复现P1 best accuracy。", "",
        "## v6 guidance干预", "",
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
        "", f"三个训练seed的置换效应见上表，跨seed平均为 {report['guided_permutation_delta_mean_pp']:+.2f} pp；guidance置零跨seed平均为 {report['guided_guidance_zero_delta_mean_pp']:+.2f} pp。", "",
        "## Deep置零", "", "| Seed | v6变化 | Matched control变化 |", "|---:|---:|---:|",
    ]
    controls = {row["seed"]: row for row in control}
    for row in guided:
        lines.append(f"| {row['seed']} | {row['deep_zero_delta_pp']:+.2f} pp | {controls[row['seed']]['deep_zero_delta_pp']:+.2f} pp |")
    lines += [
        "", "v6与control的两个目标block在完整validation上deep logits零值比例均为0。",
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
        "# v6诊断结论与CSGHA路线终止判断", "",
        "## 结论", "",
        "1. v6六个目标block的contribution绝对均值均为bounded绝对均值的约0.25倍，说明`abs(tanh(alpha))`仍接近1，固定±0.25 cap确实成为活动上限；tanh饱和率保持约0.34%–1.12%。",
        "2. guidance置零在seeds42/43/44分别变化−0.22/−0.24/−0.08个百分点，跨seed平均−0.18；guidance在v6 checkpoint内部仍有小幅净帮助。",
        "3. 三次无自身配对置换跨seed平均变化约−0.24个百分点，仍支持输入相关性，但远弱于v5的−1.72；cap同时削弱了有用与有害的浅层耦合。",
        "4. v6 deep-zero变化为−0.78/−1.16/−2.72个百分点，平均约−1.55；只略高于v5约−1.43，仍远弱于matched control约−10.47。限制幅度没有恢复接近control的deep分支依赖。",
        "5. 正式训练结果中v6对matched control为−0.12/−0.14/−0.20个百分点，均值−0.153±0.042且胜出0/3。信号工作与任务收益之间不存在稳定正向关系。", "",
        "## 路线判断", "",
        "停止CSGHA加性channel-logit guidance路线，不开发v7，也不继续扫描cap、归一化、激活或位置。v3–v6作为负结果与机制消融归档；后续主线转为预算约束下的阶段感知稀疏注意力部署。",
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
