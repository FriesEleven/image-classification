# CSGHA v3–v6 负结果与机制消融归档

## 结论

CSGHA 的跨阶段加性 channel-logit 引导路径至此停止，不进入 v7。v3–v6 的检查点干预证明模型确实使用过输入相关的浅层描述，但四轮受控修改都没有得到稳定、可复现的任务收益。该系列保留为负结果和机制消融，不作为论文正向主方法。

以下数值均为 CIFAR-10 best validation accuracy 的按 seed 配对差值；v4–v6 使用完全匹配的独立 `hybrid_leaky` 对照，未评估官方 test。

| 版本 | 单变量机制 | 配对差值（pp） | 均值 ± 样本标准差（pp） | 胜出 | 机制诊断 |
|---|---|---:|---:|---:|---|
| v3 | 有界跨阶段引导 | +0.22 / −0.34 / +0.34 | +0.073（历史汇总） | 2/3 | 置换平均下降约0.57pp；deep 分支大面积硬零 |
| v4 | deep ReLU→LeakyReLU | +0.56 / +0.08 / −0.22 | +0.140 ± 0.393 | 2/3 | deep 硬零消失，但引导 `tanh` 饱和约96%–97% |
| v5 | 引导投影输出 RMS 归一化 | −0.26 / −0.64 / +0.02 | −0.293 ± 0.331 | 1/3 | 饱和降至约0.29%–1.76%，置换效应增至−1.72pp，但 deep 依赖降至−1.433pp |
| v6 | v5 上增加固定0.25幅度上限 | −0.12 / −0.14 / −0.20 | −0.153 ± 0.042 | 0/3 | 上限生效；引导置零仅−0.18pp、置换约−0.244pp，deep-zero约−1.553pp |

## 可支持的机制陈述

1. v3 的浅层描述与最终预测存在输入配对依赖，但“使用了信息”不等于“提高了准确率”。
2. v4 修复了目标 CBAM deep channel MLP 的硬失活，却没有产生三个 seed 一致的收益。
3. v5 直接修复了投影输出饱和，并增强了输入配对效应；与此同时，deep 分支依赖进一步减弱且任务结果变差，说明更强的跨阶段耦合不是充分条件。
4. v6 的固定上限确实把引导限制为小残差，但输入耦合也明显减弱，deep 分支依赖没有恢复到匹配对照，且三个 seed 全部落后。

因此最符合证据的解释是：当前“浅层描述直接加到深层 channel logits”形成了分支间的替代/竞争，信号整形可以改变内部依赖，却没有建立稳定的分类优势。继续扫描上限、归一化、激活或位置会扩大事后调参空间，不能提高主张可信度。

## 论文使用边界

- 可作为诚实的机制消融或负结果：展示 dead branch、饱和、分支替代与任务收益之间并非单调关系。
- 不可声称 CSGHA 稳定优于独立 SE+CBAM，也不可只展示有利 seed。
- v4–v6 是 validation-only；在方法选择完成前不得补看官方 test。
- v6 检查点级干预不是重新训练消融，必须与从头训练的配对结果分开表述。

## 证据入口

- v4 正式审计与诊断：[`reports/audits/2026-08-31-csgha-v4-retry1`](../../audits/2026-08-31-csgha-v4-retry1/experiment_summary.md)、[`reports/diagnostics/2026-08-31-csgha-v4-retry1`](../../diagnostics/2026-08-31-csgha-v4-retry1/findings.md)
- v5 正式审计与诊断：[`reports/audits/2026-08-31-csgha-v5-serial-s1`](../../audits/2026-08-31-csgha-v5-serial-s1/experiment_summary.md)、[`reports/diagnostics/2026-08-31-csgha-v5-serial-s1`](../../diagnostics/2026-08-31-csgha-v5-serial-s1/findings.md)
- v6 正式审计与诊断：[`reports/audits/2026-09-01-csgha-v6-serial-v6s1`](../../audits/2026-09-01-csgha-v6-serial-v6s1/experiment_summary.md)、[`reports/diagnostics/2026-09-01-csgha-v6-serial-v6s1`](../../diagnostics/2026-09-01-csgha-v6-serial-v6s1/findings.md)

原始 manifest、checkpoint、配对预测和诊断数组保留在服务器 `artifacts/`，不提交 GitHub。审计报告中的 SHA-256 用于定位精确证据。
