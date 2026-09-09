# 本地论文材料入口

本地实验仓库：`/Users/Saul/Jlu/paper/classification-code`。

服务器来源：`/root/autodl-tmp/image-classification`，SSH 端口 50799。已同步的 Git 基准提交为 `31137299f32316ce7a7db2968c00ff28d5f2fbe1`。本次补齐 Git 未收录的论文分析材料，保留本地原有文件；没有重新运行训练或测试评估。

## 阅读顺序

1. [最终论文撰写说明](final_paper_writing_brief_20260908.md)：最新结果、准确贡献边界、方法版本、正文结构、英文结果素材与表图验收要求。
2. [正式精简证据归档](../reports/experiments/2026-09-08-paper-readiness-final/README.md)：最新结果表、时延图、审计与 P7 官方结果。
3. [历史写作交接](paper_writing_handoff_2026-09-07.md)：P1–P7 的协议和历史测试入口；最新数字优先使用第一项。

## 文件位置

以下均相对本地实验仓库。

| 用途 | 路径 |
|---|---|
| 最新公平基线选择结果 | `artifacts/analyses/paper_readiness_20260908_r1/fair_comparison.json` |
| 最新基线逐 seed 数据 | 同目录 `fair_comparison_seed_metrics.csv` |
| 全部 A0–A4 消融与训练耗时出处 | 同目录 `ablation_seed_tradeoffs.csv`、`ablation_aggregate_tradeoffs.csv` |
| 最新正式时延表与 PDF/PNG 图 | 同目录 `latency_report/` |
| 逐轮时延、p95、吞吐等源表 | 同目录 `latency/tables/round_metrics.csv` |
| 最新原始计时、workload 与正确性记录 | 同目录 `latency/raw/`、`latency/workloads/`、`latency/checks/` |
| 最新完整性清单与执行状态 | 同目录 `output_hashes.json`、`batch_status.json` |
| P5 最终分析与评分诊断 | `artifacts/analyses/early_exit_p5ab_20260905_092909/` |
| P5 复用的开发集 logits | `artifacts/analyses/early_exit_p5ab_20260905_085355/`，具体路径按 manifest |
| P5-C 原始消融分析与 logits | `artifacts/analyses/early_exit_p5c_20260905_143000/` |
| P6 停止边界 | `artifacts/analyses/early_exit_p6_source_20260906/` |
| P7 source / target | `artifacts/analyses/early_exit_p7_source_20260907/`、`early_exit_p7_target_20260907/` |
| P7 已有官方结果与预测数组 | `artifacts/analyses/early_exit_p7_official_test/` |
| 上一次完整时延测量 | `artifacts/analyses/early_exit_p8_v3_20260905_192249_293181/` |
| 历史失败/中间分析 | `artifacts/analyses/` 中其他阶段目录；按状态使用，不并入正式重测 |
| 训练批次清单 | `artifacts/sweeps/<batch>/manifest.json` |
| 训练摘要、配置、划分、provenance 与训练曲线 | `artifacts/runs/<experiment_id>/` 下已下载的 JSON、YAML 和 `training.csv` |
| 历史锁定测试、协议、精简回执 | `reports/experiments/` |
| 旧论文表图与证据索引 | `reports/paper/`；按最新说明更新后再入稿 |
| 分析、制表、绘图源码 | `scripts/analysis/`、`scripts/visualization/` |

## 使用说明

新下载的 `artifacts/` 文件仍被 Git 忽略。需要备份这些预测和计时数组时，应单独保存本地文件，不能只依赖 Git push。

本次没有下载服务器 checkpoint 或数据集。原来就在本地的 checkpoint/数据未删除。现有材料适合结果复算、论文制表和绘图；依赖未下载 checkpoint 的模型推理、架构加载或完整实验重跑不属于这次离线材料包的保证范围。

清单和回执中的 `/root/autodl-tmp/image-classification` 是原始服务器路径。编写本地读取脚本时将此前缀映射到本地仓库，不修改原始证据文件以“修复”路径，否则其哈希会改变。

官方预测数组用于冻结策略的统计、复核与披露，不用于重新选阈值、模型或 seed。论文应完整保留评分基线的接近/更优场景、A3 的计算取舍、P6/P7 失败边界以及 GPU 大 batch 减速。

准备阶段目录 `paper_readiness_preflight_20260908/` 未下载；其中旧时延链接不属于本次正式重测。

## 本次下载验收

- 新增分析文件 4,951 个，约293 MB；训练元数据及曲线903个，约77 MB；训练批次清单25个，约0.64 MB。统计为本次实际新增传输，不包括原来已有文件。
- 最新正式批次 `output_hashes.json` 中1,818个文件全部通过SHA-256核验。
- P5-A/B的15份、P5-C的24份开发集预测缓存全部通过各自manifest中的SHA-256核验。
- P7官方结果SHA-256仍为 `ac009a3203a3579997c8a5665b63ba1d766cc685f53d865b24017c3700afb749`。
- 对整个已同步分析目录再次执行只读内容校验，与服务器没有发现文件内容差异；准备阶段目录仍按约定排除。
- 本地最新批次状态为 `completed`，时延审计为 `passed`。此次没有运行任何模型推理。
