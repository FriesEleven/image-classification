# 最终论文撰写内容说明：实验冻结版（2026-09-08）

## 1. 本文用途与最终判断

本文件供另一个写作对话直接利用已完成实验，组织英文论文、结果表图和结论。它不是已经完成排版、查新和编译的论文。目标期刊沿用用户指定的 Springer The Journal of Supercomputing；本文不预测送审或录用概率。

服务器连接：`ssh -p 50799 root@connect.westb.seetacloud.com`。
实验根目录：`/root/autodl-tmp/image-classification`。下文实验路径均相对此目录。
先读本文件，再按需查 `docs/paper_writing_handoff_2026-09-07.md` 和根目录 `AGENTS.md`。涉及最新策略比较和时延数字时，以本文件列出的 20260908 正式批次为准；旧交接的历史证据入口继续有效。

**现在可以停止实验并进入论文撰写，但论文主张必须从“全面优于强基线的方法”收紧为“跨模型 seed 的共享经验约束早退及可复现的系统取舍”。** 本批增强了证据的公平性和复现性，并没有让原定三项“明确优势”全部成立：

| 原目标 | 本批实际支持程度 | 论文处理 |
|---|---|---|
| 在公平风险/计算条件下明确胜过 entropy、margin | 不支持普遍优势；仅有局部可行性/计算取舍 | 报告接近、局部优势和反例，不称 SOTA 或全面胜出 |
| A3 准确率、计算节省、训练复杂度的有说服力取舍 | 有完整消融；去 KD 提高部分最终头均值，但减少计算节省；无受控训练加速证明 | 写“简化配方的取舍”，不写“A3 支配 A4” |
| 明确场景下可复现实测收益并报告减速 | 支持 RTX3080Ti、CIFAR-10、batch1 的模型内时延收益；其他场景有边界 | 报告两次测量与全 batch 结果，不能外推端到端服务加速 |

负结果不自动意味着拒稿，但也不能靠改名或文字包装替代创新性。现有材料足够撰写一篇证据完整、结论克制的稿件；“高送审率”或“达到录用标准”不是实验能直接保证的结论。

用户希望记录的原定目标为“全面优于强基线”。该表述在本文件中仅作为目标保留，完成状态为“现有实验未支持”，不得作为摘要、结果或结论中的已实现事实。例如，CIFAR-10 同风险比较中 entropy 节省36.6373%，高于MSP的36.5434%，且accuracy相同；放宽CIFAR-100近似20%计算组中margin accuracy为57.3000%，高于MSP的57.2467%。这些反例必须与局部优势同时记录。

## 2. 最终批次完成与审计

唯一正式新增输出根目录（下称 R）：

`artifacts/analyses/paper_readiness_20260908_r1/`

- 北京时间 2026-09-08 08:28:28 开始，12:06:01 结束，总计约 3 小时 37 分 32 秒。
- `batch_status.json`：四阶段全部 `completed`，返回码均为 0。
- 公平比较约 33 秒，A3 汇总不到 1 秒，时延重测约 3 小时 37 分钟，审计约 4 秒。
- 本次撰写前逐一核验 `output_hashes.json` 中 **1,818 个文件，哈希不匹配数为 0**。
- `latency_report/analysis_receipt.json`：`passed`，`errors=[]`；1,200 份原始计时数组、300 份 workload、300 份 correctness receipt、1,200,000 个时延观测。
- 检查的 117,120 次样本执行中，路由、argmax 预测、logit 容差错误均为 0。此数不是独立图像数，也不表示所有计时样本都经过逐一正确性审计。
- 未新增训练、未优化 P7、未新增 official/external test 评估。已有 P7 结果只核验结果文件，不重新执行评估或选择阈值。
- `paper_readiness_preflight_20260908/` 是准备阶段诊断输出，其中 `latency` 指向旧实验；**不得当成本次独立重测**。

保留启动脚本、分析脚本、全部结果和失败历史。不要重复启动 `scripts/launch_paper_readiness.py`，不要删除不利 seed。

## 3. 论文定位、研究问题与贡献措辞

推荐工作标题：

> Shared-Threshold Early Exiting Across Model Retraining Seeds: Empirical Risk Constraints and Hardware Trade-offs

建议组织为三个研究问题：

1. 在多个 source 模型上选出的共享阈值，在未参与阈值选择的重训模型上能否保持预设经验风险和计算条件？
2. 不同评分函数及训练组件如何改变准确率、类别风险与计算节省，而非只改变最终头精度？
3. 算术运算节省在什么设备和 batch 场景下转化为实际模型推理时延收益，又在哪里失效？

贡献可写为共享选择/停止规则的明确实现、同协议评分比较及组件分析、配套实际续算实现和完整性能边界。不要宣称 early exit、MSP、KD 或辅助头是原创；方法相对既有研究的独特性须在写作时用原始文献确认。本批并未测量逐模型重新校准的运维成本，不能声称已经实证降低重校准时长。

建议在摘要与讨论中主动说明：评分函数无普遍赢家；约束是经验性质；算术节省不保证硬件加速；独立图像验证存在失败。P7 不能被藏到无关补充材料中。

## 4. 方法与实验设置必须写清的内容

### 4.1 模型版本不能混写

| 版本 | CIFAR 训练结构 | KD | 证据范围 |
|---|---|---|---|
| A0 | 独立基线 | 无早退 KD | 配对参照；非 A3 的 final-only 路径 |
| A1 | exit8 | 无 | P5-C 开发集消融 |
| A2 | exit8 | 有 | P5-C 开发集消融 |
| A3 | exit8 + 训练辅助 exit16 | 无 | P5-C 开发集、P8 时延 |
| A4 | exit8 + 训练辅助 exit16 | 有 | 历史完整配方、P5-C 对照及历史锁定测试 |

部署只在 exit8 决策，未早退样本从已算特征继续，跳过训练专用辅助头。ImageNet-100 的 A3 辅助位置是 exit15，不是 CIFAR 的 exit16。

P1/P2/P4 的历史锁定测试和本批 P5-A/B 缓存比较属于历史完整配方，不能改名成 A3 的 CIFAR official test。本批未填补 A3 的 CIFAR 独立官方精度评估缺口。不要将 A4 的策略节省和 A3 的时延节省拼成同一行单一模型结果。

### 4.2 评分、阈值与成本

- MSP 为最大 softmax 概率；entropy 使用负预测熵，分数越高越早退；margin 是前两名 softmax 概率差，**不是 logit margin，也不是 KL 相对熵**。本批没有测试 KL 散度阈值，不能把“熵阈值基线”改写为“相对熵方法”。
- 本批每个评分采用一致的 source pooled 分位数候选规则：1,001 个分位点、去重后加 source 全 fallback 候选。不是各方法使用完全相同的数值阈值。
- 同风险选择在每个 source seed 上约束总体、balanced、最差类别 accuracy drop，并最大化最小计算节省；总体和 balanced 预算为 0，最差类别预算为 CIFAR-10/严格 CIFAR-100 的 0 或放宽 CIFAR-100 的 4 pp。
- source/target seed：CIFAR-10 为 54–56→57–59；严格 CIFAR-100 为 60–62→63–65；放宽确认则为 60–62→66–68。新模型 seed 不应自动称为独立新图像验证。
- 本批 exploratory 阈值如 MSP 0.9833906228、0.8934412829 不等于历史正式阈值 0.984、0.903；不回写历史 protocol 或结果。
- 本批已见历史结果后进行，是 **post-hoc development-only reanalysis**。执行前锁定本批代码并不使它成为从未见过结果的独立预注册研究。

校正后的节省采用：

`S = 1 - [q × C_early + (1-q) × (C_final + C_exit_head)] / C_final`。

其中 q 是早退比例，early 路径已经含出口头，fallback 仍需付出一次出口头成本。Conv/Linear MAC 不包含 softmax、entropy/margin、路由等全部运算，因此本批“近似相同计算量”不是完全相同实际运行成本，更不是相同时延。

### 4.3 近似计算匹配的精确定义

source 平均节省需在 10%、20%、30% 目标的 ±0.25 pp 内；锁定后迁移到 target，不再调参。target 三方法平均节省最大差 ≤0.5 pp 才标记匹配。本批九个 case×budget 组均通过这一判定。

这只是 **平均 Conv/Linear MAC 在容差内近似匹配**，不是逐 seed 精确匹配；target 也不保证仍在 nominal target ±0.25 pp 内。例如放宽 CIFAR-100 的 nominal 20% 组实际为约 20.90%–21.11%。主表必须同时列 nominal 和 actual。

## 5. 公平策略比较：最终数值与解释

来源：R 的 `fair_comparison.json` 和 `fair_comparison_seed_metrics.csv`。下表 target accuracy 和节省为三个 seed 均值；最差类别列为三个 seed 中的最大下降，不是其均值。数字单位依次为 %、%、pp。

### 5.1 相同 source 风险约束下选出的锁定策略

| Case | 评分 | Target accuracy | 校正 MAC 节省 | 最大最差类别下降 | Target 风险通过 | 每 seed ≥15% 节省 |
|---|---|---:|---:|---:|---|---|
| CIFAR-10 | MSP | 87.0200 | 36.5434 | 0 | 是 | 是 |
| CIFAR-10 | Entropy | 87.0200 | 36.6373 | 0 | 是 | 是 |
| CIFAR-10 | Margin | 87.0200 | 36.4196 | 0 | 是 | 是 |
| CIFAR-100 strict | MSP | 56.5933 | 7.1051 | 0 | 是 | 否 |
| CIFAR-100 strict | Entropy | 56.5933 | 7.2840 | 0 | 是 | 否 |
| CIFAR-100 strict | Margin | 56.5867 | 6.9908 | 0 | 是 | 否 |
| CIFAR-100 relaxed | MSP | 57.5200 | 24.4635 | 4 | 是 | 是 |
| CIFAR-100 relaxed | Entropy | 57.5267 | 26.2035 | 6 | 否 | 是 |
| CIFAR-100 relaxed | Margin | 57.5133 | 24.1665 | 4 | 是 | 是 |

可以得出的结论：CIFAR-10 上三种评分几乎相同，entropy 的节省略高于 MSP。严格 CIFAR-100 上所有方法风险可行时节省都不足 15%，不推翻 P3 停止决定。放宽 CIFAR-100 上 MSP 比 margin 多节省约 0.297 pp，entropy 节省更高但 target 最差类别约束失效。

不能写“同样 target 风险下 MSP 优于 entropy”：entropy 在此处 target 风险并不相同。正确措辞是“相同 source 风险约束选择后，target 可行性不同”。不能把约 0.3 pp 的局部节省差异称为统计显著或大幅优势。

### 5.2 近似匹配 target 平均 MAC 的完整比较

每格为 `accuracy / 实际节省 / 三 seed 最大最差类别下降`，单位 `% / % / pp`。

| Case | Nominal 节省 | MSP | Entropy | Margin |
|---|---:|---|---|---|
| CIFAR-10 | 10% | 87.0200 / 10.8754 / 0 | 87.0200 / 10.8529 / 0 | 87.0200 / 10.9092 / 0 |
| CIFAR-10 | 20% | 87.0200 / 20.1509 / 0 | 87.0200 / 20.1283 / 0 | 87.0200 / 20.1058 / 0 |
| CIFAR-10 | 30% | 87.0200 / 29.6065 / 0 | 87.0200 / 29.5802 / 0 | 87.0200 / 29.6215 / 0 |
| CIFAR-100 strict | 10% | 56.6000 / 9.4809 / 2 | 56.6000 / 9.5952 / 2 | 56.6067 / 9.6904 / 2 |
| CIFAR-100 strict | 20% | 57.1867 / 19.9440 / 2 | 57.1533 / 19.8678 / 2 | 57.2333 / 20.3019 / 2 |
| CIFAR-100 strict | 30% | 57.6733 / 30.2167 / 6 | 57.6400 / 30.2890 / 8 | 57.8867 / 30.0720 / 6 |
| CIFAR-100 relaxed | 10% | 56.6600 / 9.9873 / 0 | 56.6533 / 10.1244 / 0 | 56.6800 / 10.1511 / 0 |
| CIFAR-100 relaxed | 20% | 57.2467 / 20.8997 / 2 | 57.1867 / 20.9225 / 4 | 57.3000 / 21.1091 / 4 |
| CIFAR-100 relaxed | 30% | 57.8333 / 30.9096 / 6 | 57.6000 / 31.1419 / 8 | 57.9267 / 30.6431 / 6 |

严格 CIFAR-100 的这些 compute-only 点全部不满足零最差类别风险；放宽 case 的 30% 点三方法也全部越过 4 pp。不能只报告 accuracy 上升而省略风险违规。放宽 20% 点 MSP 的最差类别下降小于另两者，但 margin 的平均 accuracy 更高；这仍是取舍，不是全面胜出。

本批只实现 MSP/预测熵/概率 margin 三个基线。旧 PCEE 为本地 adapter，不是官方实现；旧 UCB/Hoeffding 结果未经相应范围和多阈值选择的统计保证验证，不能作为已复现的正式风险控制算法来宣称胜出。本批没有完成对所有文献强基线的对比。

## 6. A3 与全部消融：写成取舍，不能只挑最终头

来源：R 的 `ablation_seed_tradeoffs.csv`、`ablation_aggregate_tradeoffs.csv`、`ablation_scope.json`；全部 30 个历史 A0–A4 seed 保留，未新增训练。以下为均值，正式排版时从原文件添加 sample SD。

| 数据集 | 版本 | Final validation accuracy (%) | Calibration policy accuracy (%) | 校正 policy MAC 节省 (%) |
|---|---|---:|---:|---:|
| CIFAR-10 | A0 | 87.2733 | — | — |
| CIFAR-10 | A1 | 88.0667 | 87.2467 | 33.4353 |
| CIFAR-10 | A2 | 88.1333 | 87.4400 | 37.0389 |
| CIFAR-10 | A3 | 88.0400 | 86.9867 | 32.9286 |
| CIFAR-10 | A4 | 87.8467 | 87.1133 | 36.4271 |
| CIFAR-100 | A0 | 55.7267 | — | — |
| CIFAR-100 | A1 | 56.4600 | 56.6200 | 15.2227 |
| CIFAR-100 | A2 | 56.7200 | 57.0267 | 21.6954 |
| CIFAR-100 | A3 | 57.5333 | 57.6867 | 15.1161 |
| CIFAR-100 | A4 | 56.9467 | 57.5067 | 23.8619 |

Final validation 与 calibration policy 不是同一评估子集，不能直接相减来声称路由损失。路由相对最终头的 drop 要读同一 calibration 子集的 reference/drop 字段。

A3 相对 A4：

- CIFAR-10 最终头均值 +0.1933 pp，但 policy accuracy -0.1267 pp、MAC 节省 -3.4985 pp。
- CIFAR-100 最终头均值 +0.5867 pp，policy accuracy +0.1800 pp，但 MAC 节省 -8.7459 pp。
- A2 在 CIFAR-10 的最终头、policy accuracy 和节省均值都高于 A3；必须保留这一对照。不要再称 A3 是“两个数据集一致最佳配置”。
- A3/A4 模型参数相同：CIFAR-10 总参数 2,238,942，出口头 2,260；CIFAR-100 总参数 2,374,572，出口头 22,600。去 KD 简化训练目标，不会自动减少这一模型的参数量。
- KD 在这里使用最终头作为蒸馏目标，不能写成 A3 消除了一个原本单独训练的外部 teacher，或因此省去一次 teacher 训练。

历史每 seed 平均训练墙钟秒数如下，只作带硬件标签的描述：

| 数据集 | A0 | A1 | A2 | A3 | A4 |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | 1208.00 | 929.33 | 927.00 | 936.33 | 1390.33 |
| CIFAR-100 | 945.00 | 933.33 | 934.33 | 947.33 | 940.67 |

CIFAR-10 历史 A0/A4 来自 RTX4090D，新增 A1–A3 来自 RTX3080Ti；原始逐行 `training_gpu`、manifest 和 SHA 必须随表保留。即使 GPU 型号一致，历史运行的软件/负载/CPU 设置也不是本批受控配对训练。CIFAR-100 的 A3 时间还略高于 A4，因此不能据此称去 KD 已实证加速训练。没有受控训练峰值显存、训练 FLOPs 或能耗测量，缺失必须写为 unavailable，不能补造。

结论用语应为：A3 是消除 KD 目标的简化候选，选择它要接受提前出口效果和节省下降，收益依赖数据集。历史 factorial 表仍可用来说明最终精度效应，但其中历史无 fallback-overhead 的节省口径不能与本批校正节省混标。

## 7. 实测系统结果：明确部署场景与全部减速

主证据：R 的 `latency_report/`；原始证据：R 的 `latency/`。

主场景限定为 RTX3080Ti、CIFAR-10、batch1、驻留输入的模型内 FP32 推理。不是实际线上到达流/队列的端到端服务实测，也不是 ImageNet-100 部署测试。CPU 为同机 12 个 Torch 线程，不自动等于“12 个物理核”。

计时比较同一 A3 checkpoint 的 actual_dynamic 与 final_only，非独立训练的 A0。TF32 关闭，采用固定数值设置；2 数据集×3 seed×2 设备×5 batch×5 轮×4 mode；每轮 100 warmup、1,000 timed calls。mode 顺序配对随机化，工作负载保留。预处理、主机到设备输入传输、服务排队，以及计时调用外的路径回传不包含在时延范围内。

每个 seed 内先汇总五轮，再报告三个 seed 的均值±样本 SD。以下为 `100 × (1 - dynamic/final_only)`，正值为时延节省、负值为减速。SD 单位是 pp，不是置信区间。

| 设备/数据集 | b1 | b4 | b8 | b16 | b32 |
|---|---:|---:|---:|---:|---:|
| GPU / CIFAR-10 | 16.909±1.702 | -11.782±2.115 | -18.456±1.314 | -20.899±0.342 | -20.654±1.312 |
| GPU / CIFAR-100 | -1.765±1.991 | -18.545±0.794 | -20.602±0.368 | -21.022±0.455 | -20.819±1.160 |
| CPU / CIFAR-10 | 27.914±0.992 | 7.417±1.608 | 3.322±1.349 | 9.931±0.891 | 4.895±1.047 |
| CPU / CIFAR-100 | 9.918±0.912 | -4.718±1.716 | -1.314±1.289 | 1.144±1.557 | -1.684±1.190 |

独立重测支持原先 GPU/CIFAR-10/b1 的方向：首次 17.14%±1.45 pp，本次 16.91%±1.70 pp。两次复用模型 seed，不能合并称为六个独立训练 seed，也不要选更有利的一次作为唯一结果。建议本批作为主表，旧完整测量作为重复性补充。

GPU/CIFAR-100/b1 本次为负点估计，不能继续复用旧报告“接近持平”的数值。所有 GPU 大 batch 都减速。CPU/CIFAR-100/b16 的 +1.144% 小于 seed SD，不能描述为稳定收益。某些 CPU/CIFAR-10 正点估计也不能自动转成统计显著结论。

必须至少报告 mean latency、p95、throughput 和 savings，但要区分汇总口径：本批 `aggregate_summary.csv` 直接提供 mean latency 与 saving；p95/throughput 需从 `latency/tables/round_metrics.csv` 按先 seed 内后跨 seed 的规则汇总，不能把平均 p95 写成合并所有请求后的总体 p95。三个 seed、重复计时、相同图像的依赖结构均需交代。

节省 16.91% 不等于“1.1691× speedup”；如需报告倍率，应从配对时延比按明确规则计算。平均节省与平均倍率也不应直接互相转换而不说明聚合方式。

分支、compaction、kernel launch 等是解释减速的可能因素；本批没有消融剥离各因素的因果贡献，不要写成已证明的原因。强制 fallback mode 可用于分析额外开销，但并不独自识别所有瓶颈。CUDA allocated-memory 与 CPU lifetime RSS 不是同口径，遥测快照不是能量测量。

## 8. 必须保留的 P3/P6/P7 边界

- P3：严格 CIFAR-100 经验类别风险与 ≥15% 节省不能同时满足，保持 `stop_without_test`。
- P6：`reports/experiments/2026-09-06-early-exit-p6-source-analysis/analysis_receipt.json` 为 `stop_without_target`，12 个 source 训练已完成，但不能称 ResNet-18 成功完成 target/test 泛化。
- P7：固定 ImageNet-100，不是 Tiny ImageNet、ImageNet-1K 或任意 100 类。A3 使用 exit8+训练辅助 exit15，source seeds77–79，target80–82，阈值 0.850；source/target 共用校准图像，独立图像验证才是已有官方 5,000 张图像。

P7 官方结果路径：`artifacts/analyses/early_exit_p7_official_test/test_results.json`。本次核验 SHA-256 仍为 `ac009a3203a3579997c8a5665b63ba1d766cc685f53d865b24017c3700afb749`，未重新执行 evaluator。

| Seed | A0 (%) | A3 final (%) | Policy (%) | Overall drop (pp) | Worst-class drop (pp) | MAC saving (%) |
|---|---:|---:|---:|---:|---:|---:|
| 80 | 81.64 | 82.82 | 82.52 | 0.30 | 4 | 14.3598 |
| 81 | 81.58 | 82.40 | 81.94 | 0.46 | 8 | 14.3036 |
| 82 | 81.18 | 82.14 | 81.92 | 0.22 | 4 | 14.5173 |

均值：A0 81.4667%，A3 final 82.4533%，policy 82.1267%，MAC saving 14.3936%。Policy 对 A0 +0.6600 pp，但相对自身 final -0.3267 pp。三个 seed 节省均不足预设15%，seed81 最差类别下降8pp超过4pp，`all_policy_gates_passed=false`。

每类50张导致一个样本对应2pp，这解释指标粒度，不是豁免失败的理由。P7 的原定成功目标没有实现，本批也没有修复它；结论应说明“校准/模型 seed 可行不保证独立图像上的全约束通过”。

## 9. 正文结构与可直接改写的英文结果段落

### 正文结构

1. Introduction：部署时计算约束与模型重训变化；提出共享经验规则及其验证问题，不以预测录用或“全新注意力网络”开篇。
2. Related Work：动态/多出口网络、置信评分、经验与统计风险控制、实际硬件推理；逐条核验原始论文，不把本地 adapter 冒充公开算法。
3. Method：模型版本、单出口续算、训练目标、共享阈值、每 seed 经验约束、停止规则、校正 MAC 公式。
4. Experimental Protocol：数据划分与模型/图像独立性、历史正式锁定与本批后验分析的区别、基线候选规则、计算容差、全部硬件和统计单位。
5. Results：历史锁定迁移证据→公平评分比较→全消融取舍→P3/P6/P7 边界→实际系统测量与复现。
6. Discussion and Limitations：局部收益、无普遍评分优势、有限 seed/数据集/架构、A3 CIFAR official 缺口、非理论风险保证、非端到端时延、无受控训练加速证据。
7. Conclusion：总结有条件的可复现收益，不宣称所有原目标已完成。

### 评分比较段落

> Under identical source-side empirical risk constraints, the three confidence scores produced similar target performance on CIFAR-10: all achieved 87.02% accuracy, with corrected MAC savings of 36.54%, 36.64%, and 36.42% for MSP, predictive entropy, and probability margin, respectively. On the relaxed CIFAR-100 confirmation cohort, MSP retained the four-percentage-point worst-class budget with 24.46% MAC savings. Entropy obtained higher savings (26.20%) but exceeded that budget on a target seed. These results indicate a score-dependent transfer trade-off rather than uniform superiority of MSP.

### 消融段落

> Removing the distillation objective in A3 increased mean final-head validation accuracy relative to A4 by 0.19 and 0.59 percentage points on CIFAR-10 and CIFAR-100, respectively. However, corrected policy MAC savings decreased by 3.50 and 8.75 percentage points. A3 therefore represents a simpler training objective, not a universally dominant configuration. Historical training durations were not collected as a controlled timing comparison and do not establish a training-speed advantage.

### 系统结果段落

> In a fresh replication on the RTX 3080 Ti, batch-one CIFAR-10 inference reduced model-only latency by 16.91% ± 1.70 percentage points relative to the same A3 checkpoint's final-only path, consistent with the earlier 17.14% ± 1.45-point result. All measured GPU batch sizes from four to thirty-two incurred slowdowns on both datasets. CPU results were dataset- and batch-dependent. These measurements demonstrate a reproducible benefit for a bounded workload, rather than general end-to-end acceleration.

### 独立验证边界段落

> On the fixed ImageNet-100 official validation subset, the locked policy saved 14.39% MACs on average and remained 0.66 percentage points more accurate than the independently trained baseline. Nevertheless, all three seeds fell below the preregistered 15% savings floor, and one exceeded the four-point worst-class degradation budget. This failure limits any claim that source calibration and model-seed transfer alone ensure constraint satisfaction on independent images.

以上段落是结果写作素材，不替代带真实引用的完整摘要/方法/相关工作。摘要可择取约16.9%的主场景收益及独立验证限制；不要只放有利数字而在正文中更改主张。

## 10. 最终表图、证据清单与写作验收

| 资产 | 最低内容 | 数据入口 |
|---|---|---|
| T1 protocol/version | dataset、split、seed、版本、阈值、development/official、GPU | 各阶段 protocol；旧交接4节；R/launch_receipt.json |
| T2 历史锁定测试 | P1/P2/CIFAR-10.1/P4；明确 A4/历史配方 | 旧交接4.1中的 locked/test/external 结果文件 |
| T3 公平评分 | 同 source 风险与近似 MAC 两个面板；actual saving及风险失败 | R/fair_comparison.json；fair_comparison_seed_metrics.csv |
| T4 完整消融 | A0–A4 final、policy、校正 saving、参数、SD；训练时间加范围警告 | R/ablation_*tradeoffs.csv |
| T5 边界 | P3/P6 stop、P7 source/target/official分列、每seed失败 | 各正式回执及P7 test_results.json |
| T6 系统性能 | 全20个 device/dataset/batch组合、mean/p95/throughput/saving | R/latency_report/；latency/tables/round_metrics.csv |
| F1 方法图 | 单部署出口、训练辅助头、source选阈值→target冻结→独立图像测试 | 实际模型/选择代码与protocol |
| F2 风险/计算图 | 三评分、source与target区分；只有离散点时不要伪造连续完整前沿 | R/fair_comparison_seed_metrics.csv |
| F3 消融取舍图 | A1–A4 accuracy–saving配对，标明calibration；不要与final validation混画 | R/ablation_seed_tradeoffs.csv |
| F4 时延图 | 全batch、0线、负值、seed SD；主图本批，旧轮次补充 | R/latency_report/actual_latency_saving_by_batch.pdf |

关键证据文件：R 下 `batch_status.json`、`launch_receipt.json`、`output_hashes.json`、`fair_comparison.json`、`ablation_scope.json`、`latency_report/analysis_receipt.json`。旧P5-C、P6、P7和历史official回执继续保留，不能用一个新 completed 状态代替它们的科学判定。

为论文生成版本化 `evidence_manifest` 和 claim-to-evidence 表时应保存实际输入路径及SHA。本说明已完成本批产物核验与P7结果哈希核验；没有声称重新审计全部P1–P4原始数组。历史测试数字在入稿前仍需沿原始回执核对，不要仅抄旧中文叙述。

最终写作验收清单：

- [ ] 英文主稿完成，结论与本说明边界一致；作者/单位/基金等使用真实信息。
- [ ] 方法版本、阈值来源、source/target/official，以及模型seed/图像独立性逐项标明。
- [ ] 所有均值、SD、pp与%从机器可读文件计算；三seed不变成百万个独立重复。
- [ ] 历史与本批MAC口径分开；不把时延节省、倍率、吞吐增益当作同一数字。
- [ ] T1–T6、F1–F4完整；全部减速、失效约束、A2反例及P7失败不遗漏。
- [ ] 不声称普遍优于entropy/margin、理论风险保证、训练加速、跨架构普遍泛化或端到端服务加速。
- [ ] 相关工作引用真实，期刊当前模板及声明要求通过出版社官方资料核验；本说明没有替代这一步。
- [ ] 编译LaTeX，检查引用/交叉引用，渲染最终PDF检查表图裁切、字体和排版。
- [ ] 交付源稿、BibTeX、PDF表图、生成脚本、源表、证据清单、最终PDF。

以上剩余项是论文制作与审阅工作，不是再次训练的默认理由。只有作者决定扩大科学主张时，才讨论额外证据需求；不得为了让当前主张更好看而删seed、改测试阈值或重复试验。

## 11. 可直接发送给写作对话的指令

```text
请连接 ssh -p 50799 root@connect.westb.seetacloud.com，在
/root/autodl-tmp/image-classification 中先读取 AGENTS.md 和
docs/final_paper_writing_brief_20260908.md，再参考旧交接证据入口。

实验已冻结。正式新增批次为 artifacts/analyses/paper_readiness_20260908_r1，
四阶段完成、1818份产物哈希核验无差异，时延审计通过。不要重跑实验、下载数据、
重访official/external evaluator，或利用测试数据选择阈值。

请利用现有材料完成目标为 The Journal of Supercomputing 的英文稿件及可追溯表图。
论文主线是共享经验风险约束早退的模型seed迁移与硬件取舍，不是全面击败强基线。
保留entropy/margin接近或更优、A3节省下降、A2反例、P6停止、P7官方门槛失败和GPU大batch减速。
主系统数字采用本批GPU/CIFAR10/b1节省16.91%±1.70pp，范围是驻留模型内计时，非端到端服务。
历史A4官方精度证据不能改名成A3；新比较是后验development分析，不是新官方测试。

先定位实际LaTeX目录并保护作者改动，再做证据表、完整表图、方法、实验、讨论、摘要和引言，
核验原始文献与出版社要求，编译并逐页检查PDF。完成说明中的写作验收项；
不将实验完成等同于录用保证，不重复询问已经授权的正常写作步骤。
```
