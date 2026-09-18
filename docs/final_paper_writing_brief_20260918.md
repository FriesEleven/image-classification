# 最终论文内容说明：2026-09-18 补充实验完成版

本文件更新 `docs/final_paper_writing_brief_20260908.md` 和 `docs/paper_writing_handoff_2026-09-07.md` 的现状判断。历史结果继续保留；涉及 A3 CIFAR 官方精度、同模型工作点、共享/逐模型策略对照时以本文件为准。

实验根目录 `/root/autodl-tmp/image-classification`，SSH端口50799。当前只在服务器整理材料，没有新增下载到本地。目标期刊沿用 The Journal of Supercomputing；本次不进行格式/文献查新或论文编译。

## 1. 实验已完成，进入写作

- M0输入盘点与冻结、M1开发缓存比较已在启动前完成；M2/M3北京时间2026-09-18 09:27:41至09:44:46执行，总计约17分04秒。此前1–3小时是保守预算，实际这次只测GPU/batch1、两个模式，明显短于历史CPU/全batch批次。
- 正式原始输出：`artifacts/analyses/minimal_core_evidence_20260918_r1/`。
- 精简实验归档：`reports/experiments/2026-09-18-minimal-core-evidence/`。
- 论文直接使用的CSV/LaTeX、图和证据索引：`reports/paper/final-20260918/`。
- 批次为completed、技术审计passed。核验94个原始产物哈希，没有差异；额外从保存数组复算M1全部72条工作点、M2六模型，检查60份原始计时的p95和吞吐，一致。
- M2对六个模型各10000张官方图像做实际singleton正确性检查，共60000次样本执行，route、argmax和logit tolerance错误均为0。六模型复用两个benchmark的图像，不称60000张不同图像。
- 本次整理只读保存数组与结果，没有重新运行模型或访问官方图片。
- 一次性官方访问标记、启动日志、原始数组永久保留。不要重复启动本批；无需为了让所有指标通过而继续实验。

## 2. 本轮填补什么，什么仍然不成立

本轮已经补齐：共享策略与逐模型重校准的直接对照、类别约束消融，以及六个固定A3 checkpoint在CIFAR官方benchmark上的补充锁定评估与同测试数据范围的精度—MAC—时延工作点。

旧稿中“没有A3 CIFAR官方精度评估”“缺少共享与独立标定直接比较”“没有同checkpoint完整工作点”需要更新。本次不是新模型seed迁移验证，也不是项目完全未接触过的独立benchmark确认。不能因此删除历史数据曝光、有限seed、ImageNet-100失败或硬件适用范围限制。

**没有证明全面优于强基线，且没有证明max-min目标优于mean目标。** S/A在本协议下相同；逐模型重校准在放宽CIFAR-100上通过类别预算而共享策略失败。A3/CIFAR-100节省边缘seed失败、平均时延没有收益；CIFAR-10平均时延改善也没有改善p95。

主线仍建议为：共享经验约束早退的重训模型取舍，以及计算节省、类别风险与真实硬件性能的边界。不可将early exit、MSP、KD/辅助头本身写成原创；创新性需由真实相关工作对照论证。

## 3. M1：共享、逐模型、平均目标与去类别约束

全部M1属于历史完整配方A4的事后开发分析，主评分MSP，不改名A3。S为共享max-min+类别约束，I为逐模型单独标定，A为共享mean+相同约束，C为共享max-min去类别约束。

每模型按图像ID对应到分层2500张F和2500张E。CIFAR-100 confirmation与source不是相同5000张；重叠ID继承同一F/E归属，新ID填满每类50/50配额。PCG64 seed20260918与实际ID/哈希已经冻结，source-F/target-E三个队列的交集均为0。图像此前用于历史研究，只称本轮拟合/审计分离。

Source-F主表记录拟合结果，target-E主表记录审计结果，两面板均在 `shared_policy_manuscript.csv/.tex`。下表列target-E均值，最差类别列为跨seed最大值。风险通过指所有seed满足原定总体/balanced/类别预算；节省通过指每seed≥15%。

| 队列 | 策略 | Accuracy (%) | 校正MAC节省 (%) | Worst-class最大下降 (pp) | 风险通过 | 节省通过 |
|---|---|---:|---:|---:|---|---|
| CIFAR-10 | S | 86.6800 | 36.2657 | 0 | 是 | 是 |
| CIFAR-10 | I | 86.7067 | 46.4983 | 2.4 | 否 | 是 |
| CIFAR-10 | A | 86.6800 | 36.2657 | 0 | 是 | 是 |
| CIFAR-10 | C | 86.7067 | 48.6980 | 0.8 | 否 | 是 |
| CIFAR-100 strict | S | 56.6933 | 9.6980 | 4 | 否 | 否 |
| CIFAR-100 strict | I | 56.8933 | 14.8838 | 4 | 否 | 否 |
| CIFAR-100 strict | A | 56.6933 | 9.6980 | 4 | 否 | 否 |
| CIFAR-100 strict | C | 57.5600 | 57.0025 | 16 | 否 | 是 |
| CIFAR-100 relaxed | S | 57.2400 | 24.1056 | 8 | 否 | 是 |
| CIFAR-100 relaxed | I | 57.0667 | 21.7906 | 4 | 是 | 是 |
| CIFAR-100 relaxed | A | 57.2400 | 24.1056 | 8 | 否 | 是 |
| CIFAR-100 relaxed | C | 56.8667 | 57.0025 | 36 | 否 | 是 |

S/A/C在source拟合一个共享阈值，target重校准0次；I在target的各自F上重新标定，维护3个target阈值。因此I有额外target拟合信息。这些不是相同信息条件的迁移优劣比较，也未测量人工维护时间。

可写：CIFAR-10共享策略以更低节省换得审计集零类别下降；放宽CIFAR-100逐模型重校准有更强的审计可行性；移除类别约束取得更高节省但可能导致严重类别损失。严格CIFAR-100不成功，不以relaxed替代其失败。

S/A同阈值有结构原因：一个共享标量阈值下，每seed的早退率/节省随阈值升高而不增加；min和mean节省在相同可行候选集合上均偏好较低阈值。按规定并列规则，本协议选择相同候选。max-min可作为选择定义保留，不能作为被此消融证实优于mean的贡献。

候选统一为1001个F分位点、去重加全fallback点；I使用各自F、相同搜索分辨率。总体及balanced预算0，类别预算0/4pp，判定比例容差1e-12。风险与节省判定分列，完整门槛均值不能替代逐seed判定。

## 4. M2：A3官方精度—MAC—实际时延完整工作点

固定对象：CIFAR-10 seeds54/55/56、阈值0.984；CIFAR-100 seeds66/67/68、阈值0.903。部署exit8，exit16只用于训练，无KD。参照是相同checkpoint的final-only，非独立A0。

精度及MAC统计所有10000张官方test图像；计时来自同一test范围，固定PCG64 permutation前1000个图像ID，跨模型/轮次使用相同子集。五轮配对，每模式100warmup、1000timed calls，只有final-only和actual_dynamic。TF32关闭、确定性FP32、严格GPU同步。输入驻留GPU；不含预处理、输入传输、排队或计时外路径回传。

### 全部seed，保留失败

| Dataset/seed | Final (%) | Policy (%) | Worst-class drop (pp) | 全test MAC节省 (%) | 平均时延节省 (%) | 风险/节省门槛 |
|---|---:|---:|---:|---:|---:|---|
| CIFAR-10/54 | 86.88 | 86.88 | 0 | 32.844121 | 17.734032 | 通过/通过 |
| CIFAR-10/55 | 87.26 | 87.26 | 0 | 33.243894 | 17.734171 | 通过/通过 |
| CIFAR-10/56 | 87.10 | 87.11 | 0 | 32.742770 | 21.832285 | 通过/通过 |
| CIFAR-100/66 | 56.98 | 57.34 | 2 | 15.180790 | 0.091955 | 通过/通过 |
| CIFAR-100/67 | 57.40 | 57.96 | 2 | 15.306438 | -1.829183 | 通过/通过 |
| CIFAR-100/68 | 57.45 | 57.61 | 1 | 14.998029 | -2.750967 | 通过/失败 |

Seed68距离15%下限只差约0.001971pp，属于非常接近门槛的失败，不能夸大为大幅失效；也不能四舍五入为15.00%后改判通过。校正fallback额外出口成本已计入，判定使用原始小数。

### 三seed均值±sample SD

| 指标 | CIFAR-10 | CIFAR-100 |
|---|---:|---:|
| Final accuracy (%) | 87.0800±0.1908 | 57.2767±0.2581 |
| Policy accuracy (%) | 87.0833±0.1914 | 57.6367±0.3109 |
| 全test早退率 (%) | 58.5267±0.4706 | 26.7267±0.2715 |
| 全test MAC节省 (%) | 32.9436±0.2650 | 15.1618±0.1551 |
| Final平均时延 (ms) | 4.7570±0.4636 | 4.6395±0.0398 |
| Dynamic平均时延 (ms) | 3.8386±0.2538 | 4.7091±0.1077 |
| 配对平均时延节省 (%) | 19.1002±2.3661 | -1.4961±1.4504 |
| Final平均轮内p95 (ms) | 5.1500±0.6592 | 4.9850±0.0326 |
| Dynamic平均轮内p95 (ms) | 5.6702±0.3911 | 5.7509±0.1551 |
| 配对轮内p95节省 (%) | -10.6537±6.4100 | -15.4626±2.5428 |

百分比指标的SD用pp，ms指标SD用ms；n=3个训练seed。先每seed汇总五轮再跨seed。配对saving从每轮比值计算；不能拿表中跨seed平均时延相除替代，p95也不能由平均p95相除替代配对汇总。

CIFAR-10是平均时延的有界场景收益，p95未受益；不适合据本批直接主张改善严格尾时延SLA。CIFAR-100一个seed平均时延略正，另两个减速，没有稳定平均收益；两数据集的p95都变慢。p95变化原因没有被因果剥离，勿将特定kernel/compaction原因写成已证实结论。

计时子集与全test统计范围不同：CIFAR-10计时子集平均MAC节省33.5667%，全test32.9436%；CIFAR-100分别14.7658%和15.1618%。不能用子集数字代替全test门槛，也不能宣称每张官方图像都被计时。完整工作点可以同表展示，但caption必须标明这两个范围。

## 5. 条件配对统计

本次整理新增2000次分层配对图像bootstrap，PCG64 seed20260918，从保存预测计算policy相对自身final的准确率差。跨模型的相同图像同时重采样，保留模型间图像依赖：

- CIFAR-10：+0.003333pp，95%区间[0.000000,0.010000]pp；主要为一个模型挽回一个样本，勿称大幅精度提升。
- CIFAR-100：+0.360000pp，95%区间[0.263333,0.456667]pp。

区间条件于这三个固定模型及当前benchmark，不是对未来重训模型总体的区间，也不能消除历史曝光或选择偏差，更不是100类同时风险保证。最差类别仍报告原始计数和经验最大值。

M1的I/A/C对S的配对bootstrap已经存在原始summary文件中；论文按该文件使用，不混淆source-F拟合统计与target-E审计统计。

## 6. 论文具体修改清单

| 部分 | 应更新的内容 |
|---|---|
| Method | 共享阈值、逐模型I的信息条件、F/E图像ID划分、类别约束、校正MAC；说明S/A在单阈值协议下等价 |
| Setup | 六模型及checkpoint哈希、固定阈值、正式授权、历史benchmark曝光、官方全量/计时子集、数值设置 |
| Results | 新增source-F/target-E两面板共享对照；新增同A3工作点seed表及均值表；披露seed68边缘失败和p95减速 |
| Abstract | 可以用CIFAR-10平均时延节省19.10%作为本轮官方图像范围例子，写明model-only/batch1；不要改成端到端或尾时延加速 |
| Discussion | 共享重校准取舍、类别约束的作用、mean/min目标的不可区分性、MAC与mean/p95性能不一致 |
| Limitations | 更新已补齐的A3官方精度缺口；继续保留n=3、历史曝光、非新模型迁移、有限设备/工作负载及P6/P7边界 |

旧P8和20260908结果使用calibration图像，继续作为另一个明确scope的测量保留；本次official workload的19.10%不能当作同一工作负载第三次复现去合并。旧RTX3080Ti/GPU batch4–32减速、CPU数据和P7官方失败继续完整披露。

本文不修改P1/P4历史A4official标签，不推翻P3严格stop，也不将P6包装为跨架构成功。新A3评估只填补相应模型精度证据，不能让A4/A3混成一个统一方法结果。

### 可直接改写的英文结果段落

> The shared threshold preserved the zero empirical risk budget on the CIFAR-10 audit partition, saving 36.27% MACs, whereas individually recalibrated thresholds saved 46.50% but violated that budget. On relaxed CIFAR-100, the shared policy exceeded the four-point worst-class budget, while individual recalibration satisfied it with lower MAC savings. The comparison quantifies a dataset-dependent calibration trade-off; individual policies use target-model fitting information unavailable to the shared policy.

> For the six locked A3 checkpoints, supplementary evaluation on the official CIFAR benchmarks closed the same-model accuracy–compute evidence gap. On CIFAR-10, the policy achieved 87.08% accuracy and saved 32.94% MACs, with a 19.10% ± 2.37-point reduction in mean model-only latency on a resident singleton subset. However, paired per-round p95 latency worsened by 10.65% ± 6.41 points. On CIFAR-100, average MAC savings were 15.16%, but one seed fell marginally below the 15% floor; mean and p95 latency savings were negative. Arithmetic savings therefore did not imply either universal acceleration or improved tail latency.

> The mean-saving and minimum-saving objectives selected identical shared thresholds. With a single nested threshold policy and a common feasible set, both objectives favor increasing early-exit coverage. This comparison does not establish a distinct advantage of the minimum-seed objective.

## 7. 文件入口和交付

从 `reports/paper/final-20260918/README.md` 开始。主要文件：

- `shared_policy_manuscript.csv/.tex`：source-F和target-E两面板、全部策略/队列、均值SD、风险及节省条件。
- `a3_joint_seed_manuscript.csv`：六seed精度/MAC/mean/p95/吞吐、子集与全test范围、checkpoint哈希。
- `a3_joint_summary_manuscript.csv`、`a3_joint_manuscript.tex`：三seed均值SD、p95负收益、条件bootstrap。
- `shared_policy_tradeoffs.pdf`：离散target-E点，类别预算失败用红色；S/A重叠有真实原因，不人为分离。
- `recalculation_audit.json`、`evidence_manifest.json`、`claim_to_evidence.md`：核验和逐项声明来源。

结果生产脚本 `scripts/analysis/build_minimal_core_paper_materials.py` 只读取保存结果，并拒绝覆盖已有输出。不要为重新排版重复执行official evaluator。变更排版应单独创建版本化输出，不改冻结原始产物。

剩余工作是将新表图并入真实LaTeX稿、修改相应方法/结果/摘要/局限、核验引用与期刊要求、编译渲染验收。此次整理未编译论文，未推送Git，未执行新的长实验。没有必须继续训练的默认需求；若作者扩大主张，再明确新证据需求。
