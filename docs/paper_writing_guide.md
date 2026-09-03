# 当前早退实验论文写作与交付指南

## 1. 文档定位

本文是下一段对话完成整篇论文的唯一执行指南。目标不是继续修补旧的静态注意力论文，而是保留
Springer `sn-article` 模板、作者与单位等元数据，围绕当前已经冻结的风险约束早退实验进行结构性重写。

本文覆盖：

- 论文主线、贡献边界和禁止表述；
- 按章节的写作目标、必写事实和证据来源；
- 主表、迁移表、消融/边界表和复杂度表的字段；
- 全部论文图片的 PDF 输出规范；
- 从冻结结果生成表图、改写 LaTeX、编译 PDF 和逐页验收的顺序；
- 可直接交给下一段对话的启动提示。

冻结实验代码与结果的基础提交为
`2244cae03555d38fb8d4e42cdf46d066d3119886`。证据包说明已在后续提交
`4fb3b1f28c09ff5b43e690e59c0259dba73bfa5e` 中加入。写作过程中允许新增制表、绘图和论文资产，
但不得改写冻结策略、正式实验记录或 official-test 结果。

## 2. 开始工作前必须读取的材料

下一段对话开始时，按以下顺序完整读取：

1. `AGENTS.md`；
2. `docs/paper_writing_guide.md`（本文）；
3. `docs/paper_evidence_bundle.md`；
4. `docs/handoff.md` 的第 28 节；文件顶部旧状态可能滞后，以第 28 节和机器可读结果为准；
5. `docs/early_exit_p0_plan.md`、`docs/early_exit_p1_plan.md`、
   `docs/early_exit_p2_plan.md`、`docs/early_exit_cifar10_1_v6_plan.md`、
   `docs/early_exit_p3_plan.md`、`docs/early_exit_p4_plan.md`；
6. 第 5 节列出的正式 JSON、审计和时延档案；
7. 旧论文目录自己的 `docs/handoff.md` 和 `sn-article.tex`。

在任何修改前同时检查：

```bash
git status --short --branch
git rev-parse HEAD
```

服务器实验仓库根目录是 `/root/autodl-tmp/image-classification`。旧论文在此前主机上的实际目录为
`/Users/heyi.suo/out/sn-article-template`，但另一台电脑上路径可能不同，必须先定位实际
`sn-article.tex`，不能继续使用旧交接中的 `/Users/salt/...` 绝对路径。

旧论文目录此前不是 Git 仓库。开始大改前，应先复制一份不可变备份，或将论文目录纳入独立版本控制；
不得覆盖唯一原稿后才发现需要回退。

## 3. 总体决策：结构性重写，而不是局部修补

旧稿研究问题是 SE/CBAM 的静态分层部署，方法名曾使用 PSHA-Net、Hybrid-Attention 或 CSGHA。
当前正证据回答的是另一问题：在轻量分类器中，能否在类别经验风险和预算约束下选择跨独立训练 seed
共享的阶段早退策略，并在冻结策略后迁移到新模型版本、自然分布转移和第二训练数据集。

因此：

- 标题、摘要、关键词、Introduction、Related Work、Method、Experiments、Discussion、Limitations 和
  Conclusion 均重新写；
- 只保留模板、作者/单位/Funding、必要的 MobileNetV2 和 CIFAR 基础事实；
- 旧主表、旧注意力消融、旧精度—FLOPs 气泡图、Grad-CAM、t-SNE 和静态注意力架构图全部退出正文；
- CSGHA 与预算阶段稀疏注意力只作为负结果/研究转向背景，最多在 Discussion 或附录简要呈现；
- 不得把旧静态注意力的 81--84% 数字与当前统一训练协议下的 86--88% CIFAR-10 数字放进同一主表。

预计旧正文只有 15%--30% 的通用材料可压缩后复用，科学内容应视为新论文。

## 4. 推荐论文定位、名称与贡献边界

### 4.1 推荐标题

首选标题：

```text
Risk-Constrained Shared Early Exiting for Budget-Aware Lightweight Image Classification
```

可选更保守标题：

```text
An Empirical Study of Cross-Seed Risk-Constrained Early Exiting for Lightweight Classifiers
```

最终标题不得再出现 `hierarchical attention`、`primary-secondary attention` 或 `CSGHA`。没有手机真机
测量，因此不要在标题中使用 `mobile real-time deployment`。

方法简称如确有需要，可统一使用临时名 `RCSEE`（Risk-Constrained Shared Early Exiting）。在定稿前
全文搜索并保证只出现一个名称；如果不需要品牌化，直接使用 `the proposed shared early-exit policy`
更稳妥。

### 4.2 唯一允许的核心主张

论文应将贡献收缩为以下三点：

1. 在总体准确率、balanced accuracy 和最差类别经验下降约束下，从多个独立训练 seed 上选择一个
   共享早退阈值，并以最差 seed 的计算节省作为优先目标；
2. 使用严格分离的模型选择、策略校准/确认和 official test 边界，并把失败停止规则、后验诊断与
   独立确认完整公开；
3. 在 CIFAR-10、未见重训 seed、CIFAR-10.1 v6 和 CIFAR-100 上验证冻结策略的适用范围，并用真实
   分阶段实现的服务器 GPU 配对剖析补充 MAC 代理。

### 4.3 明确禁止的主张

不得写：

- “首次提出 early exit / KD / softmax threshold / class-wise threshold”；
- “理论零风险”“统计保证零下降”或“distribution-free guarantee”；
- “在所有 seed、数据集或 backbone 上泛化”；
- “达到 SOTA”，因为当前没有同协议复现的近期 early-exit 方法；
- “真实移动端实时部署”，因为只做了服务器 GPU 测量；
- “MAC 节省等同于时延或能耗节省”；
- “P3 通过”或“P3 official test 结果”，因为 P3 固定为 `stop_without_test`；
- “CIFAR-100 是全项目第一次盲测”，因为历史无关 baseline 曾暴露其 test；正确说法是
  “P4 模型和方法锁定后的首次 official-test 评估”；
- “exit16 已被证明必要”，因为没有移除 exit16 的独立训练消融；
- “KD 是本文创新”，它只是固定训练 recipe。

经验零下降必须写成 `no empirical degradation was observed on the finite evaluation set`，并同时说明
每类样本数。`significant` 只在有正式统计检验时使用；当前三 seed 结果优先写 `consistent`、`observed`
或直接报告数值。

## 5. 数值唯一来源与冻结结果账本

### 5.1 证据优先级

任何论文数值按以下优先级读取：

1. `reports/experiments/**/test_results.json`、`external_results.json`、`confirmation.json`；
2. 对应 `source_index.json`、official test lock 和正式 audit；
3. 服务器 ignored `artifacts/` 中的 checkpoint、split、logits/routes；
4. `docs/handoff.md` 仅用于解释历史，不作为抄写数值的首选来源；
5. 旧论文表格、旧 CSV 和旧图片不得作为当前结果来源。

所有制表脚本先读 JSON 中的原始小数，再在最终展示层统一四舍五入。不能从本节已经四舍五入的数字
反推或覆盖源数据。

### 5.2 CIFAR-10 P1b 方法锁定 test

来源：

```text
reports/experiments/2026-09-02-early-exit-p1b/locked_selection.json
reports/experiments/2026-09-02-early-exit-p1b/test_results.json
reports/audits/2026-09-02-early-exit-p1b/audit_results.json
```

| 指标 | 冻结结果 |
|---|---:|
| baseline test accuracy | 86.83 ± 0.27% |
| multi-exit final-head test accuracy | 87.03 ± 0.18% |
| locked-policy test accuracy | 87.04 ± 0.18% |
| policy vs. baseline | +0.203 ± 0.429 pp |
| policy vs. final head | +0.010 ± 0.010 pp |
| early-route fraction | 64.85 ± 0.30% |
| MAC saving | 36.51 ± 0.17% |
| worst-class drop vs. final | 0.00 ± 0.00 pp |
| shared threshold | 0.984 |

阈值 `0.984` 仅由 seeds 54/55/56 的三个独立 5k calibration 集共同选择。official test 上候选阈值数
为 0，不能再重新搜索。

### 5.3 P2 未见模型版本迁移

来源：

```text
reports/experiments/2026-09-03-early-exit-p2a-transfer/transfer_results.json
reports/audits/2026-09-03-early-exit-p2a/audit_results.json
```

在 seeds 57/58/59 上继续使用 P1b 阈值 `0.984`，没有重新校准：

| 指标 | 冻结结果 |
|---|---:|
| final-head paired validation gain | +0.220 ± 0.171 pp |
| early-route fraction | 64.71 ± 2.48% |
| MAC saving | 36.43 ± 1.40% |
| maximum overall drop | 0 pp |
| maximum balanced drop | 0 pp |
| maximum worst-class drop | 0 pp |

P2 的 5k transfer 集与 P1b calibration 使用同一索引，只证明跨模型版本迁移，不是独立数据分布。

### 5.4 CIFAR-10.1 v6 一次性外部分布验证

来源：

```text
reports/experiments/2026-09-03-early-exit-p2-cifar10-1-v6/external_results.json
reports/data/2026-09-03-cifar10-1-v6/source_receipt.json
```

六个模型版本 seeds 54--59 的合并描述：

| 指标 | 冻结结果 |
|---|---:|
| baseline accuracy | 76.76 ± 0.64% |
| final-head accuracy | 76.68 ± 0.67% |
| locked-policy accuracy | 76.69 ± 0.67% |
| policy vs. final head | +0.008 ± 0.020 pp |
| early-route fraction | 49.01 ± 1.93% |
| MAC saving | 27.59 ± 1.09% |
| worst-class drop vs. final | 0.00 ± 0.00 pp |

CIFAR-10.1 v6 有 2,000 张图、每类 200 张。它是 CIFAR-10-like 自然分布转移，不得写成广泛 OOD
鲁棒性证明。

### 5.5 CIFAR-100 P3 失败边界

来源：

```text
reports/experiments/2026-09-03-early-exit-p3-cifar100/selection.json
reports/diagnostics/2026-09-03-early-exit-p3-boundary-v2/diagnostic.json
reports/diagnostics/2026-09-03-early-exit-p3-class-guard-v2/diagnostic.json
```

必须保留以下事实：

- multi-exit final head 相对 matched baseline 在 seeds 60--65 上 6/6 胜出，平均
  `+2.173 ± 0.397 pp`；
- 原预注册的零总体、零 balanced、零最差类别下降和至少 15% MAC 节省不能同时满足；
- P3 正式状态固定为 `stop_without_test`，official CIFAR-100 test 没有打开；
- 严格零最差类别下降时最多只有约 6.7% MAC 节省；
- 允许 2 pp 最差类别下降时，最低风险动态点只有 13.01% MAC 节省，仍低于 15% 门槛；
- 达到至少 15% MAC 节省需要放宽到 4 pp 最差类别经验下降；
- 预测类别保护在严格零风险下无可行解，放宽一张样本的方案也未迁移到 target seeds。

除“原冻结 P3 失败”外，其余风险边界与 class-guard 搜索均是看到失败后的 calibration-only 诊断，
正文必须标注 `post-hoc diagnostic`，不能冒充预注册消融或 test 结论。

### 5.6 CIFAR-100 P4 独立确认与方法锁定 test

来源：

```text
reports/experiments/2026-09-03-early-exit-p4-design/locked_policy.json
reports/experiments/2026-09-03-early-exit-p4-cifar100/confirmation.json
reports/experiments/2026-09-03-early-exit-p4-cifar100/official_test_lock.json
reports/experiments/2026-09-03-early-exit-p4-cifar100-test/test_results.json
reports/audits/2026-09-03-early-exit-p4a/audit_results.json
```

P4 使用全新 split seed 20260904、训练 seeds 66/67/68，并在训练前冻结阈值 `0.903`。P4 confirmation
候选阈值数为 0，不允许逐模型校准。

独立 confirmation：

- final-head paired validation gain：`+1.220 ± 0.987 pp`；
- early-route：41.94/41.54/42.40%；
- MAC saving：23.91/23.68/24.17%；
- policy 相对 final head 总体准确率提高：1.10/0.90/0.82 pp；
- 每个 seed 的最差类别经验下降均为 4 pp；
- 所有冻结 confirmation gate 通过。

方法锁定 official test：

| 指标 | 冻结结果 |
|---|---:|
| baseline test accuracy | 55.93 ± 0.14% |
| multi-exit final-head test accuracy | 57.51 ± 0.18% |
| locked-policy test accuracy | 58.10 ± 0.03% |
| final head vs. baseline | +1.580 ± 0.320 pp |
| policy vs. baseline | +2.163 ± 0.114 pp |
| policy vs. final head | +0.583 ± 0.212 pp |
| early-route fraction | 42.39 ± 0.30% |
| MAC saving | 24.17 ± 0.17% |
| worst-class drop by seed | 3/4/2 pp |
| shared threshold | 0.903 |

P4 test 上策略相对最终头的总体和 balanced accuracy 均提高；worst-class 仍有有限下降。这正是平均收益
和类别风险边界并存的结果，不得只报告总体提升而隐藏逐类风险。

### 5.7 参数、MAC 与 RTX4090D 时延

| 数据集/模型 | 参数量 | 出口头增量 |
|---|---:|---:|
| CIFAR-10 baseline | 2,236,682 | 0 |
| CIFAR-10 multi-exit | 2,238,942 | 2,260（约 0.101%） |
| CIFAR-100 baseline | 2,351,972 | 0 |
| CIFAR-100 multi-exit | 2,374,572 | 22,600（约 0.961%） |

MAC 路径代理：

| 数据集 | exit8 | exit16 | final |
|---|---:|---:|---:|
| CIFAR-10 | 2,676,864 | 5,234,688 | 6,124,928 |
| CIFAR-100 | 2,682,624 | 5,249,088 | 6,240,128 |

最终部署策略只使用 `exit8 → final`；exit16 是训练辅助头，推理时跳过。MAC 统计只计 Conv/Linear，
忽略 BN、激活、池化、内存访问和调度，因此正文统一写 `MAC proxy` 或 `Conv/Linear MACs`，不要无说明
写成完整 FLOPs 或真实时延。

RTX4090D 部署剖析来源：

```text
reports/profiles/2026-09-02-early-exit-p1b-rtx4090d/profile.json
```

三 seed 平均：final-only `4.3545 ms`，冻结策略期望时延 `3.3073 ms`，节省 `24.91%`，加速
`1.334×`。这是 batch-1、同步 wall-clock、隔离路径后按冻结 test 路由比例加权的期望时延；不是整批
真实数据流测量，不是手机测试，也没有能耗。

## 6. 数据协议和硬件的准确写法

### 6.1 数据划分

CIFAR-10 P1/P2：

- 官方 50k train 分成 40k 参数训练、5k checkpoint-selection validation、5k policy calibration/transfer；
- `split_seed=20260902`，三个子集分层、互斥并覆盖 50k；
- P1 seeds 54--56 用于选择共享阈值，P2 seeds 57--59 只接收冻结阈值；
- official 10k test 只在 P1 锁定后执行一次。

CIFAR-100 P3：

- `split_seed=20260903`；40k train、5k model-selection、5k calibration/transfer；
- source seeds 60--62，target seeds 63--65；
- 严格 gate 失败后停止，未访问 official test。

CIFAR-100 P4：

- 全新 `split_seed=20260904`；40k train、5k model-selection、5k policy confirmation；
- seeds 66--68；阈值 `0.903` 在 P4 开始前冻结，候选数为 0；
- confirmation 通过后，official 10k test 做一次方法锁定评估。

### 6.2 训练 recipe

正式 matched baseline 与 multi-exit 使用相同基础 recipe：

- 200 epochs、batch size 128；
- AdamW、OneCycleLR；配置中的 `lr=0.01` 是 OneCycle 最大学习率；
- AMP、CUDA Graph 训练，evaluation 为 eager FP32；
- 8 DataLoader workers、`prefetch_factor=8`、串行 `jobs=1`；
- best checkpoint 只由 final-head model-selection validation accuracy 决定；
- 训练时 `evaluate_test=false`、`measure_inference=false`。

论文不要用训练墙钟时间比较 GPU，也不要把 GPU 利用率当算法指标。

### 6.3 论文中的硬件表述

论文正文统一采用以下口径，不展开服务器迁移过程，也不放逐批 GPU 映射表：

> Training was performed primarily on a single NVIDIA GeForce RTX 3080 Ti GPU. Batch-1 latency was
> measured separately on a single NVIDIA GeForce RTX 4090 D GPU.

中文含义是“模型训练主要使用单张 RTX 3080 Ti；batch-1 延迟在单张 RTX 4090 D 上独立测量”。不要
写成“所有实验均 exclusively 在 3080 Ti 上完成”，也不要比较不同 GPU 上的训练墙钟时间。这样既突出
项目的主要训练平台，也不会把临时服务器调度包装成算法变量。

内部审计时仍以各 run 的 `provenance.json` 为准；这些记录只用于追溯，不复制到论文正文。P0--P4
provenance 记录的软件栈为 Python 3.12.3、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、CUDA
runtime 12.8。论文可报告统一软件栈，不报告与方法无关的 CPU interop 线程差异。

## 7. Proposed Method 章节所需技术内容

### 7.1 Backbone 与出口

主干是 torchvision MobileNetV2。分别在 `features[8]` 和 `features[16]` 后加入
`AdaptiveAvgPool2d(1) + Linear` 轻量分类头。模型训练输出顺序为 final、exit8、exit16；实际部署
策略只允许 exit8 或 final。

必须画一张新架构图，明确：

- 输入、MobileNetV2 stage/block 流；
- exit8 轻量头和 softmax confidence；
- 阈值判断；
- 早退路径直接输出；
- 未满足阈值时只继续剩余 backbone 和 final head；
- exit16 用虚线标注为 training-only auxiliary head；
- 不再出现 SE/CBAM/CSGHA 模块。

### 7.2 训练目标

正文给出：

```latex
\begin{aligned}
\mathcal{L} ={}& \operatorname{CE}(z_f,y) \\
&+ 0.2\left[0.5\operatorname{CE}(z_8,y)
+0.5T^2\operatorname{KL}(p_8^T\,\|\,\operatorname{sg}(p_f^T))\right] \\
&+ 0.3\left[0.5\operatorname{CE}(z_{16},y)
+0.5T^2\operatorname{KL}(p_{16}^T\,\|\,\operatorname{sg}(p_f^T))\right],
\qquad T=3.
\end{aligned}
```

其中 `sg` 表示 stop-gradient，final head 只接收自身 CE 梯度。不要声称该损失是新蒸馏算法；它是为了
训练可用出口并尽量保护最终头的固定 recipe。

### 7.3 共享路由策略

对 exit8 logits `z_8(x)` 定义最大 softmax 置信度：

```latex
c_8(x)=\max_k \operatorname{softmax}(z_8(x))_k.
```

部署策略：

```latex
\hat y_\theta(x)=
\begin{cases}
\arg\max_k z_8(x)_k, & c_8(x)\ge\theta,\\
\arg\max_k z_f(x)_k, & c_8(x)<\theta.
\end{cases}
```

“共享”是指同一数据集协议内的多个独立训练 seed 共用一个阈值，而不是 CIFAR-10 和 CIFAR-100 共用
同一阈值。CIFAR-10 使用 `0.984`；CIFAR-100 经 P3 失败诊断后，在全新 P4 前冻结 `0.903`。

### 7.4 风险指标与选择目标

相对同一 multi-exit 模型 final head 定义：

- overall drop：`Acc_final - Acc_policy`；
- balanced drop：各类别准确率均值之差；
- worst-class drop：`max_c(Acc_final,c - Acc_policy,c)`；
- early-route fraction：进入 exit8 的样本比例；
- MAC saving：相对 final-only 路径的期望 Conv/Linear MAC 减少比例。

CIFAR-10 P1 在每个 source seed 上要求三类 drop 均不大于 0、路由率 15%--95%。在所有可行共享阈值
中，先最大化跨 seed 最小 MAC saving，再最大化平均 saving；仍并列时选更高、更保守的阈值。

该约束是有限 calibration 样本上的经验约束。正文不能借用 conformal 或 concentration-bound 的语言，
除非代码和实验真正实现了相应统计保证。

### 7.5 计算成本

最终策略的期望 MAC：

```latex
\bar C(\theta)=r_8(\theta)C_8+[1-r_8(\theta)]C_f,
\qquad
S(\theta)=1-\frac{\bar C(\theta)}{C_f}.
```

参数表必须报告包括 training-only exit16 在内的完整模型参数量，不能为了让模型看起来更小而从统计中
删除 exit16。

### 7.6 策略选择伪代码

算法框应包含：输入多个 seed 的 calibration logits、标签、固定阈值网格、路径成本和风险预算；对每个
阈值逐 seed 计算风险与成本；保留所有 seed 同时可行的阈值；按“最差 seed saving→平均 saving→更高
阈值”词典序选择；输出一个共享阈值。P4 不是再次调用该选择器，而是直接应用已冻结的阈值。

## 8. 按章节写作计划

### 8.1 Title、Abstract 与 Keywords

摘要建议 180--230 词，按六句组织：

1. 说明轻量分类器的固定深度浪费易样本计算，但激进早退可能隐藏类别级退化；
2. 指出现有平均精度导向策略不能直接证明跨训练版本和最差类别稳健；
3. 介绍多出口 MobileNetV2 与跨 seed 共享、风险约束的预算选择策略；
4. 报告 CIFAR-10 的 `87.04 ± 0.18%`、`64.85%` 早退和 `36.51%` MAC 节省；
5. 报告 CIFAR-100 的 `58.10 ± 0.03%`、`42.39%` 早退、`24.17%` MAC 节省和 2--4 pp 最差类别边界；
6. 报告 RTX4090D `24.91%` 期望时延节省，同时限定为服务器 GPU 实证。

摘要必须同时保留收益和限制，不能写“无精度损失”概括两个数据集；CIFAR-100 存在逐类下降。

推荐关键词：

```text
early exiting; dynamic inference; class-wise risk; lightweight networks;
budget-aware inference; cross-seed robustness
```

### 8.2 Introduction

建议结构：

1. 固定深度轻量网络对所有样本使用相同计算的矛盾；
2. early exit 的潜力，以及只看平均准确率会掩盖少数类别退化；
3. 单模型单校准阈值在重训模型版本和分布转移下可能失效；
4. 本文问题定义：在有限经验风险预算下，选择跨 seed 共享、面向预算的策略；
5. 用 P0/P3 失败边界说明严格风险约束不是自动满足的，但不要在 Introduction 展开所有数字；
6. 列出三项贡献，与第 4.2 节完全一致；
7. 最后一段概述全文结构。

Introduction 不要声称“现有工作完全忽略类别风险”；应写“本文进一步聚焦于最差类别经验退化、共享
阈值的跨重训迁移和显式停止边界的组合”。

### 8.3 Related Work

分成三小节：

1. `Lightweight CNNs and efficient inference`：MobileNetV2、efficient backbone 和动态计算背景；
2. `Early-exit and sample-adaptive networks`：BranchyNet、MSDNet、Shallow-Deep Networks、USDN、
   Dynamic Perceiver 等；
3. `Calibration and risk-controlled early exiting`：Fast yet Safe、PCEE、SAFE-KD、CalexNet 以及逐类
   或选择性预测相关工作。

下一段对话必须联网查询并只使用论文原文、会议/期刊页面或作者官方仓库，核对题名、作者、年份、
venue、DOI/URL 和方法主张后再写 BibTeX。`docs/early_exit_p1_plan.md` 中的链接只是检索起点，不得
未经核验直接扩写。不要依赖博客、搜索摘要或二手综述支撑关键新颖性判断。

本节最后用一段明确区别：本文不重新设计 backbone，也不提出新的置信度或蒸馏损失；重点是共享策略
选择、最差类别经验约束、跨重训/分布验证和失败后独立确认协议。

由于没有同协议复现近期方法，建议增加一个“机制/协议比较表”，列：是否跨模型 seed 共享、是否报告
worst-class、是否分离 train/calibration/test、是否有分布转移、是否有真实硬件。表中只填经原论文核验
的性质，不混用不同数据集上的准确率。若目标期刊明确要求同协议数值 SOTA，对话应停止并请用户决定
是否新增训练，而不能编造对比结果。

### 8.4 Proposed Method

建议小节：

1. Problem formulation；
2. Multi-exit MobileNetV2；
3. Detached-final knowledge-distillation objective；
4. Class-risk-constrained shared threshold selection；
5. Dynamic inference and compute accounting；
6. Reproducibility and locked evaluation protocol。

本节使用第 7 节公式和伪代码。至少引用一张方法总览 PDF 图。避免长篇重述 MobileNetV2、softmax、
cross entropy 等教科书内容，把篇幅放在共享约束和数据边界上。

### 8.5 Experiments：Datasets and Splits

分别说明 CIFAR-10、CIFAR-10.1 v6、CIFAR-100，列出样本数、类别数、每类评估样本数、预处理和
split seed。明确：

- checkpoint-selection validation 与 policy calibration/confirmation 不同；
- P2 是相同索引上的跨重训迁移；
- CIFAR-10.1 才是独立自然分布转移；
- P4 confirmation 是新 CIFAR-100 split；
- official test 从不用于阈值或 checkpoint 选择。

### 8.6 Experiments：Implementation Details

使用第 6 节训练 recipe 和软件版本。硬件按第 6.3 节固定句表述：训练平台写单张 RTX 3080 Ti，
RTX4090D 只作为独立的 batch-1 latency profile 平台。正文和附录都不展开逐批服务器迁移；同时不要
使用 `all`、`exclusively` 等会把“主要训练平台”扩大为“每个历史 run 的唯一平台”的绝对措辞。

### 8.7 Experiments：Metrics and Statistical Reporting

报告：accuracy、balanced accuracy、worst-class drop、early-route fraction、MAC saving、参数量和
服务器 GPU latency。三 seed 结果使用 mean ± sample standard deviation；同时在附录保留全部 seed
值。CIFAR-10.1 六模型可给 all-six 汇总，并在补充表中区分 source 与 target 三 seed。

只有三个 seed 时不要依赖低功效显著性检验制造结论。重点使用配对差值、胜出 seed 数、预注册 gate
和完整值。所有 accuracy 差写 `percentage points (pp)`，不能把 pp 写成相对百分比。

### 8.8 Experiments：Main Results

先给 CIFAR-10 与 CIFAR-100 主表，逐数据集比较：

- matched MobileNetV2 baseline；
- multi-exit model 的 final head；
- locked dynamic policy。

解释两层收益：辅助训练是否损害/改善 final head，以及动态策略相对同模型 final head 的收益与成本。
CIFAR-10 的核心结论是约 65% 早退且未观察到总体/均衡/最差类别退化；CIFAR-100 的核心结论是约
42% 早退、24% MAC 节省、总体提升，同时最差类别存在 2--4 pp 边界。

不要把不同 GPU 的训练速度放进主表，也不要将旧注意力模型加入本表。

### 8.9 Experiments：Cross-Model and Distribution Transfer

先写 P2 seeds 57--59 的零重校准迁移，再写 CIFAR-10.1 v6 六模型结果。强调：

- P2 检验模型版本变化；
- CIFAR-10.1 检验自然分布转移；
- 二者都使用同一个 P1b 阈值 `0.984`，候选数为 0；
- CIFAR-10.1 准确率下降是数据集难度变化，不应与原 CIFAR-10 绝对精度横向排名。

### 8.10 Experiments：Ablation and Risk Boundary

本节不伪装成传统“每个神经模块都训练一次”的消融。建议标题为
`Policy Ablation and Empirical Risk Boundary`，包含：

1. baseline final-only、multi-exit final、exit8 和 locked policy 的功能分解；
2. source-only per-model 选择与跨 seed shared policy 的协议区别；若需数值只能从 calibration logits
   重新计算并明确 post-hoc，不得在 test 上选择；
3. P3 的 0/2/4 pp worst-class 风险预算边界；
4. 无 class guard 与 predicted-class guard 诊断；
5. P4 在新 split/seed 上对 4 pp 边界的独立确认。

没有 no-KD、无 exit16 或第二 backbone 训练消融。正文不要暗示这些已经完成；将其列入 Limitations。
如果审稿目标坚持传统组件消融，应请求用户授权新的训练批次，而不是从已有数据虚构。

### 8.11 Experiments：Accuracy--Compute Trade-off

主曲线必须来自 development 数据，而不是在 official test 上扫描阈值：

- CIFAR-10：P1 calibration seeds 54--56；
- CIFAR-100：P3 source/target calibration 作为边界探索，P4 只标冻结 `0.903` confirmation 点；
- official test 只画预先锁定的单个 marker，不绘制可被误解为 test-time selection 的完整阈值曲线。

横轴推荐 `Expected Conv/Linear MACs (% of final path)`，纵轴推荐
`Accuracy difference vs. final head (pp)`；可另加右轴或第二面板显示 worst-class drop。曲线显示三 seed
均值和范围/标准差带，并用竖线或特殊 marker 标出 `0.984`、`0.903`。图注必须说明所有非锁定点仅为
描述性 development sweep。

### 8.12 Experiments：Hardware Efficiency

单独给 RTX4090D 配对剖析表和 PDF 图。报告每 seed 与均值，写清 batch1、warm-up、重复次数、同步
wall-clock、隔离 early/fallback path、用冻结路由比例加权。只比较同一设备上的 final-only 与 policy；
不要引用旧稿 4.47/6.82 ms，也不要把静态 attention 的 RTX3080Ti profile 混入当前方法。

### 8.13 Discussion

围绕以下事实讨论：

- 为什么保守 shared threshold 在 CIFAR-10 可实现经验零下降；
- 为什么 100 类、每类校准样本更少时严格 worst-class=0 与 15% 节省冲突；
- 为什么 P3 失败后必须在新 split/seeds 上做 P4，而不能直接打开 test；
- 为什么 policy 有时能比 final head 更准：早期头与最终头存在少量互补决策；
- MAC 节省与实际 latency 节省不完全一致，因为 fallback 路径有路由/分支开销；
- 共享阈值的优势是部署简单和跨模型版本可移植，代价是相对 per-model calibration 更保守。

不要把相关性写成因果机制证明。

### 8.14 Limitations

必须明确列出：

1. 只有 MobileNetV2 一个 backbone；
2. 训练数据集只有 CIFAR-10/100，CIFAR-10.1 仍是 CIFAR-10-like；
3. 风险约束是有限样本经验约束，不是统计或 conformal guarantee；
4. CIFAR-100 每类 confirmation 50 张、test 100 张，worst-class 指标粒度较粗；
5. 4 pp 边界由 P3 失败后的诊断提出，虽然随后用 P4 独立确认，仍应透明披露；
6. 没有 no-KD、无 exit16 和第二 backbone 的训练消融；
7. 只有 RTX4090D 服务器 GPU 期望时延，无手机、能耗和 batch-size 扩展测量；
8. CIFAR-100 历史旧 baseline 曾暴露 test，因此只能称 P4 方法锁定评估；
9. 三训练 seed 的不确定性估计有限。

Limitations 不需要自我否定，但不能把未做项目写成当前贡献。

### 8.15 Conclusion

用一段总结方法，用一段总结 CIFAR-10/CIFAR-100 与时延的主要数值，再用一句限定经验风险和硬件
范围。不要引入摘要和正文没有出现的新结果；不要承诺已经完成 ImageNet、移动端或统计保证。

## 9. 论文表格清单

建议在实验仓库创建统一输出目录：

```text
reports/paper/
├── evidence_manifest.json
├── tables/
│   ├── main_results.csv
│   ├── main_results.tex
│   ├── transfer_results.csv
│   ├── transfer_results.tex
│   ├── risk_boundary.csv
│   ├── risk_boundary.tex
│   ├── complexity_latency.csv
│   ├── complexity_latency.tex
│   ├── protocol_hardware.csv
│   └── protocol_hardware.tex
└── figures/
```

### Table 1：Protocol and hardware

列：`Phase | Dataset | Split seed | Training seeds | Train/selection/policy samples | Policy role | GPU`。
P0 放附录，正文重点列 P1--P4。

### Table 2：Main results

列：

```text
Dataset | Method | Accuracy (%) | Gain vs. baseline (pp) |
Gain vs. final (pp) | Early fraction (%) | MAC saving (%) |
Worst-class drop (pp) | Params (M)
```

每个数据集三行 baseline、multi-exit final、locked policy。不可把不适用列填 0；使用破折号并在表注
说明 baseline/final 没有动态路由。

### Table 3：Cross-model and distribution transfer

列：

```text
Evaluation | Seeds | Threshold source | Recalibration |
Policy accuracy | Early fraction | MAC saving |
Overall/Balanced/Worst-class drop
```

分别列 P2 unseen retraining 和 CIFAR-10.1 source/target/all-six。

### Table 4：Policy ablation and risk boundary

列：

```text
Dataset/split | Policy variant | Risk budget | Threshold |
Min/mean saving | Max worst-class drop | Evidence type | Outcome
```

`Evidence type` 必须区分 preregistered、post-hoc diagnostic、independent confirmation。P3 与 P4 不能
合并成“同一次消融”。

### Table 5：Complexity and latency

列：

```text
Model/path | Params | Conv/Linear MACs | Early route |
Expected latency (ms) | Saving | Speedup | Hardware
```

MAC 和 latency 的测量方法必须分开注释。

附录另给 seed-level 表，确保任何 mean ± std 都能追溯到原始三个或六个值。

## 10. 论文图片清单与 PDF 强制规范

所有最终论文图片必须保存为 `.pdf`。论文 LaTeX 中新的 `\includegraphics` 只能引用 PDF，不使用 PNG、
JPEG 或 SVG。若底层数据天然为 raster，先以足够分辨率绘制，再封装为 PDF；图中的文字、线条和 marker
应尽量保持矢量。

建议文件：

```text
reports/paper/figures/method_overview.pdf
reports/paper/figures/accuracy_compute_tradeoff.pdf
reports/paper/figures/cross_seed_transfer.pdf
reports/paper/figures/risk_budget_boundary.pdf
reports/paper/figures/latency_profile_rtx4090d.pdf
reports/paper/figures/per_class_risk_appendix.pdf   # 可选
```

### Figure 1：Method overview

矢量流程图，表现 backbone、exit8、training-only exit16、final、confidence gate、shared calibration
和 deployment route。图中同时画数据边界：train→model selection→policy calibration→locked test。

### Figure 2：Accuracy--compute trade-off

双面板 CIFAR-10/CIFAR-100。横轴为 final path 百分比，纵轴为相对 final 的 accuracy difference；显示
seed 汇总带、风险可行区和锁定点。所有 development sweep 与 locked test marker 使用不同图例。

### Figure 3：Cross-seed and distribution transfer

可用点图或 forest-style 图显示 source、unseen retraining、CIFAR-10.1 的 early fraction、MAC saving 和
accuracy drop。不要用三维图或气泡大小同时编码多个不易比较的变量。

### Figure 4：Risk-budget boundary

展示 CIFAR-100 在 0/2/4 pp worst-class budget 下的最大可行 saving，并区分 P3 post-hoc 与 P4 independent
confirmation。不要让图形暗示 P3 test 被访问。

### Figure 5：RTX4090D latency

配对显示 final-only 与 expected policy latency，三 seed 点加均值；图注声明 expected policy 是隔离路径
测量按冻结 route fraction 加权。

### PDF 输出规范

- Matplotlib 使用 `savefig(..., format="pdf", bbox_inches="tight")`；
- 字体与论文一致，最小字号在最终版面上不低于 7--8 pt；
- 使用色盲友好配色，同时保证黑白打印可由线型/marker 区分；
- 不在图内写长标题，标题放 caption；
- 误差带和误差棒明确是 sample std、range 还是其他统计量；
- 百分数与 pp 不混用；
- 图例不遮挡数据；
- PDF 中嵌入字体，避免 Type 3 字体；
- 每张图生成后用 `pdfinfo`、`pdffonts` 检查，并渲染预览核对裁切和可读性。

现有 `reports/figures/accuracy_vs_*`、旧 confusion matrix 和旧论文 `picture/` 下静态注意力图片都不是
当前论文图，不能通过改标题继续使用。

## 11. 表图生成实现要求

现有 `scripts/analysis/generate_tables.py` 和 `scripts/analysis/plot_performance.py` 读取旧静态注意力 CSV，
不得直接用于当前论文。建议新增：

```text
scripts/analysis/build_early_exit_paper_tables.py
scripts/visualization/plot_early_exit_paper.py
tests/unit/test_early_exit_paper_tables.py
tests/unit/test_early_exit_paper_plots.py
```

要求：

1. 只从第 5 节机器可读 JSON 和经过哈希核验的原始数组读取；
2. 输出 `evidence_manifest.json`，记录每个输入文件路径、SHA-256、使用字段和输出文件；
3. CSV 保存未格式化的原始数值，LaTeX/图层才做展示舍入；
4. 同一数值只在一个构建函数中定义，摘要、表格和正文数字从同一生成结果复制；
5. 不调用任何 official/external test evaluator，不构造 test loader；
6. 需要曲线时优先重新计算 calibration/confirmation logits；官方 test NPZ 只读取已保存数组；
7. 不在脚本中复制模型架构，必要推理通过 `src/image_classification/`；
8. 新图只输出 PDF；
9. 表格和图片写入 `reports/paper/` 并提交 GitHub，checkpoint/logits 继续保持 ignored；
10. 为聚合、样本标准差、pp 转换、路径成本和禁止 test sweep 写单元测试。

如果 calibration logits 尚未持久化，可以从保留的 `model_best.pth`、`split_indices.json` 和公共 CIFAR
训练集做一次确定性 FP32 推理并保存到新的 ignored 分析目录。这不是新增训练，但必须记录 checkpoint、
split、代码哈希和命令。不得重新运行 P1/P4 official-test evaluator。

## 12. 旧 LaTeX 稿的清理清单

旧 `sn-article.tex` 已知包含以下问题，结构性重写时全部处理：

- 删除摘要末尾 `Brief Explanation of Optimizations: ...` 编辑残留；
- 删除 PSHA-Net、Primary-Secondary Hybrid Attention、SE-shallow/CBAM-deep 的核心方法叙事；
- 删除旧的 81.81/83.83%、91/93M FLOPs 和 4.47/6.82ms 表述；
- 删除重复的 Integration Architecture 段落与重复 `eq:residual_attention` label；
- 删除 `\nocite{*}`，只列正文实际引用；
- 删除“所有类别都提升”“t-SNE 直接证明泛化”“<30ms 即移动端实时”等过度推断；
- 删除与当前方法无关的 Top-5 主结论；
- 删除或替换全部旧静态注意力图片；
- 修正现有 overfull table、bookmark level 和引用告警；
- 更新 Limitations，不得再写“只测试 CIFAR-10”；
- 将代码 URL 保留为 GitHub 仓库；Zenodo DOI 只有在实际版本与当前代码/结果对应并能访问时才能保留。

不要在旧段落中批量替换方法名后继续使用；每段都必须重新验证其逻辑和证据。

## 13. 推荐执行顺序

### Phase A：冻结输入与建立资产目录

1. 检查代码仓库和论文目录状态；
2. 验证第 5 节输入文件及 source index；
3. 创建 `reports/paper/`；
4. 建立 `evidence_manifest.json`；
5. 备份旧 `sn-article.tex`。

完成标准：所有主张都有唯一机器可读来源，未修改任何冻结输入。

### Phase B：先生成表格数据

1. 生成 protocol/hardware 表；
2. 生成 main results 表；
3. 生成 transfer 表；
4. 生成 risk-boundary 表；
5. 生成 complexity/latency 表；
6. 对照源 JSON 手工抽查每个均值、标准差和 pp 转换。

完成标准：CSV、LaTeX 和 evidence manifest 一致，不含旧注意力数字。

### Phase C：生成全部 PDF 图片

1. 先实现方法总览；
2. 复算 development accuracy--compute 曲线；
3. 画迁移、风险边界和 latency；
4. 使用 `pdfinfo`、`pdffonts` 和渲染预览检查；
5. 将最终 PDF 复制或链接到论文项目的统一图片子目录。

完成标准：论文引用的新图全部是 PDF，无裁切、字体和标签错误。

### Phase D：按证据顺序重写正文

推荐先写 Method 和 Experiments，再写 Discussion/Limitations，最后写 Introduction、Related Work、
Conclusion 和 Abstract。这样摘要和贡献不会超出已经落表的证据。

### Phase E：参考文献核验

1. 删除未引用和无关 BibTeX；
2. 对所有 recent work 使用原始论文页面核验；
3. 逐项核对 author/title/venue/year/pages/DOI；
4. 确认每个核心论断紧邻适当引用；
5. 不用 `\nocite{*}` 填充数量。

### Phase F：编译与逐页验收

根据模板实际 bibliography 后端选择 `latexmk` 或 pdflatex→bibtex→pdflatex×2。至少检查：

```bash
rg -n 'XX|TODO|TBD|Brief Explanation|nocite\{\*\}|PSHA|CSGHA|Primary-Secondary' sn-article.tex
rg -n 'undefined|multiply defined|Overfull|Citation.*undefined|Reference.*undefined' sn-article.log
```

随后渲染 PDF 全部页面逐页检查：标题/作者、公式、表格宽度、图中文字、caption、分页、参考文献和附录。
不能只依据 LaTeX 返回码判断完成。

### Phase G：最终一致性审计

建立一张 claim-to-evidence 清单，逐项核对：

- 摘要数字 = 主表数字 = Results 数字 = Conclusion 数字；
- dataset/split/seed/threshold 在方法、实验和 caption 中一致；
- CIFAR-10 `0.984` 与 CIFAR-100 `0.903` 不混淆；
- MAC 与 latency 不混淆；
- RTX3080Ti 写作主要训练平台，RTX4090D 只写独立 latency profile 平台；
- validation、calibration、confirmation、external test 和 official test 不混淆；
- P3 post-hoc 与 P4 independent confirmation 不混淆；
- 所有图为 PDF；
- 所有引用存在且实际使用；
- 没有密码、私钥、checkpoint、logits 或数据集进入 Git。

## 14. 完整论文验收标准

只有以下条件全部满足，才能宣布整篇论文完成：

- [ ] 标题和全文已切换为风险约束共享早退主线；
- [ ] 旧 PSHA/CSGHA 主方法、旧表格和旧图片已清除；
- [ ] 主表覆盖 CIFAR-10 和 CIFAR-100，并报告 mean ± sample std；
- [ ] 单独报告 P2 与 CIFAR-10.1 的迁移结果；
- [ ] P3 失败与 P4 独立确认被如实区分；
- [ ] 训练硬件统一表述为主要使用 RTX3080Ti，RTX4090D 仅用于独立 latency profile；
- [ ] 参数、MAC、早退率、类别风险和 RTX4090D latency 定义准确；
- [ ] 传统训练级消融缺口已在 Limitations 披露；
- [ ] Related Work 已用原始来源更新，且不声称 SOTA；
- [ ] 所有论文图片均为 PDF，并完成字体/裁切检查；
- [ ] 摘要、表格、正文、结论中的数值完全一致；
- [ ] 没有在 official/external test 上重新选择策略；
- [ ] LaTeX 编译无 undefined citation/reference、重复 label 和明显 overfull；
- [ ] 最终 PDF 已逐页视觉检查；
- [ ] 生成脚本、最终 CSV/LaTeX 表和 PDF 图已提交 GitHub；
- [ ] checkpoint、数据、logits、凭据仍未提交；
- [ ] 论文源文件有可恢复版本历史或备份。

## 15. 遇到以下情况必须停止并请求用户决定

下一段对话只在以下情形暂停：

1. 机器可读 JSON、source index 与正文拟用数值冲突；
2. 保留 checkpoint/split 无法复算 development 曲线；
3. 目标期刊要求同协议近期方法定量比较，必须新增训练；
4. 用户希望把 no-KD、无 exit16 或第二 backbone 写成已验证贡献；
5. 论文目录有不明来源的用户修改且无法安全合并；
6. 编译工具或关键依赖缺失，需要安装；
7. 需要重新访问任何已锁定 official/external test evaluator；此项原则上不得授权作为普通写作步骤。

除此之外，直接推进表格、PDF 图片、正文重写、编译和质量检查，不再启动长训练。

## 16. 可直接复制给下一段对话的任务提示

```text
请严格读取当前项目 AGENTS.md、docs/paper_writing_guide.md、
docs/paper_evidence_bundle.md 和 docs/handoff.md 第28节。当前实验已经停止新增训练，
不要重跑 CIFAR-10、CIFAR-10.1 或 CIFAR-100 official/external test，也不要修改冻结阈值。

先检查服务器 /root/autodl-tmp/image-classification 的 Git 状态和正式证据完整性，然后按
paper_writing_guide 的 Phase A--G 顺序推进：建立机器可读 evidence manifest，生成当前早退论文的
主表、迁移表、风险边界消融表、复杂度/时延表和全部 PDF 图片，再对实际 sn-article.tex 做结构性
重写。旧 PSHA-Net/CSGHA 只能作为负结果背景，不得继续作为主方法。所有论文图片必须为 PDF。

正文使用英文，结果只从冻结 JSON/审计读取；如需精度—计算量曲线，只在 calibration/confirmation
数据上做描述性阈值扫描，official test 只标锁定点。准确区分 P0/P3 后验诊断、P4 独立确认、MAC
代理和 RTX4090D 时延。训练硬件统一写为主要使用单张 RTX3080Ti；不要在论文中展开临时服务器
迁移，RTX4090D 只作为独立的 batch-1 latency profile 平台。

完成 LaTeX 后重新编译并逐页检查最终 PDF，清除旧数字、编辑残留、nocite{*}、重复 label、引用错误、
表格溢出和非 PDF 图片。生成脚本、最终表格和 PDF 图提交到 image-classification GitHub；原始
checkpoint、数据集、logits、日志和凭据不得提交。持续推进直到整篇论文和最终 PDF 通过验收，只有
paper_writing_guide 第15节列出的实质阻塞才向我询问。
```
