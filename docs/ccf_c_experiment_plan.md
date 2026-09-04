# CCF-C 期刊送审导向的补实验计划

> 版本：2026-09-04 规划稿
> 适用项目：`classification-code` 中的风险约束共享早退（RCSEE）研究
> 目标：降低编辑初筛（desk reject）风险并形成可审计、可复现的期刊级证据链；本文不承诺录用或一定送外审。

## 1. 执行结论

当前工作的优势不是“又做了一个 early-exit head”，而是已经形成了较严格的模型选择、策略选择、确认、
official test 和失败停止边界。现阶段最明显的短板也不是论文页数，而是科学覆盖仍偏窄：没有同协议强基线，
没有 `no-KD` / `no-exit16` 训练级消融，只有 MobileNetV2 和 CIFAR 尺度数据，且现有时延结果是隔离路径的
路由加权期望值，不是实际动态数据流的端到端时延。

因此，下一轮实验按以下顺序进行：

1. **先做零训练的证据重算和公平策略对比**，判断 proposed selector 是否真的优于简单 shared-MSP；
2. **再做最小完整训练消融**，判断 KD 和 training-only exit16 是否各有稳定作用；
3. 通过上述 go/no-go 后，先补低训练成本的真实动态执行，再投入第二 backbone 和非 CIFAR 数据；
4. 最后才补 corruption、更多 seed、能耗等增强项。

面向“显著提高进入外审概率”的建议投稿前核心包是：

| 工作包 | 新训练 run | 是否投稿前核心 | 主要解决的审稿质疑 |
|---|---:|---|---|
| P5-A 现有证据深挖 | 0 | 是 | 证据是否充分、风险是否可解释 |
| P5-B 同 checkpoint 策略基线 | 0 | 是 | 方法是否只是普通置信度阈值 |
| P5-C 训练组件 2×2 消融 | 18 | 是 | KD、exit16 是否必要 |
| P8 实际动态硬件评测 | 0（需设备） | 是 | MAC 是否转化为真实时延收益 |
| P6 第二 backbone 完整迁移协议 | 24 | 是 | 是否只对 MobileNetV2 有效 |
| P7 非 CIFAR 规模数据 | 12 | 是 | 是否只在 toy-scale 数据成立 |
| P9 shift、更多 seed、第三 backbone | 视资源而定 | 强烈建议 | 稳定性和外部有效性 |
| **核心新增训练总计** | **54** |  | 不含失败重跑和超参探索 |

这里的 54 个 run 是“完整证据包”的规划量，不是立即全部启动的命令。P5-A/P5-B/P5-C 设有明确停止门，
若方法被简单基线支配，或 KD/exit16 没有稳定作用，应先简化方法与论文主张，而不是继续堆实验。

## 2. 当前证据基线与不可改写边界

### 2.1 本机已核查的资产

截至 2026-09-04，本机 `classification-code/artifacts` 约 1.7 GiB，包含 36 个正式训练 run，组成
18 组 baseline–multi-exit 配对：

| 历史阶段 | 数据/seed | 训练 run 数 | 可支持的结论 |
|---|---|---:|---|
| P0 | CIFAR-10，51–53 | 6 | 探索失败与方案转向，只能作形成性证据 |
| P1 | CIFAR-10，54–56 | 6 | 共享阈值 0.984 的校准与锁定 test |
| P2 | CIFAR-10，57–59 | 6 | 阈值不重校准的跨模型版本迁移 |
| P3 | CIFAR-100，60–65 | 12 | 严格风险/节省门不可行及 post-hoc 边界 |
| P4 | CIFAR-100，66–68 | 6 | 新 split 的独立确认和阈值 0.903 锁定 test |

现有资产还包括 36 份训练日志、划分索引和 best checkpoint，以及 12 份 official/external prediction NPZ。
它们足以支持训练收敛、固定锁定点的逐类行为、决策互补性、置信度/校准和路由分析。

### 2.2 已冻结的事实

以下事实只能引用，不得通过新 sweep 改写：

- CIFAR-10 P1 的 official-test 阈值固定为 `0.984`；
- CIFAR-10 P2 和 CIFAR-10.1 继续使用 `0.984`，候选阈值数为 0；
- CIFAR-100 P3 固定为 `stop_without_test`，后续风险边界只能标成 post-hoc diagnostic；
- CIFAR-100 P4 的阈值固定为 `0.903`，confirmation/test 不得重新选点；
- 已有 official/external NPZ 只允许在固定锁定点做诊断，不允许画 test-set threshold sweep；
- “经验上未观察到下降”不能改写为理论、统计或 distribution-free guarantee；
- CIFAR-100 每类有限样本造成 2 pp 级离散粒度，0/4 pp 不能被写成连续精确风险界。

更完整的冻结账本见 [paper_writing_guide.md](paper_writing_guide.md) 和
[paper_evidence_bundle.md](paper_evidence_bundle.md)。

### 2.3 可重算但不是新训练的内容

development calibration/confirmation logits 尚未统一持久化，但可以从保留的 `model_best.pth`、
`split_indices.json` 和冻结代码快照以确定性 FP32 推理重算。重算必须保存：

- checkpoint、split、代码提交、数据版本和输出文件 SHA-256；
- 样本 ID、标签和固定顺序；
- 重算结果与既有锁定 JSON 的逐项一致性检查；
- 独立脚本和单元测试，尤其检查百分点、sample standard deviation、paired counts 和禁止 test sweep。

这不授权重新运行旧 official/external evaluator，也不能删除原 access marker。

## 3. 研究问题与证据映射

补实验必须围绕明确问题组织，避免“有图就画”。建议冻结以下 RQ：

| RQ | 问题 | 最小证据 |
|---|---|---|
| RQ1 | 完整共享策略是否优于简单置信度/熵/逐模型阈值？ | 同 logits、同成本、matched-risk 与 matched-compute 对比 |
| RQ2 | KD 与 training-only exit16 是否分别有可复现作用？ | 两数据集、3 seed 的训练级 2×2 消融 |
| RQ3 | 共享策略能否跨训练 seed、backbone 和数据尺度迁移？ | source→target、第二 backbone、非 CIFAR 数据 |
| RQ4 | calibration 数量和样本扰动会怎样改变阈值、可行率与风险违例？ | 30–100 次分层重采样、校准规模敏感性 |
| RQ5 | MAC 节省能否转化为真实端到端时延/吞吐/内存收益？ | 实际动态执行、多 batch、多轮、至少两类硬件更佳 |
| RQ6 | 哪些类别和样本会被过早退出、被挽救或被伤害？ | per-class route/risk、premature/delayed、rescue/harm |
| RQ7 | 自然/合成分布移位时，固定策略是否风险升高或计算减速？ | CIFAR-10.1 + corruption severity 曲线 |

论文中的每个主张必须能反向指向一个 RQ、一项预注册实验和一份机器可读产物。

## 4. 投稿前必须完成的实验

### 4.1 P5-A：冻结证据深挖（0 次新训练）

#### 目的

把现有 36 个 run 从“几个最终均值”扩展成可解释、可统计、可审计的证据，同时不触碰锁定 test 边界。

#### 数据矩阵

- P1 CIFAR-10 seeds 54–56：development calibration + 固定 official-test 点；
- P2 seeds 57–59：固定阈值的未见模型版本迁移；
- CIFAR-10.1 seeds 54–59：固定阈值的自然分布转移；
- P3 CIFAR-100 source 60–62 / target 63–65：development-only 失败边界；
- P4 CIFAR-100 seeds 66–68：独立 confirmation + 固定 official-test 点。

#### 必算指标

1. baseline、final、exit8、exit16 和 policy 的 overall accuracy、balanced accuracy、macro-F1；
2. 每类 final/policy accuracy、accuracy delta、early fraction、样本数和经验风险；
3. route-conditioned accuracy，以及 exit8/final 一致、premature、rescue、共同错误四象限；
4. MSP、entropy、top-1 margin 的分布和错误排序能力；
5. ECE、NLL、Brier score；可靠性图仅作诊断，不能代替停止决策质量；
6. development 上的 threshold–accuracy–worst-class-risk–MAC 完整 frontier；
7. 1/2/3 个 source seed 子集的阈值敏感性；
8. calibration 样本规模 `500/1000/2500/5000` 的分层重采样，建议 50 次，资源允许做 100 次；
9. 锁定点按样本 paired bootstrap 95% CI，同时保留训练 seed 为主要复现实验单位。

#### 建议新增诊断指标

- `premature exit rate`：exit8 错而 final 对，且策略确实提前退出；
- `rescue rate`：exit8 对而 final 错，且策略提前退出；
- `delayed/missed exit rate`：exit8 对但策略拒绝退出，可再区分 final 对/错；
- `exit disagreement rate`：exit8 与 final 预测不同；
- `failure-ranking AUROC`：用停止分数区分 exit8 正确/错误的能力。

这些定义必须在代码和论文中完全一致，并公开分母，避免把不同研究的同名指标直接混用。

#### 验收与产物

- 重算的锁定点必须精确复现已有 JSON；不一致则停止并先查明原因；
- 输出 `development_logits_manifest.json`；
- 输出 `per_class_route_risk.csv/.tex`、`decision_complementarity.csv/.tex`、
  `calibration_metrics.csv/.tex`；
- 至少形成 6 幅有独立问题的图：训练收敛、逐类 route/risk、rescue–harm、可靠性/置信度、
  development frontier、seed/calibration-size sensitivity。

### 4.2 P5-B：同 checkpoint 的策略与近期方法公平对比（0 次新训练）

这是当前最关键的缺口。所有方法必须共享相同 exit8/final logits、样本、路径 MAC 和数据边界。

#### 第一层：必须实现的简单基线

| 方法 | 作用 |
|---|---|
| final-only | 精度/成本参考 |
| always-exit8 | 计算上界与精度下界 |
| random routing（matched coverage） | 判断收益是否来自置信度排序 |
| per-model MSP threshold | 不共享策略的性能上界及部署复杂度对照 |
| shared MSP，仅 overall constraint | 隔离 balanced/worst-class 约束贡献 |
| shared MSP，overall + balanced | 隔离 worst-class 约束贡献 |
| 完整 shared MSP，overall + balanced + worst-class | proposed |
| shared entropy | 检验停止统计量是否特殊 |
| shared logit margin | 第二种无额外训练的置信度基线 |
| temperature-scaled MSP | 检验简单校准能否替代完整方案 |

oracle 可以作为不可部署上界放在附录，但必须显著标注“uses unavailable labels / not deployable”。

#### 第二层：近期方法基线

优先核验并复现至少两个与当前 logits 接口兼容的方法：

- *Fast yet Safe: Early-Exiting with Risk Control*（NeurIPS 2024）；
- PCEE（2024 workshop/preprint，发表级别必须准确标注）。

若原论文或官方代码的假设与当前两出口/多 seed 场景不兼容，应写清排除理由；若只实现了受其启发的近似，
必须使用描述性名称（如 `UCB-style empirical risk baseline`），不得沿用原方法名制造“已复现”的印象。

#### 公平比较方式

- 同时给出 **matched risk** 与 **matched compute** 两种读表方式；
- 预先固定 10%、20%、30% MAC saving 行，不可达写 `infeasible`，不能删行；
- 另报告各方法自己的锁定点；
- source 只用于选择；target/shift 的候选阈值数必须为 0；
- 指标至少包括 policy accuracy、balanced accuracy、worst-class drop、min/mean saving、
  risk-violation seed 数、阈值个数、校准开销和 ECE/NLL/Brier；
- 所有复现方法记录论文版本、官方仓库、commit SHA 和适配说明。

#### Go/No-Go 1

完成 P5-A/B 后立即判断：

- 若完整策略在 matched-risk 与 matched-compute 两种视角下均被简单 shared-MSP 或 entropy 全面 Pareto 支配，
  先重构方法/题目，不进入大规模训练；
- 若 per-model threshold 更优但 shared threshold 显著降低部署和校准复杂度，则把贡献收缩为可量化的运维权衡，
  不宣称绝对性能领先；
- 若 only-overall 与完整约束表现近似，必须通过逐类风险证据说明 worst-class constraint 的必要性，否则简化方法。

#### 产物

- 1 张核心策略对比表；
- 1 张 accuracy/risk–compute Pareto 图；
- 1 张跨 seed 风险违例与 feasibility 图；
- 1 张校准开销/阈值稳定性表。

### 4.3 P5-C：训练组件 2×2 消融（18 次新训练）

在 CIFAR-10 seeds 54/55/56（split 20260902）和 CIFAR-100 seeds 66/67/68（split 20260904）上复用
现有 matched final-only 和 full runs，补齐三个缺失变体：

| 代号 | 可部署 exit8 | training-only exit16 | detached-final KD | 状态 |
|---|---|---|---|---|
| A0 | 否 | 否 | 否 | 现有 final-only |
| A1 | 是 | 否 | 否 | 新跑，exit8-only CE |
| A2 | 是 | 否 | 是 | 新跑，检验“去掉 exit16” |
| A3 | 是 | 是 | 否 | 新跑，检验“去掉 KD” |
| A4 | 是 | 是 | 是 | 现有 full model |

两数据集 × 3 seeds × 3 个新变体 = **18 个新 run**。

#### 控制变量

- baseline/multi-exit 使用相同初始化、split、增强、优化器、scheduler 和最终 head；
- best checkpoint 仍只由 final-head model-selection accuracy 决定；
- CE-only 不能简单把 `alpha=0` 后保留名义总权重造成 CE 权重变化：
  - A1 的 exit8 CE 权重固定为 full recipe 中的 `0.1`；
  - A3 的 exit8/exit16 CE 权重固定为 `0.1/0.15`；
  - A2/A4 保持 `alpha=0.5, T=3` 和既有名义权重；
- 新 run 增加 per-head 每 epoch 的 loss/accuracy 记录；现有日志缺少这些字段，不能从总体曲线反推。

#### 结果字段

- final/exit8/exit16 accuracy；
- 训练收敛速度、best epoch 和 generalization gap；
- 同一冻结选择器下的 policy accuracy、risk、early rate、MAC；
- 参数、静态额外 MAC、训练时间；
- 每 seed 原始值、paired effect、mean ± sample SD 和 95% CI。

#### 停止规则与 Go/No-Go 2

- 不得删除“坏 seed”或重开 seed；
- 若某变体 final gain 的均值 `< -0.30 pp`，或任一 seed `< -0.75 pp`，标记失败并停止其后续 benchmark evaluation；
- 动态策略不满足预注册 risk + 至少 15% saving 时写 `infeasible`；
- 若 exit16 或 KD 在两个数据集上都没有稳定作用，应删除“必要组件”叙事，并优先采用更简单模型；
- CIFAR-10/100 official test 已有历史暴露。新变体若需评估，必须先冻结全部变体，并通过一个新的、一次性的
  bundled evaluator；结果只能称为 **method-locked evaluation on a historically exposed benchmark**，不能称为
  独立盲测。不得重跑 P1/P4 旧 evaluator。是否创建此入口需要用户明确授权。

### 4.4 P6：第二 backbone 的完整 source→target 协议（24 次新训练）

默认推荐 **CIFAR-stem ResNet-18**，因为它与 MobileNetV2 的结构差异比 MobileNetV3-Small 更大，更能回答
“方法是否依赖倒残差结构”。如果目标期刊强调端侧部署，可将 MobileNetV3-Small 作为替代或第三 backbone。

#### 实验矩阵

| 数据集 | split seed | source seeds | target seeds | 每 seed 模型 |
|---|---:|---|---|---|
| CIFAR-10 | 20261001 | 71–73 | 74–76 | matched final-only + full multi-exit |
| CIFAR-100 | 20261002 | 71–73 | 74–76 | matched final-only + full multi-exit |

2 数据集 × 6 seeds × 2 模型 = **24 个 run**。

#### 预注册要点

- stem 固定为 `3×3, stride=1, no max-pool`；
- 出口不能凭层号任意挑选：先在不看标签的条件下，按累计 Conv/Linear MAC 固定
  “部署 exit ≈ 40–45% final MAC、training-only aux ≈ 80–85%”；
- 将层映射、模型图、参数量和 MAC 哈希写入 manifest 后再训练；
- 阈值只需在同 backbone + dataset 内跨 seed 共享，不要求复用 MobileNetV2 的 0.984/0.903；
- CIFAR-10 预注册 overall/balanced/worst drop 预算为 0 pp；
- CIFAR-100 可沿 P4 设定 worst-class 4 pp，overall/balanced 预算与 P4 完全一致；
- source 选择、target 候选数 0；target gate 失败即 `stop_without_test`，不得改出口或 seed 掩盖失败。

### 4.5 P7：至少一个非 CIFAR 规模数据集（12 次新训练）

实用默认是 **Tiny ImageNet-200**；若用户已有合法 ImageNet-1K 数据与足够算力，应优先采用 ImageNet-1K，
并另行冻结训练预算。ImageNet-100 是人工子集，若使用必须固定类别清单和生成脚本，不能与完整 ImageNet
结果混称。

Tiny ImageNet-200 的建议协议：

- 100k train 按类分层为 80k train / 10k model-selection / 10k policy，每类 policy 约 50；
- 官方带标签 validation 10k 只作一次 locked test；
- 输入 64×64，MobileNetV2 baseline + full multi-exit；
- source seeds 77–79，target seeds 80–82，共 `6 seeds × 2 models = 12 runs`；
- 出口位置按 64×64 下的实际累计 MAC 重新冻结；
- 在看数据前固定风险预算，例如 overall/balanced ≤ 0.5 pp、worst-class ≤ 4 pp、saving ≥ 15%；
- source/target gate 未通过时不打开 locked test，失败本身作为规模边界报告。

仅增加 CIFAR-10-C 而没有更大类别数/分辨率的训练数据，仍不足以消除 toy-scale 质疑。

### 4.6 P8：真实动态数据流硬件评测（0 次训练，需设备）

现有 RTX 4090D 结果是在 synthetic input 上分别测 early/fallback 路径，再按路由率加权。它是有价值的
机制证据，但不能替代实际 `forward_with_policy` 数据流。

#### 最低矩阵

- 设备 1：现有 RTX 4090D；
- 设备 2（强烈建议）：Jetson Orin Nano/NX，或同一机器 CPU；
- 数据：CIFAR-10 `theta=0.984`、CIFAR-100 `theta=0.903` 的 calibration/confirmation 样本；
- batch size：1/4/8/16/32；新增 ResNet-18 后至少测 batch 1/8；
- 对比：final-only、实际 dynamic forward、隔离路径加权 expected latency；
- 每组合 100 个 warm-up batches、至少 1000 个 timed batches、5 个随机顺序的 paired rounds；
- CUDA 必须同步，固定软件栈并记录温度/频率策略；
- device-resident latency 为主结果，含 H2D 结果放附录。

#### 输出

- mean、median、p95 latency，throughput，peak memory；
- 支持功耗读取时增加平均功率与 energy/sample；
- 真实路由率、MAC saving、expected latency saving、actual latency saving 分列；
- batch scaling 图、MAC–latency 相关图和原始每轮 CSV。

若动态执行比 final-only 更慢，必须如实报告并解释 kernel launch、分支和小 batch overhead，不能只展示 MAC。

## 5. 强烈建议的增强实验

### 5.1 P9-A：CIFAR-10-C / CIFAR-100-C 固定策略 shift 测试

- 不训练、不重校准，固定既有阈值；
- 完整版覆盖 15 种 corruption × severity 1–5；若先做 feasibility pilot，至少预注册 4 类 corruption × 3 档
  severity，pilot 后不得只挑有利项；
- 报 baseline/final/policy accuracy、worst-class drop、early fraction、MAC 和风险违例率；
- 画 severity–risk–compute 曲线与 corruption/route heatmap；
- 先保存数据来源、版本、文件哈希和一次性访问标记。

它回答固定阈值在 shift 下是否“风险升高或计算减速”，但不能替代 P7 的较大训练数据。

### 5.2 P9-B：增加冻结策略复制 seed（16 次新训练）

- MobileNetV2 CIFAR-10、CIFAR-100 各新增 4 个 seed；
- 每 seed baseline + full，共 `2 × 4 × 2 = 16 runs`；
- 原阈值 0.984/0.903 候选数为 0；
- 目标是把主要复现实验单位从 3 提高到 7，而不是重新选阈值；
- 任一失败都保留，并报告 seed-level failure rate；
- 最终可使用 seed 为外层、样本为内层的 hierarchical bootstrap。

### 5.3 P9-C：统计风险增强

最低限度：

- per-class Wilson 或 exact interval；
- paired bootstrap CI；
- class × seed 多重比较敏感性；
- calibration size 的重复重采样。

如果要升级为形式化风险控制方法，必须新设从未参与阈值选择的 certification split，并实现对
class × seed × candidate 同时有效的上界；这会改变方法和数据协议，需要单独立项。不得在现有 P1/P4
结果上事后加 `certified`、`guaranteed` 或 `distribution-free` 字样。

### 5.4 P9-D：资源允许时的补充项

- 第三 backbone：MobileNetV3-Small；
- exit 位置按累计 MAC 约 25/45/65% 的 sensitivity；
- head capacity、KD alpha/T 的小规模预注册敏感性；
- static smaller model 的 equal-compute 对照；
- CIFAR-LT 的少数类风险；
- edge power/energy；
- 量化与 early exit 的正交性（讨论优先，实验须独立立项）。

这些项目不能替代策略强基线、训练消融、第二 backbone、较大数据和真实动态时延。

## 6. 统一实验纪律

### 6.1 新阶段命名与隔离

- 历史 P0–P4 永久冻结；新增分析/训练从 P5 开始；
- 每批执行前提交 protocol manifest，包含 RQ、数据哈希、split、模型变体、seed、阈值候选、
  risk/saving gate、停止规则和唯一 test 入口；
- 训练默认 `evaluate_test=false`；
- baseline/multi-exit 必须 matched initialization、recipe 和 split；
- source 负责选择，target/shift 的候选数为 0；
- 失败写 `infeasible` / `stop_without_test`，不得换 seed、删 run 或在 test 上调参。

### 6.2 每个 run 的最低产物

```text
<experiment_id>/
├── config.yaml
├── resolved_config.yaml
├── manifest.json
├── provenance.json
├── split_indices.json
├── logs/training.csv
├── checkpoints/model_best.pth
├── development_logits_and_routes.npz
├── metrics.json
├── benchmark.json
└── audit.json
```

其中 `provenance.json` 至少记录 git commit、dirty status、seed、Python/CUDA/cuDNN/PyTorch、设备、数据和
checkpoint 哈希。硬件实验另保存逐轮原始 CSV，不只保存汇总均值。

### 6.3 统一指标与统计

主结果统一报告：

- accuracy、balanced accuracy、macro-F1、per-class accuracy；
- 相对 matched baseline 和 final head 的 paired delta；
- early fraction、route-conditioned accuracy、MAC、params、head overhead；
- worst-class empirical drop、risk violation count；
- ECE、NLL、Brier 和 failure-ranking AUROC；
- latency mean/median/p95、throughput、peak memory；有设备才报能耗。

统计原则：

- 原始 seed 值永远可见，汇总为 mean ± **sample** SD；
- paired difference 优先于两组独立均值；
- 样本 bootstrap 可刻画固定模型的不确定性，但不能替代训练 seed 不确定性；
- n=3 时不做“显著性”宣传；增加到 n≥5 后，也应先报效应量和 CI，再谨慎使用检验；
- 所有百分点明确写 `pp`，不能与相对百分比混用。

## 7. 阶段门与投稿判据

### Gate A：零训练可行性

进入新增训练前必须满足：

- 重算结果复现冻结 JSON；
- proposed selector 没有被简单 shared confidence baseline 全面支配；
- 在至少一个风险预算区间内存在清楚的性能—风险—计算权衡；
- P3 strict failure 能被一致解释而不篡改成正结果。

### Gate B：组件必要性

进入第二 backbone/大数据前必须满足：

- no-KD、no-exit16 和 full 的差异可复现；或论文主动简化掉无效组件；
- final-head 增益、early-head 质量和 selector 贡献被分开归因；
- 所有 seed、失败和停止记录完整。

### Gate C：外部有效性

投稿前建议至少满足：

- 一个结构明显不同的第二 backbone；
- 一个非 CIFAR 规模的数据集；
- source→target 不重校准；
- 至少一个 shift 集；
- 真实动态 batch-1 latency 有正收益，或如实将贡献限定为 MAC/期望时延。

### Gate D：送稿最低线

以下任一缺失都应推迟投稿或收缩主张：

- 无同协议强策略基线；
- 无训练组件消融；
- 仍只有单 backbone + CIFAR-10/100；
- 只有 route-weighted expected latency，却宣称实际部署加速；
- 表图数字无法追溯到机器可读证据；
- 把 empirical constraint 写成 theoretical guarantee；
- 隐藏 P3 的不可行边界。

## 8. 工作量与执行顺序

| 顺序 | 工作 | 训练量 | 完成判据 |
|---:|---|---:|---|
| 1 | 证据完整性/哈希复核，重算 development logits | 0 | 锁定点逐项复现 |
| 2 | P5-A 全部诊断 | 0 | 指标 CSV + 6 图 |
| 3 | P5-B 策略基线 | 0 | matched-risk/compute 主表 + Go/No-Go 1 |
| 4 | P5-C 训练消融 | 18 | factorial 表 + Go/No-Go 2 |
| 5 | P8 真实硬件 | 0 | raw latency + p95/throughput/memory |
| 6 | P6 ResNet-18 source→target | 24 | target 候选数 0，完整审计 |
| 7 | P7 Tiny ImageNet-200 | 12 | scale boundary 明确 |
| 8 | P9 shift/更多 seed/第三 backbone | 可选 | 根据剩余质疑补齐 |

正式估算 GPU-hours 前应先在目标设备上跑每种配置的 5-epoch pilot 并外推；pilot 只用于资源估算，不得据其
test 表现选择方法。预留约 20% 失败/调度缓冲，但失败 run 不能因“不好看”而丢弃。

## 9. 需要用户补充或确认的内容

开始 P5-A/B 不需要新增训练数据；本机已有 1.7 GiB 证据包和 36 个正式 run。开始 P5-C 以后，需要用户确认：

1. **目标期刊与期望投稿时间**：不同 CCF-C 期刊对应用场景、理论性、篇幅和实验规模侧重不同；
2. **算力预算**：可用 GPU 型号/数量、单卡连续可用时长、可接受的总 GPU-hours 和磁盘余量；
3. **数据权限**：是否允许下载 Tiny ImageNet-200、CIFAR-10-C/100-C；是否已有合法 ImageNet-1K 访问；
4. **硬件条件**：是否能提供 Jetson Orin、桌面 CPU、Android/树莓派等第二平台；
5. **第二 backbone 选择**：默认 ResNet-18；若投稿更偏端侧系统，可改为 MobileNetV3-Small；
6. **新 benchmark-evaluation bundle 授权**：新消融/新 backbone 是否允许在全部预注册和冻结后统一评估一次；
   CIFAR-10/100 结果会明确标成“历史已暴露 benchmark 上的方法锁定评估”，而非独立盲测；
7. **训练失败处置**：确认接受报告 infeasible/negative result，而不是以换 seed 或调 test 的方式“修复”。

若当前只能补一部分，优先级是：**P5-A/B → P5-C → P8 → P6 → P7 → P9**。但在最终投稿版本中，
P6/P7 至少完成一个且最好两者都完成，否则“跨模型、可泛化、面向部署”的表述必须明显收缩。

## 10. 主要方法学依据

- Fast yet Safe: Early-Exiting with Risk Control, NeurIPS 2024：
  <https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea5a63f7ddb82e58623693fd1f4933f7-Abstract-Conference.html>
- Fixing Overconfidence in Dynamic Neural Networks, WACV 2024：
  <https://openaccess.thecvf.com/content/WACV2024/html/Meronen_Fixing_Overconfidence_in_Dynamic_Neural_Networks_WACV_2024_paper.html>
- Dynamic Perceiver for Efficient Visual Recognition, ICCV 2023：
  <https://openaccess.thecvf.com/content/ICCV2023/html/Han_Dynamic_Perceiver_for_Efficient_Visual_Recognition_ICCV_2023_paper.html>
- USDN, WACV 2024：
  <https://openaccess.thecvf.com/content/WACV2024/html/Jeon_USDN_A_Unified_Sample-Wise_Dynamic_Network_With_Mixed-Precision_and_Early-Exit_WACV_2024_paper.html>
- Early-Exit Neural Networks with Nested Prediction Sets, UAI 2024：
  <https://proceedings.mlr.press/v244/jazbec24a.html>
- Two-Stage Early Exiting From Globality Towards Reliability, 2025：
  <https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cit2.70010>

本计划中的实验名称、指标定义和发表级别在执行前仍需对照论文正文及官方代码逐项核验。
