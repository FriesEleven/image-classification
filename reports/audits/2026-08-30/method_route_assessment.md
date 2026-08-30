# 论文方法路线复评：先收紧主张，再决定是否继续方法研发

评估日期：2026-08-30。依据为本次采集的 18 次正式运行及项目交接要求；这是基于当前实验的判断，不是新增文献综述。数值详情见 [实验汇总](experiment_summary.md)，来源及完整精度见 [源索引](source_index.json) 和 [审计结果](audit_results.json)。

## 1. 当前决策

**不建议将 CSGHA v3 直接冻结为最终方法，也不建议现在扩展它的 CIFAR-100 sweep 或自动进入 v4。**

最符合现有证据的论文定位，是“轻量网络内注意力放置与精度—开销关系的实证研究”；shallow 独立组合作为当前优先验证的候选，CSGHA 作为真实实现但尚未证明额外收益的机制消融保留。

这是一项实质性的研究范围调整：它降低了新模块的创新主张，**不能自动解决原审稿意见中的创新性问题，也不保证满足原定投稿要求**。若目标仍是交接文件要求的“有效新机制”，必须先补可信的机制证据，再决定是否投入下一版，而不是把已有组合重新命名。

本报告建议上述方向，但未替用户启动任何新路线或实验。

## 2. 结果改变了哪些判断

以下均为 CIFAR-10 的 best validation，单位为百分比，均值 ± 样本标准差；每组 seeds 为 42/43/44。

| 变体 | 验证准确率 | 相对 baseline 平均差值 | 当前含义 |
|---|---:|---:|---|
| Baseline | 87.69 ± 0.16 | — | 统一新协议的参照 |
| SE-shallow + CBAM-shallow | 88.27 ± 0.46 | +0.58 个百分点 | 三个 seed 均优于 baseline，是当前较强候选 |
| SE-shallow + CBAM-middle | 87.94 ± 0.53 | +0.25 个百分点 | 跨阶段方法必须面对的同位置对照 |
| CSGHA v3，middle | 88.01 ± 0.58 | +0.32 个百分点 | 对同位置控制仅 +0.073 个百分点，且未稳定胜出 |

三项关键判断：

1. **新增 guidance 的有效性尚未证实。** v3 对 matched middle 的配对差值为 `+0.22 / -0.34 / +0.34` 个百分点。不能把均值略高表述为稳定优势，更不能把 v3 相对 baseline 的全部收益归因于 guidance。
2. **v3 不是当前最强方案。** 它相对 shallow 的差值为 `-0.10 / -0.54 / -0.14`，三个 seed 全部落后，均值低 0.26 个百分点。以 seed 42 的 88.68% 单独作为主结果会掩盖这一事实。
3. **shallow 的优势仍是有限范围的经验观察。** 相对 baseline 的 `+0.92 / +0.48 / +0.34` 值得验证，但尚缺同协议 SE-only、CBAM-only、CIFAR-100 及其他方法对照，不能据此证明协同机制、普遍最优位置或跨数据集泛化。

deep 仅有 seed 42 的 88.36%，v1/v2 也各只有一次 87.50%/88.20%；它们是探索结果，不能与三次均值当作同等稳定性证据。v1→v2→v3 在 seed 42 上逐步提高，是开发轨迹，不等同于可靠的多 seed 消融链。

## 3. 证据质量与比较协议

### 3.1 完成性与代码溯源

最新六组实验确已完成：shallow / middle / v3 各补 seeds 43/44。正式 stability manifest 的起止时间为北京时间 `15:15:14–19:43:48`，六组均记录成功退出，且保存的配置和 summary 与各运行目录完全一致。

| 实验集合 | 审计 ID | 运行时 commit 来源 |
|---|---|---|
| CIFAR-10/100 baseline × 3 seeds | E01–E06 | baseline manifest：`a99a701418abf025ffaaf37d9392d16b58aa104f` |
| shallow / middle / v3 的 seeds 43/44 | E10、E11、E13、E14、E17、E18 | stability manifest：`64b79160b6a582086443eb45533311bfc1aaf1f0` |
| 三种位置 seed 42；CSGHA v1/v2/v3 seed 42 | E07–E09、E12、E15、E16 | 未记录；保留配置、日志和权重校验值，不补造 commit |

manifest 路径分别是：

- `artifacts/sweeps/mobilenetv2_baselines_20260829_162111/manifest.json`
- `artifacts/sweeps/cifar10_attention_stability_20260830_151514/manifest.json`

采集时服务器 HEAD 为 `64b7916`、工作区干净，只能说明采集时状态，不能证明过去每次运行都使用该版本。v1/v2/v3 分类综合了 run ID、配置及版本历史，但六次 standalone 的精确执行源码仍存在溯源缺口。

### 3.2 训练设置：哪些来自配置，哪些来自源码

18 份保存配置共同记录：200 epochs、batch size 128、累积 1、`lr=0.01`、AMP 开启、8 workers、prefetch 4、validation size 5,000；硬件记录均为 RTX 3080 Ti。

在已归档历史源码中，训练主流程使用 AdamW（weight decay `1e-4`，betas `0.9/0.999`）、无 label smoothing 的交叉熵及 OneCycleLR。`0.01` 是 OneCycle 的 **max_lr**，不是整个训练恒定学习率：`div_factor=25`、`final_div_factor=100`、`pct_start=0.3`，余弦退火，实际 optimizer update 后推进 scheduler，AMP 跳过更新时不推进。

模型随机初始化；训练增强包括 32×32 随机裁剪（padding 4）、水平翻转、ColorJitter、10° 随机旋转及概率 0.1 的 RandomErasing；validation/test 只做张量转换和各数据集归一化。训练不提前停止，按最高 validation accuracy 保存 best，相同时保留首次达到的 epoch。

`a99a701` 至 `64b7916` 的 data/reproducibility 文件无差异，engine 的差异位于训练完成后的可选 test 评估与 summary 字段，不改变训练循环。这支持新 baseline 与 stability 批次的协议可比性。**配置一致不等于缺失代码版本的六次运行已得到完整证明**，依赖库版本也只有批次 manifest 记录较全。

### 3.3 “三随机种子”的真实含义

当前 `config.seed` 同时传给模型训练随机数和分层划分的 `split_seed`。因此 seed 42/43/44 对应不同的 45k/5k 划分，而不是同一固定 validation set 上只改变初始化。

- 同 seed、同数据集、同划分代码下，各模型可作配对比较；本报告严格按 seed 配对，不按目录顺序配对。
- 跨 seed 的标准差包含划分与训练随机性的变化，不能写成“纯初始化方差”；不同 validation 子集来自同一训练集，也不是独立新数据集。
- 同 seed 不自动保证不同结构共享权重的初始化逐位一致；当前随机数设置也未承诺所有 CUDA 运算逐位确定性。
- seed 42 已反复用于位置筛选和 v1/v2/v3 设计，属于开发样本；43/44 是结构确定后补充的稳定性观察。三次合并结果仍应标记为探索性，不能称为预注册的独立确认。
- n=3 不足以支撑强显著性结论；本报告不以胜出次数或标准差是否重叠代替统计检验。

后续若保持当前协议，可以复用现有 baseline，但须明确报告 split seed 与 training seed 绑定。若改为固定划分、独立训练 seeds，必须另建协议版本并重跑直接对照，不能把旧均值混入新表，也不能根据效果选择有利划分。

### 3.4 Validation 与 test 严格分开

本轮 12 次非 baseline 正式运行均明确 `evaluate_test: false`。旧 baseline 没有此标志字段，但保存了 test 指标，且历史 engine 确实在训练后评估 best checkpoint，不能将字段缺失解释成 test 未使用。

已有 baseline test 为 CIFAR-10 `87.48 ± 0.23%`、CIFAR-100 `56.68 ± 0.24%`。它们不能与 hybrid 的 88.xx% validation 直接计算提升。CIFAR-100 在本次正式证据集中只有 baseline，尚不存在新方法泛化证据。

后续锁定候选、协议、checkpoint 选择与比较清单后，再对对应 best checkpoints 进行一次最终 test 报告；结果不理想也不得回到同一 test 上调方法。本次没有新增 test 评估。

## 4. CSGHA 诊断可以解释什么，不能解释什么

### 4.1 有界融合是实现事实，不是性能证明

v3 的 guidance 加项为 `tanh(alpha) × tanh(P(LN(g_s)))`，幅度不超过 1。它约束的是**新增 guidance logit**，不是所有 deep logits，也不是整个网络梯度。

已有 `csgha_v3_best_guidance.json` 在一个 128 张 validation batch 上记录：两个目标模块的 deep logits 均为 0；raw guidance 平均绝对值约 36.46/56.28；经过 tanh 后约 0.988/0.992；最终门控 sigmoid 饱和率均为 0。

诊断把 sigmoid 小于 0.05 或大于 0.95 定义为饱和。在该 batch 的 deep logits 为 0 时，新增项有界会让最终 sigmoid 落在约 `[0.269, 0.731]`，所以“饱和率为零”主要是结构约束的结果，**并不能证明特征更好、训练不再退化或泛化得到改善**。内部 tanh 输出已很接近 ±1，仍有内部梯度饱和的疑点，不能宣称已全面解决饱和。

同理，deep logits 为零仅是特定 checkpoint 和单个 batch 的观察；可能与 MLP 激活失活有关，但现有证据不足以认定原因，也不能推广到全部样本和训练过程。

### 4.2 跨版本 replay 不能代替原模型诊断

- `csgha_v1_guidance.json` 的配置 ID 已是 v2，加载 v1 权重时缺少 normalization/scale 参数，scale 为零。这是改变了执行函数的回放，不是完整 v1 原始推理；block 7 的变化还会影响后续 block 8 输入。
- `csgha_v2_replayed_with_v3_bound.json` 明确是给旧权重加入 v3 bound 的干预。可用于比较干预下张量范围，不能当作 v2 训练时状态或 v3 训练收益的证据。
- v2 和 v3 的权重键可以相同，因为 tanh 不增加参数。即使权重加载无 missing keys，也不能证明执行的是同一模型。`csgha_v2_or_v3` 不是精确版本标识。
- `csgha_v2_best_guidance.json` 没有 missing keys，记录了约 20.3%/62.5% 的门饱和率，但它仍只覆盖一个 batch，且未保存完整诊断代码版本及样本索引。
- deep magnitude 为零时，脚本用 float epsilon 作分母产生巨大 guidance/deep 比值。该比值应视为不可解释的零分母情形，不能写作“引导真实强了几百万倍”。

以上诊断原文均已保存在快照，校验值见 `source_index.json`。这些问题削弱的是现有机制解释，不会改变从训练日志复算得到的准确率排名。

### 4.3 若继续机制路线，先补不训练的诊断

下一步应先用匹配的历史实现与对应 checkpoint，严格记录代码版本、权重哈希、validation 索引和干预设置；先重算该 checkpoint 的原始 validation 指标，确认与日志相符，再在同一批样本上比较：

1. 原始 guidance、置零 guidance、跨样本打乱 guidance、训练集统计得到的固定平均描述；固定模型其余部分，统计配对预测变化和 validation 指标。
2. 各目标 block 的 deep MLP 激活、raw/bounded guidance 分布、通道门跨样本方差；至少覆盖完整 validation，并分 seed 报告。
3. 区分输入相关信息贡献、近似常量的通道重标定，以及改变幅度本身造成的作用。若打乱 guidance 也改变其分布，需加控制，不能直接视为机制被证明。

这些是**建议的下一项工作，尚未运行**。它们是已训练模型的干预分析，不等同于“无 guidance 从头训练”的核心消融。如果无法确认老版本来源，应明确采用待验证的参考实现；不能把复现接近当成原始版本的绝对证明。

## 5. 论文路线选择与推进门槛

### 路线 A：位置与精度—开销的实证研究（最贴合现有证据）

研究问题改为：在固定 MobileNetV2 与训练协议下，SE/CBAM 的不同放置是否有可复现的收益，收益与实际开销如何权衡？当前可以优先关注 shallow，但只能称为已测配置中的候选。

若用户确认该路线，按以下顺序追加工作：

1. 固定划分协议与报告规则；补 shallow 的 CIFAR-100 三 seed 验证，复用同协议 baseline，检验收益是否跨数据集保持。若失败则如实缩小结论，不能称为泛化成立。
2. 补同位置 SE-only、CBAM-only，形成 baseline / SE / CBAM / independent hybrid 的最小消融。单 seed 只能探索；支撑主结论的差值需要匹配的重复运行。CSGHA 结果作为独立的机制尝试呈现，不隐去负结果。
3. 确定最终主张后，再按 handoff 的比较范围补 ECA、Coordinate Attention 以及两个轻量 backbone；具体实现、32×32 适配和训练预算另行核对。这里引用的是交接要求，不声称它们覆盖最新全部方法。
4. 在统一输入、精度、batch、设备与软件环境下重新测参数/计算量/延迟，完成 test 锁定评估与图表。

不能提前写“主辅协同”“全局最优位置”“几乎零成本”“Pareto 最优”或“移动端实时”。这条路线是否仍适合原投稿目标需要重新判断；只是补全这些表格，并不会自然产生新机制贡献。

### 路线 B：坚持有效新机制（原交接目标尚未达到）

先做第 4.3 节的版本匹配诊断，不马上训练 v4。只有观察能支持一个明确的新假设时，才另写方法与对照计划，限定候选数和预算，经用户确认后训练。

继续进入 CIFAR-100 的最低门槛，是在预先确定的 CIFAR-10 比较协议下对 matched independent control 展示可重复的额外收益，同时正面比较最强 shallow control 的精度及成本。之后仍需 CIFAR-100、核心消融和统一近期对照，不能只保留有利数据集。

“引导从 block 2 出发”与 shallow blocks 1–2 作为目标存在前后依赖限制。不能因为 middle 便于实现 guidance，就把 middle 写成全局最佳位置。若更改源/目标位置，必须视作新候选并配新的同位置对照。

若预算内仍无法超过独立控制，应转入路线 A 或诚实报告局限；本报告不建议无期限地逐个尝试 v4/v5。

### 当前推荐的行动顺序

**先保全证据与收紧文字 → 确认是否接受经验研究定位 → 再按选定路线安排最小新增工作。** 若暂时不接受降低创新主张，下一项应是版本匹配机制诊断，不是新的全量训练。

这一决策点之前，不运行 CSGHA CIFAR-100，不开全矩阵 sweep，也不使用官方 test 选择版本。

## 6. 原稿与旧结果的处理

本次评估对照 `/Users/salt/Jlu/paper/sn-article-template/docs/handoff.md` 的要求：真实机制、稳定优于独立组合、双数据集核心消融、统一 recipe 与近期对照。当前只完成了其中部分实验准备与 CIFAR-10 探索，不能将交接清单判为完成。

`reports/tables/experiment_results_summary.csv`/`.xlsx` 中旧九行结果（例如 baseline 81.81%、ECA 83.17%）不属于本轮 18 个正式 run，当前证据集中没有对应完整来源链和同协议证明。保留原文件作为历史材料，**不与本轮数值拼接，不直接用于修订主表**。

`docs/paper_plan.md` 的 ECA prime / deep auxiliary 是早期构想，且包含预写的效果结论；它不是当前实验已证实的路线。`docs/experiment_plan.md` 中 v1–v3 的历史启动命令也不是本轮审计后继续运行的授权。

后续改稿应做到：

- 摘要和贡献只描述实际实现且得到证据支持的增量，不能把 v3 的结构存在等同于稳定有效。
- 主表统一 validation/test 标签、recipe、seeds 与代码版本；CIFAR-100 缺项保持缺项，不引用旧图填补。
- 模块参数量来自模型计数；现有 `benchmark.py` 的 FLOPs 是硬编码基值与每模块调整项的估计，不能充当严格 profiler 结果或位置间计算量证据。
- 现有 latency 在训练前随机权重模型上测得，batch 1、32×32、GPU warmup 10 次、测量 100 次。跨不同时间/实例的单次 benchmark 只作辅助记录，不能推导部署收益或移动端速度。
- 旧图表、Grad-CAM/t-SNE 及诊断图只承担相应的描述性作用，不作为泛化或因果机制的直接证明。

本次仅整理代码项目内报告和计划状态，尚未修改论文仓库正文。

## 7. 下轮实验必须补全的溯源字段

建议将后续正式运行统一经过 manifest 启动器，并保存：运行 ID、完整命令、resolved config、Git commit 与 dirty diff 校验值、明确的 architecture version、依赖版本、硬件、split seed / training seed、划分索引或其哈希、checkpoint 哈希、metric split 与 checkpoint 选择规则。

这些字段是下一轮改进要求，不是对已有实验的追溯性补造。本次只备份小型元数据并校验远端权重；释放租赁实例前，需要另做 best checkpoint 本体备份。
