# 版本匹配诊断与下一版机制假设

用户已确认坚持新机制路线。本报告承接 [20:06 的实验审计](../../audits/2026-08-30/method_route_assessment.md)，更新该审计中尚未解决的 guidance 输入信息问题。

后续执行更新：本报告中的旧v4批次已在首组epoch100后停止，产物保留；下一批使用匹配的CUDA Graph训练后端和新perf1编号。速度验证、数值差异与跨版本比较边界见 [吞吐优化报告](../2026-08-30-throughput/findings.md)，下文历史干预结果不变。

## 1. 本次诊断结论

**Guidance 确实包含并利用了输入相关信息；但该事实不等于跨阶段方法已经优于独立组合。**

五个 best checkpoint 均使用相应历史源码、严格权重加载，在各自完整 5,000 张 validation 上复现了原始准确率：v1 87.50%、v2 88.20%、v3 seeds 42/43/44 为 88.68%/87.62%/87.74%。本次没有打开官方 test dataset，也没有训练这些模型。

原始及干预结果见 [可复算结果表](results.md)，全部出处见 [机器可读证据](diagnostic_summary.json)。

v3 的关键观察：

- 全 validation 随机置换 guidance，保留描述的边际分布且排除自身配对；三个固定置换下，各训练 seed 都出现下降，平均分别为 **0.67、0.43、0.61 个百分点**。
- 换成对应训练集的固定平均 guidance 加项后，三个 seed 分别下降 **0.28、0.46、0.10 个百分点**。换成平均描述、完全置零 guidance 也均下降。
- 通道门跨样本存在明显变化，并非所有通道都是常量。即使 tanh 输出接近 ±1，其正负号仍可随输入变化；“饱和”不能直接推出“没有输入信息”。

这些结果反对“只是常量门控”的解释，支持已训练模型依赖正确配对的浅层描述。它们是 checkpoint 的干预证据：打乱也改变了浅深特征的联合分布，结果不应被解释为信息论互信息测量或“从头训练无 guidance”消融。不同版本置零/打乱后的下降幅度也不能作为方法强弱排名。

v3 对 matched independent middle 的三 seed 均值优势依然仅 0.073 个百分点，且落后最强 shallow 对照；之前关于“核心方法尚未验证”的结论没有被推翻。

## 2. 发现了什么可优化问题

完整 validation 统计显示，v3 的目标 block 7 / block 8 deep-logit 零值比例分别为：

| Seed | Block 7 | Block 8 |
|---|---:|---:|
| 42 | 100.00% | 100.00% |
| 43 | 95.14% | 99.98% |
| 44 | 83.16% | 99.92% |

对应 deep MLP 的第一层输出大多为负，ReLU 将其截为零。特别是 seed 42，在这组验证样本上，通道门的动态部分实际上只剩 guidance。其他 seeds 仍有少数样本保留 deep 分支，不能再沿用旧单 batch 诊断中的“全部为零”。

此外，v3 guidance 内部 tanh 的饱和比例约 73.90%–97.19%，但最终 sigmoid 饱和比例与它是两回事。这是另一个待研究问题，本轮不同时改它。

**目前只观察到了最终 checkpoint 的分支失活，尚未证明它是准确率不足的原因，也没有证明独立 CBAM 不存在同样问题。** 因此下一版必须配激活函数匹配的无 guidance 对照。

## 3. 本轮只检验一个优化假设

假设：deep 分支的硬 ReLU 截断妨碍了深层自身信息与浅层 guidance 的联合使用；为负半轴保留梯度可能改善这一点。

实现：只把两个目标 block 中 deep channel MLP 的 `ReLU` 换成 `LeakyReLU(negative_slope=0.1)`。这是一次工程优化候选，不将 LeakyReLU 本身作为创新点。

不改变的内容：SE blocks 1–2、guided CBAM blocks 7–8、source block 2、guide 的 LayerNorm/两层投影/ReLU/tanh、零初始化 scalar gate、空间注意力、优化器、学习率、数据增强、划分算法、epochs 和 seeds。暂不叠加新归一化、新损失或学习率搜索。

| 组别 | model_type | deep 激活 | Guidance | 参数量 |
|---|---|---|---|---:|
| 旧 middle 控制 | hybrid | ReLU | 无 | 2,238,024 |
| 旧 v3 | csgha | ReLU | v3 有界加项 | 2,239,178 |
| 新 matched control | hybrid_leaky | LeakyReLU(0.1) | 无 | 2,238,024 |
| 新 v4 候选 | csgha_v4 | LeakyReLU(0.1) | 与 v3 相同 | 2,239,178 |

旧的两行复用已审计结果；新两行各运行 seeds 42/43/44，共 **6 次新训练**。全部从头训练，不给 v3 checkpoint 换激活后直接宣称是 v4。旧 `csgha` 和 `hybrid` 默认行为保留。

新结构增加持久化的无参数版本 buffer，并记录显式 architecture version，避免“没有新增参数，所以旧权重能静默加载成新函数”的溯源问题。

## 4. 预先确定如何解释结果

主要比较是新 v4 与新 matched control 的逐 seed 差值，同时报告旧 v3/middle 结果、v4 对 v3 的变化，以及仍然更强的 shallow 对照。

- 若 v4 改善，但 matched control 改善相同或更大：只能支持激活函数优化，不支持 guidance 的额外贡献。
- 若 v4 在三个配对 seed 上一致优于新 matched control、且平均收益较旧 +0.073 个百分点扩大：支持继续研究；这仍不是统计显著性或投稿充分性的证明，进入 CIFAR-100 前还须正面比较 shallow 的精度与开销。
- 若 deep 分支活性改善而准确率不改善：推翻“解决硬截断便可带来任务收益”的当前假设，不能以更漂亮的激活图宣称成功。
- 若 deep 分支仍几乎无贡献，或仅更强地依赖 guidance：不宣称恢复了浅深联合利用。
- 如果优势只出现在 seed 42，不把它写成稳定结论；42/43/44 仍是开发过程中已使用过的 seeds，而非新的独立确认样本。

完成后应复查原始/打乱/训练均值/置零干预，以及两条分支的活性，再决定是否进入 CIFAR-100。当前不预设结果、不自动安排 v5。

## 5. 启动与记录

在服务器项目根目录一行后台启动：

```bash
python scripts/launch_csgha_v4.py
```

可先加 `--dry-run`，只校验和打印命令，不开始训练。启动器串行运行 6 组，打印 PID、日志及 `tail -f` 命令；断开 SSH 后继续。检测已存在训练进程、重复启动锁或同名输出时拒绝启动，不清理任何旧实验。

上一轮六组约 4 小时 29 分钟；本次两组首个冒烟 epoch 约 21 秒，直接线性外推会接近 7 小时，但首 epoch 包含启动开销且短程配置不能代表正式稳态。可先预留 **约 5–7 小时**，启动后用实际 epoch 耗时更新，不将其作为实测完成时间。期间不要修改源码或配置：runner 会核对源码指纹，发现变化则停止，避免同一 sweep 混入不同版本。

新增记录包括完整源码/配置快照、各文件 SHA-256、Git commit/dirty 状态、依赖版本、明确模型版本、实际 train/validation 索引、绑定的 split/training seed，以及 best checkpoint 校验值。源码快照保存在各新 sweep 的 `source_snapshot/`；不要求把当前 dirty HEAD 冒充成可单独复现的 commit。

正式六组训练尚未启动。验证状态另见本目录 [验证记录](verification.md)。

## 6. 重做与复算诊断

服务器重新执行历史诊断时使用新的输出目录：

```bash
python scripts/diagnostics/audit_csgha_information.py --output artifacts/diagnostics/csgha_information_rerun
```

该入口固定针对审计中的 v1/v2/v3 五个 checkpoint；不是 v4 结果的自动分析入口。v4 训练完成后应根据其新 manifest 和源码快照进行版本匹配分析。

本地已有原始小型证据与 7.4 MB 左右的配对预测备份，可离线复算本次报告：

```bash
python3 scripts/analysis/summarize_guidance_information.py
```

五个诊断的原始验证结果虽都复现，v1/v2/v3 的 seed 42 依然缺运行时 commit；报告只称参考历史实现匹配。v3 seeds 43/44 有运行时 manifest commit。这一区分保留在全部报告中。
