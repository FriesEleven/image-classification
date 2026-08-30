# 实验执行计划

## 当前状态（2026-08-30 审计后）

三组候选补充 seeds 43/44 的六组实验已全部完成。CIFAR-10 三 seed best validation：baseline `87.69 ± 0.16%`、independent shallow `88.27 ± 0.46%`、independent middle `87.94 ± 0.53%`、CSGHA v3 `88.01 ± 0.58%`。v3 对 matched middle 的平均提升仅 0.073 个百分点，对 shallow 三次均落后，当前不满足稳定优于独立组合的推进条件。

后续用户已确认坚持新机制路线。五个版本匹配 checkpoint 的完整 validation 干预已完成，guidance 利用了输入相关信息，但 deep 分支存在大量 ReLU 截断；已实现只改 deep 激活的 v4 和同激活独立控制。旧v4批次在控制组seed42完成epoch100后按用户要求中断，其他五组未开始，旧产物保留。详见 [诊断结果与 v4 假设](../reports/diagnostics/2026-08-30-guidance/findings.md)。不默认进入 CIFAR-100。

最新吞吐配置为 **perf2两路并行**：两份v4匹配配置均启用训练CUDA Graph，每进程1个PyTorch CPU线程，保持batch128和8个data workers，使用新 `_perf2` 实验编号，从头运行六组。真实队列短测每组稳定epoch约9.5–9.9秒、同时推进两组；同配置串行约5.3–5.5秒/组，混合双模型稳定吞吐约提升11%。串并行3轮复验权重逐参数、训练CSV字节一致。正式新批次尚未启动。服务器项目根目录一行命令：`/root/miniconda3/bin/python scripts/launch_csgha_v4.py`。见 [perf2测量与安全说明](../reports/diagnostics/2026-08-30-throughput/perf2_parallel.md)。

并行时明确禁用推理延迟测量，避免混入共享GPU竞争耗时；论文效率数据后续独占GPU统一测量。Graph要求关闭AMP权重缓存，不能视为旧eager轨迹的逐位等价续训；控制组与v4必须使用相同后端。前版perf1约5.3–5.8秒/epoch、旧eager控制约13.3秒的记录保留在 [前版优化报告](../reports/diagnostics/2026-08-30-throughput/findings.md)，不据此夸大本次并发增益。

可追溯数据与此前判断见 [实验审计入口](../reports/audits/2026-08-30/README.md) 与 [方法路线复评](../reports/audits/2026-08-30/method_route_assessment.md)。当前 seed 同时控制划分和训练，不能将三 seed 标准差解释为固定划分下的纯初始化方差。

以下保留配置说明与历史执行过程；已有启动命令不代表现在需要重跑。特别是单 batch 诊断和跨版本 checkpoint replay，不足以证明原模型失败的因果机制。

## 配置与历史执行记录

项目比较 MobileNetV2、全局 ECA、不同位置的 CBAM/SE，以及两种 SE+CBAM 混合放置策略。正式配置统一存放在 `configs/experiments/`。

正式对比统一使用 200 epochs、batch size 128、梯度累积 1、8 个持久化 data-loader workers、AdamW 和 OneCycleLR。OneCycleLR 在每次实际 optimizer update 后推进一次；与原来的 batch size 64、梯度累积 2 相比，有效 batch size 和每 epoch optimizer 更新次数保持不变。

数据集由配置中的 `dataset: cifar10|cifar100` 选择。两个数据集均使用按类别分层且由 `seed` 固定的 45,000/5,000 train/validation 划分；官方 test set 不参与 checkpoint 或位置选择。

## 单项实验

```bash
python scripts/train.py --config configs/experiments/baseline.yaml
python scripts/train.py --config configs/experiments/baseline_cifar100.yaml
python scripts/train.py --config configs/experiments/eca_global.yaml
python scripts/train.py --config configs/experiments/cbam_shallow.yaml
python scripts/train.py --config configs/experiments/cbam_middle.yaml
python scripts/train.py --config configs/experiments/cbam_deep.yaml
python scripts/train.py --config configs/experiments/se_shallow.yaml
python scripts/train.py --config configs/experiments/se_middle.yaml
python scripts/train.py --config configs/experiments/se_deep.yaml
python scripts/train.py --config configs/experiments/hybrid_se_shallow_cbam_deep.yaml
python scripts/train.py --config configs/experiments/hybrid_se_deep_cbam_shallow.yaml
```

命令行参数可以覆盖配置，例如：

```bash
python scripts/train.py --config configs/experiments/baseline.yaml --epochs 1 --experiment_name smoke
```

## 批量实验

`configs/sweeps/all_experiments.yaml` 定义了完整实验序列。先检查命令：

```bash
python scripts/run_experiments.py --dry-run
```

确认后启动：

```bash
python scripts/run_experiments.py
```

每次实验的配置、指标、延迟、日志、权重和预测都保存在 `artifacts/runs/<experiment_id>/`，同一实验不会再横跨多个输出目录。

## Validation-only 位置筛选

位置选择只使用 CIFAR-10 validation set，不评估官方 test set：

```bash
python scripts/launch_position_screening.py --dry-run
python scripts/launch_position_screening.py
```

默认启动方式会在后台独立会话中运行，断开 SSH 后仍会继续；命令会打印 PID、精简日志路径和对应的 `tail -f` 监控命令。启动器会先验证三份配置，拒绝重复进程，并拒绝覆盖已有位置实验目录。

筛选配置固定浅层 SE 在 blocks 1--2，分别比较 CBAM 位于 shallow 1--2、middle 7--8 和 deep 15--16。三者均设置 `evaluate_test: false`。确定最佳位置后，再将对应位置用于 CSGHA；当前 `configs/experiments/csgha_se_shallow_cbam_middle.yaml` 是 middle 位置的可运行候选配置。

CSGHA 的引导描述取自 block 2 经 SE 增强后的特征：`g_s = GAP(F_2^SE) ∈ R^24`。对于 middle blocks 7--8，每个 Guided-CBAM 都有独立的两层投影 `P_t: R^24 → R^6 → R^64`。v1 直接将投影结果与 CBAM logits 相加；seed 42 的最佳 validation accuracy 为 87.50%，低于相同位置的独立组合 88.46%，因此不能作为最终方法。

v2 先对 `g_s` 使用 LayerNorm，再通过可学习门控融合：`sigmoid(MLP(avg(F_t)) + MLP(max(F_t)) + tanh(alpha_t) P_t(LN(g_s)))`。它在 seed 42 上达到 88.20%，较 v1 提升 0.70 个百分点，但仍低于相同位置独立组合的 88.46%。最佳 checkpoint 的两个投影输出平均幅度达到 19.58 和 57.66，即使经过标量门控，通道门饱和率仍为 20.3% 和 62.5%。

v3 对投影输出再使用 `tanh`，将引导 logits 严格限制到 `[-1, 1]`：`sigmoid(MLP(avg(F_t)) + MLP(max(F_t)) + tanh(alpha_t) tanh(P_t(LN(g_s))))`。每个目标 block 的 `alpha_t` 独立且初始化为 0，使训练初始行为退化为普通 CBAM，然后逐步学习是否以及多强地使用有界引导。随后仍使用标准 CBAM 空间注意力。代码只传递通道描述，不传递高分辨率浅层特征。

位置筛选完成后，一行后台启动 bounded CSGHA v3 middle 候选的 validation-only 实验：

```bash
python scripts/launch_csgha_validation.py
```

可先追加 `--dry-run` 检查目标命令。CSGHA 至少需要超过相同 middle 位置的独立组合验证准确率，才能说明跨阶段引导带来额外增益；最终候选确定前仍不评估官方 test set。

引导分支诊断命令会报告 deep logits、原始/有界/门控后 guidance logits 的平均幅度与最大值，以及旧式直接相加和当前有界门控方式的 sigmoid 饱和比例：

```bash
python scripts/diagnostics/check_csgha_guidance.py --checkpoint <checkpoint> --output artifacts/diagnostics/csgha_guidance.json
```

v3 的 seed 42 validation accuracy 为 88.68%，超过相同 middle 位置独立组合的 88.46%，但略低于 shallow 独立组合的 88.78%。为避免用单次结果下结论，复用三者已有的 seed 42，并补跑 seeds 43/44：

```bash
python scripts/launch_cifar10_stability.py
```

该命令后台串行运行 3 个变体 × 2 个缺失 seeds，共 6 组 validation-only 实验。只有当 CSGHA v3 的三随机种子均值稳定超过 matched middle control，并与最强 shallow control 比较后仍有优势，才冻结结构并进入 CIFAR-100。

## 结果整理

```bash
python scripts/analysis/summarize_results.py
python scripts/analysis/statistics.py
python scripts/analysis/generate_tables.py
python scripts/analysis/plot_performance.py
```

分析脚本读取本地实验产物，并把需要纳入论文版本控制的内容写入 `reports/tables/` 和 `reports/figures/`。

## 对比维度

1. 基线与全局 ECA 的精度、参数量和延迟差异。
2. CBAM/SE 在浅层、中层和深层的放置敏感性。
3. 相同位置下 CBAM 与 SE 的效率差异。
4. 浅层 SE + 深层 CBAM 与反向放置的混合效果。
5. 准确率、参数量、估算 FLOPs 和实测推理延迟的帕累托权衡。

FLOPs 字段目前沿用项目原有的分析估算；论文定稿前建议使用固定 profiler、输入尺寸、批量大小和硬件重新测量全部模型。
