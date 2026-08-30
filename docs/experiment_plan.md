# 实验执行计划

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

CSGHA 的引导描述取自 block 2 经 SE 增强后的特征：`g_s = GAP(F_2^SE) ∈ R^24`。对于 middle blocks 7--8，每个 Guided-CBAM 都有独立的两层投影 `P_t: R^24 → R^6 → R^64`。其通道门控为 `sigmoid(MLP(avg(F_t)) + MLP(max(F_t)) + P_t(g_s))`，随后继续使用标准 CBAM 空间注意力。代码只传递通道描述，不传递高分辨率浅层特征。

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
