# 类别风险约束、跨种子共享早退：P1 独立校准确认计划

## 1. P0 结论与本批目的

P0a 唯一完成 manifest 为 `artifacts/sweeps/cifar10_early_exit_p0_serial_p0a_20260902_110000/manifest.json`，SHA-256 为 `49a40fc1afb007d7a357c4bc54fd4b96f7618cebaf6a290955bfb8fe19327cd6`。正式审计未发现完整性、配置、划分、checkpoint、源码快照或 test 边界问题。

matched baseline seeds51/52/53 的 best validation 为 87.96/88.16/88.16%，multi-exit 最终头为 88.62/88.46/88.30%；配对增益为 `+0.66/+0.30/+0.14 pp`，平均 `+0.367 ± 0.266 pp`，胜出 3/3。原冻结的双出口策略虽然平均节省 56.28% MAC，但在三个 seed 上约 100% 样本直接走 exit8，且检查子集最差类别下降最高 4.0pp，超过 3.0pp gate，故原判定严格保留为 `stop_or_redesign`。

失败后的机制诊断遍历全部 1,024 个预测类别保护集合及 43 个共享阈值。最优解不需要类别排除，只需 exit8 共享阈值 `0.9410777530842098`；它在 P0 的三个检查子集上分别让 75.44%/77.12%/75.80% 样本早退，MAC 代理节省 42.47%/43.42%/42.67%，相对最终头的总体、均衡和最差类别经验下降均为 0。该结果使用了已参与 checkpoint 选择的父 validation，并且是在原 gate 失败后发现，只能证明“保守共享阈值值得用独立数据确认”，不能作为论文结果，也不能把 P0 改写为通过。

P1 的唯一目的，是用新训练 seed 和互不重叠的模型选择/策略校准集，确认这一信号是否可复现。它不搜索新出口、不改损失、不增加模型矩阵。

## 2. 新颖性边界

早退、知识蒸馏、softmax 阈值、类别专属出口和普通风险控制都已有直接先例。例如 NeurIPS 2024 的 [Fast yet Safe](https://papers.neurips.cc/paper_files/paper/2024/file/ea5a63f7ddb82e58623693fd1f4933f7-Paper-Conference.pdf) 已给出带风险控制的早退和独立 train/calibration/test 协议；[USDN](https://openaccess.thecvf.com/content/WACV2024/papers/Jeon_USDN_A_Unified_Sample-Wise_Dynamic_Network_With_Mixed-Precision_and_Early-Exit_WACV_2024_paper.pdf) 与 [Dynamic Perceiver](https://openaccess.thecvf.com/content/ICCV2023/html/Han_Dynamic_Perceiver_for_Efficient_Visual_Recognition_ICCV_2023_paper.html) 已覆盖样本级动态计算；类别专属早退也有[既有方法](https://www.sciencedirect.com/science/article/pii/S1568494621002398)。

因此目前只保留一个较窄的候选主张：**在最差类别经验风险约束下，选择一个跨多个独立训练 seed 共享、面向部署预算的稳健路由策略，并完整公开失败停止规则与数据边界**。P1 只是确认信号，不足以证明该组合具有论文新颖性；若通过，最终方法定稿前还需更精确查新，并决定是否引入有统计保证的类别风险上界。不得把“经验下降为零”表述成总体分布上的零风险保证。

## 3. 冻结训练矩阵

模型定义沿用 P0，不做结构或损失更改：

- matched baseline：原生 torchvision MobileNetV2，仅最终头 CE；
- multi-exit：`features[8]`、`features[16]` 两个轻量头，权重 0.2/0.3，蒸馏 `alpha=0.5`、`T=3`，教师最终头停止梯度；
- best checkpoint 只由最终头 model-selection validation accuracy 决定。

新训练 seeds 为 54/55/56，共 `2模型 × 3 seeds = 6` 次，全部从头、串行、200 epochs。每个 run 都使用固定 `split_seed=20260902`，将 CIFAR-10 官方 50k train 分成：

| 用途 | 样本数 | 是否参与训练/选择 |
|---|---:|---|
| 参数训练 | 40,000 | 仅训练梯度 |
| checkpoint 模型选择 | 5,000 | 仅选 best checkpoint |
| 路由策略校准 | 5,000 | 训练过程不迭代；完成后才选阈值 |
| 官方 test | 10,000 | 本批不评估、不导出预测 |

三个子集分层、完全不重叠并覆盖官方 50k train；数据划分 seed 与训练 seed 解耦，而 DataLoader shuffle 仍随训练 seed 变化。baseline 与 multi-exit 以及三个训练 seed 使用同一份固定索引，避免数据构成差异。

其余配方冻结为 batch128、AdamW、OneCycleLR、AMP、CUDA Graph、`jobs=1`、`evaluate_test=false`、`measure_inference=false`。P1 不在训练中测延迟，避免 GPU 争用污染硬件证据。

## 4. 训练前冻结的共享策略

完成六组且先通过 manifest 审计后，`scripts/analysis/analyze_early_exit_p1.py` 只在三个 5k calibration 集上执行以下规则：

1. 部署路径只允许 `exit8 → final`；exit16 保留为训练辅助头，不参与本次路由，也不为 P1 重新搜索结构。
2. 置信度为 exit8 最大 softmax 概率；不做温度拟合，不使用类别排除表。
3. 三个训练 seed 共用**同一个**阈值，不允许每 seed 单独调参。
4. 阈值候选在看到 P1 数据前固定为 0.000–1.000、步长 0.001，另含 final-only 哨兵，共 1,002 个候选；不再使用数据分位数生成网格。
5. 每个 seed 的 calibration 都必须相对该 checkpoint 的最终头满足：总体 accuracy drop ≤0、balanced accuracy drop ≤0、最差类别 accuracy drop ≤0，且 exit8 路由比例在 15%–95%。
6. 在同时可行的阈值中，先最大化三个 seed 的最小 MAC 节省，再最大化平均节省；完全并列时选更保守的较高阈值。

最终头的先验保留 gate 为：multi-exit 相对 matched baseline 的三 seed 平均 validation 差值 ≥−0.30pp，且每个 seed ≥−0.75pp。任何 gate 失败都返回 `stop_or_redesign`，不打开官方 test、不增加 seed、不扩展数据集。

若全部通过，分析器输出并哈希唯一 `locked_policy`。之后才允许在三个 seed 的 baseline/multi-exit best checkpoint 上各评估官方 CIFAR-10 test 一次，并用同一个锁定策略报告总体、balanced、逐类最差风险、路由比例和 MAC。真实目标硬件延迟需在空闲设备上另做批量/单样本动态执行测量；MAC 仍不是论文延迟证据。

## 5. 实现和启动前验证

- 配置：`configs/experiments/early_exit_p1_{baseline,multi}.yaml`。
- Sweep：`configs/sweeps/early_exit_p1.yaml`。
- 后台启动器：`scripts/launch_early_exit_p1.py`，默认 tag=`p1a`，固定 `jobs=1`。
- 共享策略：`src/image_classification/selection/early_exit.py`。
- 完成后策略锁定：`scripts/analysis/analyze_early_exit_p1.py`。

服务器启动前验证：本次修改文件 Ruff、`compileall`、`git diff --check` 均通过；`scripts/diagnostics/check_model.py` 的 13 种配置全部前向通过；`check_data.py` 核对 CIFAR-10/100 历史 45k/5k 边界通过；完整测试为 158 passed + 3 subtests passed。唯一 warning 仍是 CUDA Graph 集成测试首次建立 cuBLAS context。另已完成真实 GPU 的 P1 一轮 40k/5k/5k 冒烟，三集合覆盖 50k 且两两无交集，训练未迭代 calibration、`test_evaluated=false`。启动器 dry-run 精确打印六个新 ID，未创建目标目录、未启动训练。

启动器拒绝已有训练进程、已有目标目录、重复锁和协议漂移；命令本身创建独立后台 session 并立即返回 PID 与日志路径，不需要额外 `nohup`。预计与 P0 同量级，约 2 小时，实际以 launcher log 为准。

唯一下一批启动命令：

```bash
cd /root/autodl-tmp/image-classification && /root/miniconda3/bin/python scripts/launch_early_exit_p1.py
```
