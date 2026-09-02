# 类别风险约束、跨种子稳健的阶段早退：P0 可行性计划

> **2026-09-02完成状态：** P0a 六组已完成并通过文件级审计。multi-exit 最终头相对 matched baseline 三个 seed 均胜，平均 `+0.367 ± 0.266 pp`；但原冻结双出口策略几乎把全部样本送到 exit8，最差类别下降在 seed51/52 达 `4.0/3.6 pp`，因此原 P0 gate 判定为 `stop_or_redesign`。事后共享阈值诊断只用于定位重设计方向，不反改 P0 结论。当前执行计划已切换到 [P1 独立校准确认计划](early_exit_p1_plan.md)，本文件以下内容作为已执行的冻结 P0 协议保留。

## 1. 方向决定

30 组预算阶段稀疏注意力探测已经否定当前静态注意力路线：九个“模块×阶段”单元相对 matched all-none 的三 seed 平均增益全部为负，冻结选择器在三档预算均返回 all-none。继续扫描注意力类型、位置或预算只会在同一负证据上扩大搜索，不再训练。

下一方向改为 MobileNetV2 的阶段早退。它把研究问题从“额外注意力能否提高精度”改为“在可审计的类别风险约束下，哪些样本可以安全地提前结束计算”。早退本身、知识蒸馏和普通置信度阈值都不是新颖贡献；若 P0 可行，论文候选贡献只能收敛到以下组合：

1. 类别均衡及最差类别风险约束，而不是只保护平均 Top-1；
2. 在多个训练 seed 上冻结、验证路由策略，报告最差 seed，而不是只选最好一次；
3. 使用真实目标硬件延迟重新校准预算，MAC 只作为开发阶段代理；
4. 明确区分模型选择、策略校准、策略评估和最终 test，避免阈值泄漏。

该方向仍有明显的新颖性风险。现有工作已经覆盖动态网络校准、多出口蒸馏、风险控制、逐类阈值和困难样本顺序训练；P0 只验证工程和信号是否存在，不作为论文正结果。进入正式实验前必须再次完成针对“类别最差风险 + 跨 seed 稳健”的精确查新。

## 2. P0 冻结模型与目标

主干是 torchvision MobileNetV2，最终分类头不变。在 `features[8]` 和 `features[16]` 后各接一个 `AdaptiveAvgPool2d(1) + Linear` 轻量头。模型前向输出顺序固定为：最终头、exit8、exit16；部署前向可在指定出口停止，不计算后续层。

训练目标为：

```text
L = CE(final, y)
  + 0.2 * [0.5 * CE(exit8, y)  + 0.5 * T^2 * KL(exit8/T || stopgrad(final)/T)]
  + 0.3 * [0.5 * CE(exit16, y) + 0.5 * T^2 * KL(exit16/T || stopgrad(final)/T)]
T = 3
```

最终头只接受自身 CE 梯度，出口蒸馏教师为 `stopgrad(final)`，防止辅助 KL 反向改变教师头。best checkpoint 仍只由最终头 validation accuracy 选择。

静态 Conv/Linear MAC 开发代理为：exit8=`2,676,864`（最终路径的43.70%），exit16=`5,234,688`（85.47%），final=`6,124,928`（100%）。该代理忽略 BN、激活、池化、内存访问和设备调度，禁止作为论文延迟证据。

## 3. P0 最小训练矩阵

只训练两个模型：

| 模型 | 说明 |
|---|---|
| matched baseline | 原生 MobileNetV2，最终头 CE |
| multi-exit | 相同主干与最终头，加 exit8/exit16 和上述辅助目标 |

两者使用新 seeds51/52/53，共6次，从头训练、串行执行：CIFAR-10 45k/5k 分层划分，200 epochs，batch128、AdamW、OneCycleLR、AMP、CUDA Graph，`evaluate_test=false`、`measure_inference=false`。保留 seeds48/49/50 不再用于已停止的注意力确认，也不挪作本批，以免旧冻结计划与新探索批次混淆。

P0 是探索实验。每个 seed 的5k validation 在分析时按类别分成2.5k策略校准和2.5k策略检查；但父级5k已经参与 best checkpoint 选择，因此子级“检查集”并不独立，任何 P0 数字都不能直接写成论文结果。官方 CIFAR-10 test 不评估、不导出预测。

## 4. 冻结策略与 go/no-go

策略按 exit8 → exit16 → final 顺序，使用每个出口最大 softmax 概率阈值。只在2.5k校准子集搜索阈值，最大化预期 MAC 节省，并同时约束相对同模型最终头：

- 总体 accuracy drop ≤0.5pp；
- balanced accuracy drop ≤0.5pp；
- 最差类别 accuracy drop ≤2.0pp。

冻结阈值后在另2.5k子集检查。只有以下 P0 gate 全部通过，才设计正式批次：

- multi-exit 最终头相对 matched baseline 的三 seed 平均差值 ≥−0.30pp；
- 每个 seed 的最终头差值均 ≥−0.75pp；
- 每个 seed 的检查子集 MAC 节省均 ≥15%；
- 每个 seed 的检查子集总体 drop 均 ≤1.0pp；
- 每个 seed 的检查子集最差类别 drop 均 ≤3.0pp。

若任一 gate 失败，先检查失败来自最终头干扰、出口不可分或阈值泛化；不得直接扩展更多 seed、数据集或出口位置。若全部通过，下一批也不是立即做论文主表，而是先冻结互不重叠的 model-selection/calibration/evaluation 数据协议、完成精确查新，并在空闲目标硬件上得到真实分支延迟。

## 5. 实现、验证与入口

- 模型：`model_type=multi_exit`，`src/image_classification/models/mobilenetv2.py`。
- 训练目标：`src/image_classification/training/objectives.py`。
- 策略：`src/image_classification/selection/early_exit.py`。
- P0 sweep：`configs/sweeps/early_exit_p0.yaml`。
- 后台启动器：`scripts/launch_early_exit_p0.py`，固定 `jobs=1`、默认 tag=`p0a`。
- 完成后分析：`scripts/analysis/analyze_early_exit_p0.py`，必须显式传入唯一完成 manifest 和全新输出目录。

服务器验证记录：修改文件 Ruff、compileall、`git diff --check` 均通过；13种模型前向通过；完整测试为143 passed、3 subtests passed；真实 GPU 上的 CUDA Graph 多输出训练测试通过；另完成一轮写入 `/tmp` 的45k/5k、1-epoch multi-exit 冒烟，未评估 test。

已执行的 P0 启动命令（不要重复运行）：

```bash
cd /root/autodl-tmp/image-classification && /root/miniconda3/bin/python scripts/launch_early_exit_p0.py
```
