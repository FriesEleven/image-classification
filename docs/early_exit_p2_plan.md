# 冻结策略跨重训版本迁移：P2 最小确认计划

## 1. 研究问题与停止条件

P1b 已经在 source training seeds 54/55/56 上选定一个跨 seed 共享策略：exit8 最大
softmax 置信度阈值 `0.984`，无类别排除，未早退样本走最终头。该策略在唯一一次
CIFAR-10 官方 test 上约 65% 早退，相对最终头没有观测到总体、balanced 或逐类退化，
并在 RTX 4090D batch1 的真实分阶段实现上得到 24.91% 期望时延节省。

这些结果足以构成论文中的一节，但不能单独支撑当前目标论文。P2 只回答一个更窄且可证伪的
问题：**在 source 模型版本上冻结的同一阈值，能否不经任何重新校准，直接迁移到独立重训的
新模型版本？**

- 若任一冻结 gate 失败：归档为迁移失败，不在 target seeds 上调阈值，不打开外部分布数据，
  并停止当前主张。
- 若全部 gate 通过：冻结 P2 结果，进入一次性 CIFAR-10.1 v6 分布转移验证；不得重新打开
  原始 CIFAR-10 官方 test。
- P2 通过本身仍不等于整篇论文完成。独立数据分布和至少一个额外模型/数据集的泛化证据仍是
  完整投稿的高优先级缺口。

## 2. 冻结训练矩阵

P2 不改模型结构、损失或训练配方，只更换训练随机种子：

| 模型 | target training seeds | 次数 |
|---|---|---:|
| MobileNetV2 matched baseline | 57, 58, 59 | 3 |
| exit8/exit16 detached-final KD multi-exit | 57, 58, 59 | 3 |

共 6 次从头训练，严格串行，200 epochs。multi-exit 仍使用出口权重 `[0.2, 0.3]`、
蒸馏 `alpha=0.5`、`T=3.0`；best checkpoint 只由最终头 validation accuracy 选择。
baseline 保留是为了区分“路由迁移”与“新 seed 下辅助训练损害最终头”这两个问题。

固定数据边界与 P1b 完全相同：`split_seed=20260902`，40,000 train / 5,000
checkpoint-selection validation / 5,000 cross-model transfer。训练过程不迭代 transfer loader，
`evaluate_test=false`，不生成官方 test 预测。transfer 的 5,000 张图与 P1b calibration 是同一组
索引，因此这里只能证明跨模型版本迁移，不能冒充独立数据分布验证。

其余配置冻结为 batch128、AdamW、OneCycleLR、AMP、CUDA Graph、8 workers、
`prefetch_factor=8`、`torch_num_threads=1`、`jobs=1`、`measure_inference=false`。

## 3. 不可变 source policy

版本化 source selection：

`reports/experiments/2026-09-02-early-exit-p1b/locked_selection.json`

SHA-256：

`8dce5d938e06ae9d7432bd6baafd0c0ed9f978678fb53767ab02eeafecc723ab`

P2 启动器会在训练前校验该哈希及以下字段：

- source seeds：54/55/56；target seeds：57/58/59，二者不重叠；
- 路径：exit8 → final，exit16 只作为训练辅助头；
- 置信度：exit8 最大 softmax probability；
- 阈值：精确为 `0.984`；
- 类别保护：空；
- target 模型重新校准：禁止。

阈值不传入训练过程。完成后的 `analyze_early_exit_p2_transfer.py` 只调用一次固定策略应用函数，
不会导入或调用阈值选择器，target 数据上考虑的候选阈值数记为 0。

## 4. 预注册判定 gate

先审计唯一 completed P2a manifest，再对 target best checkpoints 计算结果。所有 gate 均需通过：

1. multi-exit 最终头相对 matched baseline 的三 seed 平均 validation 差值 ≥ −0.30pp；
2. 每个 seed 的最终头差值 ≥ −0.75pp；
3. 每个 seed 的 transfer 总体 accuracy drop 相对最终头 ≤ 0；
4. 每个 seed 的 transfer balanced accuracy drop ≤ 0；
5. 每个 seed 的 transfer worst-class accuracy drop ≤ 0；
6. 每个 seed 的早退率在 15%–95%；
7. 每个 seed 的 MAC 代理节省 ≥ 15%。

这些是有限 5,000 样本上的经验 gate，不是分布级零风险保证，也不做显著性保证。结果同时记录
策略改变的决策数、伤害数和挽救数，以免只看聚合 accuracy 掩盖抵消。

## 5. 完成后的唯一处理顺序

1. 确认无训练/GPU残留进程，只选择唯一 `cifar10_early_exit_p2_transfer_serial_p2a_*`
   completed manifest 和对应 launcher log。
2. 用 `scripts/analysis/audit_early_exit_p2.py` 生成 ignored immutable snapshot 与版本化审计；
   核对 200 连续 epochs、清单/配置/summary、best/latest/final 哈希、固定互斥划分、源码快照、
   串行时间线、无 test 和无预测文件。
3. 只有审计 `issues={}` 才运行 `scripts/analysis/analyze_early_exit_p2_transfer.py`；它必须校验
   source selection 哈希、audit/manifest 哈希及 P1/P2 split fingerprint。
4. gate 失败则停止并归档；gate 通过则哈希锁定结果，之后再实现一次性 CIFAR-10.1 v6 evaluator。
5. 证据固化后才能删除 P2 的 `epoch_*.pth` 周期优化器快照；必须保留 best/latest/final、日志、
   配置、split、provenance、manifest/source snapshot、审计和分析输出，并写清理回执。

外部数据预先指定为 CIFAR-10.1 v6，因为它包含严格类别均衡的 2,000 张新图，便于预注册
worst-class 指标。P2 结束前不得根据其结果改成其他版本。外部评估是否执行，只由 P2 gate 决定。

## 6. 实现入口与启动

- 配置：`configs/experiments/early_exit_p2_{baseline,multi}.yaml`
- Sweep：`configs/sweeps/early_exit_p2.yaml`
- 后台启动：`scripts/launch_early_exit_p2.py`，默认 tag=`p2a`、`jobs=1`
- 完成审计：`scripts/analysis/audit_early_exit_p2.py`
- 冻结迁移分析：`scripts/analysis/analyze_early_exit_p2_transfer.py`

启动器自行建立后台 session、写 launcher log 并立即返回 PID，无需再包 `nohup`。预计与 P1b
相同，约 2.2 小时。运行期间不要修改 `src/`、`scripts/` 或 `configs/`，源码指纹变化会让 runner
拒绝混合版本继续执行。

唯一启动命令：

```bash
cd /root/autodl-tmp/image-classification && /root/miniconda3/bin/python scripts/launch_early_exit_p2.py
```
