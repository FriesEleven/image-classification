# 预算约束下的阶段感知稀疏注意力部署：执行计划

> 2026-09-02 状态：本计划已经完成并触发停止规则。30/30 次 probe1 均完成；九个注意力单元的三 seed 配对平均增益全部为负，冻结选择器在 ultra-light、balanced、relaxed 三档预算均选择 all-none。不得启动原计划 seeds48/49/50 确认批次，也不得据此扩展 CIFAR-100、第二 backbone 或 Coordinate Attention。正式归档见 `reports/negative_results/2026-09-02-budget-stage-probe1/`，后续执行方向见 `docs/early_exit_p0_plan.md`。

## 1. 研究问题

目标不再是继续修改 CSGHA，而是回答一个可部署且可复现的问题：在给定参数、阶段相关运算量和目标硬件延迟预算时，MobileNetV2 的浅层、中层和深层分别应该不放注意力，还是放 ECA、SE 或 CBAM？

v1 搜索空间固定为三个阶段包：

| 阶段 | torchvision MobileNetV2 `features` 索引 | 选择 |
|---|---|---|
| shallow | 1、2 | None / ECA / SE / CBAM |
| middle | 7、8 | None / ECA / SE / CBAM |
| deep | 15、16 | None / ECA / SE / CBAM |

每个阶段最多选择一种注意力，完整空间为 `4^3=64` 个架构。各注意力均独立作用于本阶段特征，不传跨阶段描述。Coordinate Attention 暂不进入首轮：先用仓库已有且可严格匹配的三类模块验证“预算选择”主张，避免把模块扩张和选择机制混成一个实验变量；若选择器通过确认实验，再把它作为统一协议的外部候选补入。

## 2. 选择目标

对阶段 `s`、模块 `a` 和配对 seed `i`，先训练只启用该单元的探测模型，并计算相对 all-none 成员的 best-validation 增益：

```text
delta[s,a,i] = val_acc[s,a,i] - val_acc[none,i]
mu[s,a]      = mean_i(delta[s,a,i])
sigma[s,a]   = sample_std_i(delta[s,a,i])
```

候选架构 `x` 的校准代理分数为单元均值之和；风险调整分数为：

```text
utility(x) = sum(mu[s,a]) - beta * sqrt(sum(sigma[s,a]^2))
```

默认 `beta=0.5`。选择器在每组预算下穷举全部64个候选并最大化 `utility(x)`，同时约束：

- 相对 all-none 的精确参数增量；
- 根据真实阶段特征尺寸计算的 attention-only 运算代理；
- 空闲 GPU 上、batch=1、32×32 输入的多轮实测延迟增量；
- 激活注意力的阶段数。

all-none 始终是可行候选，风险调整收益全部为负时必须返回它。这一规则防止为了“有新方法”而强行选择注意力。

单元增益的可加性只是低成本搜索代理，不是多阶段精度结论。被选架构必须从头训练确认，不能把代理预测写入论文主结果。

## 3. 最小实验矩阵

### A. 校准探测（当前待运行）

10个成员：all-none，以及 ECA/SE/CBAM × shallow/middle/deep；每个成员使用配对开发 seeds 45/46/47，共30次训练。

- CIFAR-10 45k/5k 分层划分，200 epochs；
- batch128、AdamW、OneCycleLR、AMP、CUDA Graph，与最近正式协议一致；
- `evaluate_test=false`、`measure_inference=false`；
- 单进程串行，避免此前两路 CUDA Graph+AMP 的随机原生 `SIGABRT`；
- 全部新 `probe1` ID，从头训练，不复用 CSGHA 或旧位置实验。

30次看似多，但它是得到九个单元三 seed 配对均值和方差的最小完整矩阵；删掉任一项便无法对相应阶段/模块做同等证据的选择。

### B. 选择后确认（A完成后生成固定清单）

对三档预算去重后的选择结果，加 matched all-none，使用冻结后的新 seeds 48/49/50 从头训练、validation-only。若多档预算选中同一架构，只训练一次。确认阶段报告真实配对差值，不使用加法预测替代。

### C. 主张成立后再扩展

只有至少一个预算下的选择在确认 seeds 上表现出可复现的效用—成本优势，才进入 CIFAR-100、第二轻量 backbone、Coordinate Attention 对照和最终一次 test。RTX 3080 Ti 延迟只是当前服务器的硬件代理；“部署”主张最终必须在明确的目标设备上重新剖析，不能把桌面 GPU 延迟冒充手机延迟。

## 4. 实现入口

- 统一模型超族：`model_type=stage_sparse`，实现于 `src/image_classification/models/mobilenetv2.py`。
- 64候选枚举、阶段运算代理和约束选择：`src/image_classification/selection/budget.py`。
- 空闲硬件剖析：`scripts/diagnostics/profile_stage_sparse_candidates.py`。
- 校准配置与清单：`configs/experiments/budget_probe_*.yaml`、`configs/sweeps/budget_stage_probe.yaml`。
- 唯一校准启动器：`scripts/launch_budget_stage_probe.py`，固定 `jobs=1`、默认 tag `probe1`。
- 完成后选择入口：`scripts/analysis/select_budget_stage_attention.py`。
- 默认三档预算：`configs/budgets/stage_sparse_mobilenetv2.yaml`；论文中必须报告具体数值，不能只写“低/中/高”。

## 5. 成功、停止与防泄漏规则

1. 校准和确认均只看validation；方法、预算和候选全部冻结后才能做最终test。
2. 首轮选择必须报告九个单元全部三 seed 配对差值、均值、样本标准差和胜出数。
3. 若校准选择 all-none，或确认后 selected 相对 matched all-none 没有稳定优势，保留为负结果并停止扩大矩阵。
4. 若精度代理有效但 RTX 延迟排序不稳定，先增加空闲剖析轮次，不重训模型来迎合延迟噪声。
5. 若确认成功，下一阶段优先补跨数据集和第二 backbone，而不是继续搜索更多位置或超参数。
