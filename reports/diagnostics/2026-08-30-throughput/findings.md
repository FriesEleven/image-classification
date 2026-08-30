# 训练吞吐优化与安全重启（2026-08-30）

后续更新：当前启动器已升级为perf2两路并行，详见 [最新部署与完整对照](perf2_parallel.md)。本报告保留perf1历史测量，下面的perf1运行编号不再是启动器的当前输出编号。

## 结论

已按用户要求停止旧 v4 sweep，保留全部日志和 checkpoint。新实现针对小模型的 CPU kernel 提交开销启用 **training-only CUDA Graph**，不是扩大 batch 或删减数据增强。

RTX 3080 Ti、Python 3.12.3、PyTorch 2.8.0+cu128 上，完整 CIFAR-10 45k train / 5k validation 短测的稳定 epoch 从控制模型约 **13.31s 降到 5.72s（2.33×）**。优化后的 CSGHA v4 约 **5.48s/epoch**。这些是三轮 smoke 的第 2–3 轮，不是 200 轮训练的速度保证，也不证明准确率改善。

正式六组尚未启动。服务器项目根目录执行：

```bash
/root/miniconda3/bin/python scripts/launch_csgha_v4.py
```

新批次为 `cifar10_csgha_v4_matched_perf1`，两个变体 × seeds 42/43/44，全部从头训练。所有实验 ID 新增 `_perf1_seed…`；不覆盖、拼接或续训已中断的旧结果。可追加 `--dry-run` 仅检查命令。

## 已停止的实验

- 原 manifest：`artifacts/sweeps/cifar10_csgha_v4_matched_20260830_204041/manifest.json`。
- 原主进程/进程组 35024、训练进程 35070 和同组 data-loader 子进程，经身份确认后整体 SIGINT，已确认退出。
- 中断时间：`2026-08-30T21:03:13+08:00`；控制组 seed42 最后完成 epoch100/200，best validation 85.36%。其余五组未开始。
- 原清单顶层为 `interrupted`。旧 runner 留下的首行 `running` 是未更新的历史字段，不代表进程仍存在；原 manifest 未改写。新 runner 中断时会同时更新正在执行的行。
- 原产物仍在原路径，未移动或删除；旧 source_snapshot 是本次冻结参考版本。

## 瓶颈与实施内容

原先连续采样利用率约 16–21%，训练进程主要占一颗 CPU 核。16 个 GPU-resident batch 的 profiler 记录 10,917 次 `cudaLaunchKernel` 和 1,408 次 `cuLaunchKernel`：小分辨率、小模型的 Python/CPU 提交开销明显。Profiler 自身带开销，不能把 CPU/CUDA 时间直接解释成精确的端到端百分比。

1. 指标延后到 epoch 末集中回传；保留原 loss 的 batch-mean 平均定义、概率与分类指标。后台日志仍只记录 epoch 摘要，交互进度每 50 batches 更新一次 loss。
2. 用 optimizer post-step hook 判断普通 AdamW 的实际更新，减少读取 AMP scale 的同步。溢出时仍不推进 scheduler；AMP-aware/fused optimizer 保留安全的 scale 回退路径。
3. 捕获完整 batch 的训练 forward/backward；optimizer、GradScaler、OneCycleLR 仍在图外。保持 batch128、梯度累积1、AdamW、lr0.01、数据增强、划分算法和200 epochs。
4. 捕获预热后原地恢复参数、BatchNorm buffers、CPU/CUDA RNG；CIFAR 最后72张样本走 eager，不补齐、不丢弃；validation/test 路径仍为 eager FP32。
5. 训练后端与 AMP cache 状态写入 provenance。普通配置 `cuda_graph` 默认 false，仅本次两个匹配配置启用；推理 benchmark 不使用训练图。

**只延后指标同步并没有带来可测的提速。** 真正收益来自 CUDA Graph；不把“减少同步”本身当成已验证的性能结论。

## 性能证据

同一批缓存的64个真实增强 batch，共8192张，ABBA顺序、每次3轮，取后2轮均值；不包含数据加载、validation、checkpoint I/O。

| 对比 | Eager平均秒 | 优化平均秒 | 结论 |
|---|---:|---:|---|
| 冻结旧循环 vs 延后指标同步 | 2.1624 | 2.1689 | 无明显提速，参数逐位一致 |
| 冻结旧循环 vs Graph | 2.1509 | 0.5594 | 约3.85×，但AMP缓存不同，参数轨迹不相同 |
| 同样关闭AMP缓存的 eager vs Graph | 2.1100 | 0.5485 | 约3.85×，192 steps 后控制模型全部 state_dict 逐位一致 |

后一个对比四次均为 scheduler188次更新、最终AMP scale4096，说明192 batches中的4次溢出跳步也一致。机器数据里的 `implementation: legacy` 在 `reference_no_cache: true` 时指当前 eager/no-cache 循环，而不是冻结旧循环。

完整数据 smoke：每轮45,000训练样本、5,000验证样本，包含数据增强、validation与日志/checkpoint保存，不评估test。

| 变体 | Epoch1秒 | Epoch2秒 | Epoch3秒 | 后两轮均值秒 |
|---|---:|---:|---:|---:|
| 控制组 eager、旧AMP缓存 | 20.82 | 13.31 | 13.31 | 13.31 |
| 控制组 Graph | 9.60 | 5.64 | 5.79 | 5.715 |
| CSGHA v4 Graph | 9.51 | 5.62 | 5.34 | 5.48 |

Graph首次准备分别3.88/3.84s，发生在epoch计时之前。三个完整子进程总耗时分别56.43/33.82/33.29s，包含导入、初始化、图准备与退出，不能用稳定epoch速度代替冷启动耗时。3轮 smoke 将 OneCycle 总长度设为3轮，仅用于功能和吞吐检查，不是正式模型筛选结果。

控制组Graph训练尾段4次连续遥测为SM利用率36/36/34/50%，显存697MiB；跨进程初始化期间有0%，因此不报告“整体利用率已达到50%”。数据增强、图外优化器、eager验证仍有开销，不以100%利用率作为成功标准。

## 数值行为与论文比较边界

PyTorch 的 [make_graphed_callables 官方说明](https://docs.pytorch.org/docs/stable/generated/torch.cuda.graphs.make_graphed_callables.html) 要求 AMP `cache_enabled=False`。本实现遵守该要求，不绕过限制。

共享 CBAM MLP 的 avg/max 分支在关闭权重缓存后，梯度累加的浮点舍入可改变。单步隔离检查中，forward/loss/RNG一致，但相对旧缓存模式4个共享FC权重梯度最大差异约1.22e-4；Graph与eager/no-cache的单步梯度完全一致。经过192步，对旧缓存训练的最大state差异已达6.4115，不能声称旧模型训练轨迹逐位等价。

两种模型6个完整/不足整批交替训练步骤的回归检查中，Graph捕获前后参数、BN与RNG恢复一致；最终RNG、AMP scale、scheduler更新次数、预测和验证概率一致。控制模型state逐位一致；v4只有两个guidance scalar存在约8.12e-9/1.98e-9差异，其他state逐位一致。测试仅对这两个scalar允许1e-7绝对误差，不对所有参数宽泛放松阈值。

因此：

- 新v4和匹配独立控制必须都用Graph/no-cache、同一源码快照，从头运行全部三seeds。
- 不混入旧eager/cache运行，也不从旧epoch100 checkpoint续训。
- 旧v3与新v4存在执行后端差异，不能仅用跨批次精度差把因果改善归因于LeakyReLU；若要作激活函数因果结论，还需在同一后端重跑ReLU版本。
- 本轮不修改模型机制、学习率或数据划分，不评估官方test。历史guidance干预结论仍见 [版本匹配诊断](../2026-08-30-guidance/findings.md)。

## 验证与可追溯性

- GPU服务器 `python -m pytest -q`：67 passed、3 subtests passed；只有首次cuBLAS context初始化警告。
- `scripts/diagnostics/check_model.py`：9种模型/数据集组合输出维度检查通过。
- 语法编译、此次优化文件Ruff检查通过；本地 `git diff --check` 通过。
- 启动器dry-run通过：6个新ID、validation-only、Graph后端。
- 本地/远端71个Python源码、YAML配置和pyproject逐文件SHA-256一致。
- 三个smoke的划分文件SHA-256均为 `25d918f697c7902c53b19a033fb1f9be2ab4b277abcc0dad7d6fd11dc73f3be4`。
- 两个Graph smoke的best checkpoint已在全新普通模型上strict load，使用完整5k validation/eager FP32分别复现39.18%和31.54%，与各自summary完全相同；只证明保存/恢复正确，不用3轮精度评价方法。回执为 `artifacts/diagnostics/training_perf1_checkpoint_replay.json`。

原始JSON与日志存放在本地和远端的 `artifacts/diagnostics/training_perf1_*`，每份JSON包含配置与源码/依赖来源；原始日志不纳入版本控制。关键数字与原始文件校验值见 [evidence.json](evidence.json)。正式启动后runner另存完整source_snapshot并检查整个sweep期间源码不变。
