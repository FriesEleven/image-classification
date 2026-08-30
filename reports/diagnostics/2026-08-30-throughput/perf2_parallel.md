# GPU利用率优化：perf2两路并行

## 交付

服务器项目根目录一行后台启动：

```bash
/root/miniconda3/bin/python scripts/launch_csgha_v4.py
```

共六组：独立控制与CSGHA v4各seeds42/43/44，200epochs，最多同时运行两组，完成一组自动补位。实验编号新增`_perf2_seed…`，旧epoch100中断结果和perf1短测均保留；本次只运行有限性能/正确性短测，**正式六组尚未启动**。

每组仍为batch128、梯度累积1、8个data workers、prefetch4、相同AdamW/OneCycleLR、lr0.01、相同增强与45k/5k划分。保持前版训练CUDA Graph和AMP/no-cache；每进程PyTorch CPU线程设为1。没有启用fused AdamW，也没有扩大batch或把增强换成不同的GPU实现。

显存不是GPU忙碌程度。32×32 CIFAR图像和MobileNetV2本来就小，当前限制还有CPU图像增强、图外调度和验证。GPU空闲显存没有必要强行填满；改batch还会改变BatchNorm统计和每轮optimizer更新次数，不属于无影响的硬件开关。

## 实测选择依据

硬件：RTX3080Ti 12GB，Python3.12.3，PyTorch2.8.0+cu128；容器CPU quota为1,200,000/100,000，即12核。所有测试均不评估官方test。

`scripts/diagnostics/benchmark_gpu_pipeline.py` 使用200轮OneCycle日程，仅执行前3轮。下表取第2–3轮、包含45k训练和5k验证，不含checkpoint写入。多路时报告各进程均值的平均，不把并发单组耗时当成串行回退。

| 设置 | 每组稳定epoch秒 | 平均秒/并发数（仅吞吐近似） | 峰值显存MiB |
|---|---:|---:|---:|
| 单路、8workers、1线程 | 5.37 | 5.37 | 709 |
| 单路、12workers、1线程 | 4.68 | 4.68 | 709 |
| **两路、各8workers、1线程** | **8.48** | **4.24** | **1414** |
| 两路、各6workers、1线程 | 8.36 | 4.18 | 1414 |
| 三路、各6workers、1线程 | 12.36 | 4.12 | 2119 |

三路组内耗时不均（11.40–12.87秒），部分后段已不是三路同时活动，平均值/3会高估稳态收益；相对两路未显示可靠额外收益。两路6workers与8workers差异很小；选择两路8workers，保留原来的worker分配及其增强随机序列。

缓存64batch比较：默认AdamW平均0.5446秒，fused为0.5276秒，约3%改善，但训练数值轨迹不同；完整数据仍需约5.4秒/轮，没有采用。PyTorch对[fused优化器的说明](https://docs.pytorch.org/docs/2.9/generated/torch.optim.AdamW.html)只是一般性能预期，最终以本机实测为准。

GPU每秒遥测（含初始化/验证/退出的完整短测）总体利用率均值：单路8workers为18.3%，两路8workers为31.4%，三路6workers为36.4%。若仅保留利用率≥20%的活动采样，均值分别38.8%、54.1%、51.5%；这不是整个训练平均，也不是稳定GPU占用率保证。两路峰值96%只是瞬时值，不能宣称持续96%。

## 使用最终训练引擎的公平复验

真正的`run_baselines.py --jobs 2`运行两个3轮smoke，完整执行训练、验证、TensorBoard、best/latest/final checkpoint和manifest写入，再对每个模型以相同配置串行重跑。这里总OneCycle长度为3轮，和前面的性能probe不同，**不能把这些3轮精度用于方法筛选**。

| 模型 | 串行epoch1/2/3秒 | 并行epoch1/2/3秒 | 最终state最大差异 | 完整training.csv |
|---|---|---|---:|---|
| 独立控制 | 8.89 / 5.33 / 5.35 | 13.66 / 9.53 / 9.53 | 0 | 字节一致 |
| CSGHA v4 | 9.85 / 5.38 / 5.54 | 14.00 / 9.90 / 9.53 | 0 | 字节一致 |

稳定阶段，一组控制加一组v4串行需要约10.80秒，并行约9.72秒，吞吐提升约11%。不能把单控制probe的约27%加速直接当作正式混合队列收益。两次串行子进程总耗时约64.68秒，双任务队列约46秒；这项短任务约29%的节时还包含冷启动重叠，不能线性外推为200轮节时。

按当前稳定耗时粗估，六组200轮约1.5–1.8小时，受服务器负载、不同seed及checkpoint保存影响；不是完成时间承诺。比最早约13.3秒/epoch、完全串行的版本已有明显改善，但GPU仍不会持续100%。

两种模型并行运行的最后权重与串行逐参数一致（包括BN buffers），整个训练CSV字节一致；相应best validation为35.94%和29.82%，只用来校验3轮复现。验证和test选择协议未改变。

## 启动、日志与安全

- `launch_csgha_v4.py --dry-run`打印6个新ID与两路并发设置，不训练。
- 顶层日志打印启动PID、各子任务日志路径，并转发每epoch摘要，不记录逐batch进度。
- 每组的完整stdout单独保存于该sweep的`run_logs/`，weights和training.csv仍在各自run目录。
- manifest记录每组PID、独立进程组、开始/结束时间、返回码与summary。异常默认停止其他在跑的所属任务，不启动剩余任务；退出清理仅面向本队列创建的进程组，不涉及Jupyter/VSCode/TensorBoard等服务。
- runner支持SIGINT/SIGTERM清理子任务与data workers，锁与已有进程/目录检查防止重复启动和覆盖。
- 并发时`measure_inference: false`，`benchmark.json`明确写`skipped`和null，不把受竞争污染的延迟当成论文数据；正式效率图后续需GPU独占时统一测量。
- 启动时保存源码快照和配置，运行过程中检查源码指纹。perf2与历史eager/perf1仍是不同执行批次；本轮逐参数一致性结论仅指本轮同配置串行/并行复验。

## 证据位置

- 最终服务器测试：75 passed、3 subtests passed，只有首次cuBLAS context初始化警告；相关文件Ruff、全项目语法编译、启动器dry-run通过。
- 本地与远端73个源码/配置文件逐文件SHA-256一致。
- 队列smoke：`artifacts/sweeps/smoke_perf2_queue_20260830_214703/manifest.json`，状态completed，两个子任务均completed。
- 吞吐probe、遥测：`artifacts/diagnostics/perf2_*`，本地和服务器均保留。
- 完整串并行一致性回执：`artifacts/diagnostics/perf2_serial_parallel_equality.json`。
- 筛选出的机器可读数字、原始文件校验与最终源码指纹：[perf2_evidence.json](perf2_evidence.json)。
