# CIFAR-100 P4 独立边界确认计划

## 决策背景

P3 的12组训练和文件级审计均成功，multi-exit最终头相对matched baseline在seeds60–65上全部胜出，
平均`+2.173±0.397 pp`。但原预注册策略要求三个source seed同时满足总体、balanced及100个类别中
最差类别经验下降均为0，并至少节省15% MAC；没有共享阈值满足全部条件。因此P3正式状态固定为
`stop_without_test`，不得打开P3 official test，也不得把后验策略写回P3。

只使用calibration的边界诊断发现：严格零风险时共享阈值`0.999`最多约11.8%早退、约6.7% MAC
节省；达到15% MAC门槛时，最低可见最差类别下降为4pp，即每类50张中的2张。source-only选出的
阈值`0.903`在source60–62和未参与选择的target63–65上均约40%早退、约22.8% MAC节省，总体和
balanced准确率反而提高约0.70–1.18pp，最差类别下降不超过4pp。预测类别保护在固定0.005阈值
粗网格的严格零风险搜索中没有可行解；允许一张时又不能迁移到target，因此P4仍使用全局阈值、
无类别保护。

这些都是看到P3失败后的机制诊断，不是独立证据。为避免把后验发现当结论，P4使用新模型版本和
新数据划分做一次最小确认。

## 冻结协议

- 数据：CIFAR-100；全新固定`split_seed=20260904`，40k训练、5k checkpoint-selection validation、
  5k policy confirmation，三部分互斥并覆盖官方50k训练集。
- 训练seed：66/67/68；每个seed各训练原生MobileNetV2和相同multi-exit模型，共6组。
- 模型、exit8/exit16、蒸馏损失、AdamW、OneCycleLR、batch128、200 epochs、AMP、CUDA Graph、
  8 workers、prefetch8均与P3相同；`jobs=1`串行。
- 训练期间`evaluate_test=false`、`measure_inference=false`；policy confirmation loader不参与训练，
  official CIFAR-100 test不得迭代。
- 策略在训练前固定为exit8最大softmax阈值`0.903`、无类别保护、fallback final；P4候选阈值数为0，
  禁止按seed重校准。
- 完整来源与数值锁见
  `reports/experiments/2026-09-03-early-exit-p4-design/locked_policy.json`。

## 确认门槛

先核对最终头：六组中的三对模型，multi-exit相对baseline平均validation差值必须≥−0.30pp，每个
seed必须≥−0.75pp。再将冻结策略原样应用到三个policy confirmation集合；每个seed必须同时满足：

- 总体准确率下降≤0；balanced准确率下降≤0；
- 最差类别准确率下降≤4pp（该5k分层集合中每类50张，即最多2张净下降）；
- 早退率15%–95%；MAC代理节省≥15%。

任一门槛失败即`stop_without_test`，不在seed66–68上调阈值、不换seed重跑，也不打开official test。
全部通过才固化manifest、审计、确认结果、策略、六个best checkpoint和一次性评估器哈希并提交。

若开发门槛通过，official-test门槛预先固定为：每seed总体/balanced下降≤0.50pp、最差类别下降≤4pp、
早退15%–95%、MAC节省≥15%，三seed平均总体下降≤0.20pp。测试只允许一次；无论结果正负都不
调整后重跑。通过后停止新增训练并进入论文；失败则把CIFAR-100边界作为负结果报告。

## 启动与预计耗时

启动器必须在干净、已提交的代码上运行，拒绝已有训练进程、已有目标目录或重复锁。它自身创建后台
session，不能额外套`nohup`。6组串行按P3实测预计约1.6小时。

```bash
cd /root/autodl-tmp/image-classification && /root/miniconda3/bin/python scripts/launch_early_exit_p4.py
```
