# Early-exit P1b 正式审计

本报告只使用唯一 completed P1b manifest 的六组 validation/calibration-only run。

| Seed | MobileNetV2 validation | Multi-exit final validation | 配对差值 |
|---:|---:|---:|---:|
| 54 | 87.36% | 88.20% | +0.84 pp |
| 55 | 87.10% | 87.88% | +0.78 pp |
| 56 | 87.36% | 87.46% | +0.10 pp |
| 均值 ± 样本标准差 | 87.273 ± 0.150% | 87.847 ± 0.371% | +0.573 ± 0.411 pp |

六组均连续记录 200 epochs；清单、summary、首次 best、best/latest/final、
40k/5k/5k 互斥划分、执行溯源与源码快照全部通过文件级核查。
所有 run 使用同一 split seed，且同 seed 两模型的原始划分文件完全一致。
官方 CIFAR-10 test 未评估、无预测文件、受争用训练期间未记录推理延迟。

总串行时长：`2.166` 小时。
Manifest SHA-256：`1545e85651103400c425bf3eaae0e70e4d1a4d0e413203158d2f6b6ff16c8c88`。
