# Early-exit P0a 正式审计

本报告只使用唯一 completed p0a manifest 的六组 validation-only run。

| Seed | MobileNetV2 | Multi-exit final | 配对差值 |
|---:|---:|---:|---:|
| 51 | 87.96% | 88.62% | +0.66 pp |
| 52 | 88.16% | 88.46% | +0.30 pp |
| 53 | 88.16% | 88.30% | +0.14 pp |
| 均值 ± 样本标准差 | 88.093 ± 0.115% | 88.460 ± 0.160% | +0.367 ± 0.266 pp |

六组均连续记录200 epochs，summary、首次best、best/latest/final、划分哈希及源码快照一致。
同seed两模型使用相同45k/5k划分；未评估test、未导出test预测、未记录受争用延迟。

Manifest SHA-256：`49a40fc1afb007d7a357c4bc54fd4b96f7618cebaf6a290955bfb8fe19327cd6`。
