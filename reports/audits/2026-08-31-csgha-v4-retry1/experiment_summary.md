# CSGHA v4 retry1 正式审计

本报告只由 retry1 completed manifest 的六个 run 生成；未收集 smoke、首批失败/取消目录或 test 结果。

## 核心结果

| Seed | Matched control | CSGHA v4 | v4 − control |
|---:|---:|---:|---:|
| 42 | 87.88% | 88.44% | +0.56 pp |
| 43 | 87.82% | 87.90% | +0.08 pp |
| 44 | 88.12% | 87.90% | -0.22 pp |
| 均值 ± 样本标准差 | 87.94 ± 0.16% | 88.08 ± 0.31% | +0.140 ± 0.393 pp |

v4 胜出 2/3；平均优势很小且 seed44 为负，不能声称稳定提升。

## 完整性与协议

六组均连续记录200 epochs；summary与首次best一致；best/latest/final checkpoint、划分哈希和source snapshot通过。
同seed两模型使用相同45k/5k划分；CUDA Graph、AMP no-cache、单线程和validation-only协议一致，未生成test预测。

Control参数量 2,238,024；v4参数量 2,239,178（+1,154）。FLOPs为项目解析估计，不是profiler实测。

## 历史参照边界

旧v3、Independent middle、Independent shallow来自2026-08-30审计，执行后端与retry1不同，只作历史参照；不能把跨批改善归因于LeakyReLU或guidance。

## 下一步

进入版本匹配checkpoint诊断：先严格复现完整5k validation，再做guidance置零/置换/训练均值及deep置零干预，同时统计v4与control的deep分支。

Manifest SHA-256：`020112beffbfef94cee7bafb0fc1a09b3b73f14988d3860edd25938cab1af988`；旧审计SHA-256：`117c808ffac4c5bc8c03204ff88dbbe3f0ba9b4ed3d5a2989bc3035e1ddb252d`。
