# CSGHA v6 serial-v6s1 正式审计

本报告只由 serial-v6s1 completed manifest 的六个 run 生成；未收集旧 CSGHA 批次或 test 结果。

## 核心结果

| Seed | Matched control | CSGHA v6 | v6 − control |
|---:|---:|---:|---:|
| 42 | 87.88% | 87.76% | -0.12 pp |
| 43 | 87.82% | 87.68% | -0.14 pp |
| 44 | 88.12% | 87.92% | -0.20 pp |
| 均值 ± 样本标准差 | 87.94 ± 0.16% | 87.79 ± 0.12% | -0.153 ± 0.042 pp |

v6 胜出 0/3；三个seed均为负，不能声称优于matched control。

## 完整性与协议

六组均连续记录200 epochs；summary与首次best一致；best/latest/final checkpoint、划分哈希和source snapshot通过。
同seed两模型使用相同45k/5k划分；CUDA Graph、AMP no-cache、单线程和validation-only协议一致，未生成test预测。

Control参数量 2,238,024；v6参数量 2,239,178（+1,154）。FLOPs为项目解析估计，不是profiler实测。

## 历史参照边界

旧v3、Independent middle、Independent shallow来自2026-08-30审计，只作历史参照；本批control与v5 serial-s1 control的三个best checkpoint哈希完全一致。

## 下一步

进入版本匹配checkpoint诊断：严格复现完整5k validation，再做guidance置零/置换/训练均值及deep置零干预，确认±0.25上限、deep分支恢复程度和逐样本纠错/破坏模式。

Manifest SHA-256：`44b5903955ea8b2aa7a68b1887d726dc0a06c632d85235092f8df4f3ab0866bf`；旧审计SHA-256：`117c808ffac4c5bc8c03204ff88dbbe3f0ba9b4ed3d5a2989bc3035e1ddb252d`。
