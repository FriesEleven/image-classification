# CSGHA v5 serial-s1 正式审计

本报告只由 serial-s1 completed manifest 的六个 run 生成；未收集 rms1/rms2/rms3 失败/中断目录或 test 结果。

## 核心结果

| Seed | Matched control | CSGHA v5 | v5 − control |
|---:|---:|---:|---:|
| 42 | 87.88% | 87.62% | -0.26 pp |
| 43 | 87.82% | 87.18% | -0.64 pp |
| 44 | 88.12% | 88.14% | +0.02 pp |
| 均值 ± 样本标准差 | 87.94 ± 0.16% | 87.65 ± 0.48% | -0.293 ± 0.331 pp |

v5 胜出 1/3；配对均值为负，不能声称优于matched control。

## 完整性与协议

六组均连续记录200 epochs；summary与首次best一致；best/latest/final checkpoint、划分哈希和source snapshot通过。
同seed两模型使用相同45k/5k划分；CUDA Graph、AMP no-cache、单线程和validation-only协议一致，未生成test预测。

Control参数量 2,238,024；v5参数量 2,239,178（+1,154）。FLOPs为项目解析估计，不是profiler实测。

## 历史参照边界

旧v3、Independent middle、Independent shallow来自2026-08-30审计，只作历史参照；本批control与v4 retry1 control的三个best checkpoint哈希完全一致。

## 下一步

进入版本匹配checkpoint诊断：先严格复现完整5k validation，再做guidance置零/置换/训练均值及deep置零干预，确认RMS是否降低饱和以及输入相关贡献是否被削弱。

Manifest SHA-256：`6ea44ff9ce47fd8682667419ac279c993cfbe3801bbc977e0ddd396d3100fea2`；旧审计SHA-256：`117c808ffac4c5bc8c03204ff88dbbe3f0ba9b4ed3d5a2989bc3035e1ddb252d`。
