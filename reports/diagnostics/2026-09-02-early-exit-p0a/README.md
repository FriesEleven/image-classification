# Early-exit P0a 策略诊断

本目录固定保存 P0a 的两个不同阶段，不能把后验诊断改写成预注册成功结果。

- `frozen_policy_analysis.json/.md`：原冻结 exit8→exit16→final 规则；结论为 `stop_or_redesign`。总体精度与 MAC gate 通过，但 seed51/52 最差类别下降 4.0/3.6pp，超过 3.0pp。
- `shared_threshold_diagnostic.json`：原 gate 失败后进行的机制定位。搜索全部 1,024 个预测类别保护子集和 43 个跨 seed 共享阈值，得到 exit8 阈值 `0.9410777530842098`、无保护类别、fallback final。三个检查子集约 75% 样本早退、MAC 代理节省约 42%–43%，经验总体/均衡/最差类别下降均为 0。

第二项使用了父级 validation 选出的 checkpoint，又是在查看原失败后设计，故只支持 P1 重设计，不属于独立论文证据。官方 CIFAR-10 test 未评估。

关键 SHA-256：

- 原 manifest：`49a40fc1afb007d7a357c4bc54fd4b96f7618cebaf6a290955bfb8fe19327cd6`
- 冻结策略分析 JSON：`015ff5d68a0ceaf2d21b00e2ab3450567c580f63f027622a969392a776378e68`
- 共享阈值诊断 JSON：`443345f5bb4de678a6e3757bcb8835ca803cd89f4dc235c2a621c801219b2f5e`
- 正式审计 JSON：`00fa166543f60771838516a143f5689961fb0917f278506127e36b8ed9ea991c`
