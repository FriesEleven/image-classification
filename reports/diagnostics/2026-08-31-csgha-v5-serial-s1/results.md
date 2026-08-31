# CSGHA v5 / matched control 诊断结果

完整保存的5,000张validation；无训练、无test。全部checkpoint strict load并精确复现P1 best accuracy。

## v5 guidance干预

| Seed | 原始 | Guidance置零 | 训练均值描述 | 训练均值加项 | 三次置换均值 | 置换变化 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 87.62 | 86.78 | 86.26 | 86.98 | 85.72 | -1.90 pp |
| 43 | 87.18 | 86.48 | 85.64 | 86.34 | 85.26 | -1.92 pp |
| 44 | 88.14 | 87.32 | 86.74 | 87.34 | 86.80 | -1.34 pp |

三个训练seed的置换效应见上表，跨seed平均为 -1.72 pp。

## Deep置零

| Seed | v4变化 | Matched control变化 |
|---:|---:|---:|
| 42 | -1.28 pp | -16.40 pp |
| 43 | -1.42 pp | -8.84 pp |
| 44 | -1.60 pp | -6.16 pp |

v5与control的两个目标block在完整validation上deep logits零值比例均为0。
Deep-zero变化证明分支参与预测，但不能被直接解释为因果贡献比例。

## 分支统计

| 模型 | Seed | Block | Deep |Guidance|均值 | Guidance tanh饱和 | Gate跨样本std |
|---|---:|---:|---:|---:|---:|
| csgha_v5 | 42 | 7 | 1.419 | 0.559 | 0.41% | 0.2056 |
| csgha_v5 | 42 | 8 | 0.397 | 0.565 | 0.52% | 0.1316 |
| csgha_v5 | 43 | 7 | 1.256 | 0.576 | 0.55% | 0.2104 |
| csgha_v5 | 43 | 8 | 0.455 | 0.574 | 0.29% | 0.1417 |
| csgha_v5 | 44 | 7 | 1.199 | 0.566 | 0.43% | 0.1892 |
| csgha_v5 | 44 | 8 | 3.253 | 0.465 | 1.76% | 0.1236 |
| hybrid_leaky | 42 | 7 | 5.547 | — | — | 0.2628 |
| hybrid_leaky | 42 | 8 | 32.543 | — | — | 0.1555 |
| hybrid_leaky | 43 | 7 | 5.172 | — | — | 0.3336 |
| hybrid_leaky | 43 | 8 | 11.438 | — | — | 0.1503 |
| hybrid_leaky | 44 | 7 | 5.793 | — | — | 0.3313 |
| hybrid_leaky | 44 | 8 | 19.182 | — | — | 0.1349 |

Deep与guidance绝对值不是可相加的性能贡献率；checkpoint干预也不同于从头训练消融。
原始manifest SHA-256：`7ac860f1a56d2ce96c8b9594fb515ff88bf8b3e02edb881598c85c6731f4aee3`。
