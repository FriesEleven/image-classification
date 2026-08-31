# CSGHA v4 / matched control P2诊断结果

完整保存的5,000张validation；无训练、无test。全部checkpoint strict load并精确复现P1 best accuracy。

## v4 guidance干预

| Seed | 原始 | Guidance置零 | 训练均值描述 | 训练均值加项 | 三次置换均值 | 置换变化 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 88.44 | 87.78 | 87.50 | 87.88 | 87.51 | -0.93 pp |
| 43 | 87.90 | 87.92 | 87.42 | 87.66 | 87.41 | -0.49 pp |
| 44 | 87.90 | 87.58 | 87.78 | 87.68 | 87.64 | -0.26 pp |

三个训练seed的置换平均变化均为负，跨seed平均为 -0.56 pp。

## Deep置零

| Seed | v4变化 | Matched control变化 |
|---:|---:|---:|
| 42 | -3.14 pp | -16.40 pp |
| 43 | -2.74 pp | -8.84 pp |
| 44 | -3.16 pp | -6.16 pp |

v4与control的两个目标block在完整validation上deep logits零值比例均为0，LeakyReLU消除了v3观察到的硬失活。
v4的deep-zero下降证明deep分支参与预测；它小于control的下降不能被直接解释为因果贡献比例。

## 分支统计

| 模型 | Seed | Block | Deep |Guidance|均值 | Guidance tanh饱和 | Gate跨样本std |
|---|---:|---:|---:|---:|---:|
| csgha_v4 | 42 | 7 | 6.103 | 0.988 | 96.45% | 0.2255 |
| csgha_v4 | 42 | 8 | 2.997 | 0.992 | 96.94% | 0.1403 |
| csgha_v4 | 43 | 7 | 4.250 | 0.990 | 97.15% | 0.2732 |
| csgha_v4 | 43 | 8 | 4.971 | 0.990 | 96.24% | 0.1841 |
| csgha_v4 | 44 | 7 | 4.498 | 0.914 | 96.98% | 0.2601 |
| csgha_v4 | 44 | 8 | 4.079 | 0.992 | 96.63% | 0.1720 |
| hybrid_leaky | 42 | 7 | 5.547 | — | — | 0.2628 |
| hybrid_leaky | 42 | 8 | 32.543 | — | — | 0.1555 |
| hybrid_leaky | 43 | 7 | 5.172 | — | — | 0.3336 |
| hybrid_leaky | 43 | 8 | 11.438 | — | — | 0.1503 |
| hybrid_leaky | 44 | 7 | 5.793 | — | — | 0.3313 |
| hybrid_leaky | 44 | 8 | 19.182 | — | — | 0.1349 |

Deep与guidance绝对值不是可相加的性能贡献率；checkpoint干预也不同于从头训练消融。
原始manifest SHA-256：`abf1a62f4cf5b067c2f1248ebac73fefbbd34789d1414f48d183f774cc7d1e9d`。
