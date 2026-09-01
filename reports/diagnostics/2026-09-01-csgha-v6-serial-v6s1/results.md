# CSGHA v6 / matched control 诊断结果

完整保存的5,000张validation；无训练、无test。全部checkpoint strict load并精确复现P1 best accuracy。

## v6 guidance干预

| Seed | 原始 | Guidance置零 | 训练均值描述 | 训练均值加项 | 三次置换均值 | 置换变化 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 87.76 | 87.54 | 87.66 | 87.70 | 87.55 | -0.21 pp |
| 43 | 87.68 | 87.44 | 87.42 | 87.52 | 87.39 | -0.29 pp |
| 44 | 87.92 | 87.84 | 87.58 | 87.80 | 87.69 | -0.23 pp |

三个训练seed的置换效应见上表，跨seed平均为 -0.24 pp；guidance置零跨seed平均为 -0.18 pp。

## Deep置零

| Seed | v6变化 | Matched control变化 |
|---:|---:|---:|
| 42 | -0.78 pp | -16.40 pp |
| 43 | -1.16 pp | -8.84 pp |
| 44 | -2.72 pp | -6.16 pp |

v6与control的两个目标block在完整validation上deep logits零值比例均为0。
Deep-zero变化证明分支参与预测，但不能被直接解释为因果贡献比例。

## 分支统计

| 模型 | Seed | Block | Deep |Guidance|均值 | Guidance tanh饱和 | Gate跨样本std |
|---|---:|---:|---:|---:|---:|
| csgha_v6 | 42 | 7 | 2.522 | 0.138 | 0.64% | 0.2122 |
| csgha_v6 | 42 | 8 | 0.355 | 0.141 | 0.34% | 0.0648 |
| csgha_v6 | 43 | 7 | 1.762 | 0.144 | 0.37% | 0.2180 |
| csgha_v6 | 43 | 8 | 0.508 | 0.142 | 0.56% | 0.0748 |
| csgha_v6 | 44 | 7 | 2.752 | 0.143 | 0.36% | 0.2808 |
| csgha_v6 | 44 | 8 | 4.156 | 0.124 | 1.12% | 0.1508 |
| hybrid_leaky | 42 | 7 | 5.547 | — | — | 0.2628 |
| hybrid_leaky | 42 | 8 | 32.543 | — | — | 0.1555 |
| hybrid_leaky | 43 | 7 | 5.172 | — | — | 0.3336 |
| hybrid_leaky | 43 | 8 | 11.438 | — | — | 0.1503 |
| hybrid_leaky | 44 | 7 | 5.793 | — | — | 0.3313 |
| hybrid_leaky | 44 | 8 | 19.182 | — | — | 0.1349 |

Deep与guidance绝对值不是可相加的性能贡献率；checkpoint干预也不同于从头训练消融。
原始manifest SHA-256：`83453bc7b9088f403b25a114d166c4803c48449b983f72908e56f870cfbe0313`。
