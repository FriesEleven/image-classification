# Early-exit P0 analysis

Decision: **stop_or_redesign**

| seed | baseline % | final % | final gain pp | exit8 % | exit16 % | holdout MAC saving % | policy drop pp | worst-class drop pp |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 51 | 87.96 | 88.62 | +0.66 | 88.14 | 88.56 | 56.30 | +0.72 | +4.00 |
| 52 | 88.16 | 88.46 | +0.30 | 88.18 | 88.30 | 56.30 | +0.28 | +3.60 |
| 53 | 88.16 | 88.30 | +0.14 | 88.10 | 87.88 | 56.25 | +0.32 | +1.20 |

Paired final gain: +0.367 ± 0.266 pp (sample std).
Holdout MAC saving: 56.28 ± 0.03%.

## Frozen go/no-go gates

- [x] `mean_final_gain_at_least_minus_0_003`
- [x] `each_final_gain_at_least_minus_0_0075`
- [x] `each_holdout_mac_saving_at_least_0_15`
- [x] `each_holdout_accuracy_drop_at_most_0_01`
- [ ] `each_holdout_worst_class_drop_at_most_0_03`

This is exploratory validation-only evidence. The 5k parent validation set also selected the best checkpoint, so its 50/50 child split is not independent paper evidence. The official CIFAR-10 test set remains untouched.

Next: Do not expand early-exit training; inspect the failed gate before redesign.
