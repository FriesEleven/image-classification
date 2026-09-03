# Early-exit P2 frozen-policy transfer

Decision: **ready_for_external_shift_test**

P1b threshold `0.984` is applied unchanged to unseen target seeds 57/58/59.
No target-seed threshold candidate is evaluated or selected.

| seed | baseline val % | final val % | gain pp | transfer early % | MAC saving % | overall drop pp | balanced drop pp | worst-class drop pp |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 57 | 87.42 | 87.82 | +0.40 | 66.22 | 37.28 | +0.00 | +0.00 | +0.00 |
| 58 | 87.50 | 87.70 | +0.20 | 66.06 | 37.19 | +0.00 | +0.00 | +0.00 |
| 59 | 87.46 | 87.52 | +0.06 | 61.84 | 34.81 | +0.00 | +0.00 | +0.00 |

## Frozen gates

- [x] `mean_final_validation_gain_at_least_minus_0_003`
- [x] `each_final_validation_gain_at_least_minus_0_0075`
- [x] `each_transfer_accuracy_drop_at_most_0`
- [x] `each_transfer_balanced_drop_at_most_0`
- [x] `each_transfer_worst_class_drop_at_most_0`
- [x] `each_transfer_route_is_dynamic`
- [x] `each_transfer_mac_saving_at_least_0_15`

The transfer images are the same P1b calibration indices. They test unseen model versions, not an independent data distribution; external evidence remains required.
The official CIFAR-10 test loader is not iterated by this analysis.

Next: Freeze and hash this transfer result, then build a one-shot CIFAR-10.1 v6 distribution-shift evaluation. Never reopen the original CIFAR-10 test.
