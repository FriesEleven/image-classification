# CIFAR-100 early-exit P4 independent confirmation

Decision: **ready_for_method_locked_cifar100_test**

| seed | baseline val % | final val % | gain pp | early % | MAC saving % | overall drop pp | worst-class drop pp |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 66 | 56.16 | 56.24 | +0.08 | 41.94 | 23.91 | -1.10 | +4.00 |
| 67 | 55.54 | 57.34 | +1.80 | 41.54 | 23.68 | -0.90 | +4.00 |
| 68 | 55.48 | 57.26 | +1.78 | 42.40 | 24.17 | -0.82 | +4.00 |

## Frozen gates

- [x] `mean_final_validation_gain_at_least_minus_0_003`
- [x] `each_final_validation_gain_at_least_minus_0_0075`
- [x] `each_accuracy_drop_at_most_0`
- [x] `each_balanced_drop_at_most_0`
- [x] `each_worst_class_drop_at_most_0_04`
- [x] `each_route_is_dynamic`
- [x] `each_mac_saving_at_least_0_15`

P4 considered zero threshold candidates and performed no per-model recalibration.
The official CIFAR-100 test loader is not iterated by this analysis.

Next: Freeze exact P4 evidence and evaluator hashes, then run one method-locked CIFAR-100 official-test evaluation.
