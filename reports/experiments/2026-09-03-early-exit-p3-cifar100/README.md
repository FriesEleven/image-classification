# CIFAR-100 early-exit P3 source selection and target transfer

Decision: **stop_without_test**

| cohort | seed | baseline val % | final val % | gain pp | early % | MAC saving % | overall drop pp | worst-class drop pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source | 60 | 55.50 | 56.98 | +1.48 | — | — | — | — |
| source | 61 | 54.82 | 56.80 | +1.98 | — | — | — | — |
| source | 62 | 55.18 | 57.38 | +2.20 | — | — | — | — |
| target | 63 | 54.94 | 57.44 | +2.50 | — | — | — | — |
| target | 64 | 54.82 | 57.36 | +2.54 | — | — | — | — |
| target | 65 | 55.32 | 57.66 | +2.34 | — | — | — | — |

## Frozen gates

- [x] `mean_final_validation_gain_at_least_minus_0_003`
- [x] `each_final_validation_gain_at_least_minus_0_0075`
- [ ] `shared_source_dynamic_policy_found`
- [ ] `each_source_accuracy_drop_at_most_0`
- [ ] `each_source_balanced_drop_at_most_0`
- [ ] `each_source_worst_class_drop_at_most_0`
- [ ] `each_source_route_is_dynamic`
- [ ] `each_source_mac_saving_at_least_0_15`
- [ ] `each_target_accuracy_drop_at_most_0`
- [ ] `each_target_balanced_drop_at_most_0`
- [ ] `each_target_worst_class_drop_at_most_0`
- [ ] `each_target_route_is_dynamic`
- [ ] `each_target_mac_saving_at_least_0_15`

Legacy baseline-only CIFAR-100 runs viewed official-test metrics before P3. P3 does not use those runs, and no P3 checkpoint or early-exit policy has been evaluated on the official test. Describe a later pass as method-locked, not globally blind.
The official CIFAR-100 test loader is not iterated by this analysis.

Next: Archive P3 as a second-dataset boundary result; do not inspect official CIFAR-100 test.
