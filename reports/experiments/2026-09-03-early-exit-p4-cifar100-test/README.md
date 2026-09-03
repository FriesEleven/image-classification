# CIFAR-100 P4 method-locked official-test evaluation

Decision: **paper_evidence_complete**

| seed | baseline % | final % | locked % | early % | MAC saving % | overall drop pp | worst-class drop pp | changed/harmed/rescued |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 66 | 55.78 | 57.68 | 58.07 | 42.05 | 23.97 | -0.39 | +3.00 | 226/57/96 |
| 67 | 55.96 | 57.54 | 58.09 | 42.63 | 24.30 | -0.55 | +4.00 | 260/61/116 |
| 68 | 56.06 | 57.32 | 58.13 | 42.50 | 24.23 | -0.81 | +2.00 | 269/52/133 |

## Preregistered confirmation gates

- [x] `each_accuracy_drop_at_most_0_005`
- [x] `each_balanced_accuracy_drop_at_most_0_005`
- [x] `each_worst_class_accuracy_drop_at_most_0_04`
- [x] `mean_accuracy_drop_at_most_0_002`
- [x] `each_route_is_dynamic`
- [x] `each_mac_saving_at_least_0_15`

Legacy baseline-only CIFAR-100 runs exposed test metrics before P4. P3 stopped without test. No P4 checkpoint or policy has seen official test data; describe a later pass as method-locked rather than globally blind.
No test-time threshold, class guard, seed or checkpoint selection was performed.

Next: Stop new training and proceed to frozen paper tables, figures and writing.
