# Frozen early-exit policy on CIFAR-10.1 v6

Decision: **external_shift_confirmed**

Threshold `0.984` was frozen on P1b before this external evaluation.
No external-data calibration, threshold search, class guard or model selection was performed.

| cohort | seed | baseline % | final % | locked % | early % | MAC saving % | worst-class drop pp | changed/harmed/rescued |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source | 54 | 77.35 | 77.15 | 77.15 | 49.30 | 27.75 | +0.00 | 0/0/0 |
| source | 55 | 76.80 | 77.20 | 77.20 | 49.65 | 27.95 | +0.00 | 0/0/0 |
| source | 56 | 76.70 | 76.40 | 76.40 | 49.20 | 27.70 | +0.00 | 0/0/0 |
| target | 57 | 75.55 | 76.95 | 76.95 | 50.20 | 28.26 | +0.00 | 0/0/0 |
| target | 58 | 77.00 | 76.95 | 77.00 | 50.50 | 28.43 | +0.00 | 1/0/1 |
| target | 59 | 77.15 | 75.45 | 75.45 | 45.20 | 25.45 | +0.00 | 0/0/0 |

## Frozen external gates

- [x] `each_external_accuracy_drop_at_most_0`
- [x] `each_external_balanced_drop_at_most_0`
- [x] `each_external_worst_class_drop_at_most_0`
- [x] `each_external_route_is_dynamic`
- [x] `each_external_mac_saving_at_least_0_15`

CIFAR-10.1 v6 contains 2,000 examples, exactly 200 per class.
MAC saving is an operation-count proxy; the existing RTX4090D profile remains the latency evidence.

Next: Freeze this result and run one minimal second-dataset or second-backbone confirmation before declaring the full paper evidence complete.
