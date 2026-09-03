# CIFAR-100 P3 predicted-class guard diagnostic

Decision: **no_source_class_guard_with_budget**

This is a post-failure calibration-only diagnostic and does not unlock CIFAR-100 test.

| risk budget | source candidates | threshold | protected classes | min source saving | target passed | min target saving | max target worst-class drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| strict_zero | 0 | — | — | — | — | — | — |
| one_sample | 101 | 0.845 | 5 | 25.050% | False | 25.267% | 6.000% |

Next: Do not add a class guard; archive the CIFAR-100 routing boundary.
