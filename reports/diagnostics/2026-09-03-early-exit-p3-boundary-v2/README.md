# CIFAR-100 P3 routing boundary diagnostic

The frozen P3 decision remains `stop_without_test`; this report uses calibration data only.

## Boundary summary

- Strict zero-risk policy without the 15% route floor: `found`.
- All source seeds individually admit a strict dynamic policy: `False`.
- Least-exposure shared dynamic threshold `0.997` routes at least `15.760%` and requires worst-class drop `2.000%`.
- Lowest-worst-class-risk dynamic threshold `0.987` has minimum MAC saving `13.010%` and maximum worst-class drop `2.000%`.
- With the required 15% MAC saving enforced, the lowest-risk threshold `0.903` needs maximum worst-class drop `4.000%`.

Any relaxed policy below is post hoc and requires a new independent confirmation; it cannot unlock P3 test.

| exploratory budget | found | threshold | min source saving | saving gate | max source overall drop | max source worst-class drop |
|---|---:|---:|---:|---:|---:|---:|
| one_class_sample | yes | 0.987 | 13.010% | False | -0.100% | 2.000% |
| preregistered_test_scale | yes | 0.987 | 13.010% | False | -0.100% | 2.000% |
| double_worst_class_tolerance | yes | 0.893 | 23.443% | True | -1.020% | 4.000% |
