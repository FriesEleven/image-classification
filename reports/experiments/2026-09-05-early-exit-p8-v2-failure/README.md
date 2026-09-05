# P8 v2 failure audit

P8 v2 stopped as designed at `cuda_cifar10_s55_b32_r1` before timing that
cell. The partial output contains 184 timing arrays from 46 complete paired
cells and 47 correctness receipts. It is failed-run evidence and must not be
reported as a complete benchmark or merged with a later run.

The failing receipt checked 1,024 calibration samples. Routes matched exactly
(`route_errors=0`), while one fallback prediction differed from an offline
full-batch final forward (`prediction_errors=1`). The affected calibration
sample ID was 37348. Its offline top-two margin was 0.0010271 and the dynamic
top-two margin was 0.0005934; classes 1 and 9 exchanged order. The maximum
absolute logit difference for that sample was 0.0018373, and the maximum over
the checked batches was 0.0069122.

The recorded runtime had cuDNN TF32 enabled. Repeating the exact failing
workload with cuDNN TF32 disabled reduced the batch maximum absolute logit
difference to 2.861e-6 and produced zero route and prediction differences.
Adding deterministic cuDNN produced the same result. This identifies
batch-shape-sensitive TF32 convolution as the cause, rather than a route-index
or checkpoint error.

P8 v3 therefore disables TF32, enables deterministic algorithms, freezes a
logit-level correctness tolerance, and remeasures all 1,200 arrays in a new
directory. The failed P8 v2 output, log, provenance, failing check, and partial
round table are hash-anchored in the P8 v3 protocol.
