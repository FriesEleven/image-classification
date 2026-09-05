# P8 handoff after first complete hardware benchmark

Latest server work follows `ccf_c_experiment_plan.md` and the user's explicit
authorization to extend the historical P4 evidence. Earlier P4-only stopping
language in `handoff.md` is historical. Target hardware remains RTX 3080 Ti and
the same server CPU; RTX 4090D is unavailable for new experiments.

P5-C finished all 18 new runs. Its current development-selected recipe is A3
(exit8 + training-only exit16, CE-only). The six old A4 checkpoints are retained.
With only three seeds per dataset, component effects remain descriptive; do not
claim statistical necessity or guaranteed journal acceptance.

P8 first run completed in approximately 2 h 12 min:
`artifacts/analyses/early_exit_p8_20260905_153545`.
All 900 raw files and all 900,000 timing entries passed integrity checks. See
`reports/experiments/2026-09-05-early-exit-p8-audit/README.md` for the complete
batch-size results and measurement limitations. GPU savings at batch=1 were
18.85% (CIFAR-10) and 2.75% (CIFAR-100), with average slowdowns at larger batches.
These first-run arrays remain unchanged.

P8 v2 started and then stopped as designed at `cuda_cifar10_s55_b32_r1`.
Its partial output is
`artifacts/analyses/early_exit_p8_v2_20260905_180241_841768`: 46 complete paired
cells (184 timing arrays) and 47 correctness checks. The failing check had zero
route errors and one near-tie prediction difference among 1,024 samples.
Diagnostics tied the difference to batch-shape-sensitive cuDNN TF32 convolution:
disabling TF32 reduced the maximum checked logit difference from 0.0069122 to
2.861e-6 and restored zero prediction differences. The partial timings remain
failed-run evidence and are not reused.

P8 v3 completed all 1,200 timing arrays in
`artifacts/analyses/early_exit_p8_v3_20260905_192249_293181`. The full audit
verified 1,200,000 positive finite latency observations, 300 exact workload
receipts, 600,000 timed route-count observations, and 300 correctness receipts.
Across 117,120 checked sample executions, route, prediction, and logit-tolerance
errors were all zero. Source stayed unchanged and no official test was accessed.

The report is
`reports/experiments/2026-09-05-early-exit-p8-v3-analysis/README.md`. Mean actual
GPU savings at batch 1 were 17.14% (sample SD 1.45 pp) for CIFAR-10 and -0.04%
(2.74 pp) for CIFAR-100. Every GPU batch size from 4 to 32 slowed down despite
positive MAC savings. CPU CIFAR-10 accelerated at all tested batch sizes; CPU
CIFAR-100 showed a clear gain at batch 1 and mixed/negative larger-batch results.
These are conditional hardware findings, not a universal acceleration claim.

Keep CPU lifetime RSS distinct from mode peak memory, and use the isolated-path
estimate only as a singleton mechanism diagnostic. Telemetry snapshots are not
energy measurements. The next core work package is P6, a preregistered CIFAR-stem
ResNet-18 source-to-target transfer using the simplified A3 recipe. Do not launch
P6 until its architecture mapping, MAC fractions, configs, tests, and immutable
manifest have all been reviewed and committed.
