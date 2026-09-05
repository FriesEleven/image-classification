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

P8 v3 is the clean repair. It retains the same six A3 checkpoints, thresholds,
devices, batch sizes, rounds and timing counts; disables cuDNN/matmul TF32;
enables deterministic algorithms; and freezes logit-level correctness at
rtol=1e-4, atol=1e-5. Exact route mismatches or logits outside tolerance remain
fatal. Near-tie argmax changes are counted and disclosed separately. P8 v3
remeasures all 1,200 arrays in a new output directory. Expected runtime remains
about 3–4 hours, subject to server contention.

User launch from the server project directory:

```bash
/root/miniconda3/bin/python scripts/launch_early_exit_p8_v3.py
```

After completion, validate 1,200 raw timing arrays, 300 exact workload receipts,
per-batch early counts and prediction checks; aggregate rounds within seed before
between-seed statistics. Keep CPU lifetime RSS distinct from mode peak memory.
Use paired singleton isolated-path estimates only at batch=1; larger-batch
latency is measured directly. No training, official test access, threshold tuning,
or replacement seeds are included in this batch. Avoid source edits, cleanup,
or concurrent experiments while it runs. Then continue P6/P7 planning using A3
and retain the documented hardware limitations in the manuscript.
