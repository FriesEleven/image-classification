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

P8 v2 is prepared, not started. It addresses mismatched timed-sample route
accounting, forced-fallback overhead, memory labels, and runtime provenance.
It uses the same six A3 checkpoints, thresholds 0.984/0.903, devices, batch sizes
and timing counts, plus a fourth forced-fallback mode. It retains all negative
results. Expected runtime is about 3–4 hours, subject to server contention.

Validation: targeted unit tests passed; all six real checkpoints were checked
on CPU/GPU at batch sizes 1/8/32 with zero route/prediction mismatches; three
timing iterations of each mode completed; actual Conv/Linear MAC counts matched
the retained constants. This was a functional smoke check, not a new paper result.

User launch from the server project directory:

```bash
/root/miniconda3/bin/python scripts/launch_early_exit_p8_v2.py
```

After completion, validate 1,200 raw timing arrays, 300 exact workload receipts,
per-batch early counts and prediction checks; aggregate rounds within seed before
between-seed statistics. Keep CPU lifetime RSS distinct from mode peak memory.
Use paired singleton isolated-path estimates only at batch=1; larger-batch
latency is measured directly. No training, official test access, threshold tuning,
or replacement seeds are included in this batch. Avoid source edits, cleanup,
or concurrent experiments while it runs. Then continue P6/P7 planning using A3
and retain the documented hardware limitations in the manuscript.
