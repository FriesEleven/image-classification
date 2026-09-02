# Budget-aware stage-sparse probe1: negative result

The frozen CIFAR-10 calibration sweep completed all 30 validation-only runs on
2026-09-01.  Every singleton attention unit had a negative three-seed mean gain
against its paired all-none member, and all three pre-specified hardware budgets
therefore selected all-none.  The pre-specified stopping rule is active: there
is no seeds 48/49/50 attention confirmation batch.

## Audited evidence

- Sweep manifest:
  `artifacts/sweeps/cifar10_budget_stage_probe_serial_probe1_20260901_114320/manifest.json`
  (`sha256=2b162999a7fc456792bb90f68bb7aa3bcd11fe2443067c1b37c36ff9eebf3e99`).
- The manifest contains exactly 10 candidates x seeds 45/46/47.  All 30 rows
  are completed with return code 0, continuous 200-epoch logs, matching best
  checkpoint hashes, and retained best/latest/final checkpoints.
- Every resolved configuration is CIFAR-10, 45k/5k, `stage_sparse`,
  validation-only, batch 128, AdamW/OneCycleLR, AMP plus training CUDA Graph,
  one serial job, and no per-run inference measurement.
- No run evaluated or exported official-test predictions.
- The frozen paired hardware profile is
  `artifacts/budget_selector/profiles/stage_sparse_rtx3080ti_paired_cuda_events_20260901_v5/profile.json`
  (`sha256=6f6363b5c933fc109dbd6dd67c1d2d9eaa4485756177b5244685e8dcfdd9069d`).
- Final selection evidence is
  `artifacts/budget_selector/selections/cifar10_stage_sparse_probe1_20260902_v3/selection.json`
  (`sha256=beafa7606eea343a65531d878e945bcae9d05cc431917e926453dadbe983a1fc`).
  It records hashes for the manifest, profile, frozen budget file and exact
  selector implementation.

Periodic optimizer snapshots were removed only after the file-level audit;
best/latest/final weights, CSV/TensorBoard logs, configurations, splits,
summaries, manifests and source snapshots remain.  The cleanup receipt is
`artifacts/cleanup/2026-09-02-probe1-periodic-checkpoints.json`.

## Paired validation results

The all-none accuracies for seeds 45/46/47 were 87.74/89.02/88.54%, or
88.433 +/- 0.647%.  Gains below are percentage points relative to the matched
all-none member with the same seed and split.

| Unit | Seed 45/46/47 gains | Mean +/- sample std | Wins |
|---|---:|---:|---:|
| shallow ECA | -0.08/-0.76/+0.28 | -0.187 +/- 0.528 | 1/3 |
| shallow SE | +0.04/-0.34/+0.22 | -0.027 +/- 0.286 | 2/3 |
| shallow CBAM | +0.50/-0.94/-0.12 | -0.187 +/- 0.722 | 1/3 |
| middle ECA | -0.04/-0.80/+0.36 | -0.160 +/- 0.589 | 1/3 |
| middle SE | +0.64/-0.82/+0.00 | -0.060 +/- 0.732 | 1/3 |
| middle CBAM | -0.04/-0.36/+0.06 | -0.113 +/- 0.219 | 1/3 |
| deep ECA | +0.18/-1.20/-0.06 | -0.360 +/- 0.737 | 1/3 |
| deep SE | +0.10/-0.70/+0.30 | -0.100 +/- 0.529 | 2/3 |
| deep CBAM | -0.02/-0.76/-0.38 | -0.387 +/- 0.370 | 0/3 |

The best non-empty singleton by risk-adjusted utility was shallow SE, but its
score was still about -0.170 pp (`mean - 0.5 * sample_std`).  Adding more
negative singleton utilities cannot beat the zero utility of all-none under the
frozen additive proxy.

| Budget | Feasible candidates | Selected candidate | Utility |
|---|---:|---|---:|
| ultra-light | 5 | `shallow_none__middle_none__deep_none` | 0.000 pp |
| balanced | 22 | `shallow_none__middle_none__deep_none` | 0.000 pp |
| relaxed | 58 | `shallow_none__middle_none__deep_none` | 0.000 pp |

## Decision boundary

This result rejects the current claim that independently inserting ECA, SE or
CBAM into the frozen MobileNetV2 stage packets yields a reproducible clean
accuracy--cost advantage on this CIFAR-10 protocol.  It does not prove that all
attention architectures, datasets or training recipes are universally harmful.
The additive multi-stage score was a search proxy and no multi-stage candidate
was trained.

Following the rule frozen before observing probe accuracy, do not run an
attention confirmation batch, do not expand this selector to CIFAR-100, a
second backbone or Coordinate Attention, and do not use official test data to
rescue the route.  CSGHA and the static stage-sparse selector remain honest
negative/mechanism results.  Any next positive route must pose a new research
question and use fresh development seeds rather than extending this search
post hoc.
