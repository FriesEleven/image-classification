# Claim-to-evidence audit

This ledger maps every central numerical claim in the manuscript to a frozen,
versioned evidence record.  The table/figure generators read these records but
do not instantiate evaluators, data loaders, or official-test sweeps.

| Manuscript claim | Evidence type | Frozen source |
|---|---|---|
| CIFAR-10 threshold 0.984 is shared by source seeds 54--56 | Calibration selection | `reports/experiments/2026-09-02-early-exit-p1b/locked_selection.json` |
| CIFAR-10 locked-test accuracy, early fraction, MAC saving, and zero observed worst-class drop | Method-locked official test | `reports/experiments/2026-09-02-early-exit-p1b/test_results.json` |
| Threshold 0.984 transfers to unseen retraining seeds 57--59 without recalibration | Frozen-policy model transfer | `reports/experiments/2026-09-03-early-exit-p2a-transfer/transfer_results.json` |
| Threshold 0.984 transfers once to CIFAR-10.1 v6 for all six models | Natural-shift external evaluation | `reports/experiments/2026-09-03-early-exit-p2-cifar10-1-v6/external_results.json` |
| CIFAR-100 P3 has no feasible strict-zero-risk point at the 15% saving floor | Pre-registered calibration stop | `reports/experiments/2026-09-03-early-exit-p3-cifar100/selection.json` |
| The 0/2/4 pp CIFAR-100 saving boundary is diagnostic only | Post-hoc calibration analysis | `reports/diagnostics/2026-09-03-early-exit-p3-boundary-v2/diagnostic.json` |
| A one-sample class guard does not transfer under strict zero risk | Post-hoc class-guard diagnostic | `reports/diagnostics/2026-09-03-early-exit-p3-class-guard-v2/diagnostic.json` |
| Threshold 0.903 passes a new-seed, new-split 4 pp confirmation | Independent confirmation | `reports/experiments/2026-09-03-early-exit-p4-cifar100/confirmation.json` |
| CIFAR-100 P4 locked-test accuracy, routing, MAC saving, and 2--4 pp observed worst-class drops | Method-locked official test | `reports/experiments/2026-09-03-early-exit-p4-cifar100-test/test_results.json` |
| RTX 4090 D paired batch-1 expected latency saving is 24.91% | Separate hardware profile | `reports/profiles/2026-09-02-early-exit-p1b-rtx4090d/profile.json` |

The SHA-256 values for these inputs and all generated paper tables and figures
are recorded in `reports/paper/evidence_manifest.json`.
