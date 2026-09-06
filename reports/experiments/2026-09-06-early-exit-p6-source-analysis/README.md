# P6 ResNet-18 source gate result

## Decision

The frozen source analysis completed for all 12 planned source runs: two datasets,
two variants (A0/A3), and seeds 71--73.  The final-head gate passed on every
paired seed, but the shared-policy gate failed for both datasets.  The target
seeds 74--76 and all official-test evaluation therefore remain unopened.

| Dataset | A3--A0 final validation gain | Final-head gate | Shared-policy gate | Decision |
|---|---:|---|---|---|
| CIFAR-10 | +0.527 pp mean; +0.340 pp minimum | pass | fail | `stop_without_target` |
| CIFAR-100 | +1.273 pp mean; +0.800 pp minimum | pass | fail | `stop_without_target` |

The result separates representation quality from deployable routing: the
multi-exit model improved the final classifier on all six paired seeds, but no
single source-calibrated threshold satisfied every seed's preregistered risk,
coverage, and corrected-MAC-saving constraints.

## Frozen policy boundary

The threshold grid contained 1,001 points per dataset (`0.000`--`1.000`, step
`0.001`).  No point was feasible.

- CIFAR-10: under coverage and at least 15% saving, the closest point was
  threshold `0.999`.  It achieved minimum saving `19.41%` and per-seed early
  fractions `31.34%`--`33.54%`, but its maximum overall/balanced drop was
  `0.08 pp` and its maximum worst-class drop was `0.60 pp`; the frozen budget
  was zero for all three drops.  The only zero-drop point was final-only at
  threshold `1.000`, with no early exits and a negligible negative saving from
  exit-head overhead.
- CIFAR-100: the closest coverage-and-saving point was threshold `0.950`.  It
  achieved minimum saving `15.22%` and per-seed early fractions
  `24.58%`--`25.90%`, but its maximum overall/balanced drop was `0.20 pp` and
  maximum worst-class drop was `6.00 pp`, above the frozen `0/0/4 pp` budget.
  The best point satisfying all risk limits was threshold `0.996`, with only
  `5.51%` minimum saving and `8.90%`--`9.22%` early exits.

These boundary values are post-decision diagnostics only.  They do not replace
the frozen gate, authorize a relaxed threshold, or justify target/test access.

## Audit and provenance

- Training commit: `4e286806059e6770fd1b99ffb9e243b19451cbd6`.
- Frozen analysis commit: `3e736dc`.
- Frozen protocol SHA-256:
  `bb51af3f815e7261e67bb95d5092ca3290ad0d6b200e5bb7086a052b3aff02c1`.
- Source manifests: both report `completed`, with 6/6 successful runs each.
- Training logs: 12/12 contain 200 epochs; A3 per-head logs contain 1,200 rows
  each; split sizes are 40,000/5,000/5,000 and disjoint.
- Calibration logits: six files, 5,000 samples each, with checkpoint hashes
  bound in `source_lock.json`.
- `official_test_accessed=false`; `target_data_accessed=false`.
- Analysis artifacts are bound by
  [analysis_receipt.json](analysis_receipt.json).
- Rebuildable epoch checkpoints and Python caches were removed after audit;
  best/latest/final checkpoints, logs, splits, manifests, and analysis logits
  were preserved.  The deletion receipt is in
  `reports/audits/2026-09-06-p6-source-cleanup/cleanup_receipt.json`.

## Paper-use boundary

This is a valid external-validity boundary, not a positive source-to-target
transfer result.  It supports the claims that ResNet-18 final-head training
benefited from the auxiliary recipe on these source seeds and that strict
zero-drop shared routing did not transfer to a useful-compute policy.  It does
not support claims of ResNet-18 target generalization, official-test
performance, or a universally safe threshold.
