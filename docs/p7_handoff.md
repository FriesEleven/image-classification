# P7 target completion and final evaluation handoff

The six target runs (seeds 80-82, A0/A3) completed and passed the integrity audit.
The source-selected threshold 0.850 was applied once, with zero target threshold candidates.

| Seed | A3-A0 validation gain (pp) | Policy accuracy (%) | Accuracy drop vs final (pp) | Worst-class drop (pp) | MAC saving (%) |
|---|---:|---:|---:|---:|---:|
|80|0.94|83.20|0.43|4.00|15.5014|
|81|0.28|83.24|0.31|3.00|15.3158|
|82|0.51|83.26|0.37|4.00|15.2033|

All target gates passed. The minimum saving is close to the 15% budget and two seeds reach the 4pp class-risk boundary. Do not describe this as a statistical guarantee.

Source and target share the same calibration image indices (split seed 20261003); target seeds are new model initializations, not an independent image holdout. Official validation is the pending image holdout. This dataset is a fixed 100-class subset, not ImageNet-1K.

Evidence: `reports/experiments/2026-09-07-early-exit-p7-target-analysis/` contains the audit, analysis receipt, cleanup receipt, and frozen official-test lock. Original target logits remain under `artifacts/analyses/early_exit_p7_target_20260907`.

Deleted only 60 periodic target checkpoints (1,724,722,152 bytes); all best/latest/final checkpoints and scientific evidence remain. Removed snapshots cannot be restored without retraining.

Training is finished. The only pending experimental step is one official evaluation of six fixed best checkpoints on 5,000 public ILSVRC2012 validation images. Public archives are present; no download is required. The evaluator extracts only the fixed 100 classes and records permanent started/completed markers. It refuses repeat execution. Verification alone does not read the official archives.

Launch from the repository root after reviewing the lock:

```bash
/root/miniconda3/bin/python scripts/analysis/evaluate_early_exit_p7_locked_test.py --authorize-official-test
```

After completion, audit and curate the results regardless of whether official-test risk constraints pass. Do not tune or rerun based on test outcomes. Integrate P5-A/B strategy comparisons, P5-C A3 simplification, P6 stopped-backbone boundary, P7 scale results and P8 v3 RTX 3080 Ti/CPU measurements into the paper. Older P4-only stopping language and RTX 4090D-centric writing guidance are superseded by these later receipts. P8 v3 supports conditional speedups, not universal acceleration. Finalize evidence tables and writing without new training.
