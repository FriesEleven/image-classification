# Expanded manuscript evidence map — 2026-09-09

This is a manuscript reconstruction, not a new experiment. Historical A4 records
and the original `reports/paper/evidence_manifest.json` remain unchanged.

| Claim / scope | Frozen source relative to repository | Manuscript output |
|---|---|---|
| Historical A4 CIFAR locked results | Original `reports/paper/claim_to_evidence.md` and its JSON inputs | RQ1; archived transfer/boundary figures |
| Equal-risk and approximate matched-compute scores, post-hoc development only | `reports/experiments/2026-09-08-paper-readiness-final/fair_comparison.json` and `fair_comparison_seed_metrics.csv` | Same-risk/matched tables and figures |
| Full A0–A4 factorial result, no universal A3 optimum | Same directory: `ablation_aggregate_tradeoffs.csv`, `ablation_seed_tradeoffs.csv`, `ablation_scope.json` | Ablation table, final-accuracy and policy-tradeoff figures |
| ResNet-18 stops at source selection | `reports/experiments/2026-09-06-early-exit-p6-source-analysis/analysis_receipt.json` | RQ1 source infeasibility subsection |
| ImageNet-100 target seed calibration passes on shared images | `artifacts/analyses/early_exit_p7_source_20260907/source_lock.json`, `artifacts/analyses/early_exit_p7_target_20260907/target_results.json` | Transfer figure and source/target account |
| ImageNet-100 official run completed; all saving gates fail; seed 81 class gate fails | `reports/experiments/2026-09-08-paper-readiness-final/p7_official/test_results.json` | Three-seed table, class heatmap and decision outcomes |
| Fresh A3 actual latency, all device/batch cells retained | `reports/experiments/2026-09-08-paper-readiness-final/latency_report/aggregate_summary.csv`; raw round table in frozen readiness artifact | GPU/CPU tables, p95, arithmetic comparison, replication figures |
| Calibration/subsampling/complementarity, original recipe | `artifacts/analyses/early_exit_p5ab_20260905_092909/tables/` | Appendix table and three figures |

## Audit and preservation

- The companion `evidence_manifest.json` records actual inputs and generated asset hashes.
- `recalculation_audit.json` verifies 1,818 readiness files and six P7 saved-logit hashes; no mismatch.
- P7 policy accuracy, route fractions, and all 100 class drops were recomputed only at the locked threshold 0.85; zero new threshold candidates and no model execution.
- The official result SHA-256 remains `ac009a3203a3579997c8a5665b63ba1d766cc685f53d865b24017c3700afb749`.
- `latency_distribution_summary.csv` uses round → seed → three-seed aggregation. Mean p95 is not pooled p95.
- The new manuscript generator is `sn-article-template/scripts/build_final_evidence.py` in the sibling paper directory. Generated typesetting tables and vector figures remain there; compact audit copies are preserved here.
- Large arrays/checkpoints remain ignored. This directory is not a full dataset or checkpoint backup.

## Claims explicitly disallowed

No statistical risk guarantee, no universal MSP superiority, no A3 CIFAR official-test
claim, no A3/A4 hybrid accuracy–latency point, no controlled training speedup from
historical wall times, no full ImageNet/mobile/end-to-end service acceleration.
