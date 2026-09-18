# Claim-to-evidence — 2026-09-18 supplement

Repository-relative sources below supplement the archived `reports/paper/final-20260909/` ledger. That historical ledger's statement that no A3 CIFAR official assessment exists is now superseded for these six locked checkpoints only.

| Claim | Source | Constraint on wording |
|---|---|---|
| Shared versus individual recalibration is directly compared | `artifacts/analyses/minimal_core_evidence_20260918_r1/shared_vs_individual_seed_metrics.csv` and `shared_vs_individual_summary.csv`; compact experiment copies under `reports/experiments/2026-09-18-minimal-core-evidence/` | Post-hoc historical A4 development analysis; I fits target F, others do not; no universal shared advantage |
| Fit/audit IDs are aligned and source F does not overlap target E | Original `fit_audit_partitions.json`, frozen `protocol_manifest.json`, input manifest | Already exposed historical development images; no fresh unseen-image claim |
| S/A return the same threshold and operating points | `policy_selection.json`, shared seed metrics, nested MSP threshold definition | No empirical advantage of max-min over mean; report their structural equivalence in this protocol |
| Six locked A3 checkpoints have official CIFAR accuracy/MAC results | `a3_accuracy_mac_seed_metrics.csv`, `a3_joint_operating_points.csv`, `m2_raw/*_predictions.npz` | Same checkpoint final-only reference; supplementary assessment on previously exposed benchmarks; no threshold search or new seed-transfer claim |
| CIFAR-10 mean latency saving 19.10% ±2.37pp | Original `a3_latency_round_metrics.csv` and raw timing; manuscript seed/summary CSV | RTX3080Ti singleton resident subset; three seeds, five paired rounds; not end-to-end latency |
| CIFAR-10 p95 saving -10.65% ±6.41pp | Manuscript CSV, independently verified raw round p95 | Mean of paired per-round percentile ratios; not pooled percentile or improved tail SLA |
| CIFAR-100 mean/p95 savings are negative | Manuscript CSV, original round metrics | Retain all seeds; arithmetic savings do not establish acceleration |
| CIFAR-100 seed68 misses savings floor | Raw saving `0.14998029168632432` and `analysis_receipt.json` | Very marginal failure; retain raw decision despite display rounding to15.00% |
| Policy-final conditional accuracy differences | `a3_joint_summary_manuscript.csv` and reconstruction script | 2000 paired stratified image bootstrap; fixed models; no future-model or simultaneous class-risk guarantee |
| Technical audit and independent statistical recheck pass | Original `analysis_receipt.json`, manuscript `recalculation_audit.json`, `evidence_manifest.json` | Completion is separate from scientific gate success |

Continue disclosing older entropy/margin counterexamples, A3/A4 compute tradeoffs, P3/P6 stop outcomes, P7 official failure, and historical GPU larger-batch slowdowns. Their data scope differs from this supplemental official workload.
