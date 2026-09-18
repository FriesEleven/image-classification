# Manuscript materials — 2026-09-18

Read `docs/final_paper_writing_brief_20260918.md` in the repository root for current results, methods, interpretation and manuscript edits. These materials supplement earlier historical evidence rather than overwrite it.

- `shared_policy_manuscript.csv/.tex`: all M1 source-F and target-E panels, four strategies and three cohorts.
- `a3_joint_seed_manuscript.csv`: all six fixed A3 checkpoint operating points with accuracy, corrected MACs, mean latency, p95, throughput and separate full-test/timed-subset scope.
- `a3_joint_summary_manuscript.csv`, `a3_joint_manuscript.tex`: across-seed summary, tail metrics and conditional paired bootstrap.
- `shared_policy_tradeoffs.pdf`: discrete target-E points; blue satisfies class budget and red violates it; markers distinguish S/I/A/C. S/A coincide under this single-threshold protocol.
- `recalculation_audit.json`, `evidence_manifest.json`, `claim_to_evidence.md`: provenance and claim boundaries.

Original evidence is `artifacts/analyses/minimal_core_evidence_20260918_r1/`; compact experimental reports are `reports/experiments/2026-09-18-minimal-core-evidence/`. Reconstruction script: `scripts/analysis/build_minimal_core_paper_materials.py`. No new model execution was performed for these materials.

CIFAR-10 exhibits faster mean inference but worse paired p95. CIFAR-100 does not exhibit a mean latency benefit and one seed marginally misses its MAC floor. Report these conditions together. Accuracy uses all 10000 official images/model, timing uses1000 unique official IDs repeated over five paired rounds. Three training seeds remain the model-level statistical units.

Raw arrays/checkpoints are ignored by Git and require separate preservation. The report is not a full backup. The live LaTeX manuscript has not been modified or compiled by this material-preparation step.
