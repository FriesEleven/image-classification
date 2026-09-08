# Final writing evidence archive — 2026-09-08

This directory preserves compact results for manuscript preparation. The primary writing brief is `docs/final_paper_writing_brief_20260908.md` (relative to repository root). Consult it before interpreting results.

## Source and scope

Files in this directory were copied byte-for-byte from `artifacts/analyses/paper_readiness_20260908_r1/`. `latency_report/` contains the completed fresh replication's tables, figures and audit, not preparation-stage outputs. `output_hashes.json` retains paths relative to the original artifact directory, including raw files intentionally not uploaded to Git.

`p7_official/` preserves the existing P7 `test_results.json`, `started.json` and `completed.json` receipts from `artifacts/analyses/early_exit_p7_official_test/`. No new evaluation was run for this archive.

## Findings that must travel with the results

- All four readiness stages completed. The 1,818-file original output-hash inventory was checked without mismatches during writing-brief preparation.
- The latency integrity audit passed: 1,200 raw timing files and zero routing, prediction or logit-tolerance errors in the checked executions.
- MSP does not universally outperform entropy or probability margin. Comparisons are post-hoc development-only and use empirical risk constraints or approximate mean Conv/Linear MAC matching, not exact runtime matching.
- A3 removes the distillation objective but sacrifices MAC savings relative to A4; historical wall times do not prove controlled training acceleration.
- Fresh RTX3080Ti CIFAR-10 batch-one model-only latency saving is 16.91% ± 1.70 percentage points across three seeds. All measured larger GPU batches slow down. Keep all negative cells.
- P7 official execution completed but the policy gates failed: all three seeds missed the 15% savings floor and one exceeded the worst-class degradation budget. Do not relabel this as a pass.

## Preservation boundary

Git contains compact summaries, reproducibility metadata and figures. Original datasets, model checkpoints, logits, raw timing arrays and workload arrays remain on the experiment server under ignored `data/` and `artifacts/`; this Git push is not a full backup of those materials. Do not delete server originals on the assumption that they have been uploaded.

Launch and analysis sources are `scripts/launch_paper_readiness.py` and `scripts/analysis/paper_readiness.py`. This is a completed batch: do not rerun it simply to regenerate manuscript tables. P7 evaluation markers and locked results must remain intact.
