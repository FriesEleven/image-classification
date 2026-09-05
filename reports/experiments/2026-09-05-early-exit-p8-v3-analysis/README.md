# P8 v3 completed-run audit and analysis

The completed output `artifacts/analyses/early_exit_p8_v3_20260905_192249_293181` passed the full integrity audit: 1200 raw arrays, 1,200,000 latency observations, 300 exact workload receipts, and 300 correctness receipts. Across 117,120 checked sample executions there were zero route, prediction, or logit-tolerance errors. No official test set was accessed.

Latency saving is `1 - actual_dynamic / final_only` within each paired round. Five rounds are averaged within each training seed; the table reports mean ± sample SD across the three seeds. Negative values are slowdowns. Three seeds are too few for broad significance claims.

| Device / dataset | b=1 | b=4 | b=8 | b=16 | b=32 |
|---|---:|---:|---:|---:|---:|
| RTX 3080 Ti / CIFAR-10 | 17.14% ± 1.45% | -14.73% ± 3.70% | -19.24% ± 0.32% | -20.36% ± 0.51% | -20.39% ± 0.72% |
| RTX 3080 Ti / CIFAR-100 | -0.04% ± 2.74% | -18.45% ± 0.90% | -19.51% ± 1.89% | -21.79% ± 0.58% | -22.44% ± 4.30% |
| 12-thread CPU / CIFAR-10 | 26.56% ± 1.95% | 7.58% ± 0.39% | 1.64% ± 0.82% | 7.78% ± 1.30% | 4.56% ± 0.56% |
| 12-thread CPU / CIFAR-100 | 10.14% ± 0.42% | -5.77% ± 2.14% | -3.30% ± 0.82% | 1.19% ± 1.18% | -1.51% ± 0.96% |

The deployable claim is conditional. On the RTX 3080 Ti, dynamic routing accelerated singleton CIFAR-10 inference by 17.14% ± 1.45%, versus an isolated-path estimate of 22.84%. CIFAR-100 singleton inference was effectively neutral (-0.04% ± 2.74%; isolated estimate 1.60%). Every GPU batch-size setting from 4 to 32 slowed down despite positive MAC savings, exposing compaction, branching, and kernel-launch overhead. CPU behavior was hardware- and dataset-dependent: CIFAR-10 accelerated at all measured batch sizes, whereas CIFAR-100 had a clear gain only at batch 1 and mixed/negative larger-batch behavior.

The reference is the same A3 checkpoint's final-only path, not an independently trained A0 model. CUDA memory numbers are allocated-memory scopes; CPU RSS is process-lifetime high-water only and does not support per-mode peak comparisons. Telemetry contains boundary snapshots, not integrated energy. The failed TF32 P8 v2 run remains separate and none of its timings are included here.

Machine-readable outputs: `seed_summary.csv`, `aggregate_summary.csv`, `aggregate_summary.tex`, and `analysis_receipt.json`. The figure `actual_latency_saving_by_batch.pdf` visualizes the complete result, including slowdowns.
