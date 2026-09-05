# P8 first-run audit

The run `artifacts/analyses/early_exit_p8_20260905_153545` completed all 900
seed/device/batch/round/mode cells. All 900 raw NPZ SHA-256 values, 900,000
finite positive timing values, and reported means were verified. The result JSON
SHA-256 is `dd7dbef1e43cb4e78022231f9ab62f3b05eecfab89605672403832e2190730d1`.

Observed mean paired latency savings across three seeds and five rounds:

| Device / dataset | b=1 | b=4 | b=8 | b=16 | b=32 |
|---|---:|---:|---:|---:|---:|
| RTX 3080 Ti / CIFAR-10 | 18.85% | -6.99% | -11.74% | -13.63% | -12.23% |
| RTX 3080 Ti / CIFAR-100 | 2.75% | -12.81% | -10.27% | -13.27% | -13.32% |
| CPU / CIFAR-10 | 24.64% | 4.92% | -3.27% | 3.83% | 2.66% |
| CPU / CIFAR-100 | 7.62% | -6.23% | -2.51% | 0.32% | -0.81% |

These are descriptive averages, not significance tests or a universal acceleration
claim. The reference is the A3 model's final path, not the independently trained
A0 checkpoint. All cells, including slowdowns, are retained.

## Measurement limitations requiring a versioned correction

1. The original code computes route fraction over all 5,000 calibration images
   at batch=128, but times 1,000 batches at the requested batch size, cycling from
   the beginning. At b=1 it repeatedly measures only the first 1,000 images. The
   reported MAC and expected latency are therefore not tied to timed samples.
2. Its weighted expected latency uses the final-only path as the fallback cost;
   an actual fallback also executes the early head and routing operations. A
   sample-weighted mixture of full-batch isolated paths is not a prediction of
   heterogeneous batched execution.
3. CPU ru_maxrss is a process-lifetime high-water mark, not per-mode peak memory.
   CUDA allocated peak includes resident inputs and weights. Both require explicit
   scope labels; CPU mode-specific peak comparisons are unsupported.
4. Source/software fingerprints and exact timed sample IDs were not persisted.
   Resource cleanup overlapped approximately the first five minutes of this run;
   this contention is disclosed and no affected timing cells are selectively removed.

The actual recorded latency arrays remain valid observations of the original
implementation and workload. A corrected P8 v2 will retain the same A3 weights,
thresholds, CPU thread count, devices, batch sizes, timing count and all seeds.
It will record exact timed routes and sample IDs, measure forced fallback, label
memory correctly, and compare dynamic predictions against offline routing.
No training, threshold tuning, seed replacement, or test evaluation is required.
