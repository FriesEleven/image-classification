# Early-exit P1b RTX 4090D deployment profile

Batch-1 end-to-end synchronized wall-clock timings on an otherwise idle GPU.
The implementation computes exit8 once and continues only unresolved samples; exit16 is skipped.

| seed | locked test early route | final-only ms | expected policy ms | saving | speedup |
|---:|---:|---:|---:|---:|---:|
| 54 | 65.14% | 4.0252 | 3.1402 | 24.87% | 1.331× |
| 55 | 64.88% | 4.6861 | 3.6263 | 21.51% | 1.274× |
| 56 | 64.54% | 4.3521 | 3.1554 | 28.36% | 1.396× |
| mean | 64.85% | 4.3545 | 3.3073 | 24.91% | 1.334× |

Each path is isolated with thresholds 0/2 only during timing; the expected latency is weighted by
the already frozen test route fraction at threshold 0.984. No dataset is loaded by this profiler.
This server GPU result is deployment evidence for the implementation, not mobile-device evidence.
