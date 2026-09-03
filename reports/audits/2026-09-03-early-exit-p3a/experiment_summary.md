# CIFAR-100 early-exit P3a formal audit

This report audits only the completed serial P3a manifest; P3 test data are excluded.

| cohort | seed | baseline validation | multi-exit final validation | paired delta |
|---|---:|---:|---:|---:|
| source | 60 | 55.50% | 56.98% | +1.48 pp |
| source | 61 | 54.82% | 56.80% | +1.98 pp |
| source | 62 | 55.18% | 57.38% | +2.20 pp |
| target | 63 | 54.94% | 57.44% | +2.50 pp |
| target | 64 | 54.82% | 57.36% | +2.54 pp |
| target | 65 | 55.32% | 57.66% | +2.34 pp |

Paired gain: `+2.173 ± 0.397 pp` (sample SD); wins `6/6`.
Total serial duration: `3.200` hours.
Manifest SHA-256: `ae20bcb4d988b5e44fd1112bd8935caa12ec0086e2c50617be33b49a9237a3a2`.
