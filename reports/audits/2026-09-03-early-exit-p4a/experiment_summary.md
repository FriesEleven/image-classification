# CIFAR-100 early-exit P4a formal audit

This report audits only the completed serial P4a manifest; official test data are excluded.

| seed | baseline validation | multi-exit final validation | paired delta |
|---:|---:|---:|---:|
| 66 | 56.16% | 56.24% | +0.08 pp |
| 67 | 55.54% | 57.34% | +1.80 pp |
| 68 | 55.48% | 57.26% | +1.78 pp |

Paired gain: `+1.220 ± 0.987 pp` (sample SD); wins `3/3`.
Total serial duration: `1.572` hours.
Manifest SHA-256: `f122758c8a62c4bd40ea4451097a138616745f979d595628daa003eaa866d2b7`.
