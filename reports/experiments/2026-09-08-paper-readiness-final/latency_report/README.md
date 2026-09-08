# Independent P8-v3 latency replication

Resident model-only FP32 timing, not end-to-end service latency. Positive saving is faster; negative is slowdown.
Mean and sample SD use three training seeds, each averaging five paired rounds. No universal acceleration claim.

| Device | Dataset | Batch | Saving (%) | Seed SD (pp) | Direction |
|---|---|---:|---:|---:|---|
| cpu | cifar10 | 1 | 27.914 | 0.992 | positive point estimate |
| cpu | cifar10 | 4 | 7.417 | 1.608 | positive point estimate |
| cpu | cifar10 | 8 | 3.322 | 1.349 | positive point estimate |
| cpu | cifar10 | 16 | 9.931 | 0.891 | positive point estimate |
| cpu | cifar10 | 32 | 4.895 | 1.047 | positive point estimate |
| cpu | cifar100 | 1 | 9.918 | 0.912 | positive point estimate |
| cpu | cifar100 | 4 | -4.718 | 1.716 | slowdown |
| cpu | cifar100 | 8 | -1.314 | 1.289 | slowdown |
| cpu | cifar100 | 16 | 1.144 | 1.557 | positive point estimate |
| cpu | cifar100 | 32 | -1.684 | 1.190 | slowdown |
| cuda | cifar10 | 1 | 16.909 | 1.702 | positive point estimate |
| cuda | cifar10 | 4 | -11.782 | 2.115 | slowdown |
| cuda | cifar10 | 8 | -18.456 | 1.314 | slowdown |
| cuda | cifar10 | 16 | -20.899 | 0.342 | slowdown |
| cuda | cifar10 | 32 | -20.654 | 1.312 | slowdown |
| cuda | cifar100 | 1 | -1.765 | 1.991 | slowdown |
| cuda | cifar100 | 4 | -18.545 | 0.794 | slowdown |
| cuda | cifar100 | 8 | -20.602 | 0.368 | slowdown |
| cuda | cifar100 | 16 | -21.022 | 0.455 | slowdown |
| cuda | cifar100 | 32 | -20.819 | 1.160 | slowdown |
