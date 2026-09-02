# Early-exit P1b locked official-test evaluation

The official CIFAR-10 test set was evaluated once after the shared policy was frozen.
No threshold, class guard or model choice was tuned on test.

Frozen exit-8 maximum-softmax threshold: `0.984`.

| seed | baseline % | final % | final−base pp | locked % | locked−base pp | early % | MAC saving % | worst-class drop vs final pp |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 54 | 86.58 | 87.23 | +0.65 | 87.24 | +0.66 | 65.14 | 36.67 | 0.00 |
| 55 | 86.80 | 86.94 | +0.14 | 86.94 | +0.14 | 64.88 | 36.52 | 0.00 |
| 56 | 87.12 | 86.91 | -0.21 | 86.93 | -0.19 | 64.54 | 36.33 | 0.00 |
| mean ± sample SD | 86.83 ± 0.27 | 87.03 ± 0.18 | 0.19 ± 0.43 | 87.04 ± 0.18 | 0.20 ± 0.43 | 64.85 ± 0.30 | 36.51 ± 0.17 | 0.00 ± 0.00 |

MAC saving is an architecture-level conv/linear operation proxy; it is not measured latency.
Per-class metrics and retained logits are indexed in `source_index.json`.

Selection SHA-256: `8dce5d938e06ae9d7432bd6baafd0c0ed9f978678fb53767ab02eeafecc723ab`.
Manifest SHA-256: `1545e85651103400c425bf3eaae0e70e4d1a4d0e413203158d2f6b6ff16c8c88`.
