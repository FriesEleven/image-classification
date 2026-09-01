# Stage-sparse MobileNetV2 hardware profile

This report records the pre-training cost profile used by the budget-aware selector. No model was trained and no dataset was read.

## Accepted profile

- Raw artifact: `artifacts/budget_selector/profiles/stage_sparse_rtx3080ti_paired_cuda_events_20260901_v5/profile.json`
- SHA-256: `6f6363b5c933fc109dbd6dd67c1d2d9eaa4485756177b5244685e8dcfdd9069d`
- GPU: NVIDIA GeForce RTX 3080 Ti; batch 1; input 3×32×32.
- Candidate space: all 64 assignments of None/ECA/SE/CBAM to shallow/middle/deep stage packets.
- Timing: three randomized candidate-order rounds. Every candidate is paired with an immediately adjacent all-none measurement; candidate/baseline order is randomized. Each measurement uses 20 warmups plus 100 timed forwards and CUDA Events on one stream. The budget value is the median of three paired overheads.
- All-none member: 2,236,682 parameters; the median of 189 paired baseline anchors is 3.9576 ms.

The exact raw file retains all 189 baseline anchors, every candidate round, the randomized pair order and all paired overheads. The median reduces sensitivity to one system outlier. Candidates close to a final deployment threshold must be reprofiled with more paired rounds after the architecture is frozen.

## Singleton costs

| Stage/module | Params Δ | Attention operation proxy | Paired latency Δ median | Three paired rounds |
|---|---:|---:|---:|---:|
| shallow/ECA | 6 | 11,384 | +4.79% | +4.79 / +6.39 / +4.60% |
| shallow/SE | 122 | 11,344 | +12.99% | +12.99 / +8.24 / +14.32% |
| shallow/CBAM | 276 | 65,312 | +16.75% | +16.75 / +14.00 / +17.19% |
| middle/ECA | 6 | 1,408 | +5.18% | +3.40 / +5.18 / +6.45% |
| middle/SE | 1,160 | 2,048 | +8.18% | +13.96 / +7.13 / +8.18% |
| middle/CBAM | 1,220 | 5,904 | +15.45% | +18.01 / +15.45 / +15.12% |
| deep/ECA | 6 | 1,600 | +4.88% | +5.10 / +4.29 / +4.88% |
| deep/SE | 6,740 | 7,040 | +7.66% | +16.26 / +7.66 / +5.98% |
| deep/CBAM | 6,596 | 14,916 | +16.79% | +15.01 / +20.74 / +16.79% |

The operation value is a stage-sensitive attention-only proxy. It includes pooling/comparison, learned multiply-accumulates and feature gating, but excludes nonlinearities and the unchanged backbone; it is not presented as full-model profiler FLOPs.

## Default budget feasibility

| Budget | Limits (params / ops / latency / stages) | Feasible candidates |
|---|---|---:|
| ultra_light | 512 / 100,000 / +15% / 1 | 5 |
| balanced | 2,000 / 250,000 / +30% / 2 | 22 |
| relaxed | 8,000 / 500,000 / +50% / 3 | 58 |

All three sets include all-none and at least one non-empty candidate. The selector may still choose all-none when paired validation utility is negative.

## Superseded measurement protocol

Four development profiles are retained but must not be used for selection. v1 (`bcc4a9e…`) synchronized and timed every forward with the CPU wall clock. v2 (`a111548a…`) and v3 (`73190bed…`) used CUDA Events but divided independently measured candidate and baseline medians, so GPU clock drift still moved small overhead estimates. v4 (`b0bda462…`) introduced adjacent paired baselines before any calibration accuracy existed, but its source snapshot predates the final rule that an exact utility tie must prefer all-none. All protocol iterations retained the same default-budget feasible counts (5/22/58); v5 is the paired profile whose complete source snapshot matches the launch-ready code.

RTX 3080 Ti latency is a controlled server-side proxy, not evidence of mobile-device latency. A final deployment claim requires the same frozen candidates and protocol on the declared target hardware.
