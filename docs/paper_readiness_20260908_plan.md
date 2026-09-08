# Paper-readiness serial batch implementation plan

Goal: give the user one detached serial launch command covering development-only fair strategy comparison, existing A0–A4 tradeoffs and full P8-v3 latency replication. No P7 optimization, training, downloading or official/external evaluation.

Implementation: `scripts/analysis/paper_readiness.py` contains analysis and fail-stop stage execution; `scripts/launch_paper_readiness.py` owns preflight, the inherited P8 lock, output reservation and provenance. `tests/unit/test_paper_readiness.py` covers fallback overhead, false compute matching and failure propagation. Existing benchmark and audit scripts remain unchanged.

- [x] Inspect archived P5/P5-C/P8-v3 evidence and current RTX3080Ti/local CIFAR availability.
- [x] Add failing tests for missing implementation, hand-derived MAC overhead, unmatched compute, and fail-stop receipt.
- [x] Implement analyses and launcher, run `python -m unittest tests.unit.test_paper_readiness`.
- [x] Run `python scripts/launch_paper_readiness.py --dry-run` and development-only analysis smoke checks in a separate diagnostic directory.
- [ ] Give `python scripts/launch_paper_readiness.py` to the user; do not start the timed batch.

Frozen comparison rules: MSP, negative entropy and top-two softmax probability margin use the same 1001 source-quantile rule plus all-fallback sentinel. Source risk budgets are overall/balanced zero, worst-class zero for CIFAR10/strict CIFAR100 or 4pp for relaxed CIFAR100. Select max-min saving under all source-seed constraints. For compute-only comparisons, source mean saving must be within 0.25pp of 10/20/30%; target methods must also be within 0.5pp of each other or be labeled unmatched. No target threshold fitting. These are empirical post-hoc development comparisons, not population guarantees or new-image validation; score-operator overhead is outside Conv/Linear MAC accounting.

Tradeoffs use all 30 historical A0–A4 seeds. Correct policy MACs for fallback exit-head overhead. Show final validation accuracy separately from calibration policy accuracy. Report historical training wall times with actual GPU provenance only when available; missing timing/memory stays unavailable. Do not claim controlled training speedup from mixed historical hardware.

Latency repeats frozen P8-v3 unchanged: 2 datasets × 3 seeds × 2 devices × 5 batch sizes × 5 rounds × 4 modes, 1200 raw timing files; 100 warmup and 1000 measured calls each. Prespecified primary scenario: RTX3080Ti CIFAR10 batch1, resident model-only FP32; all other cells are retained as applicability boundaries. This is not end-to-end service latency. CPU uses12 threads. Reuse no old timing observations, retain all slowdowns.

The launcher checks input hashes, six A3 checkpoints, local train-only archives, hardware, ≥3GiB space, conflicts and duplicate output. It starts four foreground subprocesses serially within one detached worker, retaining an exclusive shared P8 lock. Any execution/audit failure stops subsequent stages and preserves partial results. A negative scientific result does not stop execution. Neither old files nor untracked writing handoff are removed or overwritten.

Expected runtime remains provisional (previous 4–10h planning range); new-host CPU latency can materially alter it. No new checkpoint storage.

Verification addition: reuse the historical raw-data audit and aggregation functions but do not reuse its outcome-specific README writer. Generate all latency directions from current numbers. Preserve failed audit receipts. The audit is deliberately stricter than the benchmark tolerance rule: any argmax disagreement blocks final certification even if within numeric tolerance; no automatic retries or dropped seeds.
