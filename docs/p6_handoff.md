# P6 source-stage handoff

P8 v3 completed and passed the full integrity audit. Its hardware conclusions
are in `reports/experiments/2026-09-05-early-exit-p8-v3-analysis/README.md`.
RTX 4090D remains historical-only; all new work uses the RTX 3080 Ti.

P6 is a staged CIFAR-stem ResNet-18 source-to-target transfer. The label-free
architecture profiler enumerated every completed residual-block boundary before
training. The closest legal boundaries to the preregistered 42.5% and 82.5% MAC
targets are block 2 and block 6. Their actual path fractions are approximately
38.08% and 86.41%, respectively; no legal boundary lies within the nominal
target bands. Block 2 is deployable and block 6 is training-only.

The selected A3 recipe is carried forward unchanged in meaning: exit weights
0.1/0.15 and CE-only (`exit_distillation_alpha=0`). Each seed has a matched
final-only A0 model. CIFAR-10 uses split seed 20261001 and CIFAR-100 uses
20261002. Source seeds are 71-73; target seeds 74-76 are not launched until the
source gate selects and freezes one shared threshold per dataset.

The source batch contains 12 serial runs (2 datasets × A0/A3 × 3 seeds), 200
epochs each, CUDA Graph training, batch 128, and no official-test evaluation or
inference timing. Synthetic RTX 3080 Ti smoke tests verified CUDA Graph replay,
AMP forward/backward, optimizer updates, finite loss, measured path MACs, and
dynamic all-early/all-final routing. Based on historical epoch times plus the
ResNet smoke benchmark, allow roughly 5-7 hours, with server contention as the
main uncertainty.

Launch from the project directory:

```bash
/root/miniconda3/bin/python scripts/launch_early_exit_p6_source.py
```

After completion, retain all seeds and first validate both sweep manifests,
checkpoint/split hashes, 200-epoch logs, per-head logs, and final-head failure
gates. Select thresholds only on source calibration data using the frozen
objective. If either dataset fails its source gate, record `stop_without_target`
for that dataset; do not alter the exit, threshold grid, split, or seeds. Target
training and any official-test access require later gates; source execution does
not authorize either.
