# P5-C component ablation decision

All 18 new A1/A2/A3 runs completed on the RTX 3080 Ti, all six preregistered
training seeds were retained, and no final-head failure rule fired. Analysis
reused the six existing A0 and six existing A4 runs.

The development-only decision is to simplify the training recipe to A3:
deployable exit8 plus training-only exit16, without detached-final KD. KD had
small negative mean final-validation effects on both datasets and no stable
same-direction seed effect. Exit16 was dataset-dependent: slightly negative on
CIFAR-10 and positive on all three CIFAR-100 paired main effects.

The unchanged thresholds (`0.984` for CIFAR-10 and `0.903` for CIFAR-100) were
applied with zero new candidates. A3 passed the empirical risk and minimum-MAC
saving gate on all six seeds. No official CIFAR test or external evaluator was
executed.
