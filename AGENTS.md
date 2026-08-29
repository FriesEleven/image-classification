# Repository Guidelines

This repository trains and evaluates MobileNetV2 image classifiers on CIFAR-10, comparing ECA, CBAM, SE, and hybrid attention placements.

## Structure

- `src/image_classification/` is the reusable application package. Keep models in `models/`, dataset code in `data/`, and training/evaluation persistence in `training/`.
- `configs/experiments/` contains runnable YAML definitions. Generated, hardware-specific resolved configs belong to `artifacts/runs/<experiment_id>/`.
- `scripts/` contains thin command-line utilities grouped into `analysis/`, `visualization/`, and `diagnostics/`. Do not redefine model architectures in scripts.
- `tests/unit/` covers isolated logic; `tests/integration/` covers cross-module behavior; small image fixtures belong in `tests/fixtures/`.
- `third_party/eca_net/` preserves the upstream ECA-Net code and license. Runtime project code uses the adapted implementation in `src/image_classification/models/eca.py`.
- `data/` and `artifacts/` are local and ignored. Versioned paper tables and figures belong in `reports/`.

## Commands

Run commands from the repository root.

- `python -m pip install -e '.[analysis,visualization,dev]'` installs the project and development tools.
- `python scripts/train.py --config configs/experiments/baseline.yaml` runs one configured experiment.
- `python scripts/run_experiments.py --dry-run` validates and prints the full sweep; omit `--dry-run` to train it.
- `python scripts/diagnostics/check_model.py` runs forward passes for all model families.
- `python -m pytest` runs the test suite.
- `python -m compileall -q src scripts tests train.py` checks Python syntax.

The root `train.py` is a compatibility entry point for old commands; new behavior belongs in the package.

## Conventions

Use four-space indentation, `snake_case` functions and files, and `PascalCase` classes. Keep experiment names lowercase with underscores and encode layer positions with hyphens in generated IDs. Resolve repository paths through `image_classification.paths`; do not depend on the caller's current directory.

Generated checkpoints, logs, predictions, downloaded data, credentials, and environment-specific files must not be committed. Curate only final summaries, LaTeX tables, and publication figures into `reports/`.

For model changes, run unit tests and `scripts/diagnostics/check_model.py`. For training-loop changes, additionally run a one-epoch smoke experiment when dependencies and CIFAR-10 are available. Record configuration, hardware, validation commands, and affected reports in pull requests.
