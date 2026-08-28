# Repository Guidelines

This repository trains and evaluates MobileNetV2 image classifiers on CIFAR-10, comparing ECA, CBAM, SE, and hybrid attention modules.

## Project Structure & Module Organization

- `train.py` is the primary training, benchmarking, checkpointing, and evaluation entry point.
- `mobilenetv2_eca/ECANet/` contains the vendored ECA-MobileNetV2 implementation.
- `configs/` stores YAML snapshots of experiment settings; use names such as `se_deep_15-16_...yaml`.
- `scripts/` contains Grad-CAM, ROC, confusion-matrix, t-SNE, and analysis utilities; reusable analysis code is under `scripts/analysis/`; `tests/` contains smoke checks.
- `docs/` stores experiment and paper planning notes; `mobilenetv2_eca/ECANet/` is the vendored ECA implementation.
- `assets/gradcam/` contains sample input images. `results/` contains curated metrics, predictions, tables, and visualizations; generated models, logs, and TensorBoard runs go under ignored `artifacts/`, while the CIFAR-10 download goes under ignored `data/`.

## Build, Test, and Development Commands

There is no `requirements.txt` or package manifest. Use a Python environment with PyTorch/torchvision, PyYAML, NumPy, scikit-learn, tqdm, TensorBoard, and the analysis dependencies used by the scripts (`pandas`, `matplotlib`, `seaborn`, and `openpyxl`). Run commands from the repository root.

- `python train.py --model_type mobilenetv2 --epochs 1 --batch_size 64 --experiment_name smoke` runs a short baseline experiment, downloading CIFAR-10 to `./data` and using CUDA when available.
- `cmd /c run_all_experiments.bat` runs the predefined Windows batch of attention experiments.
- `python tests/testeca.py`, `python tests/testCUDA.py`, and `python check_network.py` perform model, accelerator, and forward-pass checks.
- `python scripts/analysis/sumary_to_csv.py`, `python scripts/analysis/performance_plots.py`, `python scripts/analysis/latex_tables.py`, and `python scripts/analysis/key_findings.py` regenerate analysis outputs from existing results.

`test.sh` currently only echoes a message and is not a test runner.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, `snake_case` for functions and variables, and `PascalCase` for model classes. Keep experiment names lowercase with underscores; encode attention positions as hyphenated values such as `hybrid_se1-2_cbam15-16`. Run tools from the repository root and write generated files under the established output directories.

## Testing Guidelines

No pytest/unittest suite or coverage threshold is configured. For model changes, run the smoke scripts and a CPU-compatible dummy forward pass; for training changes, run a one-epoch experiment and inspect `results/` plus `artifacts/`. CUDA checks are optional when no GPU is available.

## Commit & Pull Request Guidelines

The only existing commit is `Initial commit`, so no project-specific convention is established. Use short, imperative messages (for example, `Add hybrid attention experiment`). Pull requests should explain the change, record model/configuration details and hardware for experiments, include relevant metric or plot updates, and state validation commands. Do not include datasets, credentials, or large model checkpoints.
