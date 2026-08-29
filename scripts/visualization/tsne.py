"""Compare penultimate model features with t-SNE."""

import argparse
from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from image_classification.config import ExperimentConfig  # noqa: E402
from image_classification.data.cifar10 import MEAN, STD  # noqa: E402
from image_classification.models import build_model  # noqa: E402
from image_classification.paths import DATA_DIR  # noqa: E402


def _load_run(run: Path, device: torch.device):
    with (run / "config.yaml").open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    values.pop("runtime", None)
    model = build_model(ExperimentConfig(**values)).to(device).eval()
    state = torch.load(run / "checkpoints/model_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    return model


def _features_module(model):
    return model.model.features if hasattr(model, "model") else model.features


def _extract(model, loader, device):
    captured = {}
    hook = _features_module(model).register_forward_hook(lambda _module, _inputs, output: captured.update(value=output))
    features, labels = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            model(inputs.to(device))
            features.append(captured["value"].mean(dim=(2, 3)).cpu().numpy())
            labels.append(targets.numpy())
    hook.remove()
    return np.concatenate(features), np.concatenate(labels)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/figures/tsne.svg")
    args = parser.parse_args()
    dataset = datasets.CIFAR10(DATA_DIR, train=False, download=args.download, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)]))
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), min(args.samples, len(dataset)))
    loader = DataLoader(Subset(dataset, indices), batch_size=128, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    figure, axes = plt.subplots(1, len(args.runs), figsize=(7 * len(args.runs), 6), squeeze=False)
    for axis, run in zip(axes[0], args.runs):
        features, labels = _extract(_load_run(run, device), loader, device)
        embedding = TSNE(n_components=2, random_state=args.seed, init="pca", learning_rate="auto").fit_transform(features)
        scatter = axis.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=7, alpha=0.7)
        axis.set_title(run.name)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(scatter, ax=axes.ravel().tolist(), ticks=range(10))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close()
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
