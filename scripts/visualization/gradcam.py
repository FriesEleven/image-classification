"""Generate Grad-CAM without redefining project models."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from image_classification.config import ExperimentConfig  # noqa: E402
from image_classification.data.cifar10 import MEAN, STD  # noqa: E402
from image_classification.models import build_model  # noqa: E402


def _load_config(path: Path) -> ExperimentConfig:
    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    values.pop("runtime", None)
    return ExperimentConfig(**values)


def _load_weights(model: nn.Module, path: Path) -> None:
    value = torch.load(path, map_location="cpu", weights_only=False)
    state = value.get("model_state_dict", value) if isinstance(value, dict) else value
    model.load_state_dict(state)


def _target_layer(model: nn.Module, name: str | None) -> nn.Module:
    modules = dict(model.named_modules())
    if name:
        if name not in modules:
            raise ValueError(f"Unknown layer {name!r}; inspect model.named_modules()")
        return modules[name]
    convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    if not convolutions:
        raise ValueError("Model has no convolution layer")
    return convolutions[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="artifacts/runs/<experiment_id>")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", default="model_best.pth")
    parser.add_argument("--target-layer")
    parser.add_argument("--target-class", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/figures/gradcam.svg")
    args = parser.parse_args()
    config = _load_config(args.run / "config.yaml")
    model = build_model(config).eval()
    _load_weights(model, args.run / "checkpoints" / args.checkpoint)
    layer = _target_layer(model, args.target_layer)
    activations = {}

    def capture_activation(_module, _inputs, output):
        activations["value"] = output
        output.retain_grad()

    forward_hook = layer.register_forward_hook(capture_activation)
    original = Image.open(args.image).convert("RGB")
    tensor = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])(original).unsqueeze(0)
    scores = model(tensor)
    target_class = args.target_class if args.target_class is not None else int(scores.argmax(dim=1))
    model.zero_grad()
    scores[0, target_class].backward()
    forward_hook.remove()
    weights = activations["value"].grad.mean(dim=(2, 3), keepdim=True)
    heatmap = torch.relu((weights * activations["value"]).sum(dim=1)).squeeze().detach().numpy()
    heatmap -= heatmap.min()
    heatmap /= heatmap.max() + 1e-8
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(original)
    plt.imshow(heatmap, cmap="jet", alpha=0.45, extent=(0, original.width, original.height, 0))
    plt.axis("off")
    plt.title(f"class={target_class}")
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close()
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
