"""Inspect CSGHA deep/guidance logits and channel-gate saturation."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from image_classification.config import load_config
from image_classification.data import build_dataloaders
from image_classification.models import build_model
from image_classification.models.attention import CrossStageChannelAttention

DEFAULT_CONFIG = ROOT / "configs/experiments/csgha_se_shallow_cbam_middle.yaml"


def _saturation_fraction(logits: torch.Tensor) -> float:
    gates = torch.sigmoid(logits)
    saturated = torch.logical_or(gates < 0.05, gates > 0.95)
    return float(saturated.float().mean().item())


def summarize_guidance(model: torch.nn.Module, inputs: torch.Tensor) -> list[dict]:
    """Return per-target guidance statistics from one forward pass."""

    reports: list[dict] = []
    handles = []

    def capture(name: str):
        def hook(module, hook_inputs, _output):
            deep_logits, raw_guidance, gated_guidance = module.attention_logits(*hook_inputs)
            shallow_descriptor = hook_inputs[1]
            batch = shallow_descriptor.shape[0]
            legacy_guidance = module.guide_projection(shallow_descriptor).view(
                batch, module.channels, 1, 1,
            )
            deep_magnitude = deep_logits.detach().abs().mean()
            raw_magnitude = raw_guidance.detach().abs().mean()
            gated_magnitude = gated_guidance.detach().abs().mean()
            legacy_magnitude = legacy_guidance.detach().abs().mean()
            denominator = max(float(deep_magnitude.item()), torch.finfo(torch.float32).eps)
            reports.append(
                {
                    "module": name,
                    "guidance_scale_parameter": float(module.guidance_scale.detach().item()),
                    "guidance_scale_effective": float(
                        torch.tanh(module.guidance_scale.detach()).item()
                    ),
                    "deep_logits_abs_mean": float(deep_magnitude.item()),
                    "raw_guidance_logits_abs_mean": float(raw_magnitude.item()),
                    "gated_guidance_logits_abs_mean": float(gated_magnitude.item()),
                    "legacy_guidance_logits_abs_mean": float(legacy_magnitude.item()),
                    "raw_guidance_to_deep_ratio": float(raw_magnitude.item()) / denominator,
                    "gated_guidance_to_deep_ratio": float(gated_magnitude.item()) / denominator,
                    "legacy_guidance_to_deep_ratio": (
                        float(legacy_magnitude.item()) / denominator
                    ),
                    "legacy_gate_saturation_fraction": _saturation_fraction(
                        deep_logits.detach() + legacy_guidance.detach()
                    ),
                    "gated_gate_saturation_fraction": _saturation_fraction(
                        deep_logits.detach() + gated_guidance.detach()
                    ),
                }
            )

        return hook

    for name, module in model.named_modules():
        if isinstance(module, CrossStageChannelAttention):
            handles.append(module.register_forward_hook(capture(name)))
    if not handles:
        raise ValueError("Model does not contain cross-stage channel attention")
    try:
        with torch.no_grad():
            model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    return reports


def _load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> dict:
    payload = torch.load(path, map_location=device, weights_only=True)
    state_dict = payload.get("model_state_dict", payload)
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing_suffixes = (
        ".guidance_scale",
        ".guide_normalization.weight",
        ".guide_normalization.bias",
    )
    invalid_missing = [
        key for key in incompatible.missing_keys
        if not key.endswith(allowed_missing_suffixes)
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise ValueError(
            "Checkpoint is incompatible with CSGHA v2: "
            f"missing={invalid_missing}, unexpected={list(incompatible.unexpected_keys)}"
        )
    return {
        "path": str(path),
        "format": "csgha_v1" if incompatible.missing_keys else "csgha_v2",
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(["--config", str(config_path)])
    if config.model_type != "csgha":
        raise ValueError(f"Expected a CSGHA config, got {config.model_type}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device).eval()
    checkpoint = None
    if args.checkpoint:
        checkpoint = _load_checkpoint(model, args.checkpoint.resolve(), device)

    if args.synthetic:
        inputs = torch.randn(args.batch_size, 3, 32, 32, device=device)
        batch_source = "synthetic"
    else:
        loaders = build_dataloaders(
            dataset=config.dataset,
            batch_size=args.batch_size,
            num_workers=0,
            validation_size=config.validation_size,
            split_seed=config.seed,
        )
        inputs, _labels = next(iter(loaders.validation))
        inputs = inputs.to(device)
        batch_source = "validation"

    report = {
        "config": str(config_path),
        "experiment_id": config.experiment_id,
        "checkpoint": checkpoint,
        "device": str(device),
        "batch_source": batch_source,
        "batch_size": int(inputs.shape[0]),
        "modules": summarize_guidance(model, inputs),
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    if args.output:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{serialized}\n", encoding="utf-8")
        print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
