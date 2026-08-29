"""Model-size and inference-latency measurements."""

import time

import numpy as np
import torch
from torch import nn

from image_classification.config import ExperimentConfig
from image_classification.models.attention import CBAM, SEBlock
from image_classification.models.eca import ECALayer


def model_metrics(model: nn.Module, config: ExperimentConfig) -> dict:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    eca = sum(sum(parameter.numel() for parameter in module.parameters()) for module in model.modules() if isinstance(module, ECALayer))
    cbam = sum(sum(parameter.numel() for parameter in module.parameters()) for module in model.modules() if isinstance(module, CBAM))
    se = sum(sum(parameter.numel() for parameter in module.parameters()) for module in model.modules() if isinstance(module, SEBlock))
    classifier = sum(parameter.numel() for name, parameter in model.named_parameters() if "classifier" in name)
    eca_count = sum(isinstance(module, ECALayer) for module in model.modules())
    cbam_count = sum(isinstance(module, CBAM) for module in model.modules())
    se_count = sum(isinstance(module, SEBlock) for module in model.modules())
    base_flops = 91.0e6
    estimated_attention_flops = eca_count * 0.01e6 + cbam_count * 0.8e6 + se_count * 0.2e6
    return {
        "parameters_total": total,
        "parameters_trainable": trainable,
        "parameters_backbone": total - classifier - eca - cbam - se,
        "parameters_classifier": classifier,
        "parameters_eca": eca,
        "parameters_cbam": cbam,
        "parameters_se": se,
        "parameters_main_attention": cbam if config.model_type == "hybrid" else eca + cbam + se,
        "parameters_aux_attention": se if config.model_type == "hybrid" else 0,
        "num_eca_modules": eca_count,
        "num_cbam_modules": cbam_count,
        "num_se_modules": se_count,
        "flops_total": base_flops + estimated_attention_flops,
        "flops_base": base_flops,
        "flops_attention_adjustment": estimated_attention_flops,
        "model_type": config.model_type,
        "aux_positions": list(config.aux_positions),
        "se_positions": list(config.se_positions),
        "cbam_positions": list(config.cbam_positions),
        "flops_note": "FLOPs are an analytical estimate, not profiler output.",
    }


def benchmark_inference(model: nn.Module, device: torch.device, input_size: tuple[int, ...] = (1, 3, 32, 32), runs: int = 100) -> dict:
    was_training = model.training
    model.eval()
    sample = torch.randn(input_size, device=device)
    with torch.no_grad():
        for _ in range(10 if device.type == "cuda" else 2):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies = []
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
    model.train(was_training)
    mean = float(np.mean(latencies))
    return {
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else str(device).upper(),
        "batch_size": input_size[0],
        "input_resolution": f"{input_size[2]}x{input_size[3]}",
        "inference_latency_mean": mean,
        "inference_latency_std": float(np.std(latencies)),
        "throughput_fps": 1000 / mean,
        "num_runs": runs,
    }
