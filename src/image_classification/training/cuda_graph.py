"""Optional training-only CUDA Graph replay, preserving eager evaluation."""

import time

import torch
from torch.amp import autocast


def prepare_training_graph(model, batch_size: int, device: torch.device, amp: bool) -> dict:
    if device.type != "cuda":
        raise ValueError("cuda_graph requires CUDA; refusing a silent backend change")
    if getattr(model, "_training_graph_prepared", False):
        raise ValueError("Training graph is already prepared")
    original_forward = model.forward
    original_training = model.training
    original_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device)
    sample = torch.zeros(batch_size, 3, 32, 32, device=device)
    model.train()
    started = time.perf_counter()
    try:
        with autocast(device_type="cuda", enabled=amp, cache_enabled=False):
            torch.cuda.make_graphed_callables(model, (sample,))
        graphed_forward = model.forward
    except BaseException:
        model.forward = original_forward
        raise
    finally:
        # Warmup runs BatchNorm and Dropout: it must not become extra training
        # or consume any of the experiment's random sequence.
        model.load_state_dict(original_state, strict=True)
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
        model.train(original_training)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    model._training_graph_replays = 0
    model._training_graph_fallbacks = 0

    def shape_aware_forward(inputs):
        eligible = (
            model.training and torch.is_grad_enabled()
            and tuple(inputs.shape) == tuple(sample.shape)
            and inputs.dtype == sample.dtype and inputs.device == sample.device
            and inputs.requires_grad == sample.requires_grad
            and inputs.stride() == sample.stride()
            and torch.is_autocast_enabled("cuda") == amp
            and (not amp or torch.get_autocast_dtype("cuda") == torch.float16)
        )
        if eligible:
            model._training_graph_replays += 1
            return graphed_forward(inputs)
        # Keep the final 72-image CIFAR training batch: do not pad/drop samples.
        # Evaluation also stays eager, with the original float32 behavior.
        model._training_graph_fallbacks += 1
        return original_forward(inputs)

    model.forward = shape_aware_forward
    model._training_graph_prepared = True
    return {"backend": "cuda_graph_training_v1", "capture_seconds": elapsed,
            "captured_shape": list(sample.shape), "amp": amp,
            "state_and_rng_restored_after_warmup": True,
            "evaluation": "eager", "partial_batch": "eager, no drop_last"}
