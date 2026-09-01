"""Deterministic stage-aware candidate enumeration, costing and selection."""

from collections.abc import Mapping, Sequence
from itertools import product
from math import sqrt

import torch
from torch import nn

from image_classification.models.attention import CBAM, SEBlock
from image_classification.models.eca import ECALayer

STAGE_POSITIONS = {
    "shallow": (1, 2),
    "middle": (7, 8),
    "deep": (15, 16),
}
ATTENTION_CHOICES = ("none", "eca", "se", "cbam")


def _canonical_choices(choices: Mapping[str, str]) -> dict[str, str]:
    if set(choices) != set(STAGE_POSITIONS):
        raise ValueError(f"Stage choices must contain exactly {tuple(STAGE_POSITIONS)}")
    canonical = {stage: str(choices[stage]) for stage in STAGE_POSITIONS}
    unsupported = sorted(set(canonical.values()) - set(ATTENTION_CHOICES))
    if unsupported:
        raise ValueError(f"Unsupported attention choices: {unsupported}")
    return canonical


def candidate_id(choices: Mapping[str, str]) -> str:
    canonical = _canonical_choices(choices)
    return "__".join(f"{stage}_{canonical[stage]}" for stage in STAGE_POSITIONS)


def positions_from_choices(choices: Mapping[str, str]) -> dict[str, tuple[int, ...]]:
    canonical = _canonical_choices(choices)
    positions = {"eca": [], "se": [], "cbam": []}
    for stage, choice in canonical.items():
        if choice != "none":
            positions[choice].extend(STAGE_POSITIONS[stage])
    return {kind: tuple(values) for kind, values in positions.items()}


def enumerate_stage_candidates() -> list[dict]:
    candidates = []
    stages = tuple(STAGE_POSITIONS)
    for values in product(ATTENTION_CHOICES, repeat=len(stages)):
        choices = dict(zip(stages, values, strict=True))
        positions = positions_from_choices(choices)
        candidates.append(
            {
                "candidate_id": candidate_id(choices),
                "choices": choices,
                "positions": {kind: list(indices) for kind, indices in positions.items()},
                "active_stages": sum(choice != "none" for choice in values),
            }
        )
    return candidates


def candidate_from_positions(
    eca_positions: Sequence[int],
    se_positions: Sequence[int],
    cbam_positions: Sequence[int],
) -> dict:
    by_kind = {
        "eca": set(map(int, eca_positions)),
        "se": set(map(int, se_positions)),
        "cbam": set(map(int, cbam_positions)),
    }
    occupied = set().union(*by_kind.values())
    if sum(len(values) for values in by_kind.values()) != len(occupied):
        raise ValueError("Stage-sparse attention positions must be disjoint")
    allowed = set().union(*map(set, STAGE_POSITIONS.values()))
    if occupied - allowed:
        raise ValueError(f"Positions outside the stage packets: {sorted(occupied - allowed)}")

    choices = {}
    for stage, stage_positions in STAGE_POSITIONS.items():
        packet = set(stage_positions)
        matching = [kind for kind, positions in by_kind.items() if packet <= positions]
        partial = [kind for kind, positions in by_kind.items() if positions & packet and not packet <= positions]
        if partial:
            raise ValueError(f"Stage {stage} must use its complete position packet {stage_positions}")
        choices[stage] = matching[0] if matching else "none"
    expected = positions_from_choices(choices)
    if any(set(expected[kind]) != by_kind[kind] for kind in expected):
        raise ValueError("Positions do not encode one complete attention choice per stage")
    return next(candidate for candidate in enumerate_stage_candidates() if candidate["choices"] == choices)


def _module_operation_estimate(kind: str, module: nn.Module, shape: tuple[int, ...]) -> dict:
    if len(shape) != 4 or shape[0] != 1:
        raise ValueError("Attention operation estimates require a single NCHW sample")
    _batch, channels, height, width = shape
    elements = channels * height * width
    if kind == "eca" and isinstance(module, ECALayer):
        kernel = int(module.conv.kernel_size[0])
        pooling = elements
        learned = channels * kernel
        elementwise = elements
    elif kind == "se" and isinstance(module, SEBlock):
        hidden = int(module.fc[0].out_features)
        pooling = elements
        learned = 2 * channels * hidden
        elementwise = elements
    elif kind == "cbam" and isinstance(module, CBAM):
        hidden = int(module.channel_attention.fc[0].out_channels)
        kernel_height, kernel_width = module.spatial_attention.conv.kernel_size
        pooling = 4 * elements
        learned = 4 * channels * hidden + height * width * 2 * kernel_height * kernel_width
        elementwise = 2 * elements
    else:
        raise TypeError(f"Unexpected {kind} attention module: {type(module).__name__}")
    return {
        "input_shape": list(shape),
        "pooling_comparison_ops": pooling,
        "learned_multiply_accumulates": learned,
        "feature_elementwise_ops": elementwise,
        "operations_estimate": pooling + learned + elementwise,
        "parameters": sum(parameter.numel() for parameter in module.parameters()),
    }


def attention_operation_profile(
    model: nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
) -> dict:
    """Estimate stage-sensitive attention work without claiming full-model FLOPs."""
    attention_kinds = getattr(model, "attention_kinds", None)
    backbone = getattr(model, "model", None)
    if not isinstance(attention_kinds, dict) or backbone is None:
        raise TypeError("Expected a StageSparseAttentionMobileNetV2 model")

    rows: dict[int, dict] = {}
    handles = []
    for position, kind in attention_kinds.items():
        attention = backbone.features[position].stage_attention

        def record(_module, inputs, _output, *, position=position, kind=kind):
            row = _module_operation_estimate(kind, _module, tuple(inputs[0].shape))
            row.update(position=position, attention=kind)
            rows[position] = row

        handles.append(attention.register_forward_hook(record))

    try:
        parameter = next(model.parameters())
        sample = torch.zeros(input_shape, device=parameter.device, dtype=parameter.dtype)
        was_training = model.training
        model.eval()
        with torch.inference_mode():
            model(sample)
        model.train(was_training)
    finally:
        for handle in handles:
            handle.remove()

    ordered = [rows[position] for position in sorted(rows)]
    return {
        "modules": ordered,
        "learned_multiply_accumulates": sum(row["learned_multiply_accumulates"] for row in ordered),
        "pooling_comparison_ops": sum(row["pooling_comparison_ops"] for row in ordered),
        "feature_elementwise_ops": sum(row["feature_elementwise_ops"] for row in ordered),
        "operations_estimate": sum(row["operations_estimate"] for row in ordered),
        "note": (
            "Stage-sensitive attention-only operation proxy; pooling, comparisons and elementwise gates are included, "
            "but nonlinearities and the unchanged backbone are not full profiler FLOPs."
        ),
    }


def _unit_ids(choices: Mapping[str, str]) -> list[str]:
    canonical = _canonical_choices(choices)
    return [f"{stage}_{choice}" for stage, choice in canonical.items() if choice != "none"]


def score_candidates(
    profiles: Sequence[Mapping],
    unit_evidence: Mapping[str, Mapping[str, float]],
    risk_penalty: float,
) -> list[dict]:
    if risk_penalty < 0:
        raise ValueError("risk_penalty must be non-negative")
    scored = []
    for profile in profiles:
        units = _unit_ids(profile["choices"])
        missing = sorted(set(units) - set(unit_evidence))
        if missing:
            raise ValueError(f"Missing probe evidence for {missing}")
        mean_gain = sum(float(unit_evidence[unit]["mean_gain_pp"]) for unit in units)
        variance = sum(float(unit_evidence[unit]["sample_std_pp"]) ** 2 for unit in units)
        uncertainty = sqrt(variance)
        scored.append(
            {
                **dict(profile),
                "active_units": units,
                "predicted_gain_pp": mean_gain,
                "predicted_uncertainty_pp": uncertainty,
                "risk_adjusted_gain_pp": mean_gain - risk_penalty * uncertainty,
            }
        )
    return scored


def _within_budget(candidate: Mapping, budget: Mapping) -> bool:
    limits = {
        "max_parameter_delta": "parameter_delta",
        "max_attention_operations": "attention_operations_estimate",
        "max_latency_overhead_percent": "latency_overhead_percent",
        "max_active_stages": "active_stages",
    }
    for limit_name, candidate_name in limits.items():
        limit = budget.get(limit_name)
        if limit is not None and float(candidate[candidate_name]) > float(limit):
            return False
    return True


def select_candidates_for_budgets(
    scored_candidates: Sequence[Mapping],
    budgets: Sequence[Mapping],
) -> list[dict]:
    selections = []
    for budget in budgets:
        name = str(budget.get("name", ""))
        if not name:
            raise ValueError("Every budget requires a name")
        feasible = [candidate for candidate in scored_candidates if _within_budget(candidate, budget)]
        if not feasible:
            raise ValueError(f"No feasible candidate for budget {name}")
        selected = min(
            feasible,
            key=lambda candidate: (
                -float(candidate["risk_adjusted_gain_pp"]),
                -float(candidate["predicted_gain_pp"]),
                int(candidate["active_stages"]),
                int(candidate["parameter_delta"]),
                int(candidate["attention_operations_estimate"]),
                float(candidate["latency_overhead_percent"]),
                str(candidate["candidate_id"]),
            ),
        )
        selections.append(
            {
                "budget": dict(budget),
                "feasible_candidates": len(feasible),
                "selected": dict(selected),
            }
        )
    return selections
