"""Training objectives shared by single-head and exploratory multi-exit models."""

import torch
from torch.nn import functional as F

from image_classification.config import ExperimentConfig


def primary_logits(outputs: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Return the final classifier logits used for checkpoint selection and reporting."""

    if isinstance(outputs, torch.Tensor):
        return outputs
    if not outputs:
        raise ValueError("Model returned an empty output tuple")
    return outputs[0]


def training_objective(
    outputs: torch.Tensor | tuple[torch.Tensor, ...],
    targets: torch.Tensor,
    criterion,
    config: ExperimentConfig,
) -> torch.Tensor:
    """Compute final-head CE plus detached-teacher losses for configured exits."""

    final_logits = primary_logits(outputs)
    final_loss = criterion(final_logits, targets)
    if isinstance(outputs, torch.Tensor):
        return final_loss
    if config.model_type != "multi_exit":
        raise ValueError("Tuple model outputs require model_type=multi_exit")
    exit_logits = outputs[1:]
    if len(exit_logits) != len(config.exit_loss_weights):
        raise ValueError("Model output count does not match configured early-exit weights")

    temperature = config.exit_temperature
    alpha = config.exit_distillation_alpha
    teacher_probabilities = torch.softmax(final_logits.detach() / temperature, dim=1)
    loss = final_loss
    for logits, weight in zip(exit_logits, config.exit_loss_weights):
        classification_loss = criterion(logits, targets)
        distillation_loss = F.kl_div(
            torch.log_softmax(logits / temperature, dim=1),
            teacher_probabilities,
            reduction="batchmean",
        ) * temperature**2
        loss = loss + weight * (
            (1 - alpha) * classification_loss + alpha * distillation_loss
        )
    return loss
