import torch
from torch import nn

from image_classification.config import ExperimentConfig
from image_classification.training.objectives import primary_logits, training_objective


def _config():
    return ExperimentConfig(
        model_type="multi_exit",
        exit_positions=(8, 16),
        exit_loss_weights=(0.2, 0.3),
        exit_distillation_alpha=0.5,
        exit_temperature=3.0,
    )


def test_single_head_objective_preserves_cross_entropy():
    logits = torch.randn(4, 10, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3])
    criterion = nn.CrossEntropyLoss()

    assert torch.equal(
        training_objective(logits, targets, criterion, ExperimentConfig()),
        criterion(logits, targets),
    )
    assert primary_logits(logits) is logits


def test_multi_exit_objective_trains_all_heads_without_teacher_kd_gradient():
    final = torch.randn(4, 10, requires_grad=True)
    exit8 = torch.randn(4, 10, requires_grad=True)
    exit16 = torch.randn(4, 10, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3])
    criterion = nn.CrossEntropyLoss()
    reference_final = final.detach().clone().requires_grad_()
    reference_gradient = torch.autograd.grad(
        criterion(reference_final, targets), reference_final,
    )[0]

    loss = training_objective((final, exit8, exit16), targets, criterion, _config())
    loss.backward()

    assert primary_logits((final, exit8, exit16)) is final
    assert final.grad is not None
    assert exit8.grad is not None
    assert exit16.grad is not None
    # The detached final teacher means its gradient is exactly final-head CE.
    torch.testing.assert_close(final.grad, reference_gradient)
    assert loss.item() > criterion(final.detach(), targets).item()
