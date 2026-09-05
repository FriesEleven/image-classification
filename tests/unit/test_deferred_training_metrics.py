import numpy as np
import pytest
import torch
from torch import nn
from torch.amp import GradScaler

from image_classification.training.checkpoint import append_per_head_training_log
from image_classification.training.engine import _step_optimizer_and_scheduler
from image_classification.training.evaluate import EpochAccumulator, classification_metrics
from image_classification.training.optimizer_step import OptimizerStepTracker


class CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def test_deferred_metrics_match_old_batch_mean_and_probability_semantics():
    accumulator = EpochAccumulator(probabilities=True)
    batches = [torch.tensor([[1., 2., 0.], [3., 0., 1.]], requires_grad=True),
               torch.tensor([[0., 1., 3.]], requires_grad=True)]
    targets = [torch.tensor([1, 0]), torch.tensor([1])]
    losses = [nn.functional.cross_entropy(x, y) for x, y in zip(batches, targets)]
    for outputs, labels, loss in zip(batches, targets, losses):
        accumulator.update(outputs, labels, loss)
    assert all(loss.grad_fn is None for loss in accumulator.losses)
    assert all(probability.grad_fn is None for probability in accumulator.probabilities)
    metrics, labels, predictions, probabilities = accumulator.finish()
    old_loss = sum(loss.item() for loss in losses) / len(losses)
    old_labels = torch.cat(targets).numpy()
    old_predictions = torch.cat([x.argmax(1) for x in batches]).numpy()
    assert metrics == classification_metrics(old_labels, old_predictions, old_loss)
    np.testing.assert_array_equal(labels, old_labels)
    np.testing.assert_array_equal(predictions, old_predictions)
    np.testing.assert_array_equal(probabilities, torch.cat([x.detach().softmax(1) for x in batches]).numpy())


def test_empty_accumulator_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        EpochAccumulator().finish()


def test_per_head_log_records_each_split_and_output(tmp_path):
    values = {
        "per_output": [
            {"loss": 1.0, "accuracy": 0.5, "precision": 0.4, "recall": 0.5, "f1": 0.45},
            {"loss": 1.2, "accuracy": 0.4, "precision": 0.3, "recall": 0.4, "f1": 0.35},
        ]
    }
    path = tmp_path / "per_head_training.csv"
    append_per_head_training_log(path, 0, values, values, (8,))
    rows = path.read_text().splitlines()
    assert len(rows) == 5
    assert ",train,final,final," in rows[1]
    assert ",train,exit8,8," in rows[2]
    assert ",validation,final,final," in rows[3]
    assert ",validation,exit8,8," in rows[4]


def test_hook_path_does_not_read_amp_scale_and_cleans_up():
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = GradScaler("cpu", enabled=False)
    scaler.get_scale = lambda: (_ for _ in ()).throw(AssertionError("Unexpected scale synchronization"))
    scheduler = CountingScheduler()
    with OptimizerStepTracker(optimizer) as tracker:
        model(torch.ones(1, 2)).sum().backward()
        assert _step_optimizer_and_scheduler(optimizer, scheduler, scaler, tracker)
        assert scheduler.steps == 1
    assert not optimizer._optimizer_step_post_hooks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real CUDA AMP overflow test")
@pytest.mark.parametrize("fused", [False, True])
def test_real_amp_overflow_does_not_advance_scheduler(fused):
    model = nn.Linear(2, 2).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, fused=fused)
    scaler = GradScaler("cuda", init_scale=8, growth_interval=2)
    scheduler = CountingScheduler()
    with OptimizerStepTracker(optimizer) as tracker:
        assert tracker.supported is (not fused)
        for overflow in (False, True, False, False):
            before = [parameter.detach().clone() for parameter in model.parameters()]
            scaler.scale(model(torch.ones(1, 2, device="cuda")).sum()).backward()
            if overflow:
                next(model.parameters()).grad.fill_(float("inf"))
            result = _step_optimizer_and_scheduler(optimizer, scheduler, scaler, tracker)
            assert result is (not overflow)
            if overflow:
                assert all(torch.equal(x, y) for x, y in zip(before, model.parameters()))
    assert scheduler.steps == 3
