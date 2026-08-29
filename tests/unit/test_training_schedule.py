import torch
from torch import nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset

from image_classification.config import ExperimentConfig
from image_classification.training.engine import _train_epoch, optimizer_updates_per_epoch


class CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def test_optimizer_updates_include_partial_accumulation_group():
    assert optimizer_updates_per_epoch(num_batches=5, accumulation_steps=2) == 3
    assert optimizer_updates_per_epoch(num_batches=6, accumulation_steps=2) == 3


def test_scheduler_steps_after_each_optimizer_update():
    loader = DataLoader(
        TensorDataset(torch.randn(5, 2), torch.tensor([0, 1, 0, 1, 0])),
        batch_size=1,
    )
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = CountingScheduler()
    config = ExperimentConfig(epochs=1, accumulation_steps=2, amp=False)

    _train_epoch(
        model,
        loader,
        nn.CrossEntropyLoss(),
        optimizer,
        scheduler,
        GradScaler("cpu", enabled=False),
        config,
        torch.device("cpu"),
    )

    assert scheduler.steps == 3
