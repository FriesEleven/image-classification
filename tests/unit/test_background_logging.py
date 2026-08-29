import io
import sys

import torch
from torch import nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset

from image_classification.config import ExperimentConfig
from image_classification.training.engine import _train_epoch


class CountingScheduler:
    def step(self):
        return None


def test_non_interactive_training_does_not_write_batch_progress(monkeypatch, capsys):
    loader = DataLoader(
        TensorDataset(torch.randn(2, 2), torch.tensor([0, 1])),
        batch_size=1,
    )
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    _train_epoch(
        model,
        loader,
        nn.CrossEntropyLoss(),
        optimizer,
        CountingScheduler(),
        GradScaler("cpu", enabled=False),
        ExperimentConfig(epochs=1, accumulation_steps=1, amp=False),
        torch.device("cpu"),
    )

    captured = capsys.readouterr()
    assert "Training:" not in captured.err
