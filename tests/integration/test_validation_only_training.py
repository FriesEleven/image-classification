import json
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import image_classification.paths as paths_module
import image_classification.training.engine as engine
from image_classification.config import ExperimentConfig
from image_classification.training.evaluate import validate as real_validate


def test_validation_only_run_never_iterates_test_loader(monkeypatch, tmp_path):
    dataset = TensorDataset(
        torch.randn(4, 3, 2, 2),
        torch.tensor([0, 1, 0, 1]),
    )
    loaders = SimpleNamespace(
        train=DataLoader(dataset, batch_size=2),
        validation=DataLoader(dataset, batch_size=2),
        test=DataLoader(dataset, batch_size=2),
        class_names=tuple(str(index) for index in range(10)),
    )
    validation_calls = []

    def tracking_validate(model, loader, criterion, device, description="Validation"):
        validation_calls.append(description)
        if description == "Test":
            raise AssertionError("validation-only run accessed the test loader")
        return real_validate(model, loader, criterion, device, description)

    monkeypatch.setattr(paths_module, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(engine, "build_model", lambda _config: nn.Sequential(nn.Flatten(), nn.Linear(12, 10)))
    monkeypatch.setattr(engine, "build_dataloaders", lambda **_kwargs: loaders)
    monkeypatch.setattr(engine, "model_metrics", lambda _model, _config: {})
    monkeypatch.setattr(engine, "benchmark_inference", lambda _model, _device: {})
    monkeypatch.setattr(engine, "validate", tracking_validate)
    monkeypatch.setattr(
        engine,
        "save_evaluation_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("validation-only run exported test predictions")
        ),
    )
    config = ExperimentConfig(
        experiment_name="validation_only",
        epochs=1,
        batch_size=2,
        accumulation_steps=1,
        num_workers=0,
        amp=False,
        evaluate_test=False,
    )

    summary = engine.train(config)
    run_directory = tmp_path / "runs" / config.experiment_id
    saved_summary = json.loads(
        (run_directory / "summary.json").read_text(encoding="utf-8")
    )

    assert validation_calls == ["Validation"]
    assert summary == saved_summary
    assert summary["test_evaluated"] is False
    assert "test_accuracy" not in summary
    assert not (run_directory / "predictions/test.npz").exists()
