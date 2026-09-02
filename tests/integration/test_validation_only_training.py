import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import image_classification.paths as paths_module
from image_classification.config import ExperimentConfig
from image_classification.training import engine
from image_classification.training.evaluate import validate as real_validate


@pytest.mark.parametrize("measure_inference", [True, False])
def test_validation_only_run_never_iterates_test_loader(monkeypatch, tmp_path, measure_inference):
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
    def inference(_model, _device):
        assert measure_inference, "Shared-GPU training must not measure contended inference latency"
        return {}

    monkeypatch.setattr(engine, "benchmark_inference", inference)
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
        measure_inference=measure_inference,
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
    if not measure_inference:
        benchmark = json.loads((run_directory / "benchmark.json").read_text())
        assert benchmark["measurement_status"] == "skipped"
        assert benchmark["inference_latency_mean"] is None
        assert benchmark["throughput_fps"] is None


def test_formal_run_records_but_never_iterates_calibration_loader(monkeypatch, tmp_path):
    dataset = TensorDataset(
        torch.randn(4, 3, 2, 2),
        torch.tensor([0, 1, 0, 1]),
    )

    class CalibrationDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.indices = [4, 5]

        def __len__(self):
            return 2

        def __getitem__(self, _index):
            raise AssertionError("training iterated policy calibration data")

    loaders = SimpleNamespace(
        train=DataLoader(dataset, batch_size=2),
        validation=DataLoader(dataset, batch_size=2),
        calibration=DataLoader(CalibrationDataset(), batch_size=2),
        test=DataLoader(dataset, batch_size=2),
        class_names=tuple(str(index) for index in range(10)),
    )
    monkeypatch.setattr(paths_module, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        engine,
        "build_model",
        lambda _config: nn.Sequential(nn.Flatten(), nn.Linear(12, 10)),
    )
    monkeypatch.setattr(engine, "build_dataloaders", lambda **_kwargs: loaders)
    monkeypatch.setattr(engine, "model_metrics", lambda _model, _config: {})
    monkeypatch.setattr(engine, "benchmark_inference", lambda *_args: {})
    config = ExperimentConfig(
        experiment_name="formal_split_smoke",
        validation_size=5000,
        calibration_size=5000,
        split_seed=20_260_902,
        epochs=1,
        batch_size=2,
        num_workers=0,
        amp=False,
        evaluate_test=False,
        measure_inference=False,
    )

    summary = engine.train(config)
    split = json.loads(
        (tmp_path / "runs" / config.experiment_id / "split_indices.json").read_text()
    )

    assert summary["calibration_samples"] == 2
    assert split["split_seed"] == 20_260_902
    assert split["training_seed"] == config.seed
    assert split["calibration_indices"] == [4, 5]
