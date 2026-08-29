import pytest
import torch

from image_classification.config import ExperimentConfig
from image_classification.models import build_model


@pytest.mark.parametrize(
    "config",
    [
        ExperimentConfig(model_type="mobilenetv2"),
        ExperimentConfig(model_type="eca"),
        ExperimentConfig(model_type="cbam", aux_positions=(1, 2)),
        ExperimentConfig(model_type="se", aux_positions=(1, 2)),
        ExperimentConfig(model_type="hybrid", se_positions=(1, 2), cbam_positions=(15, 16)),
    ],
)
def test_model_output_shape(config):
    model = build_model(config).eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, config.num_classes)


def test_cifar100_model_has_100_class_output():
    config = ExperimentConfig(model_type="mobilenetv2", dataset="cifar100")
    model = build_model(config).eval()

    with torch.no_grad():
        output = model(torch.randn(2, 3, 32, 32))

    assert output.shape == (2, 100)
