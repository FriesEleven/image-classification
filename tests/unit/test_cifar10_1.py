from pathlib import Path

import numpy as np
import pytest
import torch

from image_classification.data.cifar import CIFAR10_MEAN, CIFAR10_STD
from image_classification.data.cifar10_1 import (
    CIFAR10_1_V6_DATA_FILE,
    CIFAR10_1_V6_LABELS_FILE,
    build_cifar10_1_v6_loader,
    validate_cifar10_1_v6_arrays,
)


def _valid_arrays():
    images = np.zeros((2000, 32, 32, 3), dtype=np.uint8)
    labels = np.repeat(np.arange(10, dtype=np.int32), 200)
    return images, labels


def test_cifar10_1_v6_schema_requires_exact_class_balance():
    images, labels = _valid_arrays()
    validate_cifar10_1_v6_arrays(images, labels)

    labels[0] = 1
    with pytest.raises(ValueError, match="200 examples per class"):
        validate_cifar10_1_v6_arrays(images, labels)


@pytest.mark.parametrize(
    ("images", "labels", "message"),
    [
        (np.zeros((1, 32, 32, 3), dtype=np.uint8), np.zeros(2000, dtype=np.int32), "images"),
        (np.zeros((2000, 32, 32, 3), dtype=np.float32), np.zeros(2000, dtype=np.int32), "images"),
        (np.zeros((2000, 32, 32, 3), dtype=np.uint8), np.zeros(2000, dtype=np.int64), "labels"),
    ],
)
def test_cifar10_1_v6_schema_rejects_wrong_shapes_or_dtypes(images, labels, message):
    with pytest.raises(ValueError, match=message):
        validate_cifar10_1_v6_arrays(images, labels)


def test_cifar10_1_v6_loader_uses_cifar10_normalization(monkeypatch, tmp_path: Path):
    images, labels = _valid_arrays()
    data_path = tmp_path / CIFAR10_1_V6_DATA_FILE
    labels_path = tmp_path / CIFAR10_1_V6_LABELS_FILE
    np.save(data_path, images, allow_pickle=False)
    np.save(labels_path, labels, allow_pickle=False)

    from image_classification.data import cifar10_1

    hashes = {
        data_path: cifar10_1.file_sha256(data_path),
        labels_path: cifar10_1.file_sha256(labels_path),
    }
    monkeypatch.setattr(cifar10_1, "CIFAR10_1_V6_DATA_SHA256", hashes[data_path])
    monkeypatch.setattr(cifar10_1, "CIFAR10_1_V6_LABELS_SHA256", hashes[labels_path])
    inputs, targets = next(iter(build_cifar10_1_v6_loader(tmp_path, batch_size=16)))

    expected = -torch.tensor(CIFAR10_MEAN) / torch.tensor(CIFAR10_STD)
    torch.testing.assert_close(inputs[0, :, 0, 0], expected)
    assert inputs.shape == (16, 3, 32, 32)
    assert targets.dtype == torch.long
