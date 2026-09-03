"""Strict loader for the frozen CIFAR-10.1 v6 external test set."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from image_classification.data.cifar import CIFAR10_MEAN, CIFAR10_STD
from image_classification.paths import DATA_DIR
from image_classification.training.provenance import file_sha256

CIFAR10_1_V6_ROOT = DATA_DIR / "cifar-10.1-v6"
CIFAR10_1_V6_DATA_FILE = "cifar10.1_v6_data.npy"
CIFAR10_1_V6_LABELS_FILE = "cifar10.1_v6_labels.npy"
CIFAR10_1_V6_DATA_SHA256 = "2997188e5816f5bd545dc77771b6227828c28146049fcecf3fa10775474cacc6"
CIFAR10_1_V6_LABELS_SHA256 = "ae40beda001693674edc94d925ee8268cfe68905f8f9aff800c8dcdfcd6c9448"
CIFAR10_1_REPOSITORY_COMMIT = "d9982abb0bfc4846b8d13a11e66b887d946205d0"
CIFAR10_CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def validate_cifar10_1_v6_arrays(images: np.ndarray, labels: np.ndarray) -> None:
    """Reject any array that is not the exact balanced v6 schema."""

    if images.shape != (2000, 32, 32, 3) or images.dtype != np.uint8:
        raise ValueError("CIFAR-10.1 v6 images must be uint8 with shape (2000, 32, 32, 3)")
    if labels.shape != (2000,) or labels.dtype != np.int32:
        raise ValueError("CIFAR-10.1 v6 labels must be int32 with shape (2000,)")
    if int(images.min()) < 0 or int(images.max()) > 255:
        raise ValueError("CIFAR-10.1 v6 image values must be in [0, 255]")
    if labels.min() != 0 or labels.max() != 9:
        raise ValueError("CIFAR-10.1 v6 labels must cover class IDs 0 through 9")
    if np.bincount(labels, minlength=10).tolist() != [200] * 10:
        raise ValueError("CIFAR-10.1 v6 must contain exactly 200 examples per class")


def load_cifar10_1_v6(root: Path = CIFAR10_1_V6_ROOT) -> tuple[np.ndarray, np.ndarray]:
    """Load only hash-pinned official v6 arrays with pickle disabled."""

    data_path = root / CIFAR10_1_V6_DATA_FILE
    labels_path = root / CIFAR10_1_V6_LABELS_FILE
    expected = {
        data_path: CIFAR10_1_V6_DATA_SHA256,
        labels_path: CIFAR10_1_V6_LABELS_SHA256,
    }
    for path, checksum in expected.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen CIFAR-10.1 v6 file: {path}")
        if file_sha256(path) != checksum:
            raise ValueError(f"CIFAR-10.1 v6 checksum mismatch: {path}")
    images = np.load(data_path, allow_pickle=False)
    labels = np.load(labels_path, allow_pickle=False)
    validate_cifar10_1_v6_arrays(images, labels)
    return images, labels


def build_cifar10_1_v6_loader(
    root: Path = CIFAR10_1_V6_ROOT,
    *,
    batch_size: int = 128,
) -> DataLoader:
    """Build a deterministic normalized loader without touching CIFAR-10 test data."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    images, labels = load_cifar10_1_v6(root)
    inputs = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float()
    inputs.div_(255.0)
    mean = torch.tensor(CIFAR10_MEAN, dtype=inputs.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, dtype=inputs.dtype).view(1, 3, 1, 1)
    inputs.sub_(mean).div_(std)
    targets = torch.from_numpy(np.ascontiguousarray(labels)).long()
    return DataLoader(
        TensorDataset(inputs, targets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
