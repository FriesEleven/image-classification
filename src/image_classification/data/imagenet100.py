"""Audited ImageNet-100 development loaders with a withheld official validation subset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from image_classification.paths import DATA_DIR

from .cifar import DatasetLoaders, LockedTestDataset, stratified_development_split_indices

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXPECTED_TRAIN_IMAGES = 126_689
EXPECTED_CLASSES = 100
LOCKED_TEST_IMAGES = 5_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _transforms() -> tuple[transforms.Compose, transforms.Compose]:
    training = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224, scale=(0.08, 1.0), interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.1),
        ]
    )
    evaluation = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return training, evaluation


def _read_manifest(root: Path) -> tuple[Path, dict]:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared ImageNet-100 manifest is missing: {path}. "
            "Run scripts/data/prepare_imagenet100.py first."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "prepared_train_pool":
        raise ValueError("ImageNet-100 train pool is not in the prepared state")
    if manifest.get("image_count") != EXPECTED_TRAIN_IMAGES:
        raise ValueError("ImageNet-100 image count differs from the frozen protocol")
    if len(manifest.get("class_names", [])) != EXPECTED_CLASSES:
        raise ValueError("ImageNet-100 class list differs from the frozen protocol")
    return path, manifest


def build_imagenet100_dataloaders(
    batch_size: int,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    validation_size: int = 10_000,
    split_seed: int = 20_261_003,
    calibration_size: int = 10_000,
    shuffle_seed: int | None = None,
    include_test: bool = False,
) -> DatasetLoaders:
    """Build fixed ImageNet-100 train/model-selection/policy loaders."""

    root = DATA_DIR / "imagenet100"
    manifest_path, manifest = _read_manifest(root)
    pool = root / "pool"
    training_transform, evaluation_transform = _transforms()
    training_data = datasets.ImageFolder(pool, transform=training_transform)
    evaluation_data = datasets.ImageFolder(pool, transform=evaluation_transform)
    class_names = tuple(manifest["class_names"])
    if tuple(training_data.classes) != class_names:
        raise ValueError("Extracted ImageNet-100 class order differs from the manifest")
    if len(training_data) != EXPECTED_TRAIN_IMAGES:
        raise ValueError("Extracted ImageNet-100 pool is incomplete")
    train_indices, validation_indices, calibration_indices = stratified_development_split_indices(
        training_data.targets,
        validation_size=validation_size,
        calibration_size=calibration_size,
        seed=split_seed,
    )
    options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        options["prefetch_factor"] = prefetch_factor
    generator = torch.Generator().manual_seed(split_seed if shuffle_seed is None else shuffle_seed)
    if include_test:
        test_root = root / "locked_test"
        if not test_root.is_dir():
            raise FileNotFoundError("Locked ImageNet-100 test has not been prepared")
        test_data = datasets.ImageFolder(test_root, transform=evaluation_transform)
        if tuple(test_data.classes) != class_names or len(test_data) != LOCKED_TEST_IMAGES:
            raise ValueError("Locked ImageNet-100 test does not match the frozen class boundary")
    else:
        test_data = LockedTestDataset(LOCKED_TEST_IMAGES)
    return DatasetLoaders(
        train=DataLoader(
            Subset(training_data, train_indices), shuffle=True, generator=generator, **options,
        ),
        validation=DataLoader(
            Subset(evaluation_data, validation_indices), shuffle=False, **options,
        ),
        calibration=DataLoader(
            Subset(evaluation_data, calibration_indices), shuffle=False, **options,
        ),
        test=DataLoader(test_data, shuffle=False, **options),
        class_names=class_names,
        dataset_manifest={
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "source_archive_sha256": manifest["source_archive_sha256"],
            "central_directory_sha256": manifest["central_directory_sha256"],
            "official_test_enumerated": include_test,
        },
    )
