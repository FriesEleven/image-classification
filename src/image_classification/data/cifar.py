"""CIFAR-10/100 datasets with a fixed train/validation/test boundary."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from image_classification.paths import DATA_DIR

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2430, 0.2610)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


@dataclass(frozen=True)
class DatasetSpec:
    dataset_class: type
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    num_classes: int


@dataclass(frozen=True)
class DatasetLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    class_names: tuple[str, ...]


DATASET_SPECS = {
    "cifar10": DatasetSpec(datasets.CIFAR10, CIFAR10_MEAN, CIFAR10_STD, 10),
    "cifar100": DatasetSpec(datasets.CIFAR100, CIFAR100_MEAN, CIFAR100_STD, 100),
}


def stratified_split_indices(
    targets: Sequence[int], validation_size: int = 5000, seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Return deterministic, class-balanced train and validation indices."""

    labels = torch.as_tensor(targets, dtype=torch.long)
    classes = torch.unique(labels, sorted=True)
    if not 0 < validation_size < len(labels):
        raise ValueError("validation_size must be smaller than the training dataset")
    if validation_size < len(classes):
        raise ValueError("validation_size must include at least one sample per class")

    generator = torch.Generator().manual_seed(seed)
    validation_per_class, remainder = divmod(validation_size, len(classes))
    train_parts: list[torch.Tensor] = []
    validation_parts: list[torch.Tensor] = []
    for offset, class_id in enumerate(classes):
        class_indices = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        class_indices = class_indices[torch.randperm(len(class_indices), generator=generator)]
        class_validation_size = validation_per_class + int(offset < remainder)
        if class_validation_size >= len(class_indices):
            raise ValueError("validation split leaves no training samples for a class")
        validation_parts.append(class_indices[:class_validation_size])
        train_parts.append(class_indices[class_validation_size:])

    train_indices = torch.cat(train_parts)
    validation_indices = torch.cat(validation_parts)
    train_indices = train_indices[torch.randperm(len(train_indices), generator=generator)]
    validation_indices = validation_indices[torch.randperm(len(validation_indices), generator=generator)]
    return train_indices.tolist(), validation_indices.tolist()


def _transforms(spec: DatasetSpec) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
            transforms.RandomErasing(p=0.1),
        ]
    )
    evaluation_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(spec.mean, spec.std)]
    )
    return train_transform, evaluation_transform


def build_dataloaders(
    dataset: str,
    batch_size: int,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    validation_size: int = 5000,
    split_seed: int = 42,
) -> DatasetLoaders:
    """Build 45k/5k train/validation loaders and the untouched 10k test loader."""

    try:
        spec = DATASET_SPECS[dataset]
    except KeyError as error:
        raise ValueError(f"Unsupported dataset: {dataset}") from error
    train_transform, evaluation_transform = _transforms(spec)
    training_data = spec.dataset_class(root=DATA_DIR, train=True, download=True, transform=train_transform)
    validation_data = spec.dataset_class(root=DATA_DIR, train=True, download=False, transform=evaluation_transform)
    test_data = spec.dataset_class(root=DATA_DIR, train=False, download=True, transform=evaluation_transform)
    train_indices, validation_indices = stratified_split_indices(
        training_data.targets, validation_size=validation_size, seed=split_seed,
    )

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_options["prefetch_factor"] = prefetch_factor
    shuffle_generator = torch.Generator().manual_seed(split_seed)
    return DatasetLoaders(
        train=DataLoader(
            Subset(training_data, train_indices), shuffle=True, generator=shuffle_generator, **loader_options,
        ),
        validation=DataLoader(Subset(validation_data, validation_indices), shuffle=False, **loader_options),
        test=DataLoader(test_data, shuffle=False, **loader_options),
        class_names=tuple(training_data.classes),
    )
