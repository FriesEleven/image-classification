"""Dataset loaders."""

from .cifar import (
    DatasetLoaders,
    LockedTestDataset,
    build_dataloaders as build_cifar_dataloaders,
    stratified_development_split_indices,
    stratified_split_indices,
)
from .imagenet100 import build_imagenet100_dataloaders


def build_dataloaders(dataset: str, **kwargs) -> DatasetLoaders:
    if dataset == "imagenet100":
        return build_imagenet100_dataloaders(**kwargs)
    return build_cifar_dataloaders(dataset=dataset, **kwargs)


# Import the historical external set only after the dispatcher exists.  That
# module imports provenance through image_classification.training, whose package
# initializer imports the training engine and then this dispatcher.
from .cifar10_1 import build_cifar10_1_v6_loader, load_cifar10_1_v6  # noqa: E402

__all__ = [
    "DatasetLoaders",
    "LockedTestDataset",
    "build_cifar10_1_v6_loader",
    "build_dataloaders",
    "load_cifar10_1_v6",
    "stratified_development_split_indices",
    "stratified_split_indices",
]
