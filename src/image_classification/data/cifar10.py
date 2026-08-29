"""CIFAR-10 transforms and data loaders."""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from image_classification.paths import DATA_DIR


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.247, 0.243, 0.261)


def build_dataloaders(batch_size: int, num_workers: int = 0) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing(p=0.1),
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_set = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)
    loader_options = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": True}
    return (
        DataLoader(train_set, shuffle=True, **loader_options),
        DataLoader(test_set, shuffle=False, **loader_options),
    )
