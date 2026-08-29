"""Verify CIFAR train/validation/test boundaries and class balance."""

import argparse
from collections import Counter

from image_classification.data import build_dataloaders


def _class_counts(subset) -> Counter:
    return Counter(subset.dataset.targets[index] for index in subset.indices)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cifar10", "cifar100", "both"), default="both")
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    names = ("cifar10", "cifar100") if args.dataset == "both" else (args.dataset,)

    for name in names:
        loaders = build_dataloaders(
            dataset=name,
            batch_size=128,
            validation_size=args.validation_size,
            split_seed=args.seed,
        )
        train_counts = _class_counts(loaders.train.dataset)
        validation_counts = _class_counts(loaders.validation.dataset)
        assert len(loaders.train.dataset) == 50_000 - args.validation_size
        assert len(loaders.validation.dataset) == args.validation_size
        assert len(loaders.test.dataset) == 10_000
        assert len(train_counts) == len(loaders.class_names)
        assert len(validation_counts) == len(loaders.class_names)
        print(
            f"{name}: train={len(loaders.train.dataset)}, validation={len(loaders.validation.dataset)}, "
            f"test={len(loaders.test.dataset)}, classes={len(loaders.class_names)}, "
            f"train/class={min(train_counts.values())}-{max(train_counts.values())}, "
            f"validation/class={min(validation_counts.values())}-{max(validation_counts.values())}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
