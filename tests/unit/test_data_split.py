from collections import Counter

import pytest

from image_classification.data import (
    stratified_development_split_indices,
    stratified_split_indices,
)


@pytest.mark.parametrize("num_classes", [10, 100])
def test_stratified_split_is_45k_5k_and_balanced(num_classes):
    targets = list(range(num_classes)) * (50_000 // num_classes)

    train_indices, validation_indices = stratified_split_indices(
        targets, validation_size=5000, seed=42,
    )

    assert len(train_indices) == 45_000
    assert len(validation_indices) == 5000
    assert set(train_indices).isdisjoint(validation_indices)
    assert len(set(train_indices) | set(validation_indices)) == 50_000
    train_counts = Counter(targets[index] for index in train_indices)
    validation_counts = Counter(targets[index] for index in validation_indices)
    assert set(train_counts.values()) == {45_000 // num_classes}
    assert set(validation_counts.values()) == {5000 // num_classes}


def test_stratified_split_is_reproducible():
    targets = list(range(10)) * 5000

    first = stratified_split_indices(targets, validation_size=5000, seed=42)
    second = stratified_split_indices(targets, validation_size=5000, seed=42)
    different_seed = stratified_split_indices(targets, validation_size=5000, seed=43)

    assert first == second
    assert first != different_seed


@pytest.mark.parametrize("num_classes", [10, 100])
def test_development_split_is_40k_5k_5k_balanced_and_disjoint(num_classes):
    targets = list(range(num_classes)) * (50_000 // num_classes)

    train, validation, calibration = stratified_development_split_indices(
        targets,
        validation_size=5000,
        calibration_size=5000,
        seed=20_260_902,
    )

    assert [len(train), len(validation), len(calibration)] == [40_000, 5000, 5000]
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(calibration)
    assert set(validation).isdisjoint(calibration)
    assert set(train) | set(validation) | set(calibration) == set(range(50_000))
    for indices, expected in ((train, 40_000), (validation, 5000), (calibration, 5000)):
        counts = Counter(targets[index] for index in indices)
        assert set(counts.values()) == {expected // num_classes}


def test_development_split_is_reproducible_and_seed_sensitive():
    targets = list(range(10)) * 5000
    arguments = (targets, 5000, 5000)

    first = stratified_development_split_indices(*arguments, seed=17)
    second = stratified_development_split_indices(*arguments, seed=17)
    different = stratified_development_split_indices(*arguments, seed=18)

    assert first == second
    assert first != different
