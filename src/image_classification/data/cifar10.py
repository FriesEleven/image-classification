"""Backward-compatible CIFAR-10 normalization imports."""

from .cifar import CIFAR10_MEAN as MEAN
from .cifar import CIFAR10_STD as STD

__all__ = ["MEAN", "STD"]
