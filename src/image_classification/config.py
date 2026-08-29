"""Experiment configuration and command-line parsing."""

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


MODEL_TYPES = ("mobilenetv2", "eca", "cbam", "se", "hybrid")
DATASETS = ("cifar10", "cifar100")
DATASET_NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


def _positions(value: str | Sequence[int] | None) -> tuple[int, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    return tuple(int(item) for item in value)


def _boolean(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str = "baseline"
    model_type: str = "mobilenetv2"
    dataset: str = "cifar10"
    validation_size: int = 5000
    batch_size: int = 64
    epochs: int = 100
    lr: float = 0.01
    amp: bool = True
    accumulation_steps: int = 2
    aux_positions: tuple[int, ...] = ()
    se_positions: tuple[int, ...] = ()
    cbam_positions: tuple[int, ...] = ()
    num_workers: int = 0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.model_type not in MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        if self.dataset not in DATASETS:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        if not 0 < self.validation_size < 50_000:
            raise ValueError("validation_size must be between 1 and 49,999")

    @property
    def num_classes(self) -> int:
        return DATASET_NUM_CLASSES[self.dataset]

    @property
    def experiment_id(self) -> str:
        if self.model_type == "hybrid":
            se = "-".join(map(str, self.se_positions))
            cbam = "-".join(map(str, self.cbam_positions))
            return f"{self.experiment_name}_hybrid_se{se}_cbam{cbam}_{self.dataset}"
        if self.model_type in {"cbam", "se"} and self.aux_positions:
            positions = "-".join(map(str, self.aux_positions))
            return f"{self.experiment_name}_{self.model_type}_pos{positions}_{self.dataset}"
        return f"{self.experiment_name}_{self.model_type}_{self.dataset}"

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in ("aux_positions", "se_positions", "cbam_positions"):
            data[key] = list(data[key])
        return data


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CIFAR attention models")
    parser.add_argument("--config", type=Path, help="YAML experiment definition")
    parser.add_argument("--experiment_name")
    parser.add_argument("--model_type", choices=MODEL_TYPES)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--validation_size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--amp", type=_boolean)
    parser.add_argument("--accumulation_steps", type=int)
    parser.add_argument("--aux_positions")
    parser.add_argument("--se_positions")
    parser.add_argument("--cbam_positions")
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--seed", type=int)
    return parser


def load_config(argv: Sequence[str] | None = None) -> ExperimentConfig:
    args = vars(_parser().parse_args(argv))
    config_path = args.pop("config")
    values: dict = {}
    if config_path:
        with config_path.open(encoding="utf-8") as handle:
            values.update(yaml.safe_load(handle) or {})
    values.update({key: value for key, value in args.items() if value is not None})
    for key in ("aux_positions", "se_positions", "cbam_positions"):
        values[key] = _positions(values.get(key))
    config = ExperimentConfig(**values)
    return config
