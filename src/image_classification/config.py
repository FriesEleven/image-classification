"""Experiment configuration and command-line parsing."""

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

MODEL_TYPES = ("mobilenetv2", "eca", "cbam", "se", "hybrid", "csgha", "hybrid_leaky", "csgha_v4")
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
    batch_size: int = 128
    epochs: int = 200
    lr: float = 0.01
    amp: bool = True
    cuda_graph: bool = False
    torch_num_threads: int = 0
    measure_inference: bool = True
    evaluate_test: bool = True
    accumulation_steps: int = 1
    aux_positions: tuple[int, ...] = ()
    se_positions: tuple[int, ...] = ()
    cbam_positions: tuple[int, ...] = ()
    guidance_position: int = 2
    guidance_reduction: int = 4
    num_workers: int = 8
    prefetch_factor: int = 4
    seed: int = 42

    def __post_init__(self) -> None:
        if self.model_type not in MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        if self.dataset not in DATASETS:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        if not 0 < self.validation_size < 50_000:
            raise ValueError("validation_size must be between 1 and 49,999")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.accumulation_steps < 1:
            raise ValueError("accumulation_steps must be at least 1")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.torch_num_threads < 0:
            raise ValueError("torch_num_threads cannot be negative; 0 uses the runtime default")
        if self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be at least 1")
        if self.guidance_reduction < 1:
            raise ValueError("guidance_reduction must be at least 1")
        positions = self.aux_positions + self.se_positions + self.cbam_positions
        if any(position < 0 or position > 18 for position in positions):
            raise ValueError("MobileNetV2 attention positions must be between 0 and 18")
        if self.model_type in {"csgha", "csgha_v4"}:
            if not self.se_positions or not self.cbam_positions:
                raise ValueError("CSGHA requires both SE and guided CBAM positions")
            if self.guidance_position not in self.se_positions:
                raise ValueError("guidance_position must be one of the CSGHA SE positions")
            if any(position <= self.guidance_position for position in self.cbam_positions):
                raise ValueError("CSGHA CBAM positions must follow guidance_position")

    @property
    def num_classes(self) -> int:
        return DATASET_NUM_CLASSES[self.dataset]

    @property
    def architecture_version(self) -> str:
        return {
            "csgha": "csgha_v3_bounded_relu",
            "csgha_v4": "csgha_v4_bounded_deep_leaky_relu_0.1",
            "hybrid_leaky": "independent_hybrid_deep_leaky_relu_0.1",
        }.get(self.model_type, f"{self.model_type}_v1")

    @property
    def experiment_id(self) -> str:
        if self.model_type in {"csgha", "csgha_v4"}:
            se = "-".join(map(str, self.se_positions))
            cbam = "-".join(map(str, self.cbam_positions))
            return (
                f"{self.experiment_name}_{self.model_type}_se{se}_guide{self.guidance_position}_"
                f"cbam{cbam}_{self.dataset}"
            )
        if self.model_type in {"hybrid", "hybrid_leaky"}:
            se = "-".join(map(str, self.se_positions))
            cbam = "-".join(map(str, self.cbam_positions))
            return f"{self.experiment_name}_{self.model_type}_se{se}_cbam{cbam}_{self.dataset}"
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
    parser.add_argument("--cuda_graph", type=_boolean)
    parser.add_argument("--torch_num_threads", type=int)
    parser.add_argument("--measure_inference", type=_boolean)
    parser.add_argument("--evaluate_test", type=_boolean)
    parser.add_argument("--accumulation_steps", type=int)
    parser.add_argument("--aux_positions")
    parser.add_argument("--se_positions")
    parser.add_argument("--cbam_positions")
    parser.add_argument("--guidance_position", type=int)
    parser.add_argument("--guidance_reduction", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--prefetch_factor", type=int)
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
