"""Centralized repository and experiment paths."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class RunPaths:
    """All generated files belonging to one experiment run."""

    experiment_id: str

    @property
    def root(self) -> Path:
        return ARTIFACTS_DIR / "runs" / self.experiment_id

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def tensorboard(self) -> Path:
        return self.root / "logs" / "tensorboard"

    @property
    def training_log(self) -> Path:
        return self.root / "logs" / "training.csv"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions"

    def create(self) -> "RunPaths":
        for path in (self.root, self.checkpoints, self.tensorboard, self.training_log.parent, self.predictions):
            path.mkdir(parents=True, exist_ok=True)
        return self
