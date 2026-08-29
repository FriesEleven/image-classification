"""Repository-local wrapper for the packaged training CLI."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_classification.cli.train import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
