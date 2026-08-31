"""Repository-local wrapper for the packaged training CLI."""

import faulthandler
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

main = import_module("image_classification.cli.train").main


if __name__ == "__main__":
    faulthandler.enable(all_threads=True)
    raise SystemExit(main())
