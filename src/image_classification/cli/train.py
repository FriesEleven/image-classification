"""Training command."""

import json
from typing import Sequence

from image_classification.config import load_config
from image_classification.training import train


def main(argv: Sequence[str] | None = None) -> int:
    summary = train(load_config(argv))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
