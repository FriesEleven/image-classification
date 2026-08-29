"""Backward-compatible training entry point.

New code should import from ``image_classification`` or use ``scripts/train.py``.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from image_classification.cli.train import main  # noqa: E402
from image_classification.models import (  # noqa: E402,F401
    BaseMobileNetV2,
    CBAMMobileNetV2,
    ECAMobileNetV2,
    HybridAttentionMobileNetV2,
    SEMobileNetV2,
)

# Compatibility aliases used by older diagnostics and notebooks.
ECA_MobileNetV2 = ECAMobileNetV2
ECA_CBAM_MobileNetV2 = CBAMMobileNetV2
ECA_SE_MobileNetV2 = SEMobileNetV2


if __name__ == "__main__":
    raise SystemExit(main())
