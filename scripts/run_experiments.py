"""Cross-platform sequential experiment launcher."""

import argparse
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, default=ROOT / "configs/sweeps/all_experiments.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    with args.sweep.open(encoding="utf-8") as handle:
        experiments = (yaml.safe_load(handle) or {}).get("experiments", [])
    if not experiments:
        raise ValueError(f"No experiments listed in {args.sweep}")
    failures = 0
    for config in experiments:
        config_path = ROOT / config
        command = [sys.executable, str(ROOT / "scripts/train.py"), "--config", str(config_path)]
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            failures += 1
            if not args.continue_on_error:
                return completed.returncode
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
