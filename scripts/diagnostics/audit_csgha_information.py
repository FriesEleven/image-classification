"""Reproduce historical checkpoints and test guidance information on validation."""

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from image_classification.diagnostics.guidance import diagnose_run, sha256
from image_classification.paths import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("reports/audits/2026-08-30/audit_results.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", default=["E07", "E08", "E09", "E10", "E11"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = PROJECT_ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Use a new diagnostic output directory: {output}")
    audit_path = PROJECT_ROOT / args.audit
    audit = json.loads(audit_path.read_text())
    rows = [row for row in audit["runs"] if row["id"] in args.ids]
    if len(rows) != len(set(args.ids)) or any(not row["variant"].startswith("CSGHA v") for row in rows):
        raise ValueError("Select audited CSGHA runs only")
    torch.set_num_threads(4)
    torch.manual_seed(7300)
    output.mkdir(parents=True)
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "audit_sha256": sha256(audit_path),
        "python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "diagnostic_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "diagnostic_sources_sha256": {
            name: sha256(PROJECT_ROOT / name) for name in (
                "scripts/diagnostics/audit_csgha_information.py",
                "src/image_classification/diagnostics/guidance.py",
            )
        },
        "status": "running", "runs": [],
    }
    manifest = output / "manifest.json"
    try:
        for row in rows:
            metadata["runs"].append(diagnose_run(row, output, workers=args.workers))
            manifest.write_text(json.dumps(metadata, indent=2) + "\n")
        metadata["status"] = "completed"
    except BaseException:
        metadata["status"] = "failed"
        raise
    finally:
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved: {manifest}", flush=True)


if __name__ == "__main__":
    main()
