"""Run version-matched v5/control checkpoint diagnostics; never train or open test data."""

import argparse
import json
import platform
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from image_classification.diagnostics.guidance import sha256
from image_classification.diagnostics.guidance_v4 import (
    diagnose_retry1_run,
    preflight_retry1_run,
)

DEFAULT_AUDIT = Path("reports/audits/2026-08-31-csgha-v5-serial-s1/audit_results.json")
DEFAULT_OUTPUT = Path("artifacts/diagnostics/csgha_v5_serial_information_20260831_v1")
DIAGNOSTIC_SOURCES = (
    "scripts/diagnostics/audit_csgha_v5_serial.py",
    "src/image_classification/diagnostics/guidance_v4.py",
    "src/image_classification/diagnostics/guidance.py",
)


def selected_rows(audit: dict) -> list[dict]:
    rows = sorted(audit["runs"], key=lambda row: (row["model_type"], row["seed"]))
    counts = Counter((row["model_type"], row["seed"]) for row in rows)
    expected = Counter((model, seed) for model in ("hybrid_leaky", "csgha_v5") for seed in (42, 43, 44))
    if counts != expected or any(row["issues"] for row in rows):
        raise ValueError("P2 requires exactly six issue-free P1 rows")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    audit_path = ROOT / args.audit
    output = ROOT / args.output
    audit = json.loads(audit_path.read_text())
    rows = selected_rows(audit)
    source_snapshot = (ROOT / audit["manifest_path"]).parent / "source_snapshot"
    plan = [{
        "run_id": row["run_id"], "model_type": row["model_type"], "seed": row["seed"],
        "conditions": ["original", "deep_zero"] + (
            ["guidance_zero", "train_mean_descriptor", "train_mean_contribution",
             "permuted_7301", "permuted_7302", "permuted_7303"]
            if row["model_type"] == "csgha_v5" else []
        ),
    } for row in rows]
    preflight = []
    for row in rows:
        verified = preflight_retry1_run(row, source_snapshot)
        preflight.append({
            "run_id": row["run_id"],
            "strict_state_dict_load": True,
            "architecture_version": verified["config"].architecture_version,
            "verified_training_sources": verified["verified_sources"],
        })
    if args.dry_run:
        if output.exists():
            raise FileExistsError(output)
        print(json.dumps({
            "audit": str(audit_path), "audit_sha256": sha256(audit_path),
            "source_snapshot": str(source_snapshot), "output": str(output),
            "official_test_evaluated": False, "training": False,
            "preflight": preflight, "plan": plan,
        }, indent=2))
        return 0
    if output.exists():
        raise FileExistsError(f"Use a new diagnostic output directory: {output}")
    output.mkdir(parents=True)
    source_directory = output / "source"
    source_hashes = {}
    for relative in DIAGNOSTIC_SOURCES:
        source = ROOT / relative
        target = source_directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_hashes[relative] = sha256(source)
    torch.set_num_threads(4)
    torch.manual_seed(7300)
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit": str(audit_path), "audit_sha256": sha256(audit_path),
        "source_snapshot": str(source_snapshot),
        "python": platform.python_version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "diagnostic_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "diagnostic_sources_sha256": source_hashes,
        "official_test_evaluated": False, "training": False,
        "status": "running", "runs": [],
    }
    manifest = output / "manifest.json"
    try:
        for row in rows:
            print(f"Diagnosing {row['model_type']} seed={row['seed']}", flush=True)
            result = diagnose_retry1_run(
                row, output, source_snapshot, batch_size=args.batch_size, workers=args.workers,
            )
            metadata["runs"].append(result)
            manifest.write_text(json.dumps(metadata, indent=2) + "\n")
        metadata["status"] = "completed"
    except BaseException:
        metadata["status"] = "failed"
        raise
    finally:
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved completed diagnostics: {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
