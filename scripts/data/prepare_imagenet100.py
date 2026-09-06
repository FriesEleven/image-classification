"""Safely prepare the frozen public ImageNet-100 training pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from image_classification.paths import DATA_DIR

PROTOCOL = ROOT / "reports/experiments/2026-09-06-early-exit-p7-imagenet100-design/protocol_manifest.json"
DEFAULT_SOURCE = Path("/root/autodl-pub/ImageNet100/imagenet100.zip")
DEFAULT_DESTINATION = DATA_DIR / "imagenet100"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def archive_inventory(archive: zipfile.ZipFile) -> tuple[list[zipfile.ZipInfo], list[dict]]:
    images = []
    records = []
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if info.is_dir():
            continue
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe archive path: {info.filename}")
        if len(path.parts) != 3 or path.parts[0] != "imagenet100":
            raise ValueError(f"Unexpected ImageNet-100 archive layout: {info.filename}")
        if path.suffix.lower() not in {".jpeg", ".jpg", ".png"}:
            raise ValueError(f"Unexpected non-image member: {info.filename}")
        images.append(info)
        records.append(
            {
                "path": "/".join(path.parts[1:]),
                "size": info.file_size,
                "crc32": f"{info.CRC:08x}",
            }
        )
    return images, sorted(records, key=lambda row: row["path"])


def execute(source: Path, destination: Path) -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    contract = protocol["dataset"]
    if destination.exists():
        manifest_path = destination / "manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError(f"Refusing to overwrite incomplete destination: {destination}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_archive_sha256") != contract["source_archive_sha256"]:
            raise ValueError("Existing prepared data has a different source hash")
        print(json.dumps(manifest, indent=2))
        return manifest
    actual_archive_hash = sha256(source)
    if actual_archive_hash != contract["source_archive_sha256"]:
        raise ValueError("Public ImageNet-100 archive hash differs from the frozen protocol")
    with zipfile.ZipFile(source) as archive:
        images, records = archive_inventory(archive)
        counts = Counter(PurePosixPath(info.filename).parts[1] for info in images)
        classes = sorted(counts)
        if classes != contract["class_names"]:
            raise ValueError("Archive class list differs from the frozen protocol")
        if len(images) != contract["image_count"] or sum(counts.values()) != contract["image_count"]:
            raise ValueError("Archive image count differs from the frozen protocol")
        central_hash = canonical_sha256(records)
        if central_hash != contract["central_directory_sha256"]:
            raise ValueError("Archive inventory differs from the frozen protocol")
        staging = destination.with_name(f".{destination.name}.partial")
        if staging.exists():
            raise FileExistsError(f"Inspect incomplete staging directory before retrying: {staging}")
        pool = staging / "pool"
        pool.mkdir(parents=True)
        for index, info in enumerate(images, start=1):
            source_path = PurePosixPath(info.filename)
            target = pool.joinpath(*source_path.parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as input_handle, target.open("xb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            if index % 5000 == 0:
                print(f"Extracted {index}/{len(images)}", flush=True)
    manifest = {
        "schema_version": 1,
        "status": "prepared_train_pool",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_archive": str(source),
        "source_archive_sha256": actual_archive_hash,
        "central_directory_sha256": central_hash,
        "image_count": len(images),
        "total_uncompressed_bytes": sum(info.file_size for info in images),
        "class_names": classes,
        "class_counts": {name: counts[name] for name in classes},
        "official_test_prepared": False,
        "official_test_accessed": False,
        "protocol_sha256": sha256(PROTOCOL),
    }
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(staging, destination)
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    execute(args.source.resolve(), args.destination.resolve())
