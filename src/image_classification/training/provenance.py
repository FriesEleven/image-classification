"""Exact source fingerprints and runtime receipts for newly executed runs."""

import hashlib
import importlib.metadata
import platform
import shutil
import subprocess
from pathlib import Path

import torch

from image_classification.paths import PROJECT_ROOT


def file_sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def source_fingerprint() -> dict[str, str]:
    files = [PROJECT_ROOT / "pyproject.toml"]
    for directory, extension in (("src", "*.py"), ("scripts", "*.py"), ("configs", "*.yaml")):
        files.extend(sorted((PROJECT_ROOT / directory).rglob(extension)))
    return {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in files}


def snapshot_sources(destination: Path, expected: dict[str, str]) -> None:
    for relative, checksum in expected.items():
        source = PROJECT_ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if file_sha256(target) != checksum:
            raise RuntimeError(f"Source changed during snapshot: {relative}")


def runtime_provenance() -> dict:
    def git(*arguments):
        result = subprocess.run(["git", *arguments], cwd=PROJECT_ROOT, capture_output=True, check=False)
        return result.stdout

    packages = {}
    for name in ("torch", "torchvision", "numpy", "PyYAML", "scikit-learn", "tensorboard"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not_installed"
    return {
        "git_commit": git("rev-parse", "HEAD").decode().strip(),
        "git_status": git("status", "--short").decode().strip(),
        "tracked_source_diff_sha256": hashlib.sha256(
            git("diff", "HEAD", "--", "src", "scripts", "configs", "pyproject.toml")
        ).hexdigest(),
        "source_sha256": source_fingerprint(), "python": platform.python_version(),
        "packages": packages, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }
