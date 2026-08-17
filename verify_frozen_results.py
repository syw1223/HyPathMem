#!/usr/bin/env python3
"""Verify the SHA-256 hashes recorded in HyPathMem frozen result archives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARCHIVES = (
    ROOT / "results/final/hypathmem_longmemeval_8.5time_v_500",
    ROOT / "results/final/hypathmem_locomo_d0_time_v2_final1540",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(archive: Path) -> tuple[int, int]:
    manifest_path = archive / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    passed = 0
    for relative, metadata in manifest["files"].items():
        path = archive / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen artifact: {path}")
        actual = sha256(path)
        expected = metadata["sha256"]
        if actual != expected:
            raise ValueError(f"Hash mismatch for {path}: {actual} != {expected}")
        passed += 1
    return passed, len(manifest["files"])


def main() -> None:
    total = 0
    for archive in ARCHIVES:
        passed, expected = verify_archive(archive)
        print(f"verified {archive.name}: {passed}/{expected}")
        total += passed
    print(f"verified {total} frozen files")


if __name__ == "__main__":
    main()
